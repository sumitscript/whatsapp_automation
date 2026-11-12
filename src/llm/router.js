// Chooses where each model call runs: Groq (several API keys, rotated) or the local Ollama model.
//
//   llmProvider = 'auto'    Groq whenever a key is usable; a rate-limited key is skipped instantly and the next key is
//                           used; when every key is exhausted (or Groq is down, or no key was added) the local model
//                           answers. Adding a key is all it takes: no other switch to flip.
//   llmProvider = 'ollama'  every call runs locally (Groq is never contacted)
//   llmProvider = 'groq'    Groq only: when no key can answer the call fails with GroqUnavailableError and the worker
//                           sends the "on a short break" message (setting groqBusyMessage) instead of a reply.
// Background work (memory folds, agenda extraction) stays on the local model while it is ready so it never spends Groq
// quota; in Groq-only mode with the local model not ready it uses Groq. Think-ahead is skipped on the Groq route.
// Exposes the same surface the pipeline uses on an LLM client: numCtx, chat(), prefill(), reusesPrefix().
import { GroqUnavailableError } from './groq.js';

export const PROVIDERS = ['auto', 'ollama', 'groq'];

export class RoutedLlm {
  /**
   * @param {{ ollama: object, groq: object, getSettings: () => object, gates: { ollama: object, groq: object },
   *           localReady?: () => boolean, logger?: object }} o
   */
  constructor({ ollama, groq, getSettings, gates, localReady = () => true, logger }) {
    Object.assign(this, { ollama, groq, getSettings, gates, localReady });
    this.log = logger;
    this.counters = { groq: 0, local: 0, fallbacks: 0, busy: 0 };
    this.lastFallback = null;
    this.lastLog = new Map();
  }

  get numCtx() {
    return this.ollama.numCtx;
  }

  /** Settings for one call: the request's own provider choice (playground, tests) wins over the live setting. */
  #settingsFor(req = {}) {
    const s = this.getSettings();
    const p = req.provider ?? s.llmProvider;
    return { llmProvider: PROVIDERS.includes(p) ? p : 'auto', groqModel: req.groqModel ?? s.groqModel };
  }

  /** Should this call try Groq at all? */
  #wantsGroq(s, priority = 1) {
    if (s.llmProvider === 'ollama') return false;
    // Background work stays local while the local model is up: no Groq quota spent on summaries.
    if (priority >= 3 && this.localReady()) return false;
    // Auto with no key at all is simply local; Groq-only with no key must say so (busy message), not silently go local.
    return s.llmProvider === 'groq' || this.groq.pool.size > 0;
  }

  /** Is Groq selected (auto or Groq only) and at least one key ready right now? */
  groqUsable() {
    const s = this.getSettings();
    return s.llmProvider !== 'ollama' && this.groq.usable();
  }

  /** Does a reply need the local model to be loaded (it will be, or may be, answered locally)? */
  needsLocal() {
    const s = this.getSettings();
    if (s.llmProvider === 'ollama') return true;
    if (s.llmProvider === 'groq') return false;
    return !this.groq.usable(); // auto: local only when no Groq key can serve right now
  }

  /** Pre-reading a prompt only helps a local model that keeps a prefix cache. */
  reusesPrefix(model) {
    return !this.#wantsGroq(this.#settingsFor(), 1) && this.ollama.reusesPrefix(model);
  }

  #note(kind, message, extra = {}) {
    const now = Date.now();
    if (now - (this.lastLog.get(kind) || 0) < 30_000) return;
    this.lastLog.set(kind, now);
    this.log?.warn(message, extra);
  }

  async prefill(req) {
    // Think-ahead only helps the local model's prefix cache; live replies on Groq gain nothing from it.
    if (this.#settingsFor(req).llmProvider !== 'ollama' && this.groq.pool.size > 0) return { promptTokens: 0, promptMs: 0, loadMs: 0, totalMs: 0, skipped: true };
    return this.gates.ollama.run(() => this.ollama.prefill(req), { priority: 4, signal: req.signal });
  }

  async *chat(req) {
    const s = this.#settingsFor(req);
    const priority = req.priority ?? (req.maxTokens > 450 ? 2 : 1);
    if (this.#wantsGroq(s, priority)) {
      const strict = s.llmProvider === 'groq';
      if (this.groq.usable()) {
        let started = false;
        try {
          await this.gates.groq.acquire({ priority, signal: req.signal });
          try {
            yield { type: 'meta', provider: 'groq', model: s.groqModel };
            for await (const ev of this.groq.chat({ ...req, model: s.groqModel })) {
              if (ev.type === 'token') started = true;
              yield ev;
            }
            this.counters.groq++;
            return;
          } finally {
            this.gates.groq.release();
          }
        } catch (err) {
          if (req.signal?.aborted || started) throw err; // cancelled, or half a reply already went out: cannot switch mid-sentence
          if (strict) return this.#busy(err);
          this.#fellBack(err);
        }
      } else if (strict) {
        return this.#busy(new GroqUnavailableError(this.groq.pool.size ? 'every Groq key is rate-limited right now' : 'no Groq key configured', { kind: this.groq.pool.size ? 'exhausted' : 'no_keys', retryAfterMs: this.groq.pool.nextAvailableMs() }));
      } else {
        this.#fellBack(new GroqUnavailableError('every Groq key is rate-limited right now', { kind: 'exhausted', retryAfterMs: this.groq.pool.nextAvailableMs() }));
      }
    }
    yield { type: 'meta', provider: 'ollama', model: req.model };
    await this.gates.ollama.acquire({ priority, signal: req.signal });
    try {
      yield* this.ollama.chat(req);
      this.counters.local++;
    } finally {
      this.gates.ollama.release();
    }
  }

  /** Groq-only mode could not answer: the caller turns this into the "short break" message. */
  #busy(err) {
    this.counters.busy++;
    this.lastFallback = { at: Date.now(), reason: String(err.message || err).slice(0, 160), kind: err.kind || 'error', busy: true };
    this.#note('busy', 'Groq-only mode: Groq cannot answer; the contact gets the short-break message', { reason: this.lastFallback.reason, retryInMs: err.retryAfterMs ?? null });
    throw err instanceof GroqUnavailableError ? err : new GroqUnavailableError(String(err.message || err), { kind: 'network' });
  }

  #fellBack(err) {
    this.counters.fallbacks++;
    this.lastFallback = { at: Date.now(), reason: String(err.message || err).slice(0, 160), kind: err.kind || 'error' };
    this.#note('fallback', 'Groq unavailable; answering with the local model', { reason: this.lastFallback.reason, retryInMs: err.retryAfterMs ?? null });
  }

  status() {
    const s = this.getSettings();
    return {
      provider: s.llmProvider,
      groqModel: s.groqModel,
      groqReady: this.groq.usable(),
      counters: { ...this.counters },
      lastFallback: this.lastFallback,
      keys: this.groq.pool.status(),
    };
  }
}
