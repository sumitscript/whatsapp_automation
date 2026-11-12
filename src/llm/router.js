// Chooses where each model call runs: Groq (several accounts, several models each) or the local Ollama model.
//
//   llmProvider = 'auto'    Groq whenever an (account, model) can serve; when every one is used up (or Groq is down, or no
//                           key was added) the local model answers. Adding a key is all it takes.
//   llmProvider = 'ollama'  every call runs locally (Groq is never contacted)
//   llmProvider = 'groq'    Groq only: when nothing can answer the call fails with GroqUnavailableError and the worker
//                           sends the "on a short break" message (setting groqBusyMessage) instead of a reply.
// Inside Groq the order is the owner's hierarchy (setting groqStrategy 'priority'): account 1 with its models in order
// (groqModels), then account 2 with its own quota for the same models, then account 3… (see key-pool.js).
// Background work (memory folds, agenda extraction) stays on the local model while it is ready so it never spends Groq
// quota; in Groq-only mode with the local model not ready it uses Groq. Think-ahead is skipped on the Groq route.
// Exposes the same surface the pipeline uses on an LLM client: numCtx, chat(), prefill(), reusesPrefix().
import { GroqUnavailableError } from './groq.js';

export const PROVIDERS = ['auto', 'ollama', 'groq'];

/** "a, b c" or ['a','b'] → ['a','b'] */
export function parseModelList(v) {
  const list = Array.isArray(v) ? v : String(v ?? '').split(/[\s,]+/);
  return [...new Set(list.map((m) => String(m).trim()).filter(Boolean))];
}

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

  /** Settings for one call: the request's own choice (playground, tests) wins over the live setting. */
  #settingsFor(req = {}) {
    const s = this.getSettings();
    const p = req.provider ?? s.llmProvider;
    let models = req.groqModels ? parseModelList(req.groqModels) : req.groqModel ? [req.groqModel] : parseModelList(s.groqModels);
    if (!models.length) models = s.groqModel ? [s.groqModel] : [];
    return { llmProvider: PROVIDERS.includes(p) ? p : 'auto', groqModels: models, strategy: s.groqStrategy === 'spread' ? 'spread' : 'priority' };
  }

  /** Keep the pool's strategy in step with the live setting (cheap; called before every use). */
  #sync(s) {
    this.groq.pool.setStrategy?.(s.strategy);
    this.groq.pool.setModels?.(s.groqModels);
  }

  /** Should this call try Groq at all? */
  #wantsGroq(s, priority = 1) {
    if (s.llmProvider === 'ollama') return false;
    // Background work stays local while the local model is up: no Groq quota spent on summaries.
    if (priority >= 3 && this.localReady()) return false;
    // Auto with no key at all is simply local; Groq-only with no key must say so (busy message), not silently go local.
    return s.llmProvider === 'groq' || this.groq.pool.size > 0;
  }

  /** Is Groq selected (auto or Groq only) and at least one (account, model) ready right now? */
  groqUsable() {
    const s = this.#settingsFor();
    this.#sync(s);
    return s.llmProvider !== 'ollama' && this.groq.usable(s.groqModels);
  }

  /** Does a reply need the local model to be loaded (it will be, or may be, answered locally)? */
  needsLocal() {
    const s = this.#settingsFor();
    if (s.llmProvider === 'ollama') return true;
    if (s.llmProvider === 'groq') return false;
    this.#sync(s);
    return !this.groq.usable(s.groqModels); // auto: local only when nothing on Groq can serve right now
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
      this.#sync(s);
      const strict = s.llmProvider === 'groq';
      if (this.groq.usable(s.groqModels)) {
        let started = false;
        try {
          await this.gates.groq.acquire({ priority, signal: req.signal });
          try {
            yield { type: 'meta', provider: 'groq', model: s.groqModels[0] }; // corrected by the client once an account/model accepts
            for await (const ev of this.groq.chat({ ...req, models: s.groqModels })) {
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
        return this.#busy(new GroqUnavailableError(this.groq.pool.size ? 'every Groq account is rate-limited right now' : 'no Groq key configured', { kind: this.groq.pool.size ? 'exhausted' : 'no_keys', retryAfterMs: this.groq.pool.nextAvailableMs(s.groqModels) }));
      } else {
        this.#fellBack(new GroqUnavailableError('every Groq account is rate-limited right now', { kind: 'exhausted', retryAfterMs: this.groq.pool.nextAvailableMs(s.groqModels) }));
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
    const s = this.#settingsFor();
    this.#sync(s);
    return {
      provider: s.llmProvider,
      groqModel: s.groqModels[0] ?? null,
      groqModels: s.groqModels,
      strategy: s.strategy,
      groqReady: this.groq.usable(s.groqModels),
      counters: { ...this.counters },
      lastFallback: this.lastFallback,
      keys: this.groq.pool.status(s.groqModels),
    };
  }
}
