// Chooses where each model call runs: Groq (several API keys, rotated) or the local Ollama model.
//
//   llmProvider = 'ollama'  every call runs locally (the original behaviour)
//   llmProvider = 'groq'    live replies go to Groq; a rate-limited key is skipped instantly and the next key is used;
//                           when every key is exhausted (or Groq is down) the call falls back to the local model
//                           (setting groqFallback, on by default). Background work (memory folds, agenda) stays on the
//                           local model while it is ready so it never spends Groq quota.
// Exposes the same surface the pipeline uses on an LLM client: numCtx, chat(), prefill(), reusesPrefix().
import { GroqUnavailableError } from './groq.js';

export class RoutedLlm {
  /**
   * @param {{ ollama: object, groq: object, getSettings: () => object, gates: { ollama: object, groq: object },
   *           localReady?: () => boolean, logger?: object }} o
   */
  constructor({ ollama, groq, getSettings, gates, localReady = () => true, logger }) {
    Object.assign(this, { ollama, groq, getSettings, gates, localReady });
    this.log = logger;
    this.counters = { groq: 0, local: 0, fallbacks: 0 };
    this.lastFallback = null;
    this.lastLog = new Map();
  }

  get numCtx() {
    return this.ollama.numCtx;
  }

  /** Settings for one call: the request's own provider choice (playground, tests) wins over the live setting. */
  #settingsFor(req = {}) {
    const s = this.getSettings();
    return {
      llmProvider: req.provider ?? s.llmProvider,
      groqModel: req.groqModel ?? s.groqModel,
      groqFallback: req.groqFallback ?? s.groqFallback,
    };
  }

  #wantsGroq(s, priority = 1) {
    if (s.llmProvider !== 'groq' || !this.groq.pool.size) return false;
    // Background work stays local while the local model is up: no Groq quota spent on summaries.
    return !(priority >= 3 && this.localReady());
  }

  /** Is the Groq route usable right now (selected and at least one key ready)? */
  groqUsable() {
    const s = this.getSettings();
    return s.llmProvider === 'groq' && this.groq.usable();
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
    if (this.#settingsFor(req).llmProvider === 'groq') return { promptTokens: 0, promptMs: 0, loadMs: 0, totalMs: 0, skipped: true };
    return this.gates.ollama.run(() => this.ollama.prefill(req), { priority: 4, signal: req.signal });
  }

  async *chat(req) {
    const s = this.#settingsFor(req);
    const priority = req.priority ?? (req.maxTokens > 450 ? 2 : 1);
    if (this.#wantsGroq(s, priority)) {
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
          if (!s.groqFallback) throw err;
          this.#fellBack(err);
        }
      } else if (s.groqFallback) {
        this.#fellBack(new GroqUnavailableError('every Groq key is rate-limited right now', { kind: 'exhausted', retryAfterMs: this.groq.pool.nextAvailableMs() }));
      } else {
        throw new GroqUnavailableError(this.groq.pool.size ? 'every Groq key is rate-limited right now' : 'no Groq key configured', { kind: 'exhausted', retryAfterMs: this.groq.pool.nextAvailableMs() });
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
      fallback: s.groqFallback,
      groqReady: this.groq.usable(),
      counters: { ...this.counters },
      lastFallback: this.lastFallback,
      keys: this.groq.pool.status(),
    };
  }
}
