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
  constructor({ ollama, groq, cloudflare = null, getSettings, gates, localReady = () => true, logger }) {
    Object.assign(this, { ollama, groq, cloudflare, getSettings, gates, localReady });
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
    const cf = String(s.cloudflareModel || '').trim();
    return { llmProvider: PROVIDERS.includes(p) ? p : 'auto', groqModels: models, cloudflareModels: cf ? [cf] : [], strategy: s.groqStrategy === 'spread' ? 'spread' : 'priority' };
  }

  /** Keep the pools' strategy and model lists in step with the live settings (cheap; called before every use). */
  #sync(s) {
    this.groq.pool.setStrategy?.(s.strategy);
    this.groq.pool.setModels?.(s.groqModels);
    this.cloudflare?.pool.setStrategy?.(s.strategy);
    this.cloudflare?.pool.setModels?.(s.cloudflareModels);
  }

  /** The hosted providers, in the order they are tried: Groq first, then Cloudflare Workers AI. */
  #tiers(s) {
    const tiers = [{ name: 'groq', client: this.groq, gate: this.gates.groq, models: s.groqModels }];
    if (this.cloudflare) tiers.push({ name: 'cloudflare', client: this.cloudflare.client, gate: this.gates.cloudflare || this.gates.groq, models: s.cloudflareModels });
    return tiers;
  }

  #keyCount() {
    return this.groq.pool.size + (this.cloudflare?.pool.size ?? 0);
  }

  /** Should this call try the hosted providers at all? */
  #wantsGroq(s, priority = 1) {
    if (s.llmProvider === 'ollama') return false;
    // Background work stays local while the local model is up: no hosted quota spent on summaries.
    if (priority >= 3 && this.localReady()) return false;
    // Auto with no key at all is simply local; cloud-only with no key must say so (busy message), not silently go local.
    return s.llmProvider === 'groq' || this.#keyCount() > 0;
  }

  #anyUsable(s) {
    return this.#tiers(s).some((t) => t.client.usable(t.models));
  }

  /** Is a hosted provider selected (auto or cloud only) and at least one (account, model) ready right now? */
  groqUsable() {
    const s = this.#settingsFor();
    this.#sync(s);
    return s.llmProvider !== 'ollama' && this.#anyUsable(s);
  }

  /** Does a reply need the local model to be loaded (it will be, or may be, answered locally)? */
  needsLocal() {
    const s = this.#settingsFor();
    if (s.llmProvider === 'ollama') return true;
    if (s.llmProvider === 'groq') return false;
    this.#sync(s);
    return !this.#anyUsable(s); // auto: local only when nothing hosted can serve right now
  }

  #nextAvailableMs(s) {
    const waits = this.#tiers(s).map((t) => t.client.pool.nextAvailableMs(t.models)).filter((n) => n != null);
    return waits.length ? Math.min(...waits) : null;
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
    if (this.#settingsFor(req).llmProvider !== 'ollama' && this.#keyCount() > 0) return { promptTokens: 0, promptMs: 0, loadMs: 0, totalMs: 0, skipped: true };
    return this.gates.ollama.run(() => this.ollama.prefill(req), { priority: 4, signal: req.signal });
  }

  async *chat(req) {
    const s = this.#settingsFor(req);
    const priority = req.priority ?? (req.maxTokens > 450 ? 2 : 1);
    if (this.#wantsGroq(s, priority)) {
      this.#sync(s);
      const strict = s.llmProvider === 'groq';
      let lastErr = null;
      for (const tier of this.#tiers(s)) {
        if (!tier.client.usable(tier.models)) continue;
        let started = false;
        try {
          await tier.gate.acquire({ priority, signal: req.signal });
          try {
            yield { type: 'meta', provider: tier.name, model: tier.models[0] }; // corrected by the client once an account/model accepts
            for await (const ev of tier.client.chat({ ...req, models: tier.models })) {
              if (ev.type === 'token') started = true;
              yield ev;
            }
            this.counters[tier.name] = (this.counters[tier.name] || 0) + 1;
            return;
          } finally {
            tier.gate.release();
          }
        } catch (err) {
          if (req.signal?.aborted || started) throw err; // cancelled, or half a reply already went out: cannot switch mid-sentence
          lastErr = err; // this provider failed before saying anything: the next one (or the local model) takes over
        }
      }
      const none = lastErr || new GroqUnavailableError(this.#keyCount() ? 'every hosted AI account is rate-limited right now' : 'no Groq or Cloudflare key configured', { kind: this.#keyCount() ? 'exhausted' : 'no_keys', retryAfterMs: this.#nextAvailableMs(s) });
      if (strict) return this.#busy(none);
      this.#fellBack(none);
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
      groqReady: this.#anyUsable(s),
      counters: { ...this.counters },
      lastFallback: this.lastFallback,
      keys: this.groq.pool.status(s.groqModels),
      cloudflare: this.cloudflare ? { model: s.cloudflareModels[0] ?? null, ready: this.cloudflare.client.usable(s.cloudflareModels), keys: this.cloudflare.pool.status(s.cloudflareModels) } : null,
    };
  }
}
