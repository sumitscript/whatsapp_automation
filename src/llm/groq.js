// Groq (OpenAI-compatible) chat client with per-account, per-model failover. Node built-ins only (global fetch).
//
// One logical request walks the pool in priority order: for each account (key) the models in priority order, so
// account 1's gpt-oss-120b, then its gpt-oss-20b, then its qwen, then account 2's gpt-oss-120b … A 429 rests only that
// (account, model) pair (or every key of the organization the message names) and the same call continues on the next pair
// with no pause; "model not found" retires the pair; an invalid key is skipped; a flaky network/5xx rests the key. When
// nothing is left the call throws GroqUnavailableError and the router falls back to the local model (or sends the break message).
import { parseDuration } from './key-pool.js';

export class GroqUnavailableError extends Error {
  constructor(message, { kind = 'exhausted', retryAfterMs = null } = {}) {
    super(message);
    this.name = 'GroqUnavailableError';
    this.kind = kind; // no_keys | exhausted | rate_limit | rejected | server | network | model
    this.retryAfterMs = retryAfterMs;
  }
}

/** The request itself is wrong (too many tokens, bad parameter…): another account would fail the same way. */
export class GroqRequestError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'GroqRequestError';
    this.status = status;
  }
}

const REASONING_HIDDEN = /^(qwen\/qwen3|qwen3|deepseek-r1)/i;
const REASONING_LOW = /gpt-oss/i;
const NOT_CHAT = /whisper|tts|guard|playai|orpheus|embed/i;
const roughTokens = (messages, maxTokens) => Math.ceil(messages.reduce((n, m) => n + String(m.content || '').length, 0) / 3.5) + (maxTokens || 0);

/** Provider's "try again in 2m3.5s" hint from an error body, when there is no retry-after header. */
function retryHint(text) {
  const m = /try again in ([\dhms.]+)/i.exec(String(text || ''));
  return m ? parseDuration(m[1]) : null;
}

/** "...in organization `org_01abc` service tier..." → org_01abc (keys of one organization share their limits). */
export function orgFromMessage(text) {
  const m = /organization\s+[`'"]?(org_[A-Za-z0-9]+)/i.exec(String(text || ''));
  return m ? m[1] : null;
}

/** Limit details out of a 429 text: "...tokens per day (TPD): Limit 200000, Used 199961, Requested 1500..." */
export function limitFromMessage(text) {
  const t = String(text || '');
  const kind = /(requests|tokens)\s+per\s+(day|minute|hour|second)/i.exec(t);
  const nums = /Limit\s+(\d+),\s*Used\s+(\d+)/i.exec(t);
  if (!kind && !nums) return null;
  return { what: kind ? `${kind[1].toLowerCase()} per ${kind[2].toLowerCase()}` : null, limit: nums ? Number(nums[1]) : null, used: nums ? Number(nums[2]) : null };
}

/** Milliseconds until the next 00:00 UTC (Cloudflare resets its daily free allowance then). */
export function msToUtcMidnight(now = Date.now()) {
  const d = new Date(now);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1) - now;
}

// One client for every OpenAI-compatible provider (Groq, Cloudflare Workers AI): they differ only in the URL, how the key is
// sent, and how "daily allowance used up" is recognised.
export class GroqClient {
  constructor({ pool, baseUrl = 'https://api.groq.com/openai/v1', timeoutMs = 30_000, logger, fetchImpl, provider = 'groq', urlFor, tokenFor, reasoning = true, exhausted, neuronsFor } = {}) {
    this.pool = pool;
    this.provider = provider;
    this.urlFor = urlFor || (() => `${this.baseUrl}/chat/completions`); // (secret) => chat completions URL
    this.tokenFor = tokenFor || ((secret) => secret); // (secret) => bearer token
    this.reasoning = reasoning; // send provider-specific reasoning flags for known reasoning models
    this.exhausted = exhausted || null; // (status, detail) => ms until the allowance returns, or null
    this.neuronsFor = neuronsFor || null; // (model, promptTokens, completionTokens) => neurons, for providers billed in them
    this.baseUrl = String(baseUrl).replace(/\/+$/, '');
    this.timeoutMs = timeoutMs;
    this.log = logger;
    this.fetch = fetchImpl || ((...a) => fetch(...a));
  }

  get label() {
    return this.provider === 'cloudflare' ? 'Cloudflare' : 'Groq';
  }

  /** At least one (account, model) pair is ready right now. */
  usable(models) {
    return this.pool.hasAvailable(models?.length ? models : this.pool.models);
  }

  #headers(secret) {
    return { authorization: `Bearer ${this.tokenFor(secret)}`, 'content-type': 'application/json' };
  }

  /** Open a streaming completion on one (account, model). Updates the pool and throws a typed error on failure. */
  async #open(pick, body, signal) {
    const ctl = new AbortController();
    let timer;
    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(() => ctl.abort(new Error(`Groq sent no data for ${Math.round(this.timeoutMs / 1000)}s`)), this.timeoutMs);
    };
    const watch = { touch: arm, stop: () => clearTimeout(timer) };
    arm();
    const sig = signal ? AbortSignal.any([signal, ctl.signal]) : ctl.signal;
    let res;
    try {
      res = await this.fetch(this.urlFor(pick.secret), { method: 'POST', headers: this.#headers(pick.secret), body: JSON.stringify(body), signal: sig });
    } catch (err) {
      watch.stop();
      if (signal?.aborted) throw err;
      this.pool.transient(pick.id, String(err.message || err));
      throw new GroqUnavailableError(`${this.label} network error: ${err.message || err}`, { kind: 'network' });
    }
    if (res.ok) {
      const h = res.headers;
      this.pool.success(pick.id, {
        model: pick.model,
        remainingTokens: Number(h.get('x-ratelimit-remaining-tokens')) || (h.get('x-ratelimit-remaining-tokens') === '0' ? 0 : undefined),
        remainingRequests: h.get('x-ratelimit-remaining-requests') != null ? Number(h.get('x-ratelimit-remaining-requests')) : undefined,
        limitTokens: h.get('x-ratelimit-limit-tokens') != null ? Number(h.get('x-ratelimit-limit-tokens')) : undefined,
        limitRequests: h.get('x-ratelimit-limit-requests') != null ? Number(h.get('x-ratelimit-limit-requests')) : undefined,
        resetTokensMs: parseDuration(h.get('x-ratelimit-reset-tokens')),
        resetRequestsMs: parseDuration(h.get('x-ratelimit-reset-requests')),
      });
      return { res, watch };
    }
    watch.stop();
    const text = await res.text().catch(() => '');
    const detail = (() => {
      try {
        return JSON.parse(text)?.error?.message || text;
      } catch {
        return text;
      }
    })();
    const msg = `${this.label} ${res.status} (${pick.model}): ${String(detail).slice(0, 200)}`;
    const allowanceMs = this.exhausted?.(res.status, String(detail));
    if (allowanceMs) {
      // the provider's daily free allowance is used up: rest this account until it comes back
      this.pool.rateLimited(pick.id, { model: pick.model, retryAfterMs: allowanceMs, reason: 'daily free allowance used up', hit: { what: 'daily free allowance', limit: null, used: null } });
      throw new GroqUnavailableError(msg, { kind: 'rate_limit', retryAfterMs: allowanceMs });
    }
    if (res.status === 429) {
      const retryAfterMs = parseDuration(res.headers.get('retry-after')) ?? retryHint(detail);
      this.pool.rateLimited(pick.id, { model: pick.model, retryAfterMs, reason: String(detail).slice(0, 120), org: orgFromMessage(detail), hit: limitFromMessage(detail) });
      throw new GroqUnavailableError(msg, { kind: 'rate_limit', retryAfterMs });
    }
    if (res.status === 401 || res.status === 403) {
      this.pool.authFailed(pick.id, `HTTP ${res.status}`);
      throw new GroqUnavailableError(msg, { kind: 'rejected' });
    }
    if (res.status === 404) {
      // "The model `x` does not exist or you do not have access to it": this account cannot use it, the others in the list may.
      this.pool.modelUnavailable(pick.id, pick.model, String(detail).slice(0, 120));
      throw new GroqUnavailableError(msg, { kind: 'model' });
    }
    if (res.status === 413) {
      // Request larger than this model's per-minute budget: another model (or account) may take it.
      this.pool.rateLimited(pick.id, { model: pick.model, retryAfterMs: 60_000, reason: 'request too large for this model' });
      throw new GroqUnavailableError(msg, { kind: 'rate_limit', retryAfterMs: 60_000 });
    }
    if (res.status === 408 || res.status >= 500) {
      this.pool.transient(pick.id, `HTTP ${res.status}`);
      throw new GroqUnavailableError(msg, { kind: 'server' });
    }
    throw new GroqRequestError(msg, res.status); // 400/422: the request is wrong, not the account
  }

  #body(model, { messages, temperature, maxTokens }) {
    const body = { model, messages, temperature, max_tokens: maxTokens, top_p: 0.9, stream: true };
    if (this.reasoning && REASONING_HIDDEN.test(model)) body.reasoning_format = 'hidden';
    if (this.reasoning && REASONING_LOW.test(model)) body.reasoning_effort = 'low';
    if (!this.reasoning) body.stream_options = { include_usage: true };
    return body;
  }

  /**
   * Streaming chat. Yields {type:'meta', provider:'groq', model, keyLabel} once an (account, model) accepted the request, then
   * {type:'token', text}…, then {type:'done', stats} like OllamaClient.chat().
   * `models` is the priority list (default: the pool's); `model` alone is a one-model list.
   * Throws GroqUnavailableError when nothing could serve it (nothing was yielded), GroqRequestError for a bad request.
   */
  async *chat({ model, models, messages, temperature = 0.6, maxTokens = 350, signal }) {
    const list = models?.length ? models : model ? [model] : this.pool.models;
    const need = roughTokens(messages, maxTokens);
    const tried = new Set();
    let last = null;
    let opened = null;
    while (!opened) {
      const pick = this.pool.pick({ exclude: tried, needTokens: need, models: list });
      if (!pick) {
        throw new GroqUnavailableError(last ? `${last.message} (nothing else to try)` : this.pool.size ? 'every Groq account is rate-limited right now' : 'no Groq API key is configured', {
          kind: last?.kind ?? (this.pool.size ? 'exhausted' : 'no_keys'),
          retryAfterMs: this.pool.nextAvailableMs(list),
        });
      }
      tried.add(`${pick.id}|${pick.model}`);
      try {
        opened = { ...(await this.#open(pick, this.#body(pick.model, { messages, temperature, maxTokens }), signal)), key: pick };
      } catch (err) {
        if (err instanceof GroqRequestError || signal?.aborted || !(err instanceof GroqUnavailableError)) throw err;
        last = err;
        // a key that was rejected or is down fails for every model: skip all of its pairs
        if (err.kind === 'rejected' || err.kind === 'network' || err.kind === 'server') tried.add(pick.id);
        this.log?.warn('Groq account/model failed, trying the next one', { account: pick.label, model: pick.model, kind: err.kind, retryAfterMs: err.retryAfterMs });
      }
    }
    const { res, watch, key } = opened;
    yield { type: 'meta', provider: this.provider, model: key.model, keyLabel: key.label };
    const started = performance.now();
    const decoder = new TextDecoder();
    let buf = '';
    let usage = null;
    let finish = null;
    let produced = 0;
    try {
      for await (const chunk of res.body) {
        watch.touch();
        buf += decoder.decode(chunk, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;
          let ev;
          try {
            ev = JSON.parse(payload);
          } catch {
            continue;
          }
          if (ev.error) throw new Error(ev.error.message || 'Groq stream error');
          const c = ev.choices?.[0];
          if (c?.delta?.content) {
            produced += c.delta.content.length;
            yield { type: 'token', text: c.delta.content };
          }
          if (c?.finish_reason) finish = c.finish_reason;
          usage = ev.x_groq?.usage || ev.usage || usage;
        }
      }
    } finally {
      watch.stop();
    }
    const promptTokens = usage?.prompt_tokens ?? need;
    const completionTokens = usage?.completion_tokens ?? Math.ceil(produced / 3.5);
    this.pool.success(key.id, { model: key.model, tokens: usage?.total_tokens || (usage?.prompt_tokens || 0) + (usage?.completion_tokens || 0) || (this.neuronsFor ? promptTokens + completionTokens : 0), neurons: this.neuronsFor ? this.neuronsFor(key.model, promptTokens, completionTokens) : 0, done: true });
    const ms = (s) => Math.round((Number(s) || 0) * 1000);
    yield {
      type: 'done',
      stats: {
        promptTokens: usage?.prompt_tokens ?? 0,
        evalTokens: usage?.completion_tokens ?? 0,
        promptMs: ms(usage?.prompt_time),
        evalMs: ms(usage?.completion_time),
        loadMs: 0,
        totalMs: ms(usage?.total_time) || Math.round(performance.now() - started),
        doneReason: finish === 'length' ? 'length' : finish || 'stop',
        provider: this.provider,
        model: key.model,
        keyLabel: key.label,
      },
    };
  }

  /** Chat-capable model ids available to the first working key (cached by the caller). */
  async listModels({ signal } = {}) {
    for (const id of this.pool.keys.keys()) {
      // not a chat call: read the secret directly so the listing is not counted as usage
      const out = await GroqClient.probe(this.pool.secretOf(id), { baseUrl: this.baseUrl, fetchImpl: this.fetch, signal, timeoutMs: 10_000 });
      if (out.ok) return out.models;
    }
    throw new GroqUnavailableError('no usable Groq key', { kind: this.pool.size ? 'exhausted' : 'no_keys' });
  }

  /** Check a key without touching any pool: GET /models. Returns { ok, status, ms, models?, error? }. */
  static async probe(secret, { baseUrl = 'https://api.groq.com/openai/v1', fetchImpl, signal, timeoutMs = 10_000 } = {}) {
    const f = fetchImpl || ((...a) => fetch(...a));
    const t0 = performance.now();
    try {
      const res = await f(`${String(baseUrl).replace(/\/+$/, '')}/models`, {
        headers: { authorization: `Bearer ${secret}` },
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs),
      });
      const ms = Math.round(performance.now() - t0);
      if (!res.ok) return { ok: false, status: res.status, ms, error: res.status === 401 ? 'Groq rejected this key (invalid or revoked)' : `Groq answered HTTP ${res.status}` };
      const j = await res.json();
      const models = (j.data || []).filter((m) => m.active !== false && !NOT_CHAT.test(m.id)).map((m) => m.id).sort();
      return { ok: true, status: 200, ms, models };
    } catch (err) {
      return { ok: false, status: 0, ms: Math.round(performance.now() - t0), error: `Could not reach Groq: ${err.message}` };
    }
  }
}
