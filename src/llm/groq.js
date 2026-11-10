// Groq (OpenAI-compatible) chat client with multi-key failover. Node built-ins only (global fetch).
//
// One logical request walks the key pool: a 429 on key A switches to key B in the same call with no pause,
// an invalid key is skipped, a flaky network/5xx marks the key for a short rest. When every key is
// unavailable the call throws GroqUnavailableError and the router falls back to the local model.
import { parseDuration } from './key-pool.js';

export class GroqUnavailableError extends Error {
  constructor(message, { kind = 'exhausted', retryAfterMs = null } = {}) {
    super(message);
    this.name = 'GroqUnavailableError';
    this.kind = kind; // no_keys | exhausted | rate_limit | rejected | server | network
    this.retryAfterMs = retryAfterMs;
  }
}

/** The request itself is wrong (unknown model, too many tokens…): another key would fail the same way. */
export class GroqRequestError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'GroqRequestError';
    this.status = status;
  }
}

const REASONING_HIDDEN = /^(qwen\/qwen3|qwen3|deepseek-r1)/i;
const REASONING_LOW = /gpt-oss/i;
const NOT_CHAT = /whisper|tts|guard|playai|embed/i;
const roughTokens = (messages, maxTokens) => Math.ceil(messages.reduce((n, m) => n + String(m.content || '').length, 0) / 3.5) + (maxTokens || 0);

/** Provider's "try again in 2m3.5s" hint from an error body, when there is no retry-after header. */
function retryHint(text) {
  const m = /try again in ([\dhms.]+)/i.exec(String(text || ''));
  return m ? parseDuration(m[1]) : null;
}

export class GroqClient {
  constructor({ pool, baseUrl = 'https://api.groq.com/openai/v1', timeoutMs = 30_000, logger, fetchImpl } = {}) {
    this.pool = pool;
    this.baseUrl = String(baseUrl).replace(/\/+$/, '');
    this.timeoutMs = timeoutMs;
    this.log = logger;
    this.fetch = fetchImpl || ((...a) => fetch(...a));
  }

  /** At least one key is ready right now. */
  usable() {
    return this.pool.hasAvailable();
  }

  #headers(secret) {
    return { authorization: `Bearer ${secret}`, 'content-type': 'application/json' };
  }

  /** Open a streaming completion on one key. Updates the pool and throws a typed error on failure. */
  async #open(key, body, signal) {
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
      res = await this.fetch(`${this.baseUrl}/chat/completions`, { method: 'POST', headers: this.#headers(key.secret), body: JSON.stringify(body), signal: sig });
    } catch (err) {
      watch.stop();
      if (signal?.aborted) throw err;
      this.pool.transient(key.id, String(err.message || err));
      throw new GroqUnavailableError(`Groq network error: ${err.message || err}`, { kind: 'network' });
    }
    if (res.ok) {
      const h = res.headers;
      this.pool.success(key.id, {
        remainingTokens: Number(h.get('x-ratelimit-remaining-tokens')) || (h.get('x-ratelimit-remaining-tokens') === '0' ? 0 : undefined),
        remainingRequests: h.get('x-ratelimit-remaining-requests') != null ? Number(h.get('x-ratelimit-remaining-requests')) : undefined,
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
    const msg = `Groq ${res.status}: ${String(detail).slice(0, 200)}`;
    if (res.status === 429) {
      const retryAfterMs = parseDuration(res.headers.get('retry-after')) ?? retryHint(detail);
      this.pool.rateLimited(key.id, { retryAfterMs, reason: String(detail).slice(0, 120) });
      throw new GroqUnavailableError(msg, { kind: 'rate_limit', retryAfterMs });
    }
    if (res.status === 401 || res.status === 403) {
      this.pool.authFailed(key.id, `HTTP ${res.status}`);
      throw new GroqUnavailableError(msg, { kind: 'rejected' });
    }
    if (res.status === 408 || res.status >= 500) {
      this.pool.transient(key.id, `HTTP ${res.status}`);
      throw new GroqUnavailableError(msg, { kind: 'server' });
    }
    throw new GroqRequestError(msg, res.status); // 400/404/413/422: not the key's fault
  }

  /**
   * Streaming chat. Yields {type:'token', text} then {type:'done', stats} like OllamaClient.chat().
   * Throws GroqUnavailableError when no key could serve it (nothing was yielded), GroqRequestError for a bad request.
   */
  async *chat({ model, messages, temperature = 0.6, maxTokens = 350, signal }) {
    const body = { model, messages, temperature, max_tokens: maxTokens, top_p: 0.9, stream: true };
    if (REASONING_HIDDEN.test(model)) body.reasoning_format = 'hidden';
    if (REASONING_LOW.test(model)) body.reasoning_effort = 'low';
    const need = roughTokens(messages, maxTokens);
    const tried = new Set();
    let last = null;
    let opened = null;
    while (!opened) {
      const key = this.pool.pick({ exclude: tried, needTokens: need });
      if (!key) {
        throw new GroqUnavailableError(last ? `${last.message} (no other key to try)` : this.pool.size ? 'every Groq key is rate-limited right now' : 'no Groq API key is configured', {
          kind: last?.kind ?? (this.pool.size ? 'exhausted' : 'no_keys'),
          retryAfterMs: this.pool.nextAvailableMs(),
        });
      }
      tried.add(key.id);
      try {
        opened = { ...(await this.#open(key, body, signal)), key };
      } catch (err) {
        if (err instanceof GroqRequestError || signal?.aborted || !(err instanceof GroqUnavailableError)) throw err;
        last = err;
        this.log?.warn('Groq key failed, trying the next one', { key: key.label, kind: err.kind, retryAfterMs: err.retryAfterMs });
      }
    }
    const { res, watch, key } = opened;
    const started = performance.now();
    const decoder = new TextDecoder();
    let buf = '';
    let usage = null;
    let finish = null;
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
          if (c?.delta?.content) yield { type: 'token', text: c.delta.content };
          if (c?.finish_reason) finish = c.finish_reason;
          usage = ev.x_groq?.usage || ev.usage || usage;
        }
      }
    } finally {
      watch.stop();
    }
    this.pool.success(key.id, { tokens: usage?.total_tokens || 0 });
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
        provider: 'groq',
        keyLabel: key.label,
      },
    };
  }

  /** Chat-capable model ids available to the first working key (cached by the caller). */
  async listModels({ signal } = {}) {
    const tried = new Set();
    for (;;) {
      const key = this.pool.pick({ exclude: tried });
      if (!key) throw new GroqUnavailableError('no usable Groq key', { kind: this.pool.size ? 'exhausted' : 'no_keys' });
      tried.add(key.id);
      const out = await GroqClient.probe(key.secret, { baseUrl: this.baseUrl, fetchImpl: this.fetch, signal, timeoutMs: 10_000 });
      if (out.ok) return out.models;
    }
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
