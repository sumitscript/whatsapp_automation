// Minimal Ollama HTTP client on top of global fetch (Node 22+). Streams NDJSON.

const THINKING_FAMILIES = /^(qwen3|deepseek-r1|magistral|gpt-oss)/i;

export class OllamaClient {
  constructor({ url, keepAlive = '-1', numCtx = 4096, numThread = 0, timeoutMs = 120_000, logger } = {}) {
    this.url = url;
    this.keepAlive = /^-?\d+$/.test(String(keepAlive)) ? Number(keepAlive) : keepAlive;
    this.numCtx = numCtx;
    this.numThread = numThread;
    this.timeoutMs = timeoutMs;
    this.log = logger;
  }

  /**
   * `idle: true` is for streamed replies: the timeout then means "no data for timeoutMs" and restarts on every
   * chunk, instead of capping the whole reply (a 700-token story on a CPU legitimately takes minutes, and a
   * fixed cap cut it off mid-sentence). Reading the prompt before the first token still has to fit in timeoutMs.
   */
  async #request(path, { method = 'GET', body, signal, timeoutMs = this.timeoutMs, idle = false } = {}) {
    let sig;
    let watch = null;
    if (idle) {
      const ctl = new AbortController();
      let timer;
      const arm = () => {
        clearTimeout(timer);
        timer = setTimeout(() => ctl.abort(new Error(`Ollama sent no data for ${Math.round(timeoutMs / 1000)}s`)), timeoutMs);
      };
      watch = { touch: arm, stop: () => clearTimeout(timer) };
      arm();
      sig = signal ? AbortSignal.any([signal, ctl.signal]) : ctl.signal;
    } else {
      const timeout = AbortSignal.timeout(timeoutMs);
      sig = signal ? AbortSignal.any([signal, timeout]) : timeout;
    }
    let res;
    try {
      res = await fetch(this.url + path, {
        method,
        headers: body ? { 'content-type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: sig,
      });
    } catch (err) {
      watch?.stop();
      throw err;
    }
    if (!res.ok) {
      watch?.stop();
      const text = await res.text().catch(() => '');
      const err = new Error(`Ollama ${method} ${path} → ${res.status}: ${text.slice(0, 300)}`);
      err.status = res.status;
      throw err;
    }
    if (watch) res.idleWatch = watch;
    return res;
  }

  async *#ndjson(res) {
    const decoder = new TextDecoder();
    let buf = '';
    try {
      for await (const chunk of res.body) {
        res.idleWatch?.touch();
        buf += decoder.decode(chunk, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (line) yield JSON.parse(line);
        }
      }
      if (buf.trim()) yield JSON.parse(buf);
    } finally {
      res.idleWatch?.stop();
    }
  }

  async health() {
    const res = await this.#request('/api/version', { timeoutMs: 3000 });
    return res.json();
  }

  async listModels() {
    const res = await this.#request('/api/tags', { timeoutMs: 5000 });
    const data = await res.json();
    return (data.models || []).map((m) => ({
      name: m.name,
      size: m.size,
      family: m.details?.family,
      parameterSize: m.details?.parameter_size,
      quantization: m.details?.quantization_level,
      modifiedAt: m.modified_at,
    }));
  }

  async loaded() {
    const res = await this.#request('/api/ps', { timeoutMs: 5000 });
    return (await res.json()).models || [];
  }

  async hasModel(name) {
    const models = await this.listModels();
    const want = name.includes(':') ? name : `${name}:latest`;
    return models.some((m) => m.name === want || m.name === name);
  }

  /** Pull a model, reporting progress events {status, completed, total}. */
  async pull(name, onProgress = () => {}) {
    const res = await this.#request('/api/pull', { method: 'POST', body: { model: name, stream: true }, timeoutMs: 6 * 3600_000 });
    let last = null;
    for await (const ev of this.#ndjson(res)) {
      if (ev.error) throw new Error(ev.error);
      last = ev;
      onProgress(ev);
    }
    return last;
  }

  /** Delete a model from the Ollama server (frees its disk space). Resolves false if it wasn't installed. */
  async deleteModel(name) {
    try {
      await this.#request('/api/delete', { method: 'DELETE', body: { model: name }, timeoutMs: 60_000 });
      return true;
    } catch (err) {
      if (err.status === 404) return false;
      throw err;
    }
  }

  /** Make sure a model exists locally (optionally pulling it). */
  async ensureModel(name, { autoPull = true, onProgress } = {}) {
    if (await this.hasModel(name)) return true;
    if (!autoPull) return false;
    this.log?.info('Pulling model (first run can take a few minutes)', { model: name });
    await this.pull(name, onProgress);
    this.log?.info('Model pulled', { model: name });
    return true;
  }

  /** Load the model into RAM and keep it resident so the first user never waits. */
  async warmup(model) {
    const started = Date.now();
    const res = await this.#request('/api/generate', {
      method: 'POST',
      body: { model, prompt: '', keep_alive: this.keepAlive, stream: false },
      timeoutMs: 10 * 60_000,
    });
    await res.json();
    return Date.now() - started;
  }

  /**
   * Can this model reuse a cached prompt prefix? Measured on the live server: llama/qwen-style models read the
   * persona + history once and the next message only pays for its own tokens (~25-30% less prompt time), but
   * gemma3 (sliding-window attention) cannot rewind its cache, so a pre-read changes nothing (6.2 s vs 6.4 s).
   */
  reusesPrefix(model) {
    return !/^gemma3/i.test(String(model || ''));
  }

  /**
   * "Think ahead": read a prompt without answering it. Evaluates the messages and stops after one token, so
   * Ollama's KV cache already holds that prefix when the real message arrives (the reply then only has to
   * read the new part). Same options as chat() so the cached prefix is reusable; abortable.
   * @returns {Promise<{promptTokens:number, promptMs:number, loadMs:number, totalMs:number}>}
   */
  async prefill({ model, messages, signal }) {
    const body = { model, messages, stream: false, keep_alive: this.keepAlive, options: this.buildOptions({ temperature: 0, maxTokens: 1 }) };
    if (THINKING_FAMILIES.test(model)) body.think = false;
    const res = await this.#request('/api/chat', { method: 'POST', body, signal, timeoutMs: 120_000 });
    const ev = await res.json();
    return {
      promptTokens: ev.prompt_eval_count ?? 0,
      promptMs: Math.round((ev.prompt_eval_duration ?? 0) / 1e6),
      loadMs: Math.round((ev.load_duration ?? 0) / 1e6),
      totalMs: Math.round((ev.total_duration ?? 0) / 1e6),
    };
  }

  buildOptions({ temperature, maxTokens, extra = {} } = {}) {
    const options = {
      temperature,
      num_predict: maxTokens,
      num_ctx: this.numCtx,
      top_p: 0.9,
      top_k: 40,
      repeat_penalty: 1.1,
      ...extra,
    };
    if (this.numThread > 0) options.num_thread = this.numThread;
    return options;
  }

  /**
   * Streaming chat. Yields {type:'token', text} and finally {type:'done', stats}.
   * @param {{ model: string, messages: {role: string, content: string}[], temperature?: number, maxTokens?: number, signal?: AbortSignal }} p
   */
  async *chat({ model, messages, temperature = 0.6, maxTokens = 350, signal }) {
    const body = {
      model,
      messages,
      stream: true,
      keep_alive: this.keepAlive,
      options: this.buildOptions({ temperature, maxTokens }),
    };
    if (THINKING_FAMILIES.test(model)) body.think = false; // reasoning traces are pure latency here
    const res = await this.#request('/api/chat', { method: 'POST', body, signal, idle: true });
    for await (const ev of this.#ndjson(res)) {
      if (ev.error) throw new Error(ev.error);
      const text = ev.message?.content;
      if (text) yield { type: 'token', text };
      if (ev.done) {
        yield {
          type: 'done',
          stats: {
            promptTokens: ev.prompt_eval_count ?? 0,
            evalTokens: ev.eval_count ?? 0,
            promptMs: Math.round((ev.prompt_eval_duration ?? 0) / 1e6),
            evalMs: Math.round((ev.eval_duration ?? 0) / 1e6),
            loadMs: Math.round((ev.load_duration ?? 0) / 1e6),
            totalMs: Math.round((ev.total_duration ?? 0) / 1e6),
            doneReason: ev.done_reason,
          },
        };
      }
    }
  }
}
