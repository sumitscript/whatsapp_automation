// Turns raw probes (Ollama version/tags/ps + the worker heartbeat) into one
// health verdict. Pure and dependency-free: used by GET /api/llm/status and by
// the gateway's alert monitor, and unit-tested without Ollama.

export const WORKER_STALE_MS = 15_000;

const nameVariants = (model) => (model.includes(':') ? [model] : [model, `${model}:latest`]);

/**
 * @param {{ model: string, version: {version?: string, error?: string}, installed: {name:string}[]|null,
 *           loaded: {name:string, size?:number, size_vram?:number, expires_at?:string}[]|null,
 *           worker: object|null, now?: number }} p
 * @returns {{ state: 'ok'|'degraded'|'down', message: string, ollamaUp: boolean, installedOk: boolean|null,
 *             resident: object|null, workerAlive: boolean }}
 */
export function deriveLlmState({ model, version, installed, loaded, worker, now = Date.now() }) {
  const names = nameVariants(model);
  const ollamaUp = !!version && !version.error;
  const workerAlive = !!worker && now - worker.at < WORKER_STALE_MS;
  const resident = (loaded || []).find((m) => names.includes(m.name)) || null;
  const installedOk = installed ? installed.some((m) => names.includes(m.name)) : null;

  let state = 'ok';
  let message = 'Ready to answer';
  if (!ollamaUp) [state, message] = ['down', `Ollama is unreachable: ${version?.error || 'no response'}`];
  else if (installedOk === false) [state, message] = ['down', `Model ${model} is not installed — pull it from Settings`];
  else if (!workerAlive) [state, message] = ['down', 'AI worker is not running (no heartbeat)'];
  else if (worker.model?.error) [state, message] = ['down', `Model preparation failed: ${worker.model.error}`];
  else if (!worker.model?.ready) [state, message] = ['degraded', 'Model is loading…'];
  else if (!resident) [state, message] = ['degraded', 'Model is not resident in RAM yet — the next reply will be slow'];

  return { state, message, ollamaUp, installedOk, resident, workerAlive };
}

/**
 * Debounces health into alert events: a "down" verdict must persist for
 * `afterMs` before it fires, and recovery only fires if "down" was announced.
 */
export class DownAlerter {
  constructor({ afterMs = 90_000 } = {}) {
    this.afterMs = afterMs;
    this.downSince = null;
    this.announced = false;
  }

  /** @returns {{ type: 'down', message: string, downForMs: number } | { type: 'recovered', downForMs: number } | null} */
  update(state, message, now = Date.now()) {
    if (state === 'down') {
      this.downSince ??= now;
      if (!this.announced && now - this.downSince >= this.afterMs) {
        this.announced = true;
        return { type: 'down', message, downForMs: now - this.downSince };
      }
      return null;
    }
    const since = this.downSince;
    const wasAnnounced = this.announced;
    this.downSince = null;
    this.announced = false;
    return wasAnnounced && since != null ? { type: 'recovered', downForMs: now - since } : null;
  }
}
