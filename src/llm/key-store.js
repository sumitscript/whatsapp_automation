// Where the Groq keys come from: the GROQ_API_KEYS environment variable plus keys added in the console
// (stored encrypted in Postgres). Keeps a KeyPool in sync, in every process that uses it.
import { decryptSecret } from '../core/secrets.js';

/** A Groq API key as pasted by a person: no spaces, letters/digits/underscore/dash, 20–200 characters (they start with gsk_). */
export const isGroqKeyFormat = (key) => /^[\w-]{20,200}$/.test(String(key || ''));

/** "gsk_a, gsk_b\ngsk_c" → ['gsk_a','gsk_b','gsk_c'] (blank and duplicate entries dropped). */
export function parseKeyList(text) {
  return [...new Set(String(text || '').split(/[\s,;]+/).map((k) => k.trim()).filter(Boolean))];
}

export class KeyStore {
  /**
   * @param {{ pool: import('./key-pool.js').KeyPool, repo: object, secret: string, envKeys?: string, logger?: object }} o
   */
  constructor({ pool, repo, secret, envKeys = '', logger }) {
    Object.assign(this, { pool, repo, secret });
    this.envKeys = parseKeyList(envKeys);
    this.log = logger;
    this.unreadable = 0;
  }

  /** Rebuild the pool from env + database. `reenable` clears "rejected" marks (the owner just edited the list). */
  async load({ reenable = false } = {}) {
    const list = this.envKeys.map((secret, i) => ({ id: `env:${i + 1}`, label: `env key ${i + 1}`, source: 'env', secret }));
    this.unreadable = 0;
    try {
      for (const row of await this.repo.listLlmKeys()) {
        const secret = decryptSecret(row.keyEnc, this.secret);
        if (!secret) {
          this.unreadable++; // JWT_SECRET changed since the key was saved
          continue;
        }
        if (list.some((k) => k.secret === secret)) continue;
        list.push({ id: `db:${row.id}`, label: row.label, source: 'console', secret });
      }
    } catch (err) {
      this.log?.warn('Could not read the stored Groq keys', { err });
    }
    this.pool.setKeys(list);
    if (reenable) this.pool.clearDisabled();
    return list.length;
  }

  /** Initial load, then refresh on `onChange()` calls and every minute (another process may have changed the list). */
  async start() {
    const n = await this.load();
    this.log?.info('Groq keys loaded', { keys: n, unreadable: this.unreadable });
    this.timer = setInterval(() => this.load().catch(() => {}), 60_000);
    this.timer.unref?.();
    return n;
  }

  onChange() {
    return this.load({ reenable: true }).catch((err) => this.log?.warn('Reloading Groq keys failed', { err }));
  }

  stop() {
    clearInterval(this.timer);
  }
}
