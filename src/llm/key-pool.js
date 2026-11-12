// A pool of API keys for one provider. Pure bookkeeping (no I/O), unit-tested.
//
// The router asks for a key, uses it, and reports what happened:
//   ready      → used round-robin, preferring the key with the most headroom left (from the rate-limit headers)
//   cooling    → hit a rate limit; skipped until its reset time (the provider's own retry-after)
//   disabled   → rejected as invalid/forbidden; skipped until the keys are reloaded
// A rate limit on one key never delays a request: the caller immediately takes the next ready key.
import { maskKey } from '../core/secrets.js';

/** "2m59.56s", "7.66s", "250ms", "1h2m3s", "12" (seconds) → milliseconds, or null. */
export function parseDuration(v) {
  if (v == null || v === '') return null;
  const s = String(v).trim();
  if (/^\d+(\.\d+)?$/.test(s)) return Math.round(parseFloat(s) * 1000);
  let ms = 0;
  let any = false;
  for (const [, n, unit] of s.matchAll(/(\d+(?:\.\d+)?)(ms|h|m|s)/g)) {
    any = true;
    ms += parseFloat(n) * { h: 3_600_000, m: 60_000, s: 1000, ms: 1 }[unit];
  }
  return any ? Math.round(ms) : null;
}

export class KeyPool {
  constructor({ now = Date.now, defaultCooldownMs = 60_000, maxCooldownMs = 6 * 3_600_000 } = {}) {
    this.now = now;
    this.defaultCooldownMs = defaultCooldownMs;
    this.maxCooldownMs = maxCooldownMs;
    this.keys = new Map(); // id → state
    this.seq = 0;
    this.usageSink = null; // (keyId, { calls, ok, tokens, limited, errors }) => void: persists usage outside this process (UsageStore)
  }

  /** Replace the key list, keeping the usage/cooldown state of keys that are still present. */
  setKeys(list) {
    const next = new Map();
    for (const k of list) {
      const old = this.keys.get(k.id);
      next.set(k.id, {
        id: k.id,
        label: k.label || k.id,
        source: k.source || 'env',
        secret: k.secret,
        hint: maskKey(k.secret),
        // a key whose secret changed gets a clean slate
        ...(old && old.secret === k.secret ? { cooldownUntil: old.cooldownUntil, disabled: old.disabled, requests: old.requests, limited: old.limited, errors: old.errors, failStreak: old.failStreak, lastUsedAt: old.lastUsedAt, usedSeq: old.usedSeq, lastError: old.lastError, remainingTokens: old.remainingTokens, remainingRequests: old.remainingRequests, tokens: old.tokens } : { usedSeq: 0, cooldownUntil: 0, disabled: false, requests: 0, limited: 0, errors: 0, failStreak: 0, lastUsedAt: 0, lastError: null, remainingTokens: null, remainingRequests: null, tokens: 0 }),
      });
    }
    this.keys = next;
  }

  /** The secret of a key (server-side use only: testing a stored key). */
  secretOf(id) {
    return this.keys.get(id)?.secret ?? null;
  }

  get size() {
    return this.keys.size;
  }

  #state(k) {
    if (k.disabled) return 'disabled';
    return k.cooldownUntil > this.now() ? 'cooling' : 'ready';
  }

  /** Any key that could serve a request right now? */
  hasAvailable() {
    return [...this.keys.values()].some((k) => this.#state(k) === 'ready');
  }

  /**
   * Choose a key. Skips `exclude`d ids (already tried for this request), cooling and disabled keys. Prefers a
   * key whose known remaining token budget covers `needTokens`, then the least recently used (round-robin).
   * @returns {{id:string, secret:string, label:string}|null}
   */
  pick({ exclude = new Set(), needTokens = 0 } = {}) {
    const ready = [...this.keys.values()].filter((k) => !exclude.has(k.id) && this.#state(k) === 'ready');
    if (!ready.length) return null;
    const roomy = ready.filter((k) => k.remainingTokens == null || k.remainingTokens >= needTokens);
    const pool = roomy.length ? roomy : ready;
    pool.sort((a, b) => a.usedSeq - b.usedSeq); // least recently used first = round-robin
    const k = pool[0];
    k.usedSeq = ++this.seq;
    k.lastUsedAt = this.now();
    k.requests++;
    this.usageSink?.(k.id, { calls: 1 });
    return { id: k.id, secret: k.secret, label: k.label };
  }

  success(id, { remainingTokens, remainingRequests, resetTokensMs, resetRequestsMs, tokens = 0, done = false } = {}) {
    const k = this.keys.get(id);
    if (!k) return;
    if (done) this.usageSink?.(id, { ok: 1, tokens }); // the reply finished (the earlier success() is just the response headers)
    k.failStreak = 0;
    k.lastError = null;
    k.tokens += tokens;
    if (remainingTokens != null) k.remainingTokens = remainingTokens;
    if (remainingRequests != null) k.remainingRequests = remainingRequests;
    // Almost out of requests or tokens: rest the key until the window resets instead of spending the next call on a 429.
    if (remainingRequests != null && remainingRequests <= 0 && resetRequestsMs) k.cooldownUntil = this.now() + Math.min(resetRequestsMs, this.maxCooldownMs);
    else if (remainingTokens != null && remainingTokens <= 0 && resetTokensMs) k.cooldownUntil = this.now() + Math.min(resetTokensMs, this.maxCooldownMs);
  }

  /** The provider said "slow down" (HTTP 429). `retryAfterMs` is its own estimate of when this key works again. */
  rateLimited(id, { retryAfterMs, reason = 'rate limit' } = {}) {
    const k = this.keys.get(id);
    if (!k) return;
    k.limited++;
    this.usageSink?.(id, { limited: 1 });
    k.lastError = reason;
    k.cooldownUntil = this.now() + Math.min(Math.max(retryAfterMs ?? this.defaultCooldownMs, 1000), this.maxCooldownMs);
  }

  /** 401/403: the key is wrong or revoked. Skipped until the key list is reloaded. */
  authFailed(id, reason = 'rejected by the provider') {
    const k = this.keys.get(id);
    if (!k) return;
    k.disabled = true;
    k.errors++;
    this.usageSink?.(id, { errors: 1 });
    k.lastError = reason;
  }

  /** Network error / 5xx / timeout: short, growing cooldown so a flaky key is retried but not hammered. */
  transient(id, reason = 'temporary error') {
    const k = this.keys.get(id);
    if (!k) return;
    k.errors++;
    this.usageSink?.(id, { errors: 1 });
    k.failStreak++;
    k.lastError = reason;
    k.cooldownUntil = this.now() + Math.min(2000 * 2 ** (k.failStreak - 1), 60_000);
  }

  /** Re-enable disabled keys (after the owner edited the key list). */
  clearDisabled() {
    for (const k of this.keys.values()) {
      k.disabled = false;
      k.lastError = null;
    }
  }

  /** Milliseconds until the first cooling key is usable again, or null when none is cooling (or all are disabled). */
  nextAvailableMs() {
    const t = this.now();
    const waits = [...this.keys.values()].filter((k) => !k.disabled).map((k) => Math.max(0, k.cooldownUntil - t));
    return waits.length ? Math.min(...waits) : null;
  }

  status() {
    const t = this.now();
    return [...this.keys.values()].map((k) => ({
      id: k.id,
      label: k.label,
      source: k.source,
      hint: k.hint,
      state: this.#state(k),
      cooldownMs: Math.max(0, k.cooldownUntil - t),
      requests: k.requests,
      limited: k.limited,
      errors: k.errors,
      tokens: k.tokens,
      remainingTokens: k.remainingTokens,
      remainingRequests: k.remainingRequests,
      lastError: k.lastError,
    }));
  }
}
