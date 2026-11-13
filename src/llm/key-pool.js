// A pool of API keys ("accounts") for one provider, with per-model limits. Pure bookkeeping (no I/O), unit-tested.
//
// Groq limits are per MODEL and per ORGANIZATION: hitting the gpt-oss-120b limit says nothing about gpt-oss-20b, and two
// keys of the same organization share their limits (adding more keys to one org adds no capacity). So the unit that gets
// "rate limited" is the pair (organization-or-key, model), not the whole key:
//
//   priority strategy (default)  accounts in the order listed, and for each account the models in priority order: the first
//                                ready (account, model) pair serves. Account 1's models are used up before account 2 is touched.
//   spread strategy              accounts take turns (least recently used first), models still in priority order.
//
// A 429 puts only that pair to rest (until the provider's own retry-after); when the message names the organization the rest
// applies to every key of that organization. 404 "model not found" retires the pair for hours. 401/403 disables the key.
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

const ANY = '*'; // the single pseudo-model used when no model list is given

export class KeyPool {
  constructor({ now = Date.now, defaultCooldownMs = 60_000, maxCooldownMs = 6 * 3_600_000, modelGoneMs = 6 * 3_600_000 } = {}) {
    this.now = now;
    this.defaultCooldownMs = defaultCooldownMs;
    this.maxCooldownMs = maxCooldownMs;
    this.modelGoneMs = modelGoneMs;
    this.keys = new Map(); // id → key state (insertion order = priority order)
    this.pairs = new Map(); // `${org-or-key}|${model}` → { cooldownUntil, remainingTokens, remainingRequests, limited, lastError, gone }
    this.models = [ANY];
    this.strategy = 'priority';
    this.seq = 0;
    this.usageSink = null; // (keyId, { calls, ok, tokens, limited, errors, model }) => void: persists usage outside this process
  }

  /** Replace the key list (its order is the priority order), keeping the state of keys that are still present. */
  setKeys(list) {
    const next = new Map();
    for (const k of list) {
      const old = this.keys.get(k.id);
      const same = old && old.secret === k.secret;
      next.set(k.id, {
        id: k.id,
        label: k.label || k.id,
        source: k.source || 'env',
        secret: k.secret,
        hint: maskKey(k.secret),
        // a key whose secret changed gets a clean slate
        usedSeq: same ? old.usedSeq : 0,
        cooldownUntil: same ? old.cooldownUntil : 0, // key-level rest (network trouble)
        disabled: same ? old.disabled : false,
        requests: same ? old.requests : 0,
        limited: same ? old.limited : 0,
        errors: same ? old.errors : 0,
        failStreak: same ? old.failStreak : 0,
        lastUsedAt: same ? old.lastUsedAt : 0,
        lastError: same ? old.lastError : null,
        tokens: same ? old.tokens : 0,
        org: same ? old.org : null, // learned from rate-limit messages
      });
    }
    this.keys = next;
  }

  setModels(models) {
    const list = [...new Set((models || []).map((m) => String(m).trim()).filter(Boolean))];
    this.models = list.length ? list : [ANY];
  }

  setStrategy(strategy) {
    this.strategy = strategy === 'spread' ? 'spread' : 'priority';
  }

  get size() {
    return this.keys.size;
  }

  /** The secret of a key (server-side use only: testing a stored key). */
  secretOf(id) {
    return this.keys.get(id)?.secret ?? null;
  }

  #scope(k) {
    return k.org || k.id;
  }

  #pair(k, model, create = false) {
    const key = `${this.#scope(k)}|${model}`;
    let p = this.pairs.get(key);
    if (!p && create) {
      p = { cooldownUntil: 0, remainingTokens: null, remainingRequests: null, limitTokens: null, limitRequests: null, resetTokensAt: 0, resetRequestsAt: 0, hit: null, limited: 0, lastError: null, gone: false };
      this.pairs.set(key, p);
    }
    return p;
  }

  /** ready | cooling | disabled for this key and model. */
  #state(k, model) {
    if (k.disabled) return 'disabled';
    const t = this.now();
    if (k.cooldownUntil > t) return 'cooling';
    const p = this.#pair(k, model);
    return p && p.cooldownUntil > t ? 'cooling' : 'ready';
  }

  #cooldownMs(k, model) {
    const t = this.now();
    return Math.max(0, k.cooldownUntil - t, (this.#pair(k, model)?.cooldownUntil ?? 0) - t);
  }

  /** Any key that could serve at least one of the models right now? */
  hasAvailable(models = this.models) {
    return [...this.keys.values()].some((k) => models.some((m) => this.#state(k, m) === 'ready'));
  }

  /**
   * Choose the next (account, model). `exclude` holds key ids (skip the whole key) or "id|model" pairs already tried for
   * this request. Prefers a pair whose known remaining token budget covers `needTokens`; if none does, any ready pair.
   * @returns {{id:string, secret:string, label:string, model:string}|null}
   */
  pick({ exclude = new Set(), needTokens = 0, models = this.models } = {}) {
    const list = models?.length ? models : [ANY];
    const order = [...this.keys.values()];
    if (this.strategy === 'spread') order.sort((a, b) => a.usedSeq - b.usedSeq); // least recently used account first
    const candidates = [];
    for (const k of order) {
      if (exclude.has(k.id)) continue;
      for (const m of list) {
        if (exclude.has(`${k.id}|${m}`) || this.#state(k, m) !== 'ready') continue;
        candidates.push({ k, m, rem: this.#pair(k, m)?.remainingTokens ?? null });
      }
    }
    if (!candidates.length) return null;
    const roomy = candidates.find((c) => c.rem == null || c.rem >= needTokens);
    const { k, m } = roomy || candidates[0];
    k.usedSeq = ++this.seq;
    k.lastUsedAt = this.now();
    k.requests++;
    this.usageSink?.(k.id, { calls: 1, model: m });
    return { id: k.id, secret: k.secret, label: k.label, model: m };
  }

  success(id, { model = ANY, remainingTokens, remainingRequests, limitTokens, limitRequests, resetTokensMs, resetRequestsMs, tokens = 0, neurons = 0, done = false } = {}) {
    const k = this.keys.get(id);
    if (!k) return;
    if (done) this.usageSink?.(id, { ok: 1, tokens, model, ...(neurons ? { neurons } : {}) }); // the reply finished (the earlier success() is just the response headers)
    k.failStreak = 0;
    k.lastError = null;
    k.tokens += tokens;
    const p = this.#pair(k, model, true);
    if (remainingTokens != null) p.remainingTokens = remainingTokens;
    if (remainingRequests != null) p.remainingRequests = remainingRequests;
    if (limitTokens != null) p.limitTokens = limitTokens;
    if (limitRequests != null) p.limitRequests = limitRequests;
    if (resetTokensMs != null) p.resetTokensAt = this.now() + resetTokensMs;
    if (resetRequestsMs != null) p.resetRequestsAt = this.now() + resetRequestsMs;
    if (remainingTokens != null || remainingRequests != null) p.hit = null;
    // Almost out of requests or tokens for this model: rest the pair until the window resets instead of spending the next call on a 429.
    if (remainingRequests != null && remainingRequests <= 0 && resetRequestsMs) p.cooldownUntil = this.now() + Math.min(resetRequestsMs, this.maxCooldownMs);
    else if (remainingTokens != null && remainingTokens <= 0 && resetTokensMs) p.cooldownUntil = this.now() + Math.min(resetTokensMs, this.maxCooldownMs);
  }

  /**
   * HTTP 429 for this model. `org` (from the provider's message) makes the rest apply to every key of that organization:
   * they share the limit.
   */
  rateLimited(id, { model = ANY, retryAfterMs, reason = 'rate limit', org = null, hit = null, daily = false } = {}) {
    const k = this.keys.get(id);
    if (!k) return;
    if (org) k.org = org;
    k.limited++;
    this.usageSink?.(id, { limited: 1, model });
    const p = this.#pair(k, model, true);
    p.limited++;
    p.lastError = reason;
    if (hit) p.hit = { ...hit, at: this.now() }; // which limit was reached (tokens per day…) and what the provider said it was
    p.cooldownUntil = this.now() + Math.min(Math.max(retryAfterMs ?? this.defaultCooldownMs, 1000), daily ? 25 * 3_600_000 : this.maxCooldownMs); // a daily allowance may rest for most of a day
  }

  /** 404 "model does not exist / no access": this account cannot use the model; stop trying it for hours. */
  modelUnavailable(id, model, reason = 'model not available to this account') {
    const k = this.keys.get(id);
    if (!k) return;
    const p = this.#pair(k, model, true);
    p.gone = true;
    p.lastError = reason;
    p.cooldownUntil = this.now() + this.modelGoneMs;
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

  /** Network error / 5xx / timeout: short, growing rest for the whole key so a flaky one is retried but not hammered. */
  transient(id, reason = 'temporary error') {
    const k = this.keys.get(id);
    if (!k) return;
    k.errors++;
    this.usageSink?.(id, { errors: 1 });
    k.failStreak++;
    k.lastError = reason;
    k.cooldownUntil = this.now() + Math.min(2000 * 2 ** (k.failStreak - 1), 60_000);
  }

  /** Re-enable disabled keys and forget "model gone" marks (after the owner edited the key or model list). */
  clearDisabled() {
    for (const k of this.keys.values()) {
      k.disabled = false;
      k.lastError = null;
    }
    for (const p of this.pairs.values()) if (p.gone) p.cooldownUntil = 0;
  }

  /** Milliseconds until the first resting pair is usable again, or null when none is resting (or every key is disabled). */
  nextAvailableMs(models = this.models) {
    const list = models?.length ? models : [ANY];
    const waits = [];
    for (const k of this.keys.values()) {
      if (k.disabled) continue;
      for (const m of list) waits.push(this.#cooldownMs(k, m));
    }
    return waits.length ? Math.min(...waits) : null;
  }

  /** What the provider last told us about this model's limits; a window whose reset time has passed counts as full again. */
  #limits(p) {
    if (!p) return {};
    const t = this.now();
    const one = (limit, remaining, resetAt) => {
      if (limit == null && remaining == null) return null;
      const fresh = resetAt && resetAt <= t; // the window has reset since we last heard
      return { limit: limit ?? null, remaining: fresh ? (limit ?? null) : remaining, resetAt: fresh ? null : resetAt || null };
    };
    return { requests: one(p.limitRequests, p.remainingRequests, p.resetRequestsAt), tokens: one(p.limitTokens, p.remainingTokens, p.resetTokensAt) };
  }

  status(models = this.models) {
    const list = models?.length ? models : [ANY];
    const all = [...this.keys.values()];
    return all.map((k, i) => {
      const perModel = list.map((m) => {
        const p = this.#pair(k, m);
        return { id: m, state: this.#state(k, m), cooldownMs: this.#cooldownMs(k, m), restUntil: this.#cooldownMs(k, m) > 0 ? this.now() + this.#cooldownMs(k, m) : null, remainingTokens: p?.remainingTokens ?? null, remainingRequests: p?.remainingRequests ?? null, limits: this.#limits(p), hit: p?.hit ?? null, limited: p?.limited ?? 0, gone: !!p?.gone, lastError: p?.lastError ?? null };
      });
      const ready = perModel.filter((x) => x.state === 'ready').length;
      const state = k.disabled ? 'disabled' : ready ? 'ready' : 'cooling';
      const first = k.org ? all.findIndex((o) => o.org === k.org) : -1;
      return {
        id: k.id,
        label: k.label,
        source: k.source,
        hint: k.hint,
        priority: i + 1,
        org: k.org ? maskKey(k.org) : null,
        sameOrgAs: first >= 0 && first < i ? all[first].label : null,
        state,
        cooldownMs: state === 'cooling' ? Math.min(...perModel.map((x) => x.cooldownMs)) : 0,
        models: perModel,
        requests: k.requests,
        limited: k.limited,
        errors: k.errors,
        tokens: k.tokens,
        remainingTokens: perModel.find((x) => x.remainingTokens != null)?.remainingTokens ?? null,
        remainingRequests: perModel.find((x) => x.remainingRequests != null)?.remainingRequests ?? null,
        lastError: k.lastError || perModel.find((x) => x.state !== 'ready' && x.lastError)?.lastError || null,
      };
    });
  }
}
