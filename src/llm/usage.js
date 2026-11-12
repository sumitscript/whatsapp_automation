// Per-key usage counters kept in Redis (per day), so they are shared by the worker and the gateway and survive restarts.
// (The key pool's own counters live in one process's memory: the console used to show the worker's only, and every deploy
// reset them to zero, which is why they never matched the numbers on console.groq.com.)
//
// Fields per key and day: calls (every attempt sent to Groq, 429s included), ok (finished replies), tokens (prompt +
// completion as Groq reports them), limited (HTTP 429), errors (rejected key, network error, 5xx).
const FIELDS = ['calls', 'ok', 'tokens', 'limited', 'errors', 'neurons']; // neurons: Cloudflare's billing unit (an estimate from the token counts)
const KEEP_DAYS = 45;

/** YYYY-MM-DD of `date` in `timeZone` (the owner's day, not UTC's). */
export function dayKey(date, timeZone = 'Asia/Kolkata') {
  return new Intl.DateTimeFormat('en-CA', { timeZone, year: 'numeric', month: '2-digit', day: '2-digit' }).format(date);
}

export class UsageStore {
  /**
   * @param {{ redis: object, keyFor: (keyId: string, day: string) => string, getTimezone?: () => string, now?: () => Date, onError?: (err: Error) => void }} o
   *   `redis` is ioredis-like: pipeline().hincrby().expire().exec() and pipeline().hgetall().exec().
   */
  constructor({ redis, keyFor, getTimezone = () => 'Asia/Kolkata', now = () => new Date(), onError = () => {} }) {
    Object.assign(this, { redis, keyFor, getTimezone, now, onError });
  }

  /** Add to today's counters. Fire and forget: usage numbers must never slow or break a reply. */
  record(keyId, { model, ...delta }) {
    const fields = Object.entries(delta).filter(([f, n]) => FIELDS.includes(f) && Number.isFinite(n) && n !== 0);
    if (!fields.length) return Promise.resolve();
    const key = this.keyFor(keyId, dayKey(this.now(), this.getTimezone()));
    const p = this.redis.pipeline();
    for (const [f, n] of fields) {
      p.hincrby(key, f, f === 'neurons' ? Math.round(n * 100) : Math.round(n)); // neurons kept in hundredths: a reply costs ~20
      // per model too (limits are per model, so is the usage worth knowing): fields "calls:<model>" / "tokens:<model>"
      if (model && model !== '*' && (f === 'calls' || f === 'tokens')) p.hincrby(key, `${f}:${model}`, Math.round(n));
    }
    p.expire(key, KEEP_DAYS * 86_400);
    return p.exec().catch((err) => this.onError(err));
  }

  /**
   * Totals per key for today, the last 7 days and the last 30 days.
   * @returns {Promise<Record<string, { today: object, d7: object, d30: object }>>}
   */
  async summary(keyIds, { days = 30 } = {}) {
    const tz = this.getTimezone();
    const t = this.now();
    const dayKeys = Array.from({ length: days }, (_, i) => dayKey(new Date(t.getTime() - i * 86_400_000), tz));
    const out = {};
    for (const id of keyIds) {
      const p = this.redis.pipeline();
      for (const d of dayKeys) p.hgetall(this.keyFor(id, d));
      let rows;
      try {
        rows = await p.exec();
      } catch (err) {
        this.onError(err);
        rows = [];
      }
      const sum = (n) => {
        const acc = { ...Object.fromEntries(FIELDS.map((f) => [f, 0])), models: {} };
        rows.slice(0, n).forEach((r) => {
          const h = r && r[1];
          if (!h) return;
          for (const f of FIELDS) acc[f] += f === 'neurons' ? (Number(h[f]) || 0) / 100 : Number(h[f]) || 0;
          for (const [field, v] of Object.entries(h)) {
            const m = /^(calls|tokens):(.+)$/.exec(field);
            if (m) (acc.models[m[2]] ??= { calls: 0, tokens: 0 })[m[1]] += Number(v) || 0;
          }
        });
        return acc;
      };
      out[id] = { today: sum(1), d7: sum(7), d30: sum(30) };
    }
    return out;
  }
}
