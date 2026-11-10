// Redis: short-term conversation memory (temp DB), memoized answers, rate
// limits, live log ring buffer, and cross-process pub/sub.
import Redis from 'ioredis';
import { config } from './config.js';

const P = config.redis.prefix;
export const keys = Object.freeze({
  ctx: (chatId) => `${P}ctx:${chatId}`,
  answer: (k) => `${P}${k}`,
  rate: (scope, id) => `${P}rl:${scope}:${id}:${Math.floor(Date.now() / 60_000)}`,
  takeover: (chatId) => `${P}takeover:${chatId}`,
  logs: `${P}logs`,
  cacheHits: (day) => `${P}stats:cache_hits:${day}`,
  waStatus: `${P}wa:status`,
  workerStatus: `${P}worker:status`,
});

/** Globs for everything stored per chat (wiped when the WhatsApp account is logged out with "delete data"). */
export const chatKeyPatterns = Object.freeze([`${P}ctx:*`, `${P}takeover:*`, `${P}rl:*`]);

export const channels = Object.freeze({
  logs: `${P}ch:logs`,
  settings: `${P}ch:settings`,
  kb: `${P}ch:kb`,
});

export function createRedis(name = 'main') {
  const client = new Redis(config.redis.url, {
    connectionName: `${config.service}:${name}`,
    maxRetriesPerRequest: 3,
    enableAutoPipelining: true,
    retryStrategy: (times) => Math.min(times * 200, 5000),
  });
  return client;
}

// Short-term conversation memory: the raw turns of the current conversation.
// The pipeline picks what fits the context window; the worker folds older turns
// into the contact's long-term memory (Postgres) instead of dropping them.
export const MAX_MEMORY_MESSAGES = 80;

export function createRedisMemory(redis, { ttlSec = config.memory.contextTtlSec } = {}) {
  return {
    async get(chatId) {
      const raw = await redis.lrange(keys.ctx(chatId), 0, -1);
      return raw.map((r) => JSON.parse(r));
    },
    async append(chatId, msgs) {
      const k = keys.ctx(chatId);
      const m = redis.multi();
      for (const msg of msgs) m.rpush(k, JSON.stringify({ role: msg.role, content: msg.content }));
      m.ltrim(k, -MAX_MEMORY_MESSAGES, -1);
      m.expire(k, ttlSec);
      await m.exec();
    },
    /** Remove the oldest n messages (after they were folded into long-term memory). */
    async trimOldest(chatId, n) {
      if (n > 0) await redis.ltrim(keys.ctx(chatId), n, -1);
    },
    async clear(chatId) {
      await redis.del(keys.ctx(chatId));
    },
  };
}

export function createRedisAnswerCache(redis, { ttlSec = config.memory.answerCacheTtlSec } = {}) {
  return {
    async get(k) {
      const raw = await redis.get(keys.answer(k));
      if (!raw) return null;
      redis.incr(keys.cacheHits(new Date().toISOString().slice(0, 10))).catch(() => {});
      return JSON.parse(raw);
    },
    async set(k, v) {
      await redis.set(keys.answer(k), JSON.stringify(v), 'EX', ttlSec);
    },
  };
}

/** Fixed-window counter; returns true when the caller is within the limit. */
export async function withinRate(redis, scope, id, limitPerMin) {
  const k = keys.rate(scope, id);
  const [[, n]] = await redis.multi().incr(k).expire(k, 61).exec();
  return n <= limitPerMin;
}

/** Ship every log entry to a capped Redis list + pub/sub (powers the UI tail). */
export function attachRedisLogSink(logger, redis, { size = config.logs.bufferSize } = {}) {
  let broken = false;
  return logger.addSink((_entry, line) => {
    if (broken) return;
    redis
      .multi()
      .lpush(keys.logs, line)
      .ltrim(keys.logs, 0, size - 1)
      .publish(channels.logs, line)
      .exec()
      .catch(() => {
        broken = true; // stop hammering a dead Redis; re-enable shortly
        setTimeout(() => (broken = false), 5000);
      });
  });
}

export async function recentLogs(redis, { limit = 500, level, service } = {}) {
  const order = { debug: 10, info: 20, warn: 30, error: 40 };
  const raw = await redis.lrange(keys.logs, 0, Math.min(limit * 3, config.logs.bufferSize) - 1);
  const out = [];
  for (const line of raw) {
    try {
      const e = JSON.parse(line);
      if (level && (order[e.level] ?? 0) < (order[level] ?? 0)) continue;
      if (service && e.service !== service) continue;
      out.push(e);
      if (out.length >= limit) break;
    } catch {
      /* skip malformed */
    }
  }
  return out.reverse(); // oldest first
}
