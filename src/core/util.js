// Small dependency-free primitives used across services.
import { createHash, randomUUID } from 'node:crypto';

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const uid = () => randomUUID();
export const sha1 = (s) => createHash('sha1').update(s).digest('hex');

/** Least-recently-used cache with optional TTL — used for in-process memoization. */
export class LRUCache {
  constructor({ max = 500, ttlMs = 0 } = {}) {
    this.max = max;
    this.ttlMs = ttlMs;
    this.map = new Map();
  }
  get(key) {
    const hit = this.map.get(key);
    if (!hit) return undefined;
    if (hit.exp && hit.exp < Date.now()) {
      this.map.delete(key);
      return undefined;
    }
    this.map.delete(key);
    this.map.set(key, hit); // refresh recency
    return hit.value;
  }
  set(key, value) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, { value, exp: this.ttlMs ? Date.now() + this.ttlMs : 0 });
    while (this.map.size > this.max) this.map.delete(this.map.keys().next().value);
    return value;
  }
  clear() {
    this.map.clear();
  }
  get size() {
    return this.map.size;
  }
}

/** memoize(fn, { key, max, ttlMs }) — caches sync results (or the promise of async ones). */
export function memoize(fn, { key = (...a) => JSON.stringify(a), max = 500, ttlMs = 0 } = {}) {
  const cache = new LRUCache({ max, ttlMs });
  const wrapped = (...args) => {
    const k = key(...args);
    const hit = cache.get(k);
    if (hit !== undefined) return hit;
    const value = fn(...args);
    if (value && typeof value.then === 'function') {
      value.catch(() => cache.map.delete(k)); // never cache failures
    }
    return cache.set(k, value);
  };
  wrapped.cache = cache;
  return wrapped;
}

/** Counting semaphore: bounds global concurrency (e.g. parallel LLM generations). */
export class Semaphore {
  constructor(n) {
    this.free = Math.max(1, n);
    this.waiters = [];
  }
  async acquire() {
    if (this.free > 0) {
      this.free--;
      return;
    }
    await new Promise((r) => this.waiters.push(r));
  }
  release() {
    const next = this.waiters.shift();
    if (next) next();
    else this.free++;
  }
  async run(fn) {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

/** Runs tasks serially per key (one chat = one lane) while different keys run in parallel. */
export class KeyedQueue {
  constructor() {
    this.tails = new Map();
  }
  push(key, task) {
    const prev = this.tails.get(key) || Promise.resolve();
    const run = prev.then(() => task());
    // The lane tail never rejects, so one failing task can't poison the next;
    // callers still observe their own task's rejection through `run`.
    const tail = run
      .catch(() => {})
      .finally(() => {
        if (this.tails.get(key) === tail) this.tails.delete(key);
      });
    this.tails.set(key, tail);
    return run;
  }
  get size() {
    return this.tails.size;
  }
}

/** Digits-only phone normalisation: "+91 98765-43210" -> "919876543210". */
export function normalizePhone(input) {
  return String(input ?? '').replace(/\D+/g, '');
}

/** Phone part of a WhatsApp id when it is phone-based ("9198...@c.us"). */
export function phoneFromWid(wid) {
  const m = /^(\d{6,})@c\.us$/.exec(String(wid ?? ''));
  return m ? m[1] : null;
}

export function clamp(n, lo, hi) {
  return Math.min(hi, Math.max(lo, n));
}

export function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}
