import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KeyedQueue, LRUCache, memoize, normalizePhone, phoneFromWid, Semaphore, sleep } from '../src/core/util.js';
import { DEFAULT_SETTINGS, ownerAliasList, sanitizeSettings } from '../src/core/settings-schema.js';

test('LRU evicts least recently used', () => {
  const c = new LRUCache({ max: 2 });
  c.set('a', 1);
  c.set('b', 2);
  c.get('a');
  c.set('c', 3);
  assert.equal(c.get('b'), undefined);
  assert.equal(c.get('a'), 1);
});

test('memoize caches and never caches failures', async () => {
  let calls = 0;
  const f = memoize(async (x) => {
    calls++;
    if (x === 'bad') throw new Error('no');
    return x * 2;
  });
  assert.equal(await f(2), 4);
  assert.equal(await f(2), 4);
  assert.equal(calls, 1);
  await assert.rejects(f('bad'));
  await assert.rejects(f('bad'));
  assert.equal(calls, 3);
});

test('KeyedQueue serialises per key, parallel across keys, survives failures', async () => {
  const q = new KeyedQueue();
  const order = [];
  const a1 = q.push('a', async () => {
    await sleep(20);
    order.push('a1');
  });
  const a2 = q.push('a', async () => {
    throw new Error('boom');
  });
  const a3 = q.push('a', async () => order.push('a3'));
  const b1 = q.push('b', async () => order.push('b1'));
  await Promise.allSettled([a1, a2, a3, b1]);
  assert.deepEqual(order, ['b1', 'a1', 'a3']);
  await assert.rejects(a2);
  await sleep(0);
  assert.equal(q.size, 0);
});

test('Semaphore bounds concurrency', async () => {
  const s = new Semaphore(2);
  let active = 0;
  let peak = 0;
  await Promise.all(
    Array.from({ length: 6 }, () =>
      s.run(async () => {
        peak = Math.max(peak, ++active);
        await sleep(5);
        active--;
      }),
    ),
  );
  assert.equal(peak, 2);
});

test('phone helpers', () => {
  assert.equal(normalizePhone('+91 98765-43210'), '919876543210');
  assert.equal(phoneFromWid('919876543210@c.us'), '919876543210');
  assert.equal(phoneFromWid('123@lid'), null);
});

test('settings validation', () => {
  const { value, errors } = sanitizeSettings({ replyMode: 'all', temperature: 0.3, maxTokens: '400', bogus: 1, historyTurns: 999 });
  assert.deepEqual(value, { replyMode: 'all', temperature: 0.3, maxTokens: 400 });
  assert.equal(errors.length, 2);
  assert.ok(ownerAliasList(DEFAULT_SETTINGS).includes('sumitscript'));
});
