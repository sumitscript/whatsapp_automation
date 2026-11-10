import test from 'node:test';
import assert from 'node:assert/strict';
import { SeenIds, sleep } from '../src/core/util.js';
import { EchoTracker } from '../src/core/echo.js';

test('SeenIds: a repeat within the ttl is reported, a new key is not', () => {
  const s = new SeenIds({ ttlMs: 1000, max: 10 });
  assert.equal(s.seen('message:a'), false);
  assert.equal(s.seen('message:a'), true);
  assert.equal(s.seen('own:a'), false, 'the kind is part of the key');
  assert.equal(s.seen('message:b'), false);
  assert.equal(s.size, 3);
});

test('SeenIds: entries expire after the ttl and the set is capped, oldest first', () => {
  let t = 0;
  const s = new SeenIds({ ttlMs: 100, max: 3, now: () => t });
  s.seen('a');
  t = 50;
  assert.equal(s.seen('a'), true);
  t = 150;
  assert.equal(s.seen('a'), false, 'expired: counted as new again');
  for (const k of ['b', 'c', 'd', 'e']) s.seen(k);
  assert.equal(s.size, 3);
  assert.equal(s.has('a'), false, 'the oldest key was evicted');
  assert.equal(s.has('e'), true);
});

test('echo: our own send is recognised by its WhatsApp id, however often the event repeats', () => {
  const e = new EchoTracker();
  e.expect('c1', 'Hello there');
  e.sent('false_c1_ABC');
  assert.equal(e.isOurs('c1', 'Hello there', 'false_c1_ABC'), true);
  assert.equal(e.isOurs('c1', 'Hello there', 'false_c1_ABC'), true, 'a duplicated event of the same message is still ours');
  assert.equal(e.isOurs('c1', 'Hello there', 'false_c1_OTHER'), true, 'the body fallback still has its count for another id');
  assert.equal(e.isOurs('c1', 'Hello there', 'false_c1_THIRD'), false, 'a different message with that text, nothing left to match: the owner typed it');
});

test('echo: an event that beats sendMessage is matched by text, and its id is remembered for the repeat', () => {
  const e = new EchoTracker();
  e.expect('c1', 'Welcome!');
  // the echo arrives before sendMessage resolved: no id known yet
  assert.equal(e.isOurs('c1', 'Welcome!', 'false_c1_W1'), true);
  // the duplicated delivery of the same event used to be mistaken for the owner typing (count already used up)
  assert.equal(e.isOurs('c1', 'Welcome!', 'false_c1_W1'), true);
  e.sent('false_c1_W1');
  assert.equal(e.isOurs('c1', 'Welcome!', 'false_c1_W1'), true);
});

test('echo: what the owner types is not ours, in any chat', () => {
  const e = new EchoTracker();
  e.expect('c1', 'Hi!');
  e.sent('id1');
  assert.equal(e.isOurs('c1', 'on my way', 'id2'), false);
  assert.equal(e.isOurs('c2', 'Hi!', 'id3'), false, 'same text, other chat');
  assert.equal(e.isOurs('c1', 'on my way', undefined), false);
});

test('echo: the text fallback expires', async () => {
  const e = new EchoTracker({ ttlMs: 20 });
  e.expect('c1', 'Hi!');
  await sleep(60);
  assert.equal(e.isOurs('c1', 'Hi!', 'x'), false);
  e.expect('c1', 'Hi!');
  e.clear();
  assert.equal(e.isOurs('c1', 'Hi!', 'y'), false, 'clear() forgets pending text echoes');
});
