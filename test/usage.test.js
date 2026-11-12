import test from 'node:test';
import assert from 'node:assert/strict';
import { UsageStore, dayKey } from '../src/llm/usage.js';
import { KeyPool } from '../src/llm/key-pool.js';
import { GroqClient } from '../src/llm/groq.js';
import { statedLanguagePreference, replyLanguage } from '../src/brain/lang.js';
import { guessLocation, isMeetingRequest, mentionsMeeting, wantsToMeet, parseWhen, formatWhen } from '../src/brain/agenda.js';
import { meetingAck } from '../src/brain/invite.js';

// ------------------------------------------------------------------ usage counters

// A fake of the ioredis calls UsageStore uses: pipeline().hincrby/expire/hgetall().exec()
function fakeRedis() {
  const data = new Map(); // key → { field: number }
  const ttl = new Map();
  return {
    data,
    ttl,
    pipeline() {
      const ops = [];
      const p = {
        hincrby: (k, f, n) => (ops.push(() => { const h = data.get(k) || {}; h[f] = (h[f] || 0) + n; data.set(k, h); return [null, h[f]]; }), p),
        expire: (k, s) => (ops.push(() => { ttl.set(k, s); return [null, 1]; }), p),
        hgetall: (k) => (ops.push(() => [null, Object.fromEntries(Object.entries(data.get(k) || {}).map(([f, n]) => [f, String(n)]))]), p),
        exec: async () => ops.map((o) => o()),
      };
      return p;
    },
  };
}

test('usage is counted per key and per day (in the owner\'s timezone) and survives a new process', async () => {
  const redis = fakeRedis();
  let now = new Date('2026-11-12T08:00:00Z'); // 13:30 IST
  const mk = () => new UsageStore({ redis, keyFor: (k, d) => `u:${k}:${d}`, getTimezone: () => 'Asia/Kolkata', now: () => now });
  const a = mk();
  await a.record('db:1', { calls: 1 });
  await a.record('db:1', { ok: 1, tokens: 1040 });
  await a.record('db:1', { calls: 1, limited: 1 });
  await a.record('db:2', { calls: 3 });
  assert.equal(dayKey(new Date('2026-11-12T20:00:00Z'), 'Asia/Kolkata'), '2026-11-13', 'the owner\'s day, not UTC\'s');
  // yesterday's traffic (the other process, before a restart)
  now = new Date('2026-11-11T08:00:00Z');
  await mk().record('db:1', { calls: 10, tokens: 16_260 });
  now = new Date('2026-11-12T08:30:00Z');
  const s = await mk().summary(['db:1', 'db:2', 'db:3']);
  assert.deepEqual({ ...s['db:1'].today, models: undefined }, { calls: 2, ok: 1, tokens: 1040, limited: 1, errors: 0, models: undefined });
  assert.equal(s['db:1'].d7.calls, 12);
  assert.equal(s['db:1'].d30.tokens, 17_300);
  assert.equal(s['db:2'].today.calls, 3);
  assert.deepEqual({ ...s['db:3'].today, models: undefined }, { calls: 0, ok: 0, tokens: 0, limited: 0, errors: 0, models: undefined }, 'a key never used reads zero');
  assert.ok([...redis.ttl.values()].every((t) => t >= 40 * 86_400), 'counters expire on their own');
});

test('usage recording never throws or blocks: bad deltas are ignored, a failing redis is swallowed', async () => {
  const seen = [];
  const broken = { pipeline: () => ({ hincrby() { return this; }, expire() { return this; }, exec: async () => { throw new Error('redis down'); } }) };
  const u = new UsageStore({ redis: broken, keyFor: (k, d) => `${k}:${d}`, onError: (e) => seen.push(e.message) });
  await u.record('k', { calls: 1 });
  await u.record('k', { bogus: 5, calls: NaN, tokens: 0 });
  assert.deepEqual(seen, ['redis down']);
});

test('the key pool reports calls, finished replies with tokens, rate limits and errors to the sink', () => {
  const events = [];
  const pool = new KeyPool();
  pool.usageSink = (id, d) => events.push([id, d]);
  pool.setStrategy('spread'); // accounts take turns, so the second pick is account b
  pool.setKeys([{ id: 'a', label: 'a', secret: 'gsk_secret_a_xxxxxxxxxxxx' }, { id: 'b', label: 'b', secret: 'gsk_secret_b_xxxxxxxxxxxx' }]);
  pool.pick(); // a
  pool.success('a', { remainingTokens: 5000 }); // response headers only: not a finished reply
  pool.success('a', { tokens: 1040, done: true });
  pool.pick(); // b
  pool.rateLimited('b', { retryAfterMs: 1000 });
  pool.transient('a');
  pool.authFailed('b');
  assert.deepEqual(events, [
    ['a', { calls: 1, model: '*' }],
    ['a', { ok: 1, tokens: 1040, model: '*' }],
    ['b', { calls: 1, model: '*' }],
    ['b', { limited: 1, model: '*' }],
    ['a', { errors: 1 }],
    ['b', { errors: 1 }],
  ]);
});

test('a Groq reply records its tokens even when only prompt/completion counts are sent', async () => {
  const sse = 'data: ' + JSON.stringify({ choices: [{ delta: { content: 'hi' } }] }) + '\n\ndata: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], x_groq: { usage: { prompt_tokens: 1000, completion_tokens: 20 } } }) + '\n\ndata: [DONE]\n\n';
  const fetchImpl = async () => new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  const events = [];
  const pool = new KeyPool();
  pool.usageSink = (id, d) => events.push([id, d]);
  pool.setKeys([{ id: 'a', label: 'a', secret: 'gsk_secret_a_xxxxxxxxxxxx' }]);
  const g = new GroqClient({ pool, baseUrl: 'http://x', fetchImpl });
  for await (const ev of g.chat({ model: 'm', messages: [{ role: 'user', content: 'hi' }] })) void ev;
  assert.deepEqual(events.filter(([, d]) => d.ok), [['a', { ok: 1, tokens: 1020, model: 'm' }]]);
});

// ------------------------------------------------------------------ language instructions

test('"Talk to me in English" (and its everyday forms) is an instruction; a plain question that mentions English is not', () => {
  for (const t of ['Talk to me in English', 'talk with me in english please', 'Speak to me in Hindi', 'reply in english', 'answer in hinglish', 'English please', 'in english', 'English only', 'hinglish mein baat karo']) {
    assert.ok(statedLanguagePreference([t]), t);
  }
  for (const t of ['what is the english word for pyaar', 'translate this to english', 'my english is weak', 'I like english movies']) {
    assert.equal(statedLanguagePreference([t]), null, t);
  }
  assert.equal(statedLanguagePreference(['Talk to me in English']).code, 'en');
  assert.equal(statedLanguagePreference(['reply in hindi', 'now talk to me in english']).code, 'en', 'newest wins');
});

test('the instruction beats a "Prefers Hinglish" note; plain English for a few messages also outgrows a note', () => {
  const notes = '- Name: Aditya\n- Prefers Hinglish';
  assert.equal(replyLanguage('Talk to me in English', ['Where taj mahal is located'], notes).code, 'en');
  assert.equal(replyLanguage('Where is the Eiffel tower', ['Talk to me in English'], notes).code, 'en', 'and it stays English afterwards');
  // fresh chat: the remembered preference still applies to the first message…
  assert.equal(replyLanguage('Where taj mahal is located', [], notes).code, 'hi-Latn');
  // …but not after the person has written English twice in a row
  assert.equal(replyLanguage('what is the capital of France', ['Where taj mahal is located today', 'tell me about the history of india'], notes).code, 'en');
  // an explicit preference said in this chat is never outgrown
  assert.equal(replyLanguage('what is the capital of France', ['I prefer hinglish', 'Where taj mahal is located today'], '').code, 'hi-Latn');
});

// ------------------------------------------------------------------ "fix a date for me"

test('asking the twin to fix a date / book a slot is a meeting request, never agreed to', () => {
  const NOW = new Date('2026-11-12T08:00:00Z');
  for (const t of ['Fix a date for me on next monday', 'book a slot for friday', 'can you fix a time for us', 'please arrange a meeting', 'set up a call tomorrow 5pm']) {
    assert.ok(isMeetingRequest(t) || wantsToMeet(t), t);
    assert.ok(mentionsMeeting(t) || wantsToMeet(t), t);
  }
  const hit = parseWhen('Fix a date for me on next monday', { now: NOW, timeZone: 'Asia/Kolkata' });
  const ack = meetingAck({ code: 'en', name: 'Aditya Soni', owner: 'Sumit Soni', dueAt: hit.dueAt, allDay: hit.allDay, timeZone: 'Asia/Kolkata' });
  assert.match(ack, /Mon, 16 Nov/);
  assert.match(ack, /What time works for you\?/);
  assert.doesNotMatch(ack, /lock|sure|done|booked|confirmed/i, 'it does not agree on the owner\'s behalf');
  assert.equal(isMeetingRequest('the fix for this bug is in the book I read'), false);
  assert.ok(formatWhen(hit.dueAt, 'Asia/Kolkata', true));
});

test('a booking request does not lend its words as a place', () => {
  for (const x of ['Fix a date for me on next monday', 'book a slot for us on friday', 'can you fix a time for me', 'Tell him hi for me']) assert.equal(guessLocation(x), null, x);
  assert.equal(guessLocation('fix a date on friday at the railway station'), 'Railway station');
});

test('usage is also kept per model, because the limits are per model', async () => {
  const redis = fakeRedis();
  const now = new Date('2026-11-12T08:00:00Z');
  const u = new UsageStore({ redis, keyFor: (k, d) => k + ':' + d, getTimezone: () => 'Asia/Kolkata', now: () => now });
  await u.record('db:1', { calls: 1, model: 'openai/gpt-oss-120b' });
  await u.record('db:1', { ok: 1, tokens: 1000, model: 'openai/gpt-oss-120b' });
  await u.record('db:1', { calls: 1, model: 'openai/gpt-oss-20b' });
  await u.record('db:1', { limited: 1, model: 'openai/gpt-oss-120b' });
  const s = await u.summary(['db:1']);
  assert.equal(s['db:1'].today.calls, 2);
  assert.equal(s['db:1'].today.limited, 1);
  assert.deepEqual(s['db:1'].today.models, { 'openai/gpt-oss-120b': { calls: 1, tokens: 1000 }, 'openai/gpt-oss-20b': { calls: 1, tokens: 0 } });
});
