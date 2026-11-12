import test from 'node:test';
import assert from 'node:assert/strict';
import { SelfChatTracker } from '../src/core/selfchat.js';
import { trySendStrategies } from '../src/core/image-send.js';
import { EchoTracker } from '../src/core/echo.js';

const ME = '918756365339@c.us';
const LID = '203976637206652@lid';

test('the phone id and the LID do not match by themselves: that was the bug', () => {
  const t = new SelfChatTracker();
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: LID, me: ME }), false);
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: '911111111111@c.us', me: ME }), false, 'another contact');
});

test('the cheap checks still work: same id both ways, or the same phone number', () => {
  const t = new SelfChatTracker();
  assert.equal(t.isSelf({ fromMe: true, from: LID, to: LID, me: ME }), true);
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: ME, me: ME }), true);
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: '918756365339@c.us', me: '918756365339:12@c.us' }), true, 'device suffix is ignored');
  assert.equal(t.isSelf({ fromMe: false, from: LID, to: LID, me: ME }), false, 'only the owner can write fromMe');
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: '', me: ME }), false);
});

test('the echo of a notice we sent teaches the self chat id, by text (event beat sendMessage) or by message id', () => {
  const t = new SelfChatTracker();
  t.rememberNotice('📅 Meeting request *#1* from Aditya');
  // the event arrives before sendMessage resolved: only the text is known
  assert.equal(t.matchesNotice('📅 Meeting request *#1* from Aditya', 'false_203976637206652@lid_ABC'), true);
  assert.equal(t.matchesNotice('something the owner typed', 'false_203976637206652@lid_XYZ'), false);
  // after sendMessage resolved the id is known too (a repeat of the same event, text changed by WhatsApp)
  t.rememberNotice('📅 Meeting request *#2*', 'true_918756365339@c.us_DEF');
  assert.equal(t.matchesNotice('different text', 'true_918756365339@c.us_DEF'), true);
  // learning the id of the chat makes every later message there recognisable without any notice
  assert.equal(t.learn(LID), true);
  assert.equal(t.learn(LID), false, 'already known');
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: LID, me: ME }), true);
  assert.equal(t.isLearned(LID), true);
});

test('notices are forgotten after their TTL, and the memory is bounded', () => {
  let now = 1000;
  const t = new SelfChatTracker({ noticeTtlMs: 100, maxNotices: 3, now: () => now });
  t.rememberNotice('old', 'id-old');
  now += 101;
  assert.equal(t.matchesNotice('old', 'id-old'), false);
  for (let i = 0; i < 10; i++) t.rememberNotice(`n${i}`, `id${i}`);
  assert.ok(t.texts.size <= 3 && t.msgIds.size <= 3);
  assert.equal(t.matchesNotice('n9', null), true, 'the newest survive');
});

test('learned ids can be loaded from storage and cleared on logout', () => {
  const t = new SelfChatTracker();
  t.load([LID, '', null]);
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: LID, me: ME }), true);
  t.clear();
  assert.equal(t.isSelf({ fromMe: true, from: ME, to: LID, me: ME }), false);
});

test('together with the echo tracker: our notice echo (under the LID) is ours, the owner\'s "ok 2" is not', () => {
  const echo = new EchoTracker();
  const self = new SelfChatTracker();
  const notice = '📅 Meeting request *#2* from Aditya Soni';
  echo.expect(ME, notice); // what the inbox registers when it sends to the phone id
  self.rememberNotice(notice);
  // the echo arrives under the LID: the text key under the phone id does not match, the notice memory does
  assert.equal(echo.isOurs(LID, notice, 'id-1'), false);
  assert.equal(self.matchesNotice(notice, 'id-1'), true);
  // the owner replies
  assert.equal(self.matchesNotice('ok 2', 'id-2'), false);
  assert.equal(echo.isOurs(LID, 'ok 2', 'id-2'), false);
});

// ------------------------------------------------------------------ image send ways

const way = (name, outcome) => ({ name, run: async () => (outcome instanceof Error ? Promise.reject(outcome) : outcome) });
const boom = (m) => new Error(`${m}\n    at s (https://static.whatsapp.net/x.js:85:180)`);

test('image send: the first way that works wins, failures are reported one line each', async () => {
  const r = await trySendStrategies([way('direct', boom('Data passed to getter must include an id property')), way('document', { id: 'M1' }), way('chat', { id: 'M2' })]);
  assert.equal(r.strategy, 'document');
  assert.deepEqual(r.sent, { id: 'M1' });
  assert.deepEqual(r.results.map((x) => [x.strategy, x.ok]), [['direct', false], ['document', true]], 'stops after the first success');
  assert.equal(r.results[0].error, 'Data passed to getter must include an id property', 'one line, no stack');
});

test('image send: the way that worked last is tried first next time', async () => {
  const calls = [];
  const mk = (name, ok) => ({ name, run: async () => { calls.push(name); if (!ok) throw new Error(`${name} failed`); return { id: name }; } });
  const ways = [mk('direct', false), mk('document', false), mk('chat', true)];
  const r = await trySendStrategies(ways, { preferred: 'chat' });
  assert.equal(r.strategy, 'chat');
  assert.deepEqual(calls, ['chat']);
});

test('image send: nothing works → no message and every error listed; diagnostics can try all ways', async () => {
  const none = await trySendStrategies([way('direct', boom('a')), way('chat', boom('b'))]);
  assert.equal(none.sent, null);
  assert.equal(none.strategy, null);
  assert.deepEqual(none.results.map((x) => x.error), ['a', 'b']);
  const all = await trySendStrategies([way('direct', { id: 'A' }), way('document', boom('x')), way('chat', { id: 'C' })], { all: true });
  assert.equal(all.strategy, 'direct');
  assert.deepEqual(all.results.map((x) => [x.strategy, x.ok]), [['direct', true], ['document', false], ['chat', true]]);
});
