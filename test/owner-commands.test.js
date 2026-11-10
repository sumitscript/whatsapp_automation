import test from 'node:test';
import assert from 'node:assert/strict';
import { ownerAsk, ownerHelp, ownerNotice, ownerPendingList, ownerResultLine, parseOwnerCommand, resolveDecision } from '../src/brain/owner-commands.js';
import { invitationData, invitationHtml } from '../src/brain/invite.js';

const decide = (action, ids = null, all = false) => ({ type: 'decide', action, ids, all });

test('the owner can answer with a number and a word, in either order, in English, Hinglish, Hindi or Gujarati', () => {
  assert.deepEqual(parseOwnerCommand('ok 12'), decide('confirm', [12]));
  assert.deepEqual(parseOwnerCommand('12 yes'), decide('confirm', [12]));
  assert.deepEqual(parseOwnerCommand('Confirm #7'), decide('confirm', [7]));
  assert.deepEqual(parseOwnerCommand('yes please 3'), decide('confirm', [3]));
  assert.deepEqual(parseOwnerCommand('confirm number 5!'), decide('confirm', [5]));
  assert.deepEqual(parseOwnerCommand('no 4'), decide('decline', [4]));
  assert.deepEqual(parseOwnerCommand('decline 9'), decide('decline', [9]));
  assert.deepEqual(parseOwnerCommand('4 nahi'), decide('decline', [4]));
  assert.deepEqual(parseOwnerCommand('haan 8 kar do'), decide('confirm', [8]));
  assert.deepEqual(parseOwnerCommand('theek hai 2'), decide('confirm', [2]));
  assert.deepEqual(parseOwnerCommand('हाँ 6'), decide('confirm', [6]));
  assert.deepEqual(parseOwnerCommand('નહીં 6'), decide('decline', [6]));
  assert.deepEqual(parseOwnerCommand('done 11'), decide('done', [11]));
  assert.deepEqual(parseOwnerCommand('✅ 5'), decide('confirm', [5]));
});

test('several requests at once, everything, and a bare answer', () => {
  assert.deepEqual(parseOwnerCommand('ok 3 5'), decide('confirm', [3, 5]));
  assert.deepEqual(parseOwnerCommand('confirm 3, 5 and 8'), decide('confirm', [3, 5, 8]));
  assert.deepEqual(parseOwnerCommand('ok all'), decide('confirm', null, true));
  assert.deepEqual(parseOwnerCommand('no all'), decide('decline', null, true));
  assert.deepEqual(parseOwnerCommand('ok'), decide('confirm'));
  assert.deepEqual(parseOwnerCommand('Yes!'), decide('confirm'));
  assert.deepEqual(parseOwnerCommand('no'), decide('decline'));
});

test('listing and help', () => {
  assert.deepEqual(parseOwnerCommand('pending'), { type: 'list' });
  assert.deepEqual(parseOwnerCommand('List'), { type: 'list' });
  assert.deepEqual(parseOwnerCommand('?'), { type: 'list' });
  assert.deepEqual(parseOwnerCommand('help'), { type: 'help' });
});

test('ordinary notes to self are not commands', () => {
  for (const t of ['buy milk tomorrow', 'ok thanks for the update', 'remind me to call mom', '12', '12/11', 'ok 12 but move it to 5pm', 'yes no', '', '   ', 'a'.repeat(200)]) {
    assert.equal(parseOwnerCommand(t), null, JSON.stringify(t));
  }
});

const item = (id, over = {}) => ({ id, kind: 'appointment', status: 'pending', title: `Meet P${id}`, contactName: `Person ${id}`, dueAt: '2026-11-11T00:30:00Z', allDay: false, location: 'Station', ...over });

test('which request does the answer apply to?', () => {
  const a = item(3);
  const b = item(5);
  const done = item(9, { status: 'done' });
  const byId = (n) => [a, b, done].find((i) => i.id === n) || null;
  // explicit numbers, including one that does not exist and one that is already done
  const r = resolveDecision(decide('confirm', [3, 99, 9]), { pending: [a, b], byId });
  assert.deepEqual(r.targets.map((t) => t.id), [3, 9]);
  assert.deepEqual(r.unknown, [99]);
  // ok all → everything pending
  assert.deepEqual(resolveDecision(decide('confirm', null, true), { pending: [a, b], byId }).targets.map((t) => t.id), [3, 5]);
  // a bare "ok" with exactly one pending request is unambiguous
  assert.deepEqual(resolveDecision(decide('confirm'), { pending: [a], byId }).targets.map((t) => t.id), [3]);
  // …with two waiting it asks which one…
  assert.equal(resolveDecision(decide('confirm'), { pending: [a, b], byId }).kind, 'ask');
  // …unless the owner replied to one of the notices
  assert.deepEqual(resolveDecision(decide('confirm'), { pending: [a, b], byId, quotedId: 5 }).targets.map((t) => t.id), [5]);
  // a bare "ok" with nothing waiting is just a note
  assert.equal(resolveDecision(decide('confirm'), { pending: [], byId }).kind, 'ignore');
  assert.equal(resolveDecision(decide('confirm', null, true), { pending: [], byId }).kind, 'ignore');
});

test('the notice carries the number and how to answer; results and lists read clearly', () => {
  const n = ownerNotice(item(12, { dueText: 'kal subah 6 baje' }), { timeZone: 'Asia/Kolkata' });
  assert.match(n, /Meeting request \*#12\* from Person 12/);
  assert.match(n, /Meet P12/);
  assert.match(n, /Wed, 11 Nov, 6:00 am/);
  assert.match(n, /📍 Station/);
  assert.match(n, /ok 12/);
  assert.match(n, /no 12/);
  assert.match(n, /just "ok"/);
  assert.match(ownerNotice(item(13, { kind: 'todo', title: 'Send the file' }), { timeZone: 'Asia/Kolkata', updated: true }), /✏️ Updated Request \*#13\*/);
  const list = ownerPendingList([item(3), item(5, { kind: 'todo', title: 'Send the file', dueAt: null })], { timeZone: 'Asia/Kolkata' });
  assert.match(list, /Waiting for you \(2\)/);
  assert.match(list, /\*#3\*/);
  assert.match(list, /no time set/);
  assert.equal(ownerPendingList([], { timeZone: 'Asia/Kolkata' }), '✅ Nothing is waiting for you.');
  assert.match(ownerAsk([item(3), item(5)], { timeZone: 'Asia/Kolkata' }), /I need the number[\s\S]*#3[\s\S]*#5[\s\S]*ok 3/);
  assert.match(ownerHelp(), /ok 12/);
  assert.match(ownerResultLine({ item: item(3), action: 'confirm', notified: true, card: true }), /✅ #3 confirmed.*Person got the confirmation and invitation card/);
  assert.match(ownerResultLine({ item: item(3, { chatId: '91@c.us' }), action: 'confirm', notified: false }), /couldn't message Person/);
  assert.match(ownerResultLine({ item: item(3, { status: 'confirmed' }), action: 'confirm', skipped: true }), /already confirmed/);
  assert.match(ownerResultLine({ item: item(3), action: 'decline', notified: true }), /🚫 #3 declined/);
  assert.match(ownerResultLine({ item: item(3), action: 'done' }), /marked done/);
});

test('any agenda item gets a card: a to-do reads TASK and Handled by, and works with no date or place', () => {
  const settings = { ownerName: 'Sumit Soni', personaName: 'eSumit', timezone: 'Asia/Kolkata' };
  const now = new Date('2026-11-10T08:00:00Z');
  const todo = { id: 21, kind: 'todo', title: 'Send the project file', dueAt: null, status: 'confirmed', contactName: 'Prachi', location: null };
  const d = invitationData(todo, settings, { now });
  assert.equal(d.kicker, 'TASK');
  assert.equal(d.hostLabel, 'Handled by');
  const html = invitationHtml(d);
  assert.match(html, />TASK</);
  assert.match(html, /Handled by/);
  assert.match(html, /Send the project file/);
  assert.doesNotMatch(html, /class="date"/);
  const meet = invitationData({ ...todo, id: 22, kind: 'appointment', dueAt: '2026-11-11T00:30:00Z' }, settings, { now });
  assert.equal(meet.kicker, 'INVITATION');
  assert.match(invitationHtml(invitationData({ ...todo, dueAt: '2026-11-14T10:00:00Z' }, settings, { code: 'hi', now })), />कार्य</);
});
