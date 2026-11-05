import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  heuristicItem, isMeetingRequest, looksLikeCommitment, parseExtraction, parseWhen, resolveItems, sameItem,
  sanitizeAgendaInput, summarizeAgenda, zonedToUtc,
} from '../src/brain/agenda.js';
import { createBrain } from '../src/brain/pipeline.js';
import { DEFAULT_SETTINGS, renderTemplate, sanitizeSettings } from '../src/core/settings-schema.js';
import { deriveLlmState, DownAlerter } from '../src/llm/status.js';

// Wednesday 4 Nov 2026, 10:00 in India (04:30 UTC).
const NOW = new Date('2026-11-04T04:30:00Z');
const tz = 'Asia/Kolkata';
const when = (text) => parseWhen(text, { now: NOW, timeZone: tz });
const iso = (r) => r?.dueAt.toISOString();

test('wall-clock time in a zone converts to the right instant', () => {
  assert.equal(zonedToUtc(2026, 11, 5, 17, 0, 'Asia/Kolkata').toISOString(), '2026-11-05T11:30:00.000Z');
  assert.equal(zonedToUtc(2026, 7, 1, 9, 0, 'America/New_York').toISOString(), '2026-07-01T13:00:00.000Z'); // DST
  assert.equal(zonedToUtc(2026, 1, 1, 9, 0, 'America/New_York').toISOString(), '2026-01-01T14:00:00.000Z');
});

test('reads "tomorrow at 5pm" and friends in the owner timezone', () => {
  assert.equal(iso(when("let's meet tomorrow at 5pm")), '2026-11-05T11:30:00.000Z');
  assert.equal(when("let's meet tomorrow at 5pm").allDay, false);
  assert.equal(iso(when('tomorrow at 5')), '2026-11-05T11:30:00.000Z'); // bare 1–6 means pm
  assert.equal(iso(when('kal shaam 6 baje')), '2026-11-05T12:30:00.000Z');
  assert.equal(iso(when('on monday at 10:30')), '2026-11-09T05:00:00.000Z');
  assert.equal(iso(when('12/11 3pm')), '2026-11-12T09:30:00.000Z');
  assert.equal(iso(when('15 dec at 11am')), '2026-12-15T05:30:00.000Z');
  assert.equal(iso(when('day after tomorrow 7:15 pm')), '2026-11-06T13:45:00.000Z');
});

test('a date without a time is an all-day item at 09:00', () => {
  const r = when('can you do next friday?');
  assert.equal(r.allDay, true);
  assert.equal(iso(r), '2026-11-06T03:30:00.000Z');
});

test('a time without a day means later today, else tomorrow', () => {
  assert.equal(iso(when('at 11am')), '2026-11-04T05:30:00.000Z'); // 10:00 now → 11:00 today
  assert.equal(iso(when('at 9am')), '2026-11-05T03:30:00.000Z'); // 09:00 already passed → tomorrow
});

test('text without a date or time yields nothing', () => {
  assert.equal(when('how are you doing'), null);
  assert.equal(when('may I ask something'), null);
  assert.equal(when(''), null);
});

test('spots messages that may hold a commitment', () => {
  for (const t of ["Let's meet tomorrow at 5pm", 'remind me to send the file', 'kal milte hain 5 baje?', 'are you free on friday?', 'deadline is 12/11']) {
    assert.equal(looksLikeCommitment(t), true, t);
  }
  for (const t of ['tell me a joke', 'what is oops concept', 'hi', 'thanks a lot', '']) {
    assert.equal(looksLikeCommitment(t), false, t);
  }
  assert.equal(isMeetingRequest("let's meet tomorrow"), true);
  assert.equal(isMeetingRequest('what is 2+2'), false);
});

test('parses the model JSON, even inside a code fence', () => {
  const raw = 'Sure!\n```json\n{"items":[{"kind":"appointment","title":"Meet Prachi","details":null,"when":"tomorrow at 5pm","location":"Cafe"},{"kind":"bogus","title":"","when":"x"}]}\n```';
  assert.deepEqual(parseExtraction(raw), [{ kind: 'appointment', title: 'Meet Prachi', details: null, when: 'tomorrow at 5pm', location: 'Cafe' }]);
  assert.equal(parseExtraction('no json here'), null);
  assert.deepEqual(parseExtraction('{"items":[]}'), []);
});

test('dates come from the phrase, the message, or recent user messages', () => {
  const [a] = resolveItems([{ kind: 'appointment', title: 'Meet', when: 'tomorrow at 5pm' }], { text: 'ok', now: NOW, timeZone: tz });
  assert.equal(a.dueAt.toISOString(), '2026-11-05T11:30:00.000Z');
  // "yes 5pm works" borrows "tomorrow" from the previous user turn
  const [b] = resolveItems([{ kind: 'appointment', title: 'Meet', when: null }], { text: 'yes 5pm works', userTexts: ['can we meet tomorrow?', 'yes 5pm works'], now: NOW, timeZone: tz });
  assert.equal(b.dueAt.toISOString(), '2026-11-05T11:30:00.000Z');
});

test('fallback item when the model is unavailable', () => {
  const it = heuristicItem({ text: "let's meet tomorrow at 5pm at the office", contactName: 'Prachi', now: NOW, timeZone: tz });
  assert.equal(it.kind, 'appointment');
  assert.equal(it.title, 'Meet Prachi');
  assert.equal(it.dueAt.toISOString(), '2026-11-05T11:30:00.000Z');
  assert.equal(heuristicItem({ text: 'hello there', contactName: 'X', now: NOW, timeZone: tz }), null);
});

test('the same proposal refined later is one item, different ones are not', () => {
  const stored = { chatId: 'a@c.us', kind: 'appointment', title: 'Meet Prachi', dueAt: '2026-11-05T11:30:00.000Z' };
  assert.equal(sameItem(stored, { ...stored, title: 'Catch up with Prachi', dueAt: '2026-11-05T12:30:00.000Z' }), true); // 1 h later
  assert.equal(sameItem(stored, { ...stored, dueAt: '2026-11-08T11:30:00.000Z', title: 'Dentist' }), false);
  assert.equal(sameItem(stored, { ...stored, chatId: 'b@c.us' }), false);
  assert.equal(sameItem(stored, { ...stored, kind: 'todo' }), false);
});

test('API input validation', () => {
  assert.deepEqual(sanitizeAgendaInput({ title: ' Call dentist ', kind: 'todo' }).errors, []);
  assert.equal(sanitizeAgendaInput({ kind: 'todo' }).errors.length, 1); // title required
  assert.equal(sanitizeAgendaInput({ title: 'x', kind: 'nope' }).errors.length, 1);
  assert.equal(sanitizeAgendaInput({ title: 'x', dueAt: 'tomorrow' }).errors.length, 1);
  const ok = sanitizeAgendaInput({ title: 'x', dueAt: '2026-11-05T11:30:00+05:30', allDay: false, chatId: '9199@c.us' });
  assert.equal(ok.value.dueAt, '2026-11-05T06:00:00.000Z');
  assert.equal(ok.value.kind, 'todo'); // default on create
  assert.deepEqual(sanitizeAgendaInput({ status: 'done' }, { partial: true }), { value: { status: 'done' }, errors: [] });
  assert.equal(sanitizeAgendaInput({ status: 'weird' }, { partial: true }).errors.length, 1);
  assert.equal(sanitizeAgendaInput({ dueAt: null }, { partial: true }).value.dueAt, null);
});

test('dashboard summary counts review, today, next 7 days, to-dos and overdue', () => {
  const items = [
    { kind: 'appointment', status: 'pending', dueAt: '2026-11-04T11:30:00.000Z', allDay: false }, // today 17:00, needs review
    { kind: 'appointment', status: 'confirmed', dueAt: '2026-11-06T05:00:00.000Z', allDay: false }, // in 2 days
    { kind: 'todo', status: 'pending', dueAt: '2026-11-03T05:00:00.000Z', allDay: false }, // overdue
    { kind: 'todo', status: 'pending', dueAt: null, allDay: false },
    { kind: 'todo', status: 'done', dueAt: '2026-11-01T05:00:00.000Z', allDay: false }, // done: ignored
    { kind: 'todo', status: 'pending', dueAt: '2026-11-04T03:30:00.000Z', allDay: true }, // all-day today: not overdue
  ];
  assert.deepEqual(summarizeAgenda(items, { now: NOW, timeZone: tz }), { review: 1, today: 2, next7: 1, openTodos: 3, overdue: 1, total: 6 });
});

test('extractAgenda: the model names the item, the code reads the date', async () => {
  const reply = '{"items":[{"kind":"appointment","title":"Meet Prachi","details":"Coffee","when":"tomorrow at 5pm","location":null}]}';
  const llm = { numCtx: 4096, async *chat() { yield { type: 'token', text: reply }; yield { type: 'done', stats: {} }; } };
  const brain = createBrain({ kb: null, llm, memory: null, cache: null });
  const turns = [{ role: 'user', content: "let's meet tomorrow at 5pm" }, { role: 'assistant', content: "I'll pass it on" }];
  const [it] = await brain.extractAgenda({ turns, text: turns[0].content, contactName: 'Prachi', settings: { ...DEFAULT_SETTINGS, model: 'x' }, now: NOW });
  assert.equal(it.title, 'Meet Prachi');
  assert.equal(it.dueAt.toISOString(), '2026-11-05T11:30:00.000Z');
});

test('extractAgenda falls back to the plain message when the model fails', async () => {
  const llm = { numCtx: 4096, async *chat() { throw new Error('ollama down'); } };
  const brain = createBrain({ kb: null, llm, memory: null, cache: null });
  const text = "let's meet tomorrow at 5pm";
  const out = await brain.extractAgenda({ turns: [{ role: 'user', content: text }], text, contactName: 'Prachi', settings: { ...DEFAULT_SETTINGS, model: 'x' }, now: NOW });
  assert.equal(out.length, 1);
  assert.equal(out[0].kind, 'appointment');
});

test('the reply model is told not to agree to meetings on the owner\'s behalf', async () => {
  const calls = [];
  const llm = { numCtx: 4096, async *chat(req) { calls.push(req); yield { type: 'token', text: 'ok' }; yield { type: 'done', stats: {} }; } };
  const { createMemoryStores } = await import('../src/brain/pipeline.js');
  const { KnowledgeBase } = await import('../src/rag/kb.js');
  const path = await import('node:path');
  const kb = await new KnowledgeBase({ dir: path.join(import.meta.dirname, 'fixtures', 'kb'), minScore: 0.5 }).load();
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  for await (const ev of brain.respond({ chatId: 'm1', text: "let's meet tomorrow at 5pm", settings: { ...DEFAULT_SETTINGS, model: 'x' } })) void ev;
  assert.match(calls[0].messages.at(-1).content, /pass it on and the real \w+ will confirm/);
});

test('personalised notice templates', () => {
  const s = { ...DEFAULT_SETTINGS, ownerName: 'Sumit Soni', personaName: 'eSumit' };
  assert.match(renderTemplate(s.unsupportedMessage, s, { name: 'Prachi Shah', kind: 'voice notes' }), /^Hey Prachi! I can only read text messages right now, so I can't open voice notes/);
  assert.match(renderTemplate(s.timeoutMessage, s, { name: 'Prachi' }), /Send \/hi whenever.*the real Sumit/);
  assert.equal(sanitizeSettings({ timezone: 'Asia/Kolkata' }).errors.length, 0);
  assert.equal(sanitizeSettings({ timezone: 'Mars/Olympus' }).errors.length, 1);
  assert.equal(sanitizeSettings({ alertOwner: 'yes' }).errors.length, 1);
});

test('LLM health verdicts', () => {
  const base = { model: 'gemma3:4b', version: { version: '0.12' }, installed: [{ name: 'gemma3:4b' }], loaded: [{ name: 'gemma3:4b', size: 1 }], worker: { at: 1000, model: { ready: true } }, now: 2000 };
  assert.equal(deriveLlmState(base).state, 'ok');
  assert.equal(deriveLlmState({ ...base, version: { error: 'ECONNREFUSED' } }).state, 'down');
  assert.equal(deriveLlmState({ ...base, installed: [] }).state, 'down');
  assert.equal(deriveLlmState({ ...base, worker: null }).state, 'down');
  assert.equal(deriveLlmState({ ...base, worker: { at: 1000, model: { error: 'boom' } } }).state, 'down');
  assert.equal(deriveLlmState({ ...base, worker: { at: 1000, model: { ready: false } } }).state, 'degraded');
  assert.equal(deriveLlmState({ ...base, loaded: [] }).state, 'degraded');
  assert.equal(deriveLlmState({ ...base, now: 1000 + 60_000 }).state, 'down'); // stale heartbeat
  assert.equal(deriveLlmState({ ...base, model: 'gemma3', installed: [{ name: 'gemma3:latest' }], loaded: [{ name: 'gemma3:latest' }] }).state, 'ok');
});

test('down alerts fire once after the delay and announce recovery', () => {
  const a = new DownAlerter({ afterMs: 90_000 });
  assert.equal(a.update('down', 'x', 0), null);
  assert.equal(a.update('down', 'x', 60_000), null);
  assert.equal(a.update('down', 'x', 95_000).type, 'down');
  assert.equal(a.update('down', 'x', 120_000), null); // not repeated
  assert.equal(a.update('ok', '', 200_000).type, 'recovered');
  assert.equal(a.update('ok', '', 210_000), null);
  // a blip that heals before the delay never alerts
  assert.equal(a.update('down', 'x', 300_000), null);
  assert.equal(a.update('ok', '', 310_000), null);
});

test('meeting-request gate: asks for a meeting, not a task that merely mentions one', () => {
  for (const t of ["let's meet tomorrow at 5pm", 'Call me at 10 am tomorrow', 'Interview is scheduled on 12/11 at 3pm, please confirm', 'Can we reschedule our meeting to friday 3pm?', 'kal shaam 5 baje milte hain?']) {
    assert.equal(isMeetingRequest(t), true, t);
  }
  for (const t of ['{"task":"summarize","text":"The meeting moved to Friday. Budget approved."}', 'summarize: the meeting is on friday', 'what is 2+2', 'tell me a joke']) {
    assert.equal(isMeetingRequest(t), false, t);
  }
});

test('loosely typed times: "six p m", "6 p.m", "five o\'clock"', () => {
  assert.equal(iso(when('tomorrow at six p m near the office')), '2026-11-05T12:30:00.000Z'); // 18:00 IST
  assert.equal(iso(when('tomorrow at 6 p.m.')), '2026-11-05T12:30:00.000Z');
  assert.equal(iso(when('tomorrow at five o\'clock')), '2026-11-05T11:30:00.000Z');
  assert.equal(iso(when('kal seven baje')), '2026-11-05T01:30:00.000Z'); // 7 → 07:00 IST (7–11 stay am)
  assert.equal(when('tomorrow at six p m').allDay, false);
});

test('a meeting request split over two messages is still caught (seen in a real chat)', async () => {
  const { looksLikeCommitmentInContext, heuristicItem } = await import('../src/brain/agenda.js');
  const details = 'Tomorrow at six p m near zudio office , vadonra , gujarat';
  assert.equal(looksLikeCommitment('I want to meet'), false); // no time on its own
  assert.equal(looksLikeCommitment(details), false); // no request on its own
  assert.equal(looksLikeCommitmentInContext(details, ['How are you', 'Good', 'I want to meet']), true);
  assert.equal(looksLikeCommitmentInContext(details, ['How are you', 'Good']), false); // nobody asked for anything
  assert.equal(looksLikeCommitmentInContext('Tomorrow is a holiday', ['I want to meet']), true); // a day counts as a time signal
  const it = heuristicItem({ text: details, contextText: 'I want to meet', contactName: 'J', now: NOW, timeZone: tz });
  assert.equal(it.kind, 'appointment');
  assert.equal(it.title, 'Meet J');
  assert.equal(it.dueAt.toISOString(), '2026-11-05T12:30:00.000Z');
});
