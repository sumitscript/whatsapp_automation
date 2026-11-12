import test from 'node:test';
import assert from 'node:assert/strict';
import { assess, availabilityConfig, clashLines, freeSlots, itemInterval, parseHHMM, slotsText } from '../src/brain/slots.js';
import { createAvailability } from '../src/brain/availability.js';
import { formatWhen } from '../src/brain/agenda.js';
import { declineText, slotTakenAck } from '../src/brain/invite.js';
import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { DEFAULT_SETTINGS, sanitizeSettings } from '../src/core/settings-schema.js';

const TZ = 'Asia/Kolkata';
const NOW = new Date('2026-11-12T08:00:00Z'); // Thu 12 Nov, 13:30 IST
const cfg = availabilityConfig({ timezone: TZ, workStart: '09:00', workEnd: '21:00', meetingMinutes: 60, meetingBufferMinutes: 15 });
const IST = (d, h, m = 0) => new Date(`2026-11-${String(d).padStart(2, '0')}T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00+05:30`);
const meet = (id, start, over = {}) => ({ id, kind: 'appointment', status: 'confirmed', title: `Meet P${id}`, contactName: `Person ${id}`, dueAt: start.toISOString(), allDay: false, ...over });

test('settings: working hours, days, durations are validated', () => {
  assert.equal(DEFAULT_SETTINGS.enforceAvailability, true);
  assert.deepEqual(sanitizeSettings({ workStart: '10:30', workEnd: '19:00', workDays: 'Mon, tue,FRI', meetingMinutes: 45, meetingBufferMinutes: 10 }).value, { workStart: '10:30', workEnd: '19:00', workDays: 'mon,tue,fri', meetingMinutes: 45, meetingBufferMinutes: 10 });
  for (const bad of [{ workStart: '25:00' }, { workStart: '9am' }, { workDays: 'funday' }, { workDays: '' }, { meetingMinutes: 0 }, { meetingBufferMinutes: -5 }]) assert.equal(sanitizeSettings(bad).errors.length, 1, JSON.stringify(bad));
});

test('config reads settings and survives nonsense', () => {
  assert.equal(parseHHMM('09:30', 0), 570);
  assert.equal(parseHHMM('nope', 7), 7);
  const c = availabilityConfig({ workStart: '20:00', workEnd: '08:00', workDays: '' });
  assert.ok(c.endMin > c.startMin, 'an inverted window is repaired');
  assert.equal(c.days.size, 7, 'no days listed means every day');
});

test('only confirmed-or-pending appointments with a time occupy time', () => {
  assert.ok(itemInterval(meet(1, IST(13, 10)), cfg));
  assert.equal(itemInterval(meet(1, IST(13, 10), { kind: 'todo' }), cfg), null);
  assert.equal(itemInterval(meet(1, IST(13, 10), { allDay: true }), cfg), null);
  assert.equal(itemInterval({ kind: 'appointment', dueAt: null }, cfg), null);
});

test('a time on top of a confirmed meeting is a hard clash; the 15-minute buffer counts', () => {
  const items = [meet(4, IST(13, 10))]; // 10:00–11:00
  assert.equal(assess(IST(13, 10), { items, cfg, now: NOW }).hard[0].id, 4, 'same time');
  assert.equal(assess(IST(13, 10, 30), { items, cfg, now: NOW }).ok, false, 'overlapping start');
  assert.equal(assess(IST(13, 9, 30), { items, cfg, now: NOW }).ok, false, 'ends inside it');
  assert.equal(assess(IST(13, 11, 0), { items, cfg, now: NOW }).ok, false, 'back to back is inside the buffer');
  assert.equal(assess(IST(13, 11, 15), { items, cfg, now: NOW }).ok, true, 'a full buffer later is fine');
  assert.equal(assess(IST(13, 8, 45), { items, cfg, now: NOW }).reasons.includes('outside_hours'), true);
  assert.equal(assess(IST(13, 10), { items, cfg, now: NOW, ignoreId: 4 }).ok, true, 'a meeting never clashes with itself');
});

test('a pending request is only a soft clash (accepted, the owner is told); declined and done never block', () => {
  const items = [meet(5, IST(13, 10), { status: 'pending' }), meet(6, IST(13, 10), { status: 'declined' }), meet(7, IST(13, 10), { status: 'done' })];
  const r = assess(IST(13, 10), { items, cfg, now: NOW });
  assert.equal(r.ok, true);
  assert.deepEqual(r.soft.map((x) => x.id), [5]);
  assert.equal(r.hard.length, 0);
});

test('blocks, days off, hours and the past', () => {
  const blocks = [{ id: 1, label: 'Lunch', startsAt: IST(13, 13).toISOString(), endsAt: IST(13, 14).toISOString() }];
  const r = assess(IST(13, 13, 30), { blocks, cfg, now: NOW });
  assert.equal(r.ok, false);
  assert.equal(r.blocked[0].label, 'Lunch');
  assert.equal(assess(IST(13, 14), { blocks, cfg, now: NOW }).ok, true, 'a block has no buffer');
  assert.deepEqual(assess(IST(13, 22), { cfg, now: NOW }).reasons, ['outside_hours']);
  assert.deepEqual(assess(IST(12, 9), { cfg, now: NOW }).reasons, ['past']);
  const weekdays = availabilityConfig({ timezone: TZ, workDays: 'mon,tue,wed,thu,fri' });
  assert.deepEqual(assess(IST(14, 11), { cfg: weekdays, now: NOW }).reasons, ['day_off'], 'Saturday');
});

test('free slots: inside hours, 30-minute steps, after the lead time, around meetings and blocks, spread over days', () => {
  const items = [meet(4, IST(13, 10))];
  const blocks = [{ id: 1, label: 'Lunch', startsAt: IST(13, 13).toISOString(), endsAt: IST(13, 14).toISOString() }];
  const slots = freeSlots({ items, blocks, cfg, now: NOW, count: 40, days: 2 });
  assert.equal(slots[0].toISOString(), IST(12, 14).toISOString(), 'today starts after now + 30 min');
  for (const s of slots) assert.equal(assess(s, { items, blocks, cfg, now: NOW }).ok, true);
  const times = slots.map((s) => s.toISOString());
  assert.ok(!times.includes(IST(13, 10).toISOString()) && !times.includes(IST(13, 11).toISOString()), 'meeting plus buffer is skipped');
  assert.ok(times.includes(IST(13, 11, 30).toISOString()));
  assert.ok(!times.includes(IST(13, 13).toISOString()) && !times.includes(IST(13, 12, 30).toISOString()), 'the lunch block is skipped');
  assert.ok(!times.includes(IST(12, 20, 30).toISOString()), 'a 60-minute meeting must end by 21:00');
  const spread = freeSlots({ items, cfg, now: NOW, count: 4, perDay: 2 });
  assert.deepEqual([...new Set(spread.map((s) => s.toISOString().slice(0, 10)))].length >= 2, true, 'suggestions spread over days');
  assert.equal(freeSlots({ cfg: availabilityConfig({ timezone: TZ, workDays: 'sat' }), now: NOW, days: 1, count: 3 }).length, 0, 'nothing when the days asked are off');
});

test('slot text groups by day in the language of the reader', () => {
  const s = [IST(13, 11, 30), IST(13, 12), IST(14, 10)];
  assert.equal(slotsText(s, TZ), 'Fri, 13 Nov: 11:30 am, 12:00 pm · Sat, 14 Nov: 10:00 am');
  assert.match(slotsText(s, TZ, 'hi'), /शुक्र/);
  assert.equal(slotsText([], TZ), '');
});

test('the owner is told exactly what a request clashes with', () => {
  const items = [meet(4, IST(13, 10)), meet(5, IST(13, 10), { status: 'pending', contactName: 'Shabby' })];
  const blocks = [{ id: 1, label: 'Travel', startsAt: IST(13, 9).toISOString(), endsAt: IST(13, 12).toISOString() }];
  const lines = clashLines(assess(IST(13, 10), { items, blocks, cfg, now: NOW }), { timeZone: TZ, formatWhen });
  assert.match(lines.join('\n'), /Clashes with \*#4\*[\s\S]*already confirmed for Fri, 13 Nov, 10:00 am/);
  assert.match(lines.join('\n'), /marked .* as busy \(Travel\)/);
  assert.match(lines.join('\n'), /\*#5\* \(Shabby\) also asked/);
  assert.deepEqual(clashLines(assess(IST(13, 15), { cfg, now: NOW }), { timeZone: TZ, formatWhen }), []);
});

test('the service loads items and blocks and respects enforceAvailability', async () => {
  const repo = {
    listAgenda: async () => [meet(4, IST(13, 10))],
    listTimeBlocks: async () => [],
  };
  let settings = { ...DEFAULT_SETTINGS, timezone: TZ };
  const av = createAvailability({ repo, getSettings: () => settings, now: () => NOW });
  assert.equal((await av.assess(IST(13, 10))).hard[0].id, 4);
  assert.equal((await av.assess(IST(13, 10), { ignoreId: 4 })).ok, true);
  const free = await av.freeSlots({ count: 3 });
  assert.equal(free.length, 3);
  const o = await av.overview();
  assert.equal(o.meetings[0].id, 4);
  assert.ok(o.slots.length > 0);
  settings = { ...settings, enforceAvailability: false };
  const off = await av.assess(IST(13, 10));
  assert.equal(off.ok, true);
  assert.equal(off.skipped, true);
});

test('replies: not available (with and without free slots) and a decline that offers times', () => {
  const p = { name: 'Aditya Soni', owner: 'Sumit Soni', dueAt: IST(13, 10), allDay: false, timeZone: TZ };
  const en = slotTakenAck({ ...p, code: 'en', slots: 'Fri, 13 Nov: 11:30 am' });
  assert.match(en, /Sorry Aditya, Sumit isn't available then \(Fri, 13 Nov, 10:00 am\)/);
  assert.match(en, /Free slots: Fri, 13 Nov: 11:30 am/);
  assert.doesNotMatch(en, /Meet|confirmed|clash/i, 'it does not say why');
  assert.doesNotMatch(slotTakenAck({ ...p, code: 'en', slots: '' }), /Free slots/);
  for (const code of ['hi-Latn', 'hi', 'mr', 'gu']) assert.match(slotTakenAck({ ...p, code, slots: 'x' }), /Aditya/, code);
  assert.match(declineText({ ...p, code: 'en', title: 'Meet Aditya', slots: 'Fri, 13 Nov: 11:30 am' }), /can't make this one[\s\S]*\nFree slots: Fri, 13 Nov: 11:30 am$/);
  assert.doesNotMatch(declineText({ ...p, code: 'en', title: 'Meet Aditya' }), /Free slots/);
});

// ------------------------------------------------------------------ through the pipeline

const kb = new KnowledgeBase({ dir: new URL('../knowledge_base', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1') });

async function ask(text, items) {
  await kb.load().catch(() => {});
  const llmCalls = [];
  const llm = { numCtx: 4096, async *chat(req) { llmCalls.push(req); yield { type: 'token', text: 'MODEL' }; yield { type: 'done', stats: {} }; } };
  const settings = { ...DEFAULT_SETTINGS, ownerName: 'Sumit Soni', model: 'x', timezone: TZ };
  const availability = createAvailability({ repo: { listAgenda: async () => items, listTimeBlocks: async () => [] }, getSettings: () => settings, now: () => NOW });
  const brain = createBrain({ kb, llm, ...createMemoryStores(), availability });
  const chunks = [];
  let done;
  for await (const ev of brain.respond({ chatId: 'c', text, settings, contactName: 'Aditya Soni' })) {
    if (ev.type === 'chunk') chunks.push(ev.text);
    else if (ev.type === 'done') done = ev;
  }
  return { chunks, done, llmCalls };
}

test('pipeline: asking for a taken time gets "not available" with free slots and is not filed; a free time is acknowledged as before', async () => {
  const items = [meet(4, IST(13, 10))];
  const taken = await ask("let's meet tomorrow at 10am", items);
  assert.equal(taken.llmCalls.length, 0);
  assert.equal(taken.done.meta.intent, 'meeting_unavailable');
  assert.match(taken.chunks[0], /isn't available then \(Fri, 13 Nov, 10:00 am\)/);
  assert.match(taken.chunks[0], /Free slots: .*Fri, 13 Nov: 11:30 am/);
  const free = await ask("let's meet tomorrow at 3pm", items);
  assert.equal(free.done.meta.intent, 'meeting');
  assert.match(free.chunks[0], /Got it Aditya!/);
  const outside = await ask("let's meet tomorrow at 11pm", items);
  assert.equal(outside.done.meta.intent, 'meeting_unavailable', 'outside working hours');
  const hinglish = await ask('kal subah 10 baje milte hain', items);
  assert.match(hinglish.chunks[0], /free nahi hain/);
  const soft = await ask("let's meet tomorrow at 3pm", [meet(9, IST(13, 15), { status: 'pending' })]);
  assert.equal(soft.done.meta.intent, 'meeting', 'a pending request at the same time does not turn the second person away');
});
