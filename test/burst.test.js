import test from 'node:test';
import assert from 'node:assert/strict';
import { looksLikeRefinement, meetingDraft, sameItem, parseWhen } from '../src/brain/agenda.js';
import { ownerDigest } from '../src/brain/owner-commands.js';
import { createAvailability } from '../src/brain/availability.js';
import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { DEFAULT_SETTINGS } from '../src/core/settings-schema.js';

const TZ = 'Asia/Kolkata';
const NOW = new Date('2026-11-12T08:00:00Z'); // Thu 12 Nov 13:30 IST
const IST = (d, h, m = 0) => new Date(`2026-11-${String(d).padStart(2, '0')}T${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00+05:30`);
const item = (over) => ({ chatId: 'c1', kind: 'appointment', title: 'Meet Sumit', dueAt: IST(16, 16, 30).toISOString(), ...over });

// ------------------------------------------------------------------ merging

test('requests for different days are different items, even with the same title (30 of them used to collapse into a few)', () => {
  const a = item({});
  assert.equal(sameItem(a, item({ dueAt: IST(17, 11).toISOString() })), false, 'another day');
  assert.equal(sameItem(a, item({ dueAt: IST(16, 19).toISOString() })), false, 'same day, hours apart: another meeting');
  assert.equal(sameItem(a, item({ dueAt: IST(16, 17).toISOString() })), true, 'within two hours: the same request');
  assert.equal(sameItem(a, item({ dueAt: IST(16, 16, 30).toISOString() })), true, 'the identical request sent twice');
  assert.equal(sameItem(a, item({ chatId: 'c2' })), false, 'another person');
  assert.equal(sameItem(a, item({ kind: 'todo' })), false);
});

test('changing an earlier request still updates it instead of adding a second one', () => {
  const a = item({});
  const later = item({ dueAt: IST(17, 11).toISOString() });
  assert.equal(sameItem(a, later, { refinement: true }), true, '"actually make it the 17th instead"');
  assert.equal(sameItem(a, item({ title: 'Lunch with Sumit', dueAt: IST(17, 11).toISOString() }), { refinement: true }), false, 'but only when it is about the same thing');
  for (const t of ['make it 6pm', 'actually the 17th instead', 'can we reschedule', 'please move it', 'Change the time to 5', 'kal ki jagah parso, badal do']) assert.equal(looksLikeRefinement(t), true, t);
  for (const t of ["I'd like to meet Sumit on 18 November at 6:30 PM at Worli", 'Could we set up a meeting on the 20th']) assert.equal(looksLikeRefinement(t), false, t);
  // to-dos without a time keep matching by title
  assert.equal(sameItem({ chatId: 'c', kind: 'todo', title: 'Send the file' }, { chatId: 'c', kind: 'todo', title: 'send the file please' }), true);
});

test('the meeting the reply already understood becomes an item at once, with the purpose in the details', () => {
  const text = "I'd like to schedule some time with Sumit on 16 November at 7:00 PM at Lower Parel, Mumbai, to discuss our system architecture.";
  const d = meetingDraft({ meeting: { dueAt: IST(16, 19).toISOString(), dueText: '16 november 7pm', place: 'Lower parel', kind: 'meet' }, text, contactName: 'Aditya Soni' });
  assert.equal(d.kind, 'appointment');
  assert.equal(d.title, 'Meet Aditya');
  assert.equal(d.dueAt.toISOString(), IST(16, 19).toISOString());
  assert.equal(d.allDay, false);
  assert.equal(d.location, 'Lower parel');
  assert.equal(d.details, text);
  assert.equal(meetingDraft({ meeting: { dueAt: IST(16, 19).toISOString(), kind: 'call' }, text: 'call me', contactName: null }).title, 'Call with contact');
});

// ------------------------------------------------------------------ owner digest

test('a burst reaches the owner as one message that keeps every number', () => {
  const items = Array.from({ length: 28 }, (_, i) => ({ id: i + 7, kind: 'appointment', status: 'pending', title: 'Meet Sumit', contactName: 'Sumit', dueAt: IST(16 + (i % 8), 11 + (i % 5)).toISOString(), location: i % 2 ? 'BKC' : null }));
  const text = ownerDigest(items, { timeZone: TZ });
  assert.match(text, /\*28 new requests\*/);
  assert.match(text, /\*#7\* 📅 Sumit: Mon, 16 Nov, 11:00 am/);
  assert.match(text, /\*#31\*/, 'the 25th listed');
  assert.match(text, /…and 3 more/);
  assert.match(text, /ok all/);
  assert.match(ownerDigest([items[0]], { timeZone: TZ }), /\*1 new request\*/);
});

// ------------------------------------------------------------------ picking an offered slot

const kb = new KnowledgeBase({ dir: new URL('../knowledge_base', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1') });

async function chat(messages, items = []) {
  await kb.load().catch(() => {});
  const llmCalls = [];
  const llm = { numCtx: 4096, async *chat(req) { llmCalls.push(req); yield { type: 'token', text: 'MODEL SAYS SOMETHING' }; yield { type: 'done', stats: {} }; } };
  const settings = { ...DEFAULT_SETTINGS, ownerName: 'Sumit Soni', model: 'x', timezone: TZ };
  const availability = createAvailability({ repo: { listAgenda: async () => items, listTimeBlocks: async () => [] }, getSettings: () => settings, now: () => NOW });
  const stores = createMemoryStores();
  const brain = createBrain({ kb, llm, ...stores, availability });
  const out = [];
  for (const text of messages) {
    const chunks = [];
    let done;
    for await (const ev of brain.respond({ chatId: 'c', text, settings, contactName: 'Sumit' })) {
      if (ev.type === 'chunk') chunks.push(ev.text);
      else if (ev.type === 'done') done = ev;
    }
    out.push({ text, reply: chunks.join('\n'), meta: done.meta });
  }
  return { out, llmCalls };
}

test('after "not available + free slots", a bare time picks the offered slot and is filed; no model says "booked"', async () => {
  const items = [{ id: 4, kind: 'appointment', status: 'confirmed', title: 'Meet Mehul', dueAt: IST(13, 10).toISOString(), allDay: false }];
  const { out, llmCalls } = await chat(["I'd like to catch up with Sumit on 13 November at 10:00 AM at BKC, Mumbai, to go through the latest code changes.", '12 pm'], items);
  assert.equal(out[0].meta.intent, 'meeting_unavailable');
  assert.match(out[0].reply, /Free slots:.*Fri, 13 Nov: 11:30 am, 12:00 pm/);
  assert.equal(out[1].meta.intent, 'meeting', '"12 pm" is answered by the template, not the model');
  assert.match(out[1].reply, /Fri, 13 Nov, 12:00 pm/, 'it resolved to the offered Friday slot, not "today"');
  assert.equal(out[1].meta.meeting.dueAt, IST(13, 12).toISOString(), 'and hands the exact time to the worker');
  assert.equal(llmCalls.length, 0);
  assert.doesNotMatch(out[1].reply, /booked/i);
});

test('a complete free request carries its meeting in the reply meta; a date without a time does not (it asks first)', async () => {
  const { out } = await chat(["Can we arrange a meeting with Sumit on 16 November at 4:30 PM at Bandra West, Mumbai, to review the budget?", 'Fix a date for me on next monday']);
  assert.equal(out[0].meta.intent, 'meeting');
  assert.equal(out[0].meta.meeting.dueAt, IST(16, 16, 30).toISOString());
  assert.equal(out[0].meta.meeting.place, 'Bandra west');
  assert.equal(out[1].meta.meeting, undefined);
  assert.match(out[1].reply, /What time works for you\?/);
});

test('thirty requests in a row: every one is answered by the template and carries its own meeting (nothing to overwrite)', async () => {
  const asks = Array.from({ length: 30 }, (_, i) => `I'd like to meet Sumit on ${14 + (i % 14)} December at ${10 + (i % 8)}:00 at Powai, Mumbai, to discuss item ${i}.`);
  const { out, llmCalls } = await chat(asks);
  assert.equal(llmCalls.length, 0);
  assert.equal(out.filter((o) => o.meta.meeting).length, 30);
  const times = new Set(out.map((o) => o.meta.meeting.dueAt));
  assert.ok(times.size >= 28, 'distinct times stay distinct');
  assert.ok(parseWhen('1 december at 10 am', { now: NOW, timeZone: TZ }));
});
