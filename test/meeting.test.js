import test from 'node:test';
import assert from 'node:assert/strict';
import { guessLocation, isMeetingRequest, looksLikeCommitment, meetingKind, mentionsMeeting, normalizeTimeText, parseWhen, formatWhen, heuristicItem } from '../src/brain/agenda.js';
import { confirmationText, declineText, escapeHtml, inviteFamily, invitationData, invitationHtml, meetingAck } from '../src/brain/invite.js';
import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { DEFAULT_SETTINGS } from '../src/core/settings-schema.js';

// Tuesday 10 Nov 2026, 13:30 in Asia/Kolkata
const NOW = new Date('2026-11-10T08:00:00Z');
const TZ = 'Asia/Kolkata';
const at = (text) => {
  const r = parseWhen(text, { now: NOW, timeZone: TZ });
  return r && { when: formatWhen(r.dueAt, TZ, r.allDay), allDay: r.allDay };
};

test('times in English, Hinglish, Hindi, Marathi and Gujarati read the same', () => {
  assert.equal(at('let us meet tomorrow at 6 in the morning').when, 'Wed, 11 Nov, 6:00 am');
  assert.equal(at('kal subah paanch baje milte hain').when, 'Wed, 11 Nov, 5:00 am', 'subah beats the "1-6 means pm" guess');
  assert.equal(at('कल सुबह ६ बजे मिलते हैं').when, 'Wed, 11 Nov, 6:00 am');
  assert.equal(at('आज शाम 5 बजे मीटिंग रखते हैं').when, 'Tue, 10 Nov, 5:00 pm');
  assert.equal(at('કાલે સાંજે 6 વાગ્યે મળીએ').when, 'Wed, 11 Nov, 6:00 pm');
  assert.equal(at('उद्या सकाळी ९ वाजता भेटू').when, 'Wed, 11 Nov, 9:00 am');
});

test('everyday Hindi clock phrases and other ways people say a time', () => {
  assert.equal(at('kal sadhe paanch baje').when, 'Wed, 11 Nov, 5:30 pm');
  assert.equal(at('kal paune 6 baje call karna').when, 'Wed, 11 Nov, 5:45 pm');
  assert.equal(at('kal sava 4 baje').when, 'Wed, 11 Nov, 4:15 pm');
  assert.equal(at('call me tomorrow 5 to 6pm').when, 'Wed, 11 Nov, 5:00 pm');
  assert.equal(at('dinner tomorrow at 1730 hrs').when, 'Wed, 11 Nov, 5:30 pm');
  assert.equal(at('meeting on the 20th at 4 pm').when, 'Fri, 20 Nov, 4:00 pm');
  assert.equal(at('somvar ko office mein milte hain 11 baje').when, 'Mon, 16 Nov, 11:00 am');
  assert.equal(at('dinner tomorrow').when, 'Wed, 11 Nov, 8:00 pm');
});

test('relative times and fuzzy days', () => {
  assert.equal(at('can we catch up in 2 hours').when, 'Tue, 10 Nov, 3:30 pm');
  assert.equal(at('2 ghante baad milte hain').when, 'Tue, 10 Nov, 3:30 pm');
  assert.equal(at('free this weekend?').when, 'Sat, 14 Nov');
  assert.equal(at('free this weekend?').allDay, true);
  assert.equal(at('next week monday lunch').when, 'Mon, 16 Nov, 1:00 pm');
  assert.equal(at('what is the capital of France'), null);
});

test('names that look like words are not read as days', () => {
  assert.equal(normalizeTimeText('Ravi and Tom are coming'), 'ravi and tom are coming');
  assert.equal(at('Ravi said hello'), null);
});

test('meeting detection works on native scripts and Hinglish, and ignores news about other people', () => {
  for (const t of ['कल सुबह ६ बजे मिलते हैं', 'આવતીકાલે સાંજે 5 વાગ્યે મળીએ', 'kal shaam ko milte hain 6 baje', "let's meet tomorrow", 'free ho kal?', 'dinner this weekend?']) {
    assert.equal(isMeetingRequest(t), true, t);
    assert.equal(looksLikeCommitment(t), true, t);
  }
  assert.equal(isMeetingRequest('Ravi bhai kal aa raha hai'), false);
  assert.equal(isMeetingRequest('summarize this: the meeting moved to Friday'), false);
  assert.equal(isMeetingRequest('what is the capital of France'), false);
  assert.equal(mentionsMeeting('I want to meet you'), true);
  assert.equal(mentionsMeeting('nice weather'), false);
});

test('places named Hinglish-style ("station pe") and English-style', () => {
  assert.equal(guessLocation('kal subah railway station pe milte hain'), 'Railway station');
  assert.equal(guessLocation('somvar ko office mein milte hain'), 'Office');
  assert.equal(guessLocation('meet at the railway station tomorrow'), 'Railway station');
  assert.equal(guessLocation('kya tum free ho'), null);
});

test('meeting kind picks call / meal / meeting; the fallback item title follows it', () => {
  assert.equal(meetingKind('call karna kal 5 baje'), 'call');
  assert.equal(meetingKind('lunch tomorrow?'), 'meal');
  assert.equal(meetingKind("let's meet"), 'meet');
  const it = heuristicItem({ text: 'dinner tomorrow at 8pm near the mall', contactName: 'Prachi Shah', now: NOW, timeZone: TZ });
  assert.equal(it.title, 'Meal with Prachi');
  assert.equal(it.kind, 'appointment');
  assert.equal(it.location, 'Mall');
});

// ---------------------------------------------------------------- replies and cards

test('language families', () => {
  assert.equal(inviteFamily('hi-Latn'), 'hl');
  assert.equal(inviteFamily('gu'), 'gu');
  assert.equal(inviteFamily('ta'), 'en');
  assert.equal(inviteFamily(undefined), 'en');
});

test('meeting reply never agrees, is in their language, and asks for details when the time is unclear', () => {
  const base = { name: 'Prachi Shah', owner: 'Sumit Soni', kind: 'meet', timeZone: TZ };
  const dueAt = new Date('2026-11-11T00:30:00Z');
  const en = meetingAck({ ...base, code: 'en', dueAt, place: 'Railway station' });
  assert.match(en, /Got it Prachi!/);
  assert.match(en, /Wed, 11 Nov, 6:00 am · 📍 Railway station/);
  assert.match(en, /real Sumit/);
  assert.doesNotMatch(en, /\b(sure|yes|okay|ok|confirmed)\b/i);
  assert.match(meetingAck({ ...base, code: 'hi-Latn', dueAt }), /asli Sumit tak pahuncha diya jayega/);
  assert.match(meetingAck({ ...base, code: 'hi', dueAt }), /असली Sumit/);
  assert.match(meetingAck({ ...base, code: 'gu', dueAt }), /સાચા Sumit/);
  assert.match(meetingAck({ ...base, code: 'mr', dueAt }), /खऱ्या Sumit/);
  assert.match(meetingAck({ ...base, code: 'en' }), /Which day, time and place/, 'no time → asks');
  assert.match(meetingAck({ ...base, code: 'en', dueAt, allDay: true }), /Which day, time and place/, 'date only → asks for the time');
});

test('confirmation and decline texts', () => {
  const p = { code: 'en', name: 'Prachi Shah', owner: 'Sumit Soni', title: 'Meet Prachi', dueAt: new Date('2026-11-11T00:30:00Z'), place: 'Railway station', timeZone: TZ };
  assert.equal(confirmationText(p), '✅ Confirmed, Prachi! Sumit has accepted: Meet Prachi\n🗓 Wed, 11 Nov, 6:00 am\n📍 Railway station\nSee you then 🙂');
  assert.match(confirmationText({ ...p, code: 'hi' }), /पक्का, Prachi! Sumit ने मंज़ूरी दे दी है/);
  assert.match(declineText({ ...p, code: 'hi-Latn' }), /Sumit is baar nahi aa paayenge/);
  assert.match(declineText({ ...p, code: 'en', dueAt: null }), /the time we discussed/);
});

test('invitation card: content, escaping, and every card is different', () => {
  const settings = { ownerName: 'Sumit Soni', personaName: 'eSumit', timezone: TZ };
  const item = { id: 7, title: 'Meet <b>Prachi</b> & co', dueAt: '2026-11-11T00:30:00Z', location: 'Railway station', contactName: 'Prachi Shah', status: 'confirmed', allDay: false };
  const d = invitationData(item, settings, { code: 'en', now: NOW });
  assert.equal(d.ref, 'INV-0007');
  assert.equal(d.weekday, 'Wednesday');
  assert.equal(d.day, '11');
  assert.equal(d.time, '6:00 am');
  assert.equal(d.status, 'CONFIRMED');
  const html = invitationHtml(d);
  assert.match(html, /Meet &lt;b&gt;Prachi&lt;\/b&gt; &amp; co/);
  assert.doesNotMatch(html, /<b>Prachi/);
  assert.match(html, /Railway station/);
  assert.match(html, /Sumit Soni/);
  assert.doesNotMatch(html, /https?:\/\//, 'no external resources: it renders offline');
  assert.doesNotMatch(html, /<script/i);
  const other = invitationHtml(invitationData({ ...item, id: 8, title: 'Call Rahul' }, settings, { code: 'en', now: NOW }));
  assert.notEqual(html, other);
  assert.equal(invitationHtml(d), html, 'the same item always draws the same card');
  assert.equal(escapeHtml(`"'<&>`), '&quot;&#39;&lt;&amp;&gt;');
});

test('invitation card: all-day and unknown dates, native scripts', () => {
  const settings = { ownerName: 'Sumit Soni', personaName: 'eSumit', timezone: TZ };
  const allDay = invitationData({ id: 1, title: 'Weekend trip', dueAt: '2026-11-14T03:30:00Z', allDay: true, status: 'pending' }, settings, { now: NOW });
  assert.equal(allDay.time, 'All day');
  assert.equal(allDay.status, 'REQUESTED');
  const none = invitationData({ id: 2, title: 'Catch up', dueAt: null, status: 'confirmed' }, settings, { now: NOW });
  assert.equal(none.day, '');
  assert.doesNotMatch(invitationHtml(none), /class="date"/);
  const hi = invitationData({ id: 3, title: 'मीटिंग', dueAt: '2026-11-14T10:00:00Z', status: 'confirmed' }, settings, { code: 'hi', now: NOW });
  assert.equal(hi.weekday, 'शनिवार');
  assert.match(invitationHtml(hi), /class="native"/);
});

// ------------------------------------------------------------------ the pipeline

const kb = new KnowledgeBase({ dir: new URL('../knowledge_base', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1') });
const settings = { ...DEFAULT_SETTINGS, ownerName: 'Sumit Soni', model: 'x' };

async function replyTo(text, { chatId = 'mt', settingsOver = {}, prior = [] } = {}) {
  await kb.load().catch(() => {});
  const calls = [];
  const llm = { numCtx: 4096, async *chat(req) { calls.push(req); yield { type: 'token', text: 'MODEL REPLY' }; yield { type: 'done', stats: {} }; } };
  const { memory, cache } = createMemoryStores();
  if (prior.length) await memory.append(chatId, prior);
  const brain = createBrain({ kb, llm, memory, cache });
  const chunks = [];
  let done;
  for await (const ev of brain.respond({ chatId, text, settings: { ...settings, ...settingsOver }, contactName: 'Prachi Shah' })) {
    if (ev.type === 'chunk') chunks.push(ev.text);
    else if (ev.type === 'done') done = ev;
  }
  return { chunks, calls, done };
}

test('pipeline: a meeting request gets the instant template, no model call', async () => {
  const r = await replyTo("let's meet tomorrow at six in the morning at the railway station");
  assert.equal(r.calls.length, 0);
  assert.equal(r.chunks.length, 1);
  assert.match(r.chunks[0], /Got it Prachi!/);
  assert.match(r.chunks[0], /6:00 am/);
  assert.match(r.chunks[0], /Railway station/);
  assert.equal(r.done.meta.intent, 'meeting');
  assert.equal(r.done.meta.model, 'template');
});

test('pipeline: Devanagari and Gujarati requests are answered in their own script', async () => {
  const hi = await replyTo('कल सुबह ६ बजे मिलते हैं');
  assert.equal(hi.calls.length, 0);
  assert.match(hi.chunks[0], /असली Sumit/);
  const gu = await replyTo('કાલે સાંજે 6 વાગ્યે મળીએ');
  assert.equal(gu.calls.length, 0);
  assert.match(gu.chunks[0], /સાચા Sumit/);
});

test('pipeline: Hinglish request', async () => {
  const r = await replyTo('kal subah 6 baje railway station pe milte hain');
  assert.equal(r.calls.length, 0);
  assert.match(r.chunks[0], /Theek hai Prachi!/);
  assert.match(r.chunks[0], /Railway station/);
});

test('pipeline: a later message that only carries the time still gets the reply', async () => {
  const r = await replyTo('tomorrow at 5pm near the office', { prior: [{ role: 'user', content: 'I want to meet you' }, { role: 'assistant', content: 'Sure, when?' }] });
  assert.equal(r.calls.length, 0);
  assert.match(r.chunks[0], /5:00 pm/);
});

test('pipeline: questions about plans and ordinary chat still go to the model; the template can be turned off', async () => {
  assert.equal((await replyTo('what are you doing tomorrow?')).calls.length, 1);
  assert.equal((await replyTo('how does a closure work in javascript')).calls.length, 1);
  assert.equal((await replyTo("let's meet tomorrow at 5pm", { settingsOver: { meetingAck: false } })).calls.length, 1);
});

// ---------------------------------------------- bare hour after a day word, pass-on, and wider intent gating

import { isPassOnRequest } from '../src/brain/agenda.js';
import { passOnAck } from '../src/brain/invite.js';

test('a bare hour right after a day word is a time; dates and counts are not', () => {
  assert.equal(at('tomorrow 3').when, 'Wed, 11 Nov, 3:00 pm');
  assert.equal(at('I want to meet mehul urgently tomorrow 3').when, 'Wed, 11 Nov, 3:00 pm');
  assert.equal(at('kal 5 milte hain').when, 'Wed, 11 Nov, 5:00 pm');
  assert.equal(at('tonight 8').when, 'Tue, 10 Nov, 8:00 pm');
  assert.equal(at('tomorrow 11').when, 'Wed, 11 Nov, 11:00 am');
  // not times
  for (const t of ['tomorrow 12/11', 'monday 3rd', 'tomorrow 3 people', 'tomorrow 15 nov', 'friday 2 days']) {
    const r = at(t);
    assert.equal(r.allDay, true, `${t} must stay date-only`);
  }
  assert.equal(at('tomorrow 3 pm').when, 'Wed, 11 Nov, 3:00 pm', 'explicit am/pm still wins');
});

test('pass-on requests in English, Hinglish and Devanagari', () => {
  const names = ['sumit', 'soni'];
  for (const t of [
    'I have a client meeting and I want to join.So remind him',
    'please tell him I called',
    'tell Sumit the file is ready',
    'let him know I am waiting',
    'can you inform sumit about the invoice',
    'ask him to call me back',
    'pass it on to him please',
    'unhe bol dena ki main aaya tha',
    'usse bolo mujhe call kare',
    'sumit ko bata dena',
    'उन्हें बोल देना कि मैं आया था',
  ]) assert.equal(isPassOnRequest(t, { ownerNames: names }), true, t);
  for (const t of ['I will tell him myself', "i'll ask him tomorrow", 'what is the capital of France', 'summarize this: tell him to leave', 'hello?', 'how are you']) {
    assert.equal(isPassOnRequest(t, { ownerNames: names }), false, t);
  }
});

test('pass-on reply: noted, passed on, nothing promised, in their language', () => {
  const p = { name: 'Prachi Shah', owner: 'Sumit Soni' };
  const en = passOnAck({ ...p, code: 'en' });
  assert.match(en, /^Noted Prachi!/);
  assert.match(en, /pass it on to the real Sumit/);
  assert.doesNotMatch(en, /\b(sure|will do|confirmed|yes)\b/i);
  assert.match(passOnAck({ ...p, code: 'hi-Latn' }), /asli Sumit tak pahuncha di jayegi/);
  assert.match(passOnAck({ ...p, code: 'hi' }), /असली Sumit/);
  assert.match(passOnAck({ ...p, code: 'mr' }), /खऱ्या Sumit/);
  assert.match(passOnAck({ ...p, code: 'gu' }), /સાચા Sumit/);
});

test('pipeline: naming the owner (about_owner) or opening with "Hello?" still gets the meeting template', async () => {
  const mehul = { ownerName: 'Mehul Gosavi', ownerAliases: 'mehul' };
  const a = await replyTo('I want to meet mehul urgently tomorrow 3', { settingsOver: mehul });
  assert.equal(a.calls.length, 0);
  assert.equal(a.done.meta.intent, 'meeting');
  assert.match(a.chunks[0], /Wed, 11 Nov|3:00 pm/);
  assert.match(a.chunks[0], /real Mehul/);
  const b = await replyTo('I want to meet Sumit we have a client meeting today at 8 pm');
  assert.equal(b.calls.length, 0);
  assert.match(b.chunks[0], /8:00 pm/);
  const c = await replyTo('Hello?\nI want to meet him tomorrow 3 pm\nHello?');
  assert.equal(c.calls.length, 0);
  assert.match(c.chunks[0], /3:00 pm/);
});

test('pipeline: "remind him" gets the pass-on template; coding and ordinary chat still reach the model', async () => {
  const r = await replyTo('I have a client meeting and I want to join.So remind him');
  assert.equal(r.calls.length, 0);
  assert.equal(r.done.meta.intent, 'pass_on');
  assert.match(r.chunks[0], /^Noted Prachi!/);
  const hi = await replyTo('unhe bol dena ki main aaya tha');
  assert.equal(hi.calls.length, 0);
  assert.match(hi.chunks[0], /asli Sumit/);
  assert.equal((await replyTo("I'll tell him myself")).calls.length, 1);
  assert.equal((await replyTo('write a python function to reverse a string')).calls.length, 1);
  assert.equal((await replyTo('remind him', { settingsOver: { meetingAck: false } })).calls.length, 1, 'switched off with meetingAck');
});
