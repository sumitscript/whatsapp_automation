import test from 'node:test';
import assert from 'node:assert/strict';
import { guessLocation, guessLocationFromContext, isMeetingRequest, looksLikeCommitment, meetingKind, mentionsMeeting, normalizeTimeText, parseWhen, formatWhen, heuristicItem } from '../src/brain/agenda.js';
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
  assert.match(en, /Wed, 11 Nov, 6:00 am\n📍 Railway station/);
  assert.match(en, /real Sumit/);
  assert.doesNotMatch(en, /\b(sure|yes|okay|ok|confirmed)\b/i);
  assert.match(meetingAck({ ...base, code: 'hi-Latn', dueAt }), /asli Sumit tak pahuncha diya/);
  assert.match(meetingAck({ ...base, code: 'hi', dueAt }), /असली Sumit/);
  assert.match(meetingAck({ ...base, code: 'gu', dueAt }), /સાચા Sumit/);
  assert.match(meetingAck({ ...base, code: 'mr', dueAt }), /खऱ्या Sumit/);
  assert.match(meetingAck({ ...base, code: 'en' }), /When would you like to meet, and how/, 'no time → asks');
  const dateOnly = meetingAck({ ...base, code: 'en', dueAt, allDay: true, place: 'BKC Mumbai' });
  assert.match(dateOnly, /What time works for you?/, 'date only → asks only for the time');
  assert.match(dateOnly, /Wed, 11 Nov/, '…and says what was understood');
  assert.match(dateOnly, /BKC Mumbai/);
  assert.notEqual(dateOnly, meetingAck({ ...base, code: 'en' }), 'a follow-up question is not word-for-word the first one');
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
  assert.match(en, /pass this on to the real Sumit/);
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

// ------------------------------------------------- the 12 Nov chat: swallowed period, word dates, places, "milna hai"

import { wantsToMeet } from '../src/brain/agenda.js';

const NOW2 = new Date('2026-11-12T05:11:00Z'); // Thu 12 Nov, 10:41 in Asia/Kolkata
const at2 = (text) => {
  const r = parseWhen(text, { now: NOW2, timeZone: TZ });
  return r && formatWhen(r.dueAt, TZ, r.allDay);
};

test('a full stop right after "p m" does not glue the next word on ("9pmat")', () => {
  assert.equal(normalizeTimeText('nine p m.At sayaji'), '9pm at sayaji');
  assert.equal(normalizeTimeText('at 9 p.m. at the mall').replace(/\s+/g, ' '), 'at 9pm at the mall');
  assert.equal(normalizeTimeText('6 pm tomorrow'), '6pm tomorrow', 'no dot, nothing changes');
  assert.equal(at2("Let's meet on twelfth november nine p m.At sayaji pawar par"), 'Thu, 12 Nov, 9:00 pm', 'the message that used to ask for the details again');
});

test('days of the month in words: ordinals and cardinals, with or without "of"', () => {
  assert.equal(at2('meet on twenty first of november at 5 pm'), 'Sat, 21 Nov, 5:00 pm');
  assert.equal(at2('first dec 6 pm lunch'), 'Tue, 1 Dec, 6:00 pm');
  assert.equal(at2('november twelve 4pm'), 'Thu, 12 Nov, 4:00 pm');
  assert.equal(at2('twelve november 4pm'), 'Thu, 12 Nov, 4:00 pm');
  assert.equal(at2('thirtieth november 4pm'), 'Mon, 30 Nov, 4:00 pm');
  assert.equal(at2('on the thirty first at 3 pm'), 'Thu, 31 Dec, 3:00 pm', 'November has no 31st: the next month that has one');
  assert.equal(at2('12th november 9pm'), 'Thu, 12 Nov, 9:00 pm');
  // a clock time next to a month name is not a date, and "the second" alone is the unit of time
  assert.equal(at2('november nine p m'), 'Thu, 12 Nov, 9:00 pm', 'a clock time after a month name is 9 pm today, not the 9th of November');
  assert.equal(normalizeTimeText('wait a second then at 5'), 'wait a second then at 5');
  assert.match(normalizeTimeText('twelve november nine p m'), /^12 november 9pm$/);
});

test('the place stops where the purpose starts, and "add" typed for "at" after a time is read as "at"', () => {
  assert.equal(guessLocation("Let's meet on 20th November morning 10 am at beyond 7 caffe for bank financial statements"), 'Beyond 7 caffe');
  assert.equal(guessLocation('meet at the railway station about the contract'), 'Railway station');
  assert.equal(guessLocation('lunch at pizza hut regarding the project tomorrow'), 'Pizza hut');
  assert.equal(guessLocation('On 12 november 9 p m add sayaji bagh'), 'Sayaji bagh');
  assert.equal(guessLocation('tomorrow add the office'), 'Office');
  assert.equal(guessLocation('please add sayaji bagh to the list'), null, '"add" with no date/time before it is just the word add');
  assert.equal(guessLocation("Let's meet on twelfth november nine p m.At sayaji pawar par"), 'Sayaji pawar', 'a trailing Hinglish "par" is dropped');
});

test('wanting to meet with no time yet ("I want to meet you", "milna hai") is detected; small talk is not', () => {
  for (const t of ['I want to meet you', 'I really need to meet urgently', 'i wanna meet him', 'We would like to meet', 'Mujhe aapse milna hai', 'mujhe unse milna hai', 'milna hai', 'meet karna hai', 'mulakat karni hai', 'मुझे मिलना है', 'mujhe milna tha']) {
    assert.equal(wantsToMeet(t), true, t);
  }
  for (const t of ['nice to meet you', 'Glad to meet you!', 'I need to see the report', 'summarize this: I want to meet you', 'how are you', 'what is the capital of France']) {
    assert.equal(wantsToMeet(t), false, t);
  }
});

test('pipeline: "Mujhe aapse milna hai" gets the "which day, time and place?" reply, not a model answer', async () => {
  const hi = await replyTo('Mujhe aapse milna hai');
  assert.equal(hi.calls.length, 0);
  assert.equal(hi.done.meta.intent, 'meeting');
  assert.match(hi.chunks[0], /Kab milna hai, aur kaise: Google Meet, call ya in person/);
  const en = await replyTo('I want to meet you');
  assert.equal(en.calls.length, 0);
  assert.match(en.chunks[0], /When would you like to meet, and how/);
  assert.equal((await replyTo('nice to meet you')).calls.length, 1, 'small talk still goes to the model');
  assert.equal((await replyTo('Mujhe aapse milna hai', { settingsOver: { meetingAck: false } })).calls.length, 1);
});

test('pipeline: the twelfth-november message with the swallowed period now gets the full reply with date, time and place', async () => {
  const r = await replyTo("Let's meet on twelfth november nine p m.At sayaji pawar par");
  assert.equal(r.calls.length, 0);
  assert.match(r.chunks[0], /9:00 pm/);
  assert.match(r.chunks[0], /Sayaji pawar/);
  assert.doesNotMatch(r.chunks[0], /par$/m);
});

test('the heuristic item and the extraction fallback use the place named in the current message', () => {
  const it = heuristicItem({ text: 'On 12 november 9 p m add sayaji bagh', contextText: "Let's meet at sayaji pawar", contactName: 'Aditya Soni', now: NOW2, timeZone: TZ });
  assert.equal(it.location, 'Sayaji bagh');
  assert.equal(formatWhen(it.dueAt, TZ, it.allDay), 'Thu, 12 Nov, 9:00 pm');
});

test('rescheduled confirmations say "updated" in every language and keep the new time and place', () => {
  const p = { name: 'Aditya Soni', owner: 'Sumit Soni', title: 'Meet Aditya', dueAt: new Date('2026-11-26T04:30:00Z'), place: 'Beyond 7 caffe', timeZone: TZ };
  assert.equal(confirmationText({ ...p, code: 'en', rescheduled: true }), '🔁 Updated, Aditya! Sumit has confirmed the new time: Meet Aditya\n🗓 Thu, 26 Nov, 10:00 am\n📍 Beyond 7 caffe\nSee you then 🙂');
  assert.match(confirmationText({ ...p, code: 'hi-Latn', rescheduled: true }), /^🔁 Update, Aditya! Sumit ne naya time confirm kiya hai/);
  assert.match(confirmationText({ ...p, code: 'hi', rescheduled: true }), /^🔁 बदलाव, Aditya!/);
  assert.match(confirmationText({ ...p, code: 'mr', rescheduled: true }), /^🔁 बदल, Aditya!/);
  assert.match(confirmationText({ ...p, code: 'gu', rescheduled: true }), /^🔁 ફેરફાર, Aditya!/);
  assert.match(confirmationText({ ...p, code: 'en' }), /^✅ Confirmed, Aditya!/, 'without the flag it is the normal confirmation');
});

test('"Tell him hi for me" never lends its words as a place; a meeting message does', () => {
  assert.equal(guessLocation('Tell him hi for me'), null);
  assert.equal(guessLocation('Schedule a meet with him for 10am tomorrow regarding a prospectus client'), null);
  assert.equal(guessLocationFromContext(['Tell him hi for me'], { now: NOW, timeZone: TZ }), null);
  assert.equal(guessLocationFromContext(['hello', 'I want to meet her at bkc mumbai', 'ok'], { now: NOW, timeZone: TZ }), 'Bkc mumbai');
});

// ------------------------------------------------------------------ what / how, typos, the date that was asked

test('a meeting names its platform and its purpose: Google Meet, WebEx, in person, "code review"', async () => {
  const { meetingMode, meetingTopic, meetingTitle } = await import('../src/brain/agenda.js');
  assert.equal(meetingMode('Can i have a meeting with sumit tommorow on google meet'), 'Google Meet');
  assert.equal(meetingMode('I want on 20tth nov for code review on WebEx'), 'WebEx');
  assert.equal(meetingMode("let's do it in person"), 'In person');
  assert.equal(meetingMode('a phone call tomorrow'), 'Phone call');
  assert.equal(meetingMode('see you tomorrow at 5'), null);
  assert.equal(meetingTopic('I want on 20tth nov for code review on WebEx'), 'Code review');
  assert.equal(meetingTopic('lets meet on 22 nov 4pm in person for project discussion'), 'Project discussion');
  assert.equal(meetingTopic('can we meet tomorrow'), null);
  assert.equal(meetingTitle({ topic: 'Code review', contactName: 'Sugam Sharma' }), 'Code review with Sugam');
  assert.equal(meetingTitle({ contactName: 'Sugam Sharma' }), 'Meet Sugam');
});

test('typos people really make are read: "20tth nov", "14 noc 9 0m", "I wany to meet Sumit"', async () => {
  const { parseWhen, wantsToMeet, isMeetingRequest, normalizeTimeText } = await import('../src/brain/agenda.js');
  const now = new Date('2026-11-13T09:00:00Z');
  assert.equal(parseWhen('I want on 20tth nov for code review', { now })?.dueAt.toISOString().slice(0, 10), '2026-11-20');
  const t = parseWhen('14 noc 9 0m', { now });
  assert.equal(t.dueAt.toISOString(), '2026-11-14T15:30:00.000Z'); // 9 pm IST
  assert.ok(wantsToMeet('I wany to meet Sumit'));
  assert.ok(isMeetingRequest('I want on 20tth nov for code review on WebEx'));
  assert.equal(isMeetingRequest('what is the capital of france'), false);
  assert.match(normalizeTimeText('I will not 5 now 6'), /not 5 now 6/, 'real words are not turned into months');
});

test('a place must come from the person\'s words, never from one of our own replies', async () => {
  const { placeFromTheirWords } = await import('../src/brain/agenda.js');
  assert.equal(placeFromTheirWords('Asli Sumit tak pahuncha diya jayega', ['meeting tomorrow on google meet']), null);
  assert.equal(placeFromTheirWords('Railway station', ['meet me at the railway station at 6']), 'Railway station');
  assert.equal(placeFromTheirWords('Sayaji Garden', ['meet tomorrow']), null, 'invented by the model');
});

test('the reply to a meeting request shows what and how, and asks how when it is not said', async () => {
  const { meetingAck } = await import('../src/brain/invite.js');
  const base = { code: 'en', name: 'Sugam Sharma', owner: 'Sumit Soni', dueAt: new Date('2026-11-20T15:30:00Z'), allDay: false, timeZone: 'Asia/Kolkata' };
  const ack = meetingAck({ ...base, topic: 'Code review', mode: 'WebEx' });
  assert.match(ack, /Fri, 20 Nov, 9:00 pm\n📝 Code review · 💻 WebEx/);
  const askTime = meetingAck({ ...base, allDay: true });
  assert.match(askTime, /What time works for you\? And how: Google Meet, a call or in person\?/);
  assert.doesNotMatch(meetingAck({ ...base, allDay: true, mode: 'Google Meet' }), /And how/, 'already said');
});

test('a clear English sentence is answered in English even when an old note says "Prefers Hinglish"; short words still follow the note', async () => {
  const { replyLanguage } = await import('../src/brain/lang.js');
  const notes = '- Name: Sugam\n- Prefers Hinglish';
  assert.equal(replyLanguage('Can i have a meeting with sumit tomorrow on google meet', [], notes).code, 'en');
  assert.equal(replyLanguage('Tell me about your Projects', [], notes).code, 'en');
  assert.equal(replyLanguage('ok', [], notes).code, 'hi-Latn');
  assert.equal(replyLanguage('can you help', ['I prefer hinglish'], '').code, 'hi-Latn', 'said in this chat: kept');
});

test('pipeline: "meeting tomorrow on google meet" then just "9 PM" books tomorrow 9 pm on Google Meet, in English, and the done-meta carries what/how', async () => {
  await kb.load().catch(() => {});
  const calls = [];
  const llm = { numCtx: 4096, async *chat(req) { calls.push(req); yield { type: 'token', text: 'MODEL REPLY' }; yield { type: 'done', stats: {} }; } };
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  const send = async (text) => {
    const chunks = [];
    let done;
    for await (const ev of brain.respond({ chatId: 'gm', text, settings: { ...settings, timezone: 'Asia/Kolkata' }, contactName: 'Banjesh Patel', longTermMemory: '- Prefers Hinglish' })) {
      if (ev.type === 'chunk') chunks.push(ev.text);
      else if (ev.type === 'done') done = ev;
    }
    await memory.append('gm', [{ role: 'user', content: text }, { role: 'assistant', content: chunks.join('\n') }]);
    return { chunks, done };
  };
  const first = await send('Can i have a meeting with sumit tomorrow on google meet');
  assert.match(first.chunks[0], /^Noted Banjesh!/, 'English message, English reply, even though the notes say Hinglish');
  assert.match(first.chunks[0], /Google Meet/);
  assert.match(first.chunks[0], /What time works for you\?/);
  assert.doesNotMatch(first.chunks[0], /And how/, 'it already said Google Meet');
  const second = await send('9 PM');
  assert.equal(calls.length, 0, 'no model call for either');
  const m = second.done.meta.meeting;
  const day = (iso) => new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(new Date(iso));
  const tomorrow = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(new Date(Date.now() + 86_400_000));
  assert.equal(day(m.dueAt), tomorrow, 'the day that was asked about, not today');
  assert.equal(new Date(m.dueAt).toLocaleTimeString('en-GB', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }), '21:00');
  assert.equal(m.mode, 'Google Meet');
});
