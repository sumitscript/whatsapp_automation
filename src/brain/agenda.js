// Agenda: appointments and to-dos that people ask of the owner in chats
// ("let's meet tomorrow at 5", "remind me to send the file by Friday").
//
// Everything here is pure and dependency-free (unit-tested). Dates are read
// deterministically with Intl — a 3B model is unreliable at date math — and the
// LLM is only used to decide *what* the commitment is (title, kind, details).
import { estimateTokens } from '../rag/text.js';

export const KINDS = ['appointment', 'todo'];
export const STATUSES = ['pending', 'confirmed', 'declined', 'done'];

// ---------------------------------------------------------------- time zones

const dtfCache = new Map();
function dtf(timeZone) {
  let f = dtfCache.get(timeZone);
  if (!f) {
    f = new Intl.DateTimeFormat('en-US', { timeZone, year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric', second: 'numeric', hourCycle: 'h23', weekday: 'short' });
    dtfCache.set(timeZone, f);
  }
  return f;
}

const WD = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 };

/** Wall-clock parts of an instant in a zone. */
export function localParts(date, timeZone) {
  const o = {};
  for (const p of dtf(timeZone).formatToParts(date)) o[p.type] = p.value;
  return { y: +o.year, m: +o.month, d: +o.day, h: +o.hour % 24, mi: +o.minute, s: +o.second, wd: WD[o.weekday] ?? 0 };
}

const offsetMs = (date, timeZone) => {
  const p = localParts(date, timeZone);
  return Date.UTC(p.y, p.m - 1, p.d, p.h, p.mi, p.s) - Math.floor(date.getTime() / 1000) * 1000;
};

/** The instant at which the wall clock in `timeZone` reads y-m-d h:mi. */
export function zonedToUtc(y, m, d, h, mi, timeZone) {
  const guess = Date.UTC(y, m - 1, d, h, mi);
  let t = guess - offsetMs(new Date(guess), timeZone);
  t = guess - offsetMs(new Date(t), timeZone); // second pass handles DST edges
  return new Date(t);
}

const addDays = (y, m, d, n) => {
  const x = new Date(Date.UTC(y, m - 1, d + n));
  return { y: x.getUTCFullYear(), m: x.getUTCMonth() + 1, d: x.getUTCDate(), wd: x.getUTCDay() };
};

/** "Tue, 5 Nov, 5:00 pm" (date only when allDay). */
export function formatWhen(date, timeZone, allDay = false) {
  if (!date) return 'no date';
  const d = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(d.getTime())) return 'no date';
  const opts = { timeZone, weekday: 'short', day: 'numeric', month: 'short' };
  if (!allDay) Object.assign(opts, { hour: 'numeric', minute: '2-digit', hour12: true });
  return new Intl.DateTimeFormat('en-IN', opts).format(d);
}

// ------------------------------------------------------------- reading times

const WEEKDAY_FULL = { sunday: 0, monday: 1, tuesday: 2, wednesday: 3, thursday: 4, friday: 5, saturday: 6 };
const WEEKDAY_ABBR = { sun: 0, mon: 1, tue: 2, tues: 2, wed: 3, thu: 4, thur: 4, thurs: 4, fri: 5, sat: 6 };
const MONTHS = { jan: 1, january: 1, feb: 2, february: 2, mar: 3, march: 3, apr: 4, april: 4, may: 5, jun: 6, june: 6, jul: 7, july: 7, aug: 8, august: 8, sep: 9, sept: 9, september: 9, oct: 10, october: 10, nov: 11, november: 11, dec: 12, december: 12 };
const MONTH_RE = Object.keys(MONTHS).sort((a, b) => b.length - a.length).join('|');

const EVENING = /\b(evening|night|tonight|shaam|raat|afternoon|dopahar|pm)\b/;

const NUMBER_WORDS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12 };

/**
 * People type times loosely: "six p m", "6 p.m", "five o'clock", "at seven baje".
 * Lower-cases and rewrites those into the digit forms the date reader understands.
 */
export function normalizeTimeText(text) {
  return String(text ?? '')
    .toLowerCase()
    .replace(/\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b(?=\s*(?:o['’]?clock|[ap]\s?\.?\s?m\b|baje|bje))/g, (w) => String(NUMBER_WORDS[w]))
    .replace(/(\d)\s*([ap])\s?\.?\s?m\b\.?/g, '$1$2m');
}

/**
 * Read a date/time from free text ("tomorrow at 5pm", "kal shaam 6 baje",
 * "next friday", "12/11 10:30"). Returns null when the text has no date or time.
 * Bare hours: 1–6 mean pm, 7–11 mean am unless an evening word is present.
 * @returns {{ dueAt: Date, allDay: boolean, matched: string } | null}
 */
export function parseWhen(text, { now = new Date(), timeZone = 'Asia/Kolkata' } = {}) {
  const t = normalizeTimeText(text);
  if (!t.trim()) return null;
  const today = localParts(now, timeZone);
  const matched = [];
  const take = (m) => m && matched.push(m[0].trim());

  // ---- day ----
  let day = null; // {y,m,d}
  let m;
  if ((m = /\bday after tomorrow\b|\bparso\b/.exec(t))) {
    day = addDays(today.y, today.m, today.d, 2);
    take(m);
  } else if ((m = /\b(tomorrow|tmrw|tmr|tommorow|tommorrow|tomorow|kal)\b/.exec(t))) {
    day = addDays(today.y, today.m, today.d, 1);
    take(m);
  } else if ((m = /\b(today|tonight|aaj)\b/.exec(t))) {
    day = addDays(today.y, today.m, today.d, 0);
    take(m);
  }
  if (!day && (m = /\b(?:(next|this|coming|on|by)\s+)?(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/.exec(t))) {
    day = nextWeekday(today, WEEKDAY_FULL[m[2]]);
    take(m);
  }
  if (!day && (m = /\b(next|this|coming|on|by)\s+(sun|mon|tues?|wed|thur?s?|fri|sat)\b/.exec(t))) {
    day = nextWeekday(today, WEEKDAY_ABBR[m[2]]);
    take(m);
  }
  if (!day && (m = /\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b/.exec(t))) {
    const dd = +m[1];
    const mm = +m[2];
    if (dd >= 1 && dd <= 31 && mm >= 1 && mm <= 12) {
      day = explicitDate(today, dd, mm, m[3] ? +m[3] : null);
      take(m);
    }
  }
  if (!day && (m = new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s*(?:of\\s+)?(${MONTH_RE})\\b`).exec(t))) {
    day = explicitDate(today, +m[1], MONTHS[m[2]], null);
    take(m);
  }
  if (!day && (m = new RegExp(`\\b(${MONTH_RE})\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b`).exec(t))) {
    day = explicitDate(today, +m[2], MONTHS[m[1]], null);
    take(m);
  }

  // ---- time ----
  let hour = null;
  let minute = 0;
  if ((m = /\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])/.exec(t))) {
    const h = +m[1];
    if (h >= 1 && h <= 12) {
      hour = (h % 12) + (m[3].startsWith('p') ? 12 : 0);
      minute = m[2] ? +m[2] : 0;
      take(m);
    }
  }
  if (hour == null && (m = /\b([01]?\d|2[0-3]):([0-5]\d)\b/.exec(t))) {
    hour = +m[1];
    minute = +m[2];
    take(m);
  }
  if (hour == null && (m = /\b(?:at|by|around)\s+(\d{1,2})(?:[:.]([0-5]\d))?\b(?!\s*(?:st|nd|rd|th|\/|-))|\b(\d{1,2})(?:[:.]([0-5]\d))?\s*(?:baje|bje|o['’]?clock)\b/.exec(t))) {
    const h = +(m[1] ?? m[3]);
    minute = +(m[2] ?? m[4] ?? 0);
    if (h >= 1 && h <= 23) {
      hour = h <= 6 || (h < 12 && EVENING.test(t)) ? h + 12 : h;
      take(m);
    }
  }
  if (hour == null) {
    const part = /\b(noon|midday|midnight|morning|subah|subha|afternoon|dopahar|evening|shaam|tonight|night|raat)\b/.exec(t);
    if (part) {
      const w = part[1];
      hour = { noon: 12, midday: 12, midnight: 0, morning: 9, subah: 9, subha: 9, afternoon: 15, dopahar: 15, evening: 18, shaam: 18, tonight: 20, night: 20, raat: 20 }[w];
      take(part);
    }
  } else if (hour < 12 && /\b(evening|night|shaam|raat|afternoon|dopahar)\b/.test(t) && !/\b(a\.?m\.?)\b/.test(t)) {
    hour += 12;
  }

  if (!day && hour == null) return null;

  const hasDay = !!day;
  let allDay = false;
  if (hour == null) {
    hour = 9;
    minute = 0;
    allDay = true;
  }
  if (!day) {
    // time only: later today, otherwise tomorrow
    const nowMin = today.h * 60 + today.mi;
    day = hour * 60 + minute > nowMin ? { y: today.y, m: today.m, d: today.d } : addDays(today.y, today.m, today.d, 1);
  }
  return { dueAt: zonedToUtc(day.y, day.m, day.d, hour, minute, timeZone), allDay, hasDay, matched: matched.join(' ') };
}

function nextWeekday(today, wd) {
  let off = (wd - today.wd + 7) % 7;
  if (off === 0) off = 7;
  return addDays(today.y, today.m, today.d, off);
}

function explicitDate(today, d, m, y) {
  let year = y ? (y < 100 ? 2000 + y : y) : today.y;
  const probe = Date.UTC(year, m - 1, d);
  if (!y && probe < Date.UTC(today.y, today.m - 1, today.d)) year += 1; // "12 jan" said in December
  return { y: year, m, d };
}

// ------------------------------------------------------- spotting commitments

const MEET =
  /\b(meet|meeting|meetup|catch ?up|call|calling|lunch|dinner|breakfast|coffee|visit|appointment|schedule[d]?|reschedule|interview|demo|discussion|sync|session|hangout|come over|coming|milte|milenge|milna|mulakat|bulana|aana|aaunga|aaoge)\b/i;
const TODO =
  /\b(remind(?:er)?|remember to|don'?t forget|dont forget|todo|to-do|deadline|due (?:on|by|date)|submit|send me|share (?:the|your|it)|follow ?up|call ?back|yaad dil\w*|yaad rakh\w*)\b/i;
const STRONG_TIME =
  /\b(today|tonight|tomorrow|tmrw|tomm?orow|tommorrow|day after|kal|aaj|parso|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|weekend|\d{1,2}\s*(?::\d{2})?\s*(?:am|pm)|\d{1,2}:\d{2}|\d{1,2}\s*baje)\b/i;
const ASK = /\b(free|available|availability|let'?s|lets|shall we|can we|can you|could you|will you|are you (?:coming|free)|come|aana|aa jao|milte)\b/i;

/** Cheap gate so the LLM only reads messages that might hold a commitment. */
export function looksLikeCommitment(text) {
  const s = normalizeTimeText(text);
  if (s.length < 4 || s.length > 1500) return false;
  return TODO.test(s) || (MEET.test(s) && STRONG_TIME.test(s)) || (STRONG_TIME.test(s) && ASK.test(s));
}

/**
 * Same gate, but the request and its details often arrive as separate messages:
 * "I want to meet" … then "tomorrow at six near the office". A message that carries
 * a time counts when a recent earlier message from the same person asked for something.
 * @param {string} text the latest message
 * @param {string[]} [previousUserTexts] this person's earlier messages, oldest first
 */
export function looksLikeCommitmentInContext(text, previousUserTexts = []) {
  if (looksLikeCommitment(text)) return true;
  const recent = previousUserTexts.slice(-3).join('\n');
  if (!recent || String(text ?? '').length > 400) return false;
  return STRONG_TIME.test(normalizeTimeText(text)) && (MEET.test(recent) || ASK.test(recent) || TODO.test(recent));
}

/** Gate for prompting the *reply* model to not agree on the owner's behalf. */
// A task on pasted text ("summarize this: the meeting moved to Friday") mentions meetings but asks for none.
const TEXT_TASK = /\b(summari[sz]e|summary|translate|explain|rewrite|proofread|what(?:'|’)?s on my)\b/i;
const DIRECT_ASK = /\b(call|ping|meet|text|message|ring) me|please confirm|pls confirm|confirm (it|this|that|the)\b|can you confirm/i;
// "What are you doing tomorrow?", "are you free tonight", "aaj kya kar rahe ho": the twin has no real
// schedule, so a small model makes one up ("wrapping up some coding challenges"). Pinned by a rule in the prompt.
const PLANS_Q = /\bwhat(?:'|’)?s?\s+(?:(?:are|r)\s+)?(?:you|u)\s+(?:doing|up\s+to)\b|\bwhat(?:'|’)?s\s+(?:your|ur)\s+(?:plan|plans|schedule)\b|\bwhat\s+(?:are\s+)?(?:your|ur)\s+plans\b|\b(?:are|r)\s+(?:you|u)\s+(?:free|busy|available)\b|\b(?:kya|kaha|kahan)\s+(?:kar\s+rah[ae]|ho|hai)\b|\bplans?\s+(?:for|on)\s+(?:today|tonight|tomorrow|weekend|sunday|saturday)\b/i;
export function asksAboutPlans(text) {
  return PLANS_Q.test(String(text || ''));
}

export function isMeetingRequest(text) {
  const s = String(text ?? '');
  if (TEXT_TASK.test(s)) return false;
  return (MEET.test(s) && (STRONG_TIME.test(s) || ASK.test(s))) || (STRONG_TIME.test(s) && (ASK.test(s) || DIRECT_ASK.test(s))) || (MEET.test(s) && DIRECT_ASK.test(s));
}

// ---------------------------------------------------------------- extraction

const clip = (s, n) => (String(s).length > n ? `${String(s).slice(0, n - 1)}…` : String(s));
const firstName = (s) => String(s || 'the owner').trim().split(/\s+/)[0];

/** Chat messages for the LLM extraction call. */
export function buildExtractionMessages({ ownerName, contactName, turns, now = new Date(), timeZone = 'Asia/Kolkata' }) {
  const owner = firstName(ownerName);
  const who = firstName(contactName || 'them');
  const lines = [];
  let budget = 700;
  for (let i = turns.length - 1; i >= 0 && lines.length < 6; i--) {
    const line = `${turns[i].role === 'user' ? who : `${owner}'s AI`}: ${clip(turns[i].content, 300)}`;
    budget -= estimateTokens(line);
    if (budget < 0) break;
    lines.unshift(line);
  }
  const today = new Intl.DateTimeFormat('en-GB', { timeZone, weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).format(now);
  return [
    {
      role: 'system',
      content: `You read a WhatsApp chat and list the commitments that ${who} asks of, or proposes to, ${owner}: meetings, calls, visits, deadlines, favours, reminders. Ignore small talk, general questions and anything ${owner}'s AI said on its own.
Reply with ONLY JSON: {"items":[{"kind":"appointment"|"todo","title":"short, e.g. Meet ${who}","details":"one short line or null","when":"the exact time words they used, or null","location":"place or link, or null"}]}
"appointment" = something at a time/place (meet, call, visit). "todo" = something ${owner} must do. Never invent times or places; copy their words. If there is nothing, reply {"items":[]}.`,
    },
    { role: 'user', content: `Today is ${today} (${timeZone}).\n\nChat:\n${lines.join('\n')}` },
  ];
}

/** Parse the model's JSON (tolerates code fences / chatter around it). */
export function parseExtraction(raw) {
  const s = String(raw ?? '');
  const a = s.indexOf('{');
  const b = s.lastIndexOf('}');
  if (a < 0 || b <= a) return null;
  let data;
  try {
    data = JSON.parse(s.slice(a, b + 1));
  } catch {
    return null;
  }
  if (!Array.isArray(data?.items)) return null;
  const str = (v, n) => (typeof v === 'string' && v.trim() && !/^(null|none|n\/a|-)$/i.test(v.trim()) ? clip(v.trim(), n) : null);
  return data.items
    .slice(0, 3)
    .map((it) => ({
      kind: KINDS.includes(it?.kind) ? it.kind : 'todo',
      title: str(it?.title, 120),
      details: str(it?.details, 500),
      when: str(it?.when, 80),
      location: str(it?.location, 160),
    }))
    .filter((it) => it.title);
}

/**
 * Attach dates to extracted items. The model's "when" phrase is tried first,
 * then the latest message, then the last few user messages together (a
 * "yes, 5pm works" reply borrows "tomorrow" from the previous turn).
 */
export function resolveItems(items, { text, userTexts = [], now = new Date(), timeZone = 'Asia/Kolkata' }) {
  const opts = { now, timeZone };
  return items.map((it) => {
    // Prefer a reading that names a day; a bare time ("yes 5pm works") borrows the day from earlier turns.
    const tries = [it.when && parseWhen(it.when, opts), parseWhen(text, opts), parseWhen([...userTexts.slice(-3), text].join('\n'), opts)].filter(Boolean);
    const hit = tries.find((t) => t.hasDay) || tries[0];
    return {
      ...it,
      dueAt: hit?.dueAt ?? null,
      allDay: hit?.allDay ?? false,
      dueText: it.when || hit?.matched || null,
    };
  });
}

/** Item built without the LLM, when the message is clearly a meeting request. */
export function heuristicItem({ text, contextText = '', contactName, now = new Date(), timeZone = 'Asia/Kolkata' }) {
  const hit = parseWhen(text, { now, timeZone });
  const meet = MEET.test(text) || MEET.test(contextText); // "I want to meet" may be an earlier message
  if (!hit && !TODO.test(text)) return null;
  if (meet && hit) {
    return { kind: 'appointment', title: `Meet ${firstName(contactName || 'contact')}`, details: clip(text.trim(), 300), when: hit.matched, dueAt: hit.dueAt, allDay: hit.allDay, dueText: hit.matched, location: null };
  }
  return { kind: 'todo', title: clip(text.trim(), 80), details: null, when: hit?.matched ?? null, dueAt: hit?.dueAt ?? null, allDay: hit?.allDay ?? false, dueText: hit?.matched ?? null, location: null };
}

// ---------------------------------------------------------------- duplicates

export const normalizeTitle = (t) =>
  String(t ?? '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();

/** Is `b` (new) the same commitment as `a` (stored)? Same chat + kind, and either close in time or alike in words. */
export function sameItem(a, b) {
  if ((a.chatId ?? null) !== (b.chatId ?? null) || a.kind !== b.kind) return false;
  if (a.dueAt && b.dueAt && Math.abs(new Date(a.dueAt) - new Date(b.dueAt)) <= 2 * 3600_000) return true;
  const wa = new Set(normalizeTitle(a.title).split(' ').filter(Boolean));
  const wb = new Set(normalizeTitle(b.title).split(' ').filter(Boolean));
  if (!wa.size || !wb.size) return false;
  let same = 0;
  for (const w of wa) if (wb.has(w)) same++;
  return same / Math.max(wa.size, wb.size) >= 0.6;
}

// -------------------------------------------------------------- API plumbing

const ISO = /^\d{4}-\d{2}-\d{2}T/;

/**
 * Validate a create (`partial: false`) or update (`partial: true`) body.
 * @returns {{ value: object, errors: string[] }}
 */
export function sanitizeAgendaInput(body, { partial = false } = {}) {
  const b = body && typeof body === 'object' ? body : {};
  const value = {};
  const errors = [];
  const has = (k) => b[k] !== undefined;
  const text = (k, max, { required = false } = {}) => {
    if (!has(k)) return required && !partial ? errors.push(`${k} is required`) : undefined;
    if (b[k] === null || (typeof b[k] === 'string' && !b[k].trim())) {
      if (required) errors.push(`${k} must not be empty`);
      else value[k] = null;
      return;
    }
    if (typeof b[k] !== 'string' || b[k].length > max) return errors.push(`${k} must be text up to ${max} characters`);
    value[k] = b[k].trim();
  };
  text('title', 200, { required: true });
  text('details', 1000);
  text('location', 200);
  if (has('kind')) KINDS.includes(b.kind) ? (value.kind = b.kind) : errors.push(`kind must be one of ${KINDS.join(', ')}`);
  else if (!partial) value.kind = 'todo';
  if (has('status')) STATUSES.includes(b.status) ? (value.status = b.status) : errors.push(`status must be one of ${STATUSES.join(', ')}`);
  if (has('allDay')) typeof b.allDay === 'boolean' ? (value.allDay = b.allDay) : errors.push('allDay must be true or false');
  if (has('dueAt')) {
    if (b.dueAt === null || b.dueAt === '') value.dueAt = null;
    else if (typeof b.dueAt === 'string' && ISO.test(b.dueAt) && !Number.isNaN(new Date(b.dueAt).getTime())) value.dueAt = new Date(b.dueAt).toISOString();
    else errors.push('dueAt must be an ISO date-time or null');
  }
  if (has('chatId')) {
    if (b.chatId === null) value.chatId = null;
    else if (typeof b.chatId === 'string' && /^[\w.\-:@]{3,120}$/.test(b.chatId)) value.chatId = b.chatId;
    else errors.push('chatId is invalid');
  }
  return { value, errors };
}

/** Dashboard numbers. Items are the stored rows (camelCase). */
export function summarizeAgenda(items, { now = new Date(), timeZone = 'Asia/Kolkata' } = {}) {
  const today = localParts(now, timeZone);
  const dayKey = (d) => {
    const p = localParts(new Date(d), timeZone);
    return p.y * 10_000 + p.m * 100 + p.d;
  };
  const todayKey = today.y * 10_000 + today.m * 100 + today.d;
  const weekKey = (() => {
    const e = addDays(today.y, today.m, today.d, 7);
    return e.y * 10_000 + e.m * 100 + e.d;
  })();
  const out = { review: 0, today: 0, next7: 0, openTodos: 0, overdue: 0, total: items.length };
  for (const it of items) {
    const open = it.status === 'pending' || it.status === 'confirmed';
    if (it.kind === 'appointment' && it.status === 'pending') out.review++;
    if (!open) continue;
    if (it.kind === 'todo' && it.status === 'pending') out.openTodos++;
    if (!it.dueAt) continue;
    const k = dayKey(it.dueAt);
    if (k === todayKey) out.today++;
    else if (k > todayKey && k <= weekKey) out.next7++;
    if (it.allDay ? k < todayKey : new Date(it.dueAt) < now) out.overdue++;
  }
  return out;
}
