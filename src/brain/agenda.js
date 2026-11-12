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
const MORNING = /\b(morning|subah|subha)\b/;

const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12,
  // Hindi / Hinglish number words ("paanch baje", "chhe subah")
  ek: 1, do: 2, teen: 3, char: 4, chaar: 4, paanch: 5, panch: 5, paach: 5, chhe: 6, chhah: 6, chah: 6, che: 6, saat: 7, aath: 8, nau: 9, das: 10, gyarah: 11, barah: 12, baarah: 12,
};
const NUMBER_WORD_RE = Object.keys(NUMBER_WORDS).sort((a, b) => b.length - a.length).join('|');

// Native-script words → the Latin tokens the rest of this file reads. Hindi, Marathi and Gujarati cover
// most of the chats this twin gets; a script that is missing here still works through the LLM extraction.
const NATIVE_WORDS = {
  // Hindi / Marathi (Devanagari)
  कल: 'kal', आज: 'aaj', परसों: 'parso', परसो: 'parso', परवा: 'parso', उद्या: 'kal',
  सुबह: 'subah', सवेरे: 'subah', सकाळी: 'subah', शाम: 'shaam', संध्याकाळी: 'shaam', रात: 'raat', रात्री: 'raat', दोपहर: 'dopahar', दुपारी: 'dopahar',
  बजे: 'baje', वाजता: 'baje', वाजले: 'baje',
  मिलना: 'milna', मिलते: 'milte', मिलेंगे: 'milenge', मिलें: 'milein', मिलूंगा: 'milunga', मिलूँगा: 'milunga', मिलूंगी: 'milunga', मुलाकात: 'mulakat', भेटू: 'milna', भेटूया: 'milna', भेटायचं: 'milna', भेटायचे: 'milna',
  मीटिंग: 'meeting', मिटिंग: 'meeting', कॉल: 'call', फोन: 'phone', बैठक: 'meeting',
  // "मुझे मिलना है" (I want to meet)
  है: 'hai', मुझे: 'mujhe', करना: 'karna', करनी: 'karni', चाहता: 'chahta', चाहती: 'chahti', आपसे: 'aapse', तुमसे: 'tumse',
  सोमवार: 'monday', मंगलवार: 'tuesday', बुधवार: 'wednesday', गुरुवार: 'thursday', बृहस्पतिवार: 'thursday', शुक्रवार: 'friday', शनिवार: 'saturday', रविवार: 'sunday',
  // "tell him" in Hindi/Marathi/Gujarati (for isPassOnRequest)
  उन्हें: 'unhe', उनको: 'unko', उसे: 'usse', उसको: 'usko', बोल: 'bol', बोलो: 'bolo', बोलना: 'bolna', बता: 'bata', बताओ: 'batao', बताना: 'batana', कह: 'keh', कहो: 'kaho', कहना: 'kehna', देना: 'dena', दीजिए: 'dijiye',
  त्यांना: 'unhe', सांगा: 'bolo', सांग: 'bolo', તેમને: 'unhe', કહેજો: 'bolo', કહી: 'keh', દેજો: 'dena', જણાવજો: 'batao',
  एक: 'ek', दो: 'do', तीन: 'teen', चार: 'char', पांच: 'paanch', पाँच: 'paanch', छह: 'chhe', छः: 'chhe', छ: 'chhe', सात: 'saat', आठ: 'aath', नौ: 'nau', दस: 'das', ग्यारह: 'gyarah', बारह: 'barah',
  साढ़े: 'sadhe', साढे: 'sadhe', सवा: 'sava', पौने: 'paune', डेढ़: 'dedh', ढाई: 'dhai',
  // Gujarati
  આવતીકાલ: 'kal', કાલે: 'kal', આજે: 'aaj', પરમ: 'parso', સવારે: 'subah', બપોરે: 'dopahar', સાંજે: 'shaam', રાત્રે: 'raat', વાગ્યે: 'baje', વાગે: 'baje',
  મળવું: 'milna', મળીએ: 'milte', મળશું: 'milenge', મળીશું: 'milenge', મુલાકાત: 'mulakat', મીટિંગ: 'meeting', કોલ: 'call',
  સોમવાર: 'monday', મંગળવાર: 'tuesday', બુધવાર: 'wednesday', ગુરુવાર: 'thursday', શુક્રવાર: 'friday', શનિવાર: 'saturday', રવિવાર: 'sunday',
  સાડા: 'sadhe',
};
const NATIVE_RULES = Object.entries(NATIVE_WORDS)
  .sort((a, b) => b[0].length - a[0].length)
  .map(([w, to]) => [new RegExp(`(?<![\\p{L}\\p{M}])${w}(?![\\p{L}\\p{M}])`, 'gu'), ` ${to} `]);
// Latin spellings of Hindi/Gujarati/Marathi weekdays and phrases people type every day.
const LATIN_WORDS = [
  // Names like "Ravi" or "Tom" are not weekdays or "tomorrow": only the unambiguous spellings are mapped.
  [/\b(?:somvar|somwar|somvaar)\b/g, 'monday'], [/\b(?:mangalvar|mangalwar|mangalvaar)\b/g, 'tuesday'], [/\b(?:budhvar|budhwar|budhvaar)\b/g, 'wednesday'],
  [/\b(?:guruvar|guruwar|guruvaar|brihaspativar|brihaspatiwar|veervar|virvar)\b/g, 'thursday'], [/\b(?:shukravar|shukrawar|shukravaar)\b/g, 'friday'],
  [/\b(?:shanivar|shaniwar|shanivaar)\b/g, 'saturday'], [/\b(?:ravivar|raviwar|ravivaar|itwar|itvar)\b/g, 'sunday'],
  [/\b(?:agle|agla|agli)\s+(?:hafte|hafta|week|saptah)\b/g, 'next week'], [/\b(?:is|iss|isi)\s+weekend\b/g, 'weekend'],
  [/\b(?:aavtikal|aavatikal|avtikal|udya|uddya|2morrow|2moro|2mrw|tomrw)\b/g, 'tomorrow'], [/\b(?:savere|savare)\b/g, 'subah'], [/\bparson\b/g, 'parso'],
];

// Days of the month typed or spoken as words: "twelfth november", "twenty first of november", "first dec",
// "november twelve", "on the thirty first". Folded into digits ("12th november") before the date reader runs.
const DAY_UNITS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19 };
const DAY_ORDINALS = { first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8, ninth: 9, tenth: 10, eleventh: 11, twelfth: 12, thirteenth: 13, fourteenth: 14, fifteenth: 15, sixteenth: 16, seventeenth: 17, eighteenth: 18, nineteenth: 19 };
const DAY_TENS = { twenty: 20, thirty: 30 };
const ORDINAL_PHRASE = `(?:(?:twenty|thirty)[\\s-]+)?(?:${Object.keys(DAY_ORDINALS).join('|')})|twentieth|thirtieth`;
const CARDINAL_PHRASE = `(?:(?:twenty|thirty)(?:[\\s-]+(?:one|two|three|four|five|six|seven|eight|nine))?|${Object.keys(DAY_UNITS).join('|')})`;
const dayOrdinalSuffix = (n) => (n % 100 >= 11 && n % 100 <= 13 ? 'th' : { 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] || 'th');
const TIME_CUE_AHEAD = String.raw`(?!\s*(?:[ap]\s?\.?\s?m\b|o['’]?clock|baje|bje|(?:in the |at )?(?:morning|afternoon|evening|night|subah|shaam|raat|dopahar)\b))`;

function ordinalValue(phrase) {
  const w = phrase.trim().split(/[\s-]+/);
  let n = null;
  if (w.length === 2 && DAY_TENS[w[0]] && DAY_ORDINALS[w[1]] <= 9) n = DAY_TENS[w[0]] + DAY_ORDINALS[w[1]];
  else if (w[0] === 'twentieth') n = 20;
  else if (w[0] === 'thirtieth') n = 30;
  else if (w.length === 1) n = DAY_ORDINALS[w[0]] ?? null;
  return n && n >= 1 && n <= 31 ? n : null;
}
function cardinalValue(phrase) {
  const w = phrase.trim().split(/[\s-]+/);
  let n = null;
  if (DAY_TENS[w[0]]) n = DAY_TENS[w[0]] + (w[1] ? DAY_UNITS[w[1]] : 0);
  else if (w.length === 1) n = DAY_UNITS[w[0]] ?? null;
  return n && n >= 1 && n <= 31 ? n : null;
}
function foldDayWords(s) {
  const ord = (n) => `${n}${dayOrdinalSuffix(n)}`;
  return s
    .replace(new RegExp(`\\b(${ORDINAL_PHRASE})\\s+(?:of\\s+)?(?=(?:${MONTH_RE})\\b)`, 'g'), (m, p) => (ordinalValue(p) ? `${ord(ordinalValue(p))} ` : m))
    .replace(new RegExp(`\\b(${MONTH_RE})\\s+(?:the\\s+)?(${ORDINAL_PHRASE})\\b`, 'g'), (m, mon, p) => (ordinalValue(p) ? `${mon} ${ord(ordinalValue(p))}` : m))
    // "on/by/till the twelfth" with no month ("the second" alone is left alone: it is usually the unit of time)
    .replace(new RegExp(`\\b(on|by|till|until)\\s+(?:the\\s+)?(${ORDINAL_PHRASE})\\b`, 'g'), (m, w, p) => (p.trim() !== 'second' && ordinalValue(p) ? `${w} the ${ord(ordinalValue(p))}` : m))
    .replace(new RegExp(`\\b(${CARDINAL_PHRASE})\\s+(?:of\\s+)?(?=(?:${MONTH_RE})\\b)`, 'g'), (m, p) => (cardinalValue(p) ? `${cardinalValue(p)} ` : m))
    // "november nine p m" is a time, not the 9th: only fold when no clock word follows
    .replace(new RegExp(`\\b(${MONTH_RE})\\s+(${CARDINAL_PHRASE})\\b${TIME_CUE_AHEAD}`, 'g'), (m, mon, p) => (cardinalValue(p) ? `${mon} ${cardinalValue(p)}` : m));
}

/**
 * People type times loosely: "six p m", "6 p.m", "five o'clock", "at seven baje", "sadhe paanch baje",
 * "५ बजे", "5 to 6pm", "1730 hrs". Lower-cases and rewrites those into the digit forms the date reader
 * understands, folding Devanagari/Gujarati words and digits into the same Latin tokens first.
 */
export function normalizeTimeText(text) {
  let s = String(text ?? '')
    .toLowerCase()
    // native digits → 0-9
    .replace(/[०-९]/g, (d) => String(d.charCodeAt(0) - 0x0966))
    .replace(/[૦-૯]/g, (d) => String(d.charCodeAt(0) - 0x0ae6))
    .replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - 0x0660));
  for (const [re, to] of NATIVE_RULES) s = s.replace(re, to);
  for (const [re, to] of LATIN_WORDS) s = s.replace(re, to);
  s = s.replace(/\s{2,}/g, ' ');
  s = foldDayWords(s);
  const cue = String.raw`(?=\s*(?:o['’]?clock|[ap]\s?\.?\s?m\b|baje|bje|(?:in the |at )?(?:morning|afternoon|evening|night|subah|shaam|raat|dopahar)\b))`;
  s = s
    .replace(new RegExp(String.raw`\b(${NUMBER_WORD_RE})\b${cue}`, 'g'), (w) => String(NUMBER_WORDS[w]))
    // Hindi fractions: "sadhe 5 baje" = 5:30, "sava 5" = 5:15, "paune 6" = 5:45, "dedh baje" = 1:30, "dhai baje" = 2:30
    .replace(/\bsadhe\s+(\d{1,2})\b/g, '$1:30')
    .replace(/\bsava\s+(\d{1,2})\b/g, '$1:15')
    .replace(/\bpaune\s+(\d{1,2})\b/g, (_, n) => `${(+n + 11) % 12 || 12}:45`)
    .replace(/\bdedh(?=\s*baje)/g, '1:30')
    .replace(/\bdhai(?=\s*baje)/g, '2:30')
    // "9 p m", "9 p.m." -> 9pm. A full stop right after it ("9 p m.At the cafe") is a sentence break: keep a space so
    // the next word is not glued on ("9pmat").
    .replace(/(\d)\s*([ap])\s?\.?\s?m\b(\.?)/g, (_, d, ap, dot) => `${d}${ap}m${dot ? ' ' : ''}`)
    // military time: "1730 hrs" -> 17:30
    .replace(/\b([01]\d|2[0-3])([0-5]\d)\s*(?:hrs|hours|hr)\b/g, '$1:$2')
    // "5 to 6pm" / "5-6 pm" → the start time; the suffix comes from the end unless the start would pass it ("11-1pm")
    .replace(/\b(\d{1,2})(?::(\d{2}))?\s*(?:to|-|–|—|se|till|until)\s*(\d{1,2})(?::(\d{2}))?([ap]m)\b/g, (_, h1, m1, h2, m2, ap) => {
      const suffix = +h1 > +h2 && ap === 'pm' && +h1 !== 12 ? 'am' : ap;
      return `${h1}${m1 ? `:${m1}` : ''}${suffix} to ${h2}${m2 ? `:${m2}` : ''}${ap}`;
    })
    // "6 in the morning" / "chhe subah" -> 6am; "5 in the evening" -> 5pm (the hour needs a number in front)
    .replace(/\b(\d{1,2})(:\d{2})?\s*(?:in the |at )?(?:morning|subah)\b/g, '$1$2am')
    .replace(/\b(\d{1,2})(:\d{2})?\s*(?:in the |at )?(?:afternoon|evening|night|shaam|raat|dopahar)\b/g, '$1$2pm');
  return s;
}

// "at the railway station", "near zudio office, vadodara": a place named in the message, for when the model leaves
// `location` empty. Only the first place phrase; a time ("at 6") never counts.
// The place ends where the purpose starts: "at beyond 7 caffe for bank financial statements" → "beyond 7 caffe".
const PLACE_END = String.raw`(?=\s+(?:at|by|around|tomorrow|today|tonight|on|for|about|regarding|because|to discuss|so that|jahan|ke liye|ke baare|ke bare|taaki)\b|[,.;\n!?]|$)`;
const PLACE = new RegExp(String.raw`\b(?:at|near|outside|opposite|in front of|behind|inside)\s+(?:the\s+)?([a-z][^,.;\n!?]{2,50}?)${PLACE_END}`, 'i');
// Typed too fast: "on 12 november 9 p m add sayaji bagh" means "at". Only when "add" directly follows a date or time.
const WHEN_BEFORE_ADD = String.raw`(?:\d{1,2}\s*(?::\d{2})?\s*[ap]m|\d{1,2}:\d{2}|\d{1,2}\s*baje|today|tonight|tomorrow|kal|aaj|parso|sunday|monday|tuesday|wednesday|thursday|friday|saturday|(?:${MONTH_RE})(?:\s+\d{1,2}(?:st|nd|rd|th)?)?|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:${MONTH_RE}))`;
const PLACE_ADD = new RegExp(String.raw`\b${WHEN_BEFORE_ADD}\s+add\s+(?:the\s+)?([a-z][^,.;\n!?]{2,50}?)${PLACE_END}`, 'i');
// Hinglish puts the place first: "railway station pe milte hain", "office ke bahar", "ghar par". Up to 3 words
// before the postposition, minus time words and pronouns.
const POSTPOSITION = /\b((?:[a-z][a-z'-]*\s+){0,3}[a-z][a-z'-]*)\s+(?:pe|par|mein|me|ke paas|ke bahar|ke samne|ke pass|ke andar)\b/;
const NOT_PLACE = new Set(['kal', 'aaj', 'parso', 'subah', 'shaam', 'raat', 'dopahar', 'baje', 'am', 'pm', 'tomorrow', 'today', 'tonight', 'at', 'on', 'ko', 'se', 'tak', 'tum', 'aap', 'main', 'mai', 'hum', 'kya', 'abhi', 'ye', 'yeh', 'wo', 'vo', 'is', 'us', 'mera', 'meri', 'mere', 'tumhare', 'aapke', 'ki', 'ka', 'ke', 'hai', 'hain', 'ho', 'to', 'toh', 'bhi', 'next', 'week', 'weekend', 'milte', 'milna', 'milenge', 'baat', 'call', 'meeting', 'meet', 'and', 'the', 'ek', 'do', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
  // English function words: "Tell him hi for me" has the Hinglish-looking "... me" but no place in it
  'him', 'her', 'his', 'hi', 'hello', 'hey', 'for', 'tell', 'say', 'ask', 'with', 'you', 'your', 'my', 'our', 'we', 'it', 'this', 'that', 'please', 'pls', 'ok', 'okay', 'yes', 'no', 'let', 'lets', 'give', 'me',
  // asking for a booking is not a place: "fix a date for me", "book a slot for us"
  'fix', 'a', 'an', 'date', 'time', 'slot', 'book', 'set', 'up', 'arrange', 'appointment', 'schedule', 'plan', 'us', 'them', 'him', 'can', 'could', 'would', 'will', 'pls']);
export function guessLocation(text) {
  const norm = normalizeTimeText(text);
  const m = PLACE_ADD.exec(norm) || PLACE.exec(norm);
  if (m) {
    const v = m[1].trim().replace(/\s+/g, ' ').replace(/\s+(?:par|pe|pr|mein|me)$/, ''); // "sayaji pawar par" → "sayaji pawar"
    return v.length < 3 ? null : v.charAt(0).toUpperCase() + v.slice(1);
  }
  const h = POSTPOSITION.exec(norm.replace(/[,.;!?]/g, ' '));
  if (!h) return null;
  const words = h[1].split(/\s+/).filter((w) => !/^\d/.test(w));
  while (words.length && NOT_PLACE.has(words[0])) words.shift();
  while (words.length && NOT_PLACE.has(words.at(-1))) words.pop();
  const v = words.slice(-3).join(' ');
  if (v.length < 3 || words.some((w) => NOT_PLACE.has(w) && w !== 'ke')) return null;
  return v.charAt(0).toUpperCase() + v.slice(1);
}

/**
 * A place named in EARLIER messages, newest first: only messages that are about a meeting or carry a date/time count, so a
 * chat line like "tell him hi for me" can never lend its words as the place.
 */
export function guessLocationFromContext(texts, { now = new Date(), timeZone = 'Asia/Kolkata' } = {}) {
  for (const t of [...texts].reverse()) {
    if (!mentionsMeeting(t) && !parseWhen(t, { now, timeZone })) continue;
    const p = guessLocation(t);
    if (p) return p;
  }
  return null;
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
  // "on the 12th", "by 20th": a day of the month with no month name
  if (!day && (m = /\b(?:on|by|till|until|the)\s+(\d{1,2})(?:st|nd|rd|th)\b/.exec(t)) && +m[1] >= 1 && +m[1] <= 31) {
    const dom = +m[1];
    // This month if the day is still ahead, else next month; skip months that do not have that day ("the 31st" in November).
    let ym = { y: today.y, m: today.m };
    if (dom < today.d) ym = today.m === 12 ? { y: today.y + 1, m: 1 } : { y: today.y, m: today.m + 1 };
    for (let tries = 0; tries < 3 && dom > new Date(Date.UTC(ym.y, ym.m, 0)).getUTCDate(); tries++) ym = ym.m === 12 ? { y: ym.y + 1, m: 1 } : { y: ym.y, m: ym.m + 1 };
    day = { y: ym.y, m: ym.m, d: dom };
    take(m);
  }
  // "this weekend" → Saturday; "next week" / "agle hafte" → Monday of next week (both all-day unless a time follows)
  if (!day && (m = /\bweekend\b/.exec(t))) {
    day = addDays(today.y, today.m, today.d, (6 - today.wd + 7) % 7);
    take(m);
  }
  if (!day && (m = /\bnext week\b/.exec(t))) {
    day = addDays(today.y, today.m, today.d, (1 - today.wd + 7) % 7 || 7);
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
  if (hour == null && (m = /\b([01]?\d|2[0-3]):([0-5]\d)\b(\s*baje)?/.exec(t))) {
    hour = +m[1];
    minute = +m[2];
    // "sadhe 5 baje" (5:30) is an everyday clock reading, not a 24-hour time: apply the same am/pm guess as "5 baje"
    if (m[3] && hour >= 1 && hour <= 11) hour = hour <= 6 && !MORNING.test(t) ? hour + 12 : hour < 12 && EVENING.test(t) && !MORNING.test(t) ? hour + 12 : hour;
    take(m);
  }
  if (hour == null && (m = /\b(?:at|by|around)\s+(\d{1,2})(?:[:.]([0-5]\d))?\b(?!\s*(?:st|nd|rd|th|\/|-))|\b(\d{1,2})(?:[:.]([0-5]\d))?\s*(?:baje|bje|o['’]?clock)\b/.exec(t))) {
    const h = +(m[1] ?? m[3]);
    minute = +(m[2] ?? m[4] ?? 0);
    if (h >= 1 && h <= 23) {
      // 1–6 mean pm ("5 baje") unless the message says morning ("subah 5 baje")
      hour = (h <= 6 && !MORNING.test(t)) || (h < 12 && EVENING.test(t) && !MORNING.test(t)) ? h + 12 : h;
      take(m);
    }
  }
  // A bare hour right after a day word: "tomorrow 3", "kal 5", "monday 11", "tonight 8". Same am/pm guess as "5 baje".
  // Not a date ("tomorrow 12/11", "monday 3rd") and not a count ("tomorrow 3 people").
  if (hour == null && (m = /\b(?:today|tonight|tomorrow|tmrw|tomm?orow|tommorrow|kal|aaj|parso|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:at\s+)?(\d{1,2})(?:[:.]([0-5]\d))?\b(?!\s*(?:st|nd|rd|th|\/|-|%|\d|people|persons?|guys|friends|items?|days?|hours?|hrs?|mins?|minutes?|km|rs|rupees|times|more))/.exec(t)) && !new RegExp(`^\\s*(?:of\\s+)?(?:${MONTH_RE})\\b`).test(t.slice(m.index + m[0].length))) {
    const h = +m[1];
    if (h >= 1 && h <= 23) {
      minute = m[2] ? +m[2] : 0;
      hour = (h <= 6 && !MORNING.test(t)) || (h < 12 && EVENING.test(t) && !MORNING.test(t)) ? h + 12 : h;
      take(m);
    }
  }
  if (hour == null && !day) {
    // "in 2 hours", "after 30 min", "2 ghante baad", "3 din mein": relative to now
    const rel = /\b(?:in|after|within)\s+(\d{1,3}|an?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?)\b|\b(\d{1,3})\s*(minutes?|mins?|ghante|ghanta|hours?|hrs?|din|dino|hafte|weeks?)\s*(?:baad|bad|mein|me)\b/.exec(t);
    if (rel) {
      const raw = rel[1] ?? rel[3];
      const n = /^\d+$/.test(raw) ? +raw : raw === 'a' || raw === 'an' ? 1 : NUMBER_WORDS[raw];
      const unit = rel[2] ?? rel[4];
      if (n >= 1) {
        if (/^(day|din|week|hafte)/.test(unit)) {
          const d = addDays(today.y, today.m, today.d, n * (/^(week|hafte)/.test(unit) ? 7 : 1));
          return { dueAt: zonedToUtc(d.y, d.m, d.d, 9, 0, timeZone), allDay: true, hasDay: true, matched: rel[0].trim() };
        }
        const ms = n * (/^(hour|hr|ghant)/.test(unit) ? 3600_000 : 60_000);
        return { dueAt: new Date(Math.ceil((now.getTime() + ms) / 60_000) * 60_000), allDay: false, hasDay: true, matched: rel[0].trim() };
      }
    }
  }
  if (hour == null) {
    const part = /\b(noon|midday|midnight|morning|subah|subha|afternoon|dopahar|evening|shaam|tonight|night|raat|breakfast|brunch|lunch|dinner)\b/.exec(t);
    if (part) {
      const w = part[1];
      hour = { noon: 12, midday: 12, midnight: 0, morning: 9, subah: 9, subha: 9, afternoon: 15, dopahar: 15, evening: 18, shaam: 18, tonight: 20, night: 20, raat: 20, breakfast: 9, brunch: 11, lunch: 13, dinner: 20 }[w];
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

// All gates run on normalizeTimeText(): lower-cased, with Hindi/Marathi/Gujarati words and digits folded to Latin tokens.
const MEET =
  /\b(meet|meeting|meetup|catch ?up|call|calling|lunch|dinner|breakfast|brunch|coffee|visit|appointment|schedule[d]?|reschedule|interview|demo|discussion|sync|session|hangout|come over|coming|get together|milte|milenge|milna|milein|milunga|mil sakte|mil lete|mulakat|bulana|aana|aaunga|aaungi|aaoge|baat karni|baat karte|call karna|call karo|phone karna|phone karo|malishu|malvu|bhetu|ghar aana|fix (?:a |the |me a )?(?:date|time|slot)|date fix|time fix|book (?:a |the |me a )?(?:slot|time|date|meeting|appointment)|set up (?:a )?(?:meeting|call|appointment))\b/i;
const TODO =
  /\b(remind(?:er)?|remember to|don'?t forget|dont forget|todo|to-do|deadline|due (?:on|by|date)|submit|send me|share (?:the|your|it)|follow ?up|call ?back|yaad dil\w*|yaad rakh\w*)\b/i;
const STRONG_TIME =
  /\b(today|tonight|tomorrow|tmrw|tomm?orow|tommorrow|day after|kal|aaj|parso|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|weekend|subah|shaam|dopahar|\d{1,2}\s*(?::\d{2})?\s*(?:am|pm)|\d{1,2}:\d{2}|\d{1,2}\s*baje|(?:in|after)\s+(?:\d+|an?|one|two|three|four|five|ten)\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?)|\d+\s*(?:ghante|ghanta|din|minutes?|mins?)\s*(?:baad|mein|me)|(?:on|by|the)\s+\d{1,2}(?:st|nd|rd|th))\b/i;
const ASK =
  /\b(free|available|availability|let'?s|lets|shall we|can we|can you|could you|will you|are you (?:coming|free)|come|aana|aa jao|aa sakte|aa sakta|aa sakti|aa paoge|milte|chalo|chalein|chaloge|chalte hain|free ho|available ho|kab|karte hain|rakhte hain|rakhein|fix karte|fix karein)\b/i;

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
  if (TEXT_TASK.test(String(text ?? ''))) return false;
  const s = normalizeTimeText(text);
  return (MEET.test(s) && (STRONG_TIME.test(s) || ASK.test(s))) || (STRONG_TIME.test(s) && (ASK.test(s) || DIRECT_ASK.test(s))) || (MEET.test(s) && DIRECT_ASK.test(s));
}

// "remind him", "tell Sumit …", "let him know", "pass it on", "unhe bol dena", "usse bolo": the sender wants the
// owner to hear something. The twin cannot act on it, but it can say it was noted and will be passed on.
const PASS_VERBS = 'remind|tell|inform|notify|message|text|ping|ask|convey';
const HINGLISH_PASS = /\b(?:unhe|unko|usse|usko|unse)\s+(?:bol|bolo|bolna|bata|batao|batana|keh|kaho|kehna|kahna|message|msg|sandesh|yaad)\b|\b(?:bol|bata|keh)\s+(?:dena|do|dijiye|dijiyega)\s+(?:ki|unhe|usse)\b|\bmessage\s+(?:de\s+dena|dena|pahucha\w*)\b/;
const PASS_IT_ON = /\bpass\s+(?:it|this|that|the message|my message|on)\s*(?:on|along|over|to)?\b|\bforward\s+(?:it|this|that)\s+to\b|\bgive\s+(?:him|her|them)\s+(?:a\s+|my\s+)?(?:message|msg|note)\b|\bconvey\b.*\b(?:to|him|her)\b|\blet\s+(?:him|her|them)\s+know\b/;
const passOnRes = new Map();
function passOnRe(names) {
  const key = names.join('|');
  let re = passOnRes.get(key);
  if (!re) {
    const who = ['him', 'her', 'them', 'the owner', ...names.filter((n) => /^[\p{L}][\p{L} .'-]{1,30}$/u.test(n)).map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))].join('|');
    re = new RegExp(`\\b(${PASS_VERBS})\\s+(?:${who})\\b|\\b(?:${who})\\s+ko\\s+(?:bol|bata|keh|message|msg)`, 'g');
    passOnRes.set(key, re);
  }
  return re;
}

/**
 * Does the sender ask us to tell/remind/inform the owner of something? Pure.
 * @param {string} text
 * @param {{ ownerNames?: string[] }} [opts] the owner's name and aliases (lower-case ok)
 */
export function isPassOnRequest(text, { ownerNames = [] } = {}) {
  if (TEXT_TASK.test(String(text ?? ''))) return false;
  const s = normalizeTimeText(text);
  if (s.length < 4 || s.length > 600) return false;
  if (HINGLISH_PASS.test(s) || PASS_IT_ON.test(s)) return true;
  const re = passOnRe(ownerNames.map((n) => String(n).toLowerCase().trim()).filter(Boolean));
  re.lastIndex = 0;
  let m;
  while ((m = re.exec(s))) {
    // "I'll tell him", "i can ask him": the sender is the one who will do it
    const before = s.slice(Math.max(0, m.index - 14), m.index);
    if (/\bi(?:['’]ll|\s*will|\s*shall|\s*can|\s*would|\s*might|\s*am going to|['’]m going to)?\s*$/.test(before)) continue;
    return true;
  }
  return false;
}

/** The text talks about meeting/calling/eating together (no time needed). */
export function mentionsMeeting(text) {
  return !TEXT_TASK.test(String(text ?? '')) && MEET.test(normalizeTimeText(text));
}

// "I want to meet you", "mujhe aapse milna hai", "milna hai", "meet karna hai", "मुझे मिलना है": the sender wants a
// meeting but names no time. The twin answers with the "which day, time and place?" template instead of the model
// (a 3B model replied "Mujhe bhi tumse milna hai. Kaisi jagah ho rahi thi?"). "Nice to meet you" does not count.
const WANTS_TO_MEET = [
  /\b(?:i|we|id|i'd)\s*(?:would\s+|really\s+|just\s+|urgently\s+|do\s+)*(?:want|wanna|need|have|like|wish|hope)\s+(?:to\s+)?(?:meet|catch\s*up|see\s+(?:you|him|her|them)|talk\s+(?:to|with)\s+(?:you|him|her|them)|speak\s+(?:to|with)\s+(?:you|him|her|them))\b/,
  /\b(?:wanna|want\s+to|need\s+to)\s+meet\b/,
  /\bmil(?:na|ne|unga|ungi)\s+(?:hai|tha|thi|chahta|chahti|chahte|ka\s+man|ka\s+mann|ki\s+iccha)\b/,
  /\b(?:mujhe|mereko|mujhko|hume|humein)\b[^.?!]{0,30}\bmil(?:na|ne|unga|ungi)\b/,
  /\b(?:meet|mulakat|baat|call)\s+(?:karna|karni|karne)\s+(?:hai|tha|thi|chahta|chahti)\b/,
  // "fix a date for me", "book a slot", "set up a call": asking the twin to pin down a time is asking for a meeting
  /\b(?:fix|book|arrange|set\s+up)\s+(?:a\s+|the\s+|me\s+a\s+)?(?:date|time|slot|meeting|appointment|call)\b/,
];
export function wantsToMeet(text) {
  if (TEXT_TASK.test(String(text ?? ''))) return false;
  const s = normalizeTimeText(text);
  if (/\bnice\s+to\s+meet\b|\bglad\s+to\s+meet\b|\bpleased\s+to\s+meet\b/.test(s)) return false;
  return WANTS_TO_MEET.some((re) => re.test(s));
}

/** What sort of get-together: a call, a meal or a plain meeting. Used for titles and the invitation card. */
export function meetingKind(text) {
  const s = normalizeTimeText(text);
  if (/\b(call|calling|call karna|call karo|phone karna|phone karo|baat karni|baat karte)\b/.test(s)) return 'call';
  if (/\b(lunch|dinner|breakfast|brunch|coffee)\b/.test(s)) return 'meal';
  return 'meet';
}

const KIND_TITLE = { call: 'Call with', meal: 'Meal with', meet: 'Meet' };

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
  const norm = normalizeTimeText(text);
  const meet = MEET.test(norm) || MEET.test(normalizeTimeText(contextText)); // "I want to meet" may be an earlier message
  if (!hit && !TODO.test(norm)) return null;
  if (meet && hit) {
    const kind = meetingKind(`${contextText}\n${text}`);
    return { kind: 'appointment', title: `${KIND_TITLE[kind]} ${firstName(contactName || 'contact')}`, details: clip(text.trim(), 300), when: hit.matched, dueAt: hit.dueAt, allDay: hit.allDay, dueText: hit.matched, location: guessLocation(text) };
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
