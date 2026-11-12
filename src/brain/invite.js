// Meeting replies and the invitation card. Pure and dependency-free (unit-tested).
//
// A 3B model cannot be trusted with "never agree to a meeting for the owner" in Hindi, Gujarati or
// Hinglish, so replies to meeting requests are templates in the contact's language: instant, always
// correct, and they never commit the owner. The invitation card is plain HTML/CSS that the gateway
// renders to a PNG with the headless Chromium it already runs for WhatsApp.
import { localParts } from './agenda.js';

// ------------------------------------------------------------------ languages

/** Template family for a detected language code (see brain/lang.js). */
export function inviteFamily(code) {
  const c = String(code || '');
  if (c === 'hi-Latn') return 'hl';
  if (c === 'hi' || c === 'mr' || c === 'gu') return c;
  return 'en';
}

const LOCALE = { en: 'en-IN', hl: 'en-IN', hi: 'hi-IN', mr: 'mr-IN', gu: 'gu-IN' };

const KIND_LABEL = {
  en: { call: 'Call request', meal: 'Meal plan', meet: 'Meeting request' },
  hl: { call: 'Call ki request', meal: 'Khaane ka plan', meet: 'Meeting ki request' },
  hi: { call: 'कॉल का अनुरोध', meal: 'भोजन की योजना', meet: 'मीटिंग का अनुरोध' },
  mr: { call: 'कॉलची विनंती', meal: 'जेवणाचा बेत', meet: 'मीटिंगची विनंती' },
  gu: { call: 'કોલની વિનંતી', meal: 'જમવાનો પ્લાન', meet: 'મીટિંગની વિનંતી' },
};

const TEXT = {
  en: {
    ack: "Got it {name}! 📅 {what}: {when}{place}\nI can't confirm anything for {owner} myself, so I've noted it and will pass it to the real {owner}. {owner} will confirm soon ✅",
    passOn: "Noted {name}! I can't act on this myself, so I'll pass it on to the real {owner}. {owner} will see it soon ✅",
    ask: "Noted {name}! Which day, time and place did you have in mind? I'll pass it to the real {owner} to confirm.",
    askTime: "Noted {name}! {what}: {when}{place}\nWhat time works for you? I'll pass it to the real {owner} to confirm.",
    busy: "Sorry {name}, {owner} isn't available then ({when}).\nFree slots: {slots}\nReply with one that suits you, or suggest another time.",
    busyNone: "Sorry {name}, {owner} isn't available then ({when}). Please suggest another time.",
    confirmHead: '✅ Confirmed, {name}! {owner} has accepted: {title}',
    updatedHead: '🔁 Updated, {name}! {owner} has confirmed the new time: {title}',
    confirmTail: 'See you then 🙂',
    decline: "Hey {name}, {owner} can't make this one ({title}, {when}). Feel free to suggest another time 🙏",
  },
  hl: {
    ack: 'Theek hai {name}! 📅 {what}: {when}{place}\nIse note kar liya hai aur asli {owner} tak pahuncha diya jayega. {owner} khud confirm karenge ✅',
    passOn: 'Note kar liya {name}! Yeh baat asli {owner} tak pahuncha di jayegi. {owner} jald dekh lenge ✅',
    ask: 'Note kar liya {name}! Kis din, kitne baje aur kahan milna hai? Asli {owner} tak pahuncha diya jayega, {owner} confirm karenge.',
    askTime: 'Note kar liya {name}! {what}: {when}{place}\nKitne baje ka time theek rahega? Asli {owner} tak pahuncha diya jayega, {owner} confirm karenge.',
    busy: 'Sorry {name}, {owner} us time free nahi hain ({when}).\nFree slots: {slots}\nIn mein se koi chun lo, ya koi aur time batao.',
    busyNone: 'Sorry {name}, {owner} us time free nahi hain ({when}). Koi aur time batao.',
    confirmHead: '✅ Confirm ho gaya, {name}! {owner} ne haan kar di hai: {title}',
    updatedHead: '🔁 Update, {name}! {owner} ne naya time confirm kiya hai: {title}',
    confirmTail: 'Milte hain 🙂',
    decline: 'Hey {name}, {owner} is baar nahi aa paayenge ({title}, {when}). Koi aur time batao 🙏',
  },
  hi: {
    ack: 'ठीक है {name}! 📅 {what}: {when}{place}\nइसे नोट कर लिया है और असली {owner} तक पहुँचा दिया जाएगा। {owner} खुद पुष्टि करेंगे ✅',
    passOn: 'नोट कर लिया {name}! यह बात असली {owner} तक पहुँचा दी जाएगी। {owner} जल्द देख लेंगे ✅',
    ask: 'नोट कर लिया {name}! किस दिन, कितने बजे और कहाँ मिलना है? असली {owner} तक पहुँचा दिया जाएगा, {owner} पुष्टि करेंगे।',
    askTime: 'नोट कर लिया {name}! {what}: {when}{place}\nकितने बजे का समय ठीक रहेगा? असली {owner} तक पहुँचा दिया जाएगा, {owner} पुष्टि करेंगे।',
    busy: 'क्षमा करें {name}, {owner} उस समय उपलब्ध नहीं हैं ({when})।\nखाली समय: {slots}\nइनमें से कोई चुनिए या दूसरा समय बताइए।',
    busyNone: 'क्षमा करें {name}, {owner} उस समय उपलब्ध नहीं हैं ({when})। कृपया दूसरा समय बताइए।',
    confirmHead: '✅ पक्का, {name}! {owner} ने मंज़ूरी दे दी है: {title}',
    updatedHead: '🔁 बदलाव, {name}! {owner} ने नया समय पक्का किया है: {title}',
    confirmTail: 'तब मिलते हैं 🙂',
    decline: '{name}, {owner} इस बार नहीं आ पाएँगे ({title}, {when})। कोई और समय बताइए 🙏',
  },
  mr: {
    ack: 'ठीक आहे {name}! 📅 {what}: {when}{place}\nहे नोंदवून घेतले आहे आणि खऱ्या {owner} पर्यंत पोहोचवले जाईल. {owner} स्वतः कळवतील ✅',
    passOn: 'नोंदवून घेतले {name}! हे खऱ्या {owner} पर्यंत पोहोचवले जाईल. {owner} लवकरच पाहतील ✅',
    ask: 'नोंदवून घेतले {name}! कोणत्या दिवशी, किती वाजता आणि कुठे भेटायचे? खऱ्या {owner} पर्यंत पोहोचवले जाईल, {owner} कळवतील.',
    askTime: 'नोंदवून घेतले {name}! {what}: {when}{place}\nकिती वाजता सोयीचे आहे? खऱ्या {owner} पर्यंत पोहोचवले जाईल, {owner} कळवतील.',
    busy: 'क्षमस्व {name}, {owner} त्या वेळी उपलब्ध नाहीत ({when}).\nमोकळ्या वेळा: {slots}\nयापैकी एक निवडा किंवा दुसरी वेळ सांगा.',
    busyNone: 'क्षमस्व {name}, {owner} त्या वेळी उपलब्ध नाहीत ({when}). कृपया दुसरी वेळ सांगा.',
    confirmHead: '✅ निश्चित, {name}! {owner} यांनी मान्य केले आहे: {title}',
    updatedHead: '🔁 बदल, {name}! {owner} यांनी नवीन वेळ निश्चित केली आहे: {title}',
    confirmTail: 'तेव्हा भेटू 🙂',
    decline: '{name}, {owner} यावेळी येऊ शकणार नाहीत ({title}, {when}). दुसरी वेळ सुचवा 🙏',
  },
  gu: {
    ack: 'ઠીક છે {name}! 📅 {what}: {when}{place}\nઆ નોંધી લીધું છે અને સાચા {owner} સુધી પહોંચાડવામાં આવશે. {owner} પોતે જણાવશે ✅',
    passOn: 'નોંધી લીધું {name}! આ વાત સાચા {owner} સુધી પહોંચાડવામાં આવશે. {owner} જલ્દી જોશે ✅',
    ask: 'નોંધી લીધું {name}! કયા દિવસે, કેટલા વાગ્યે અને ક્યાં મળવું છે? સાચા {owner} સુધી પહોંચાડીશું, {owner} જણાવશે.',
    askTime: 'નોંધી લીધું {name}! {what}: {when}{place}\nકેટલા વાગ્યે અનુકૂળ રહેશે? સાચા {owner} સુધી પહોંચાડીશું, {owner} જણાવશે.',
    busy: 'માફ કરજો {name}, {owner} તે સમયે ઉપલબ્ધ નથી ({when}).\nખાલી સમય: {slots}\nઆમાંથી પસંદ કરો અથવા બીજો સમય જણાવો.',
    busyNone: 'માફ કરજો {name}, {owner} તે સમયે ઉપલબ્ધ નથી ({when}). કૃપા કરીને બીજો સમય જણાવો.',
    confirmHead: '✅ પાક્કું, {name}! {owner} એ મંજૂરી આપી છે: {title}',
    updatedHead: '🔁 ફેરફાર, {name}! {owner} એ નવો સમય પાક્કો કર્યો છે: {title}',
    confirmTail: 'ત્યારે મળીએ 🙂',
    decline: '{name}, {owner} આ વખતે આવી શકશે નહીં ({title}, {when}). બીજો સમય જણાવો 🙏',
  },
};

const fill = (tpl, vars) => tpl.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? '');
const first = (s, fallback) => String(s || '').trim().split(/\s+/)[0] || fallback;

/** "Tue, 12 Nov, 6:00 am" in the language's own locale. */
export function whenLabel(date, timeZone, allDay, code = 'en') {
  if (!date) return '';
  const d = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(d.getTime())) return '';
  const opts = { timeZone, weekday: 'short', day: 'numeric', month: 'short' };
  if (!allDay) Object.assign(opts, { hour: 'numeric', minute: '2-digit', hour12: true });
  return new Intl.DateTimeFormat(LOCALE[inviteFamily(code)] || 'en-IN', opts).format(d);
}

function vars({ code, name, owner, kind = 'meet', dueAt, allDay, place, title, timeZone }) {
  const fam = inviteFamily(code);
  return {
    fam,
    v: {
      name: first(name, { en: 'there', hl: 'dost', hi: 'दोस्त', mr: 'मित्रा', gu: 'મિત્ર' }[fam]),
      owner: first(owner, 'the owner'),
      what: KIND_LABEL[fam][kind] || KIND_LABEL[fam].meet,
      when: whenLabel(dueAt, timeZone, allDay, code),
      place: place ? ` · 📍 ${place}` : '',
      title: title || '',
    },
  };
}

/**
 * The reply to a meeting request: never agrees, says it was noted and passed on.
 * With no readable day/time it asks for the details instead.
 */
export function meetingAck(p) {
  const { fam, v } = vars(p);
  if (p.dueAt && !p.allDay) return fill(TEXT[fam].ack, v);
  // a date but no time: say what was understood and ask only for the time (also keeps a repeated question from being word-for-word identical)
  if (p.dueAt && p.allDay) return fill(TEXT[fam].askTime, v);
  return fill(TEXT[fam].ask, v);
}

/**
 * The asked time is not free (a confirmed meeting, a block, outside working hours, or already past): say so without saying
 * why, and offer free slots. `slots` is the text from slots.js slotsText() ('' when there are none).
 */
export function slotTakenAck(p) {
  const { fam, v } = vars({ ...p, place: '' });
  return fill(TEXT[fam][p.slots ? 'busy' : 'busyNone'], { ...v, slots: p.slots || '' });
}

/** Reply to "tell him / remind him / pass it on": noted and passed on, never agreeing or committing for the owner. */
export function passOnAck(p) {
  const { fam, v } = vars(p);
  return fill(TEXT[fam].passOn, v);
}

/** Message that goes with the invitation card (or alone) when the owner accepts. */
export function confirmationText(p) {
  const { fam, v } = vars(p);
  // `rescheduled`: the owner already confirmed this and the time/place changed since: say "updated", not "confirmed" again.
  const lines = [fill(TEXT[fam][p.rescheduled ? 'updatedHead' : 'confirmHead'], v)];
  if (v.when) lines.push(`🗓 ${v.when}`);
  if (p.place) lines.push(`📍 ${p.place}`);
  lines.push(TEXT[fam].confirmTail);
  return lines.join('\n');
}

const SLOTS_LABEL = { en: 'Free slots:', hl: 'Free slots:', hi: 'खाली समय:', mr: 'मोकळ्या वेळा:', gu: 'ખાલી સમય:' };

/** `slots` (from slots.js slotsText) adds the owner's free times so the person can pick another one. */
export function declineText(p) {
  const { fam, v } = vars({ ...p, dueAt: p.dueAt });
  const text = fill(TEXT[fam].decline, { ...v, when: v.when || { en: 'the time we discussed', hl: 'jo time baat hua tha', hi: 'जो समय तय हुआ था', mr: 'ठरलेली वेळ', gu: 'નક્કી થયેલો સમય' }[fam] });
  return p.slots ? `${text}\n${SLOTS_LABEL[fam]} ${p.slots}` : text;
}

// ------------------------------------------------------------------- the card

const CARD_LABELS = {
  en: { invitation: 'INVITATION', confirmed: 'CONFIRMED', pending: 'REQUESTED', when: 'When', where: 'Where', who: 'With', host: 'Hosted by', allDay: 'All day', footer: "Sent by {persona}, {owner}'s AI twin" },
  hl: { invitation: 'INVITATION', confirmed: 'CONFIRMED', pending: 'REQUESTED', when: 'Kab', where: 'Kahan', who: 'Kiske saath', host: 'Host', allDay: 'Poora din', footer: "{persona} ne bheja, {owner} ka AI twin" },
  hi: { invitation: 'निमंत्रण', confirmed: 'पक्का', pending: 'अनुरोध', when: 'कब', where: 'कहाँ', who: 'किसके साथ', host: 'मेज़बान', allDay: 'पूरे दिन', footer: '{persona} द्वारा भेजा गया, {owner} का AI ट्विन' },
  mr: { invitation: 'आमंत्रण', confirmed: 'निश्चित', pending: 'विनंती', when: 'कधी', where: 'कुठे', who: 'कोणासोबत', host: 'यजमान', allDay: 'दिवसभर', footer: '{persona} कडून, {owner} चा AI ट्विन' },
  gu: { invitation: 'આમંત્રણ', confirmed: 'પાક્કું', pending: 'વિનંતી', when: 'ક્યારે', where: 'ક્યાં', who: 'કોની સાથે', host: 'યજમાન', allDay: 'આખો દિવસ', footer: '{persona} તરફથી, {owner} નો AI ટ્વિન' },
};

// A to-do/request card reads "TASK" and "Handled by" instead of "INVITATION" and "Hosted by".
const TASK_LABELS = {
  en: { task: 'TASK', handler: 'Handled by' },
  hl: { task: 'TASK', handler: 'Kaun karega' },
  hi: { task: 'कार्य', handler: 'ज़िम्मेदार' },
  mr: { task: 'कार्य', handler: 'जबाबदार' },
  gu: { task: 'કાર્ય', handler: 'જવાબદાર' },
};

const ACCENTS = ['#2f6fed', '#0f9d8a', '#d9622b', '#7a4fe0', '#c0397a', '#1f8a4c', '#b7791f', '#2b7a9b'];

const hashOf = (s) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
};
const rng = (seed) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

export const escapeHtml = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

/**
 * Everything the card shows, from an agenda row. Every card is different: the content, an accent
 * colour and the decorative circles are derived from the item, so no two invitations look alike.
 */
export function invitationData(item, settings, { code = 'en', now = new Date() } = {}) {
  const tz = settings.timezone || 'Asia/Kolkata';
  const fam = inviteFamily(code);
  const L = CARD_LABELS[fam];
  const due = item.dueAt ? new Date(item.dueAt) : null;
  const loc = LOCALE[fam];
  const parts = due && !Number.isNaN(due.getTime()) ? due : null;
  const fmt = (o) => (parts ? new Intl.DateTimeFormat(loc, { timeZone: tz, ...o }).format(parts) : '');
  const ref = `INV-${String(item.id ?? 0).padStart(4, '0')}`;
  const seed = hashOf(`${ref}|${item.title}|${item.dueAt}`);
  const confirmed = item.status === 'confirmed';
  const issued = new Intl.DateTimeFormat('en-IN', { timeZone: tz, day: 'numeric', month: 'short', year: 'numeric' }).format(now);
  const owner = String(settings.ownerName || 'the owner').trim();
  return {
    fam,
    labels: L,
    kicker: item.kind === 'todo' ? TASK_LABELS[fam].task : L.invitation,
    hostLabel: item.kind === 'todo' ? TASK_LABELS[fam].handler : L.host,
    ref,
    seed,
    accent: ACCENTS[seed % ACCENTS.length],
    status: confirmed ? L.confirmed : L.pending,
    confirmed,
    title: item.title || 'Meeting',
    weekday: fmt({ weekday: 'long' }),
    day: parts ? String(localParts(parts, tz).d) : '',
    monthYear: fmt({ month: 'long', year: 'numeric' }),
    time: parts ? (item.allDay ? L.allDay : fmt({ hour: 'numeric', minute: '2-digit', hour12: true })) : '',
    place: item.location || '',
    contact: item.contactName || item.phone || '',
    owner,
    persona: settings.personaName || 'eSumit',
    issued,
    footer: fill(L.footer, { persona: settings.personaName || 'eSumit', owner: first(owner, 'the owner') }),
  };
}

const LOGO = (c) => `<svg viewBox="0 0 24 24" width="64" height="64"><circle cx="8.6" cy="12" r="6" fill="none" stroke="${c}" stroke-width="1.8"/><circle cx="15.4" cy="12" r="6" fill="none" stroke="${c}" stroke-width="1.8"/><path d="M12 7.1A6 6 0 0 1 12 16.9A6 6 0 0 1 12 7.1z" fill="${c}"/></svg>`;

/** The full HTML document for a 1080×1350 card. No external resources: it renders offline. */
export function invitationHtml(d) {
  const rand = rng(d.seed);
  const circles = Array.from({ length: 5 }, () => {
    const r = 70 + Math.round(rand() * 150);
    return `<circle cx="${Math.round(rand() * 1080)}" cy="${Math.round(rand() * 470)}" r="${r}" fill="none" stroke="#fff" stroke-opacity="${(0.1 + rand() * 0.14).toFixed(2)}" stroke-width="${2 + Math.round(rand() * 4)}"/>`;
  }).join('');
  const row = (label, value, icon) =>
    value
      ? `<div class="row"><div class="ico">${icon}</div><div><div class="k">${escapeHtml(label)}</div><div class="v">${escapeHtml(value)}</div></div></div>`
      : '';
  const dot = '<span class="dot"></span>';
  const titleSize = d.title.length > 90 ? 40 : d.title.length > 60 ? 46 : d.title.length > 34 ? 60 : 78;
  return `<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1080px;height:1350px;background:#f3f1ec}
body{font-family:"Noto Sans","Noto Sans Devanagari","Noto Sans Gujarati","Noto Color Emoji",system-ui,sans-serif;color:#1d1d1b;-webkit-font-smoothing:antialiased}
.card{position:relative;width:1080px;height:1350px;background:#fbfaf7;overflow:hidden}
.top{position:relative;height:560px;background:${d.accent};color:#fff;padding:64px 72px}
.top svg.deco{position:absolute;inset:0;width:1080px;height:560px}
.brand{position:relative;display:flex;align-items:center;justify-content:space-between}
.brand .name{display:flex;align-items:center;gap:16px;font-size:38px;font-weight:600;letter-spacing:.2px}
.pill{border:3px solid rgba(255,255,255,.85);border-radius:999px;padding:10px 28px;font-size:24px;font-weight:700;letter-spacing:3px}
.kicker{position:relative;margin-top:92px;font-size:26px;font-weight:600;letter-spacing:7px;opacity:.85}
.title{position:relative;margin-top:22px;font-size:${titleSize}px;line-height:1.12;font-weight:700;letter-spacing:-.5px;max-width:900px;overflow-wrap:anywhere;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:3;overflow:hidden}
.body{padding:0 72px}
.date{display:flex;align-items:flex-end;justify-content:space-between;margin-top:-86px;position:relative;background:#fff;border-radius:28px;padding:44px 52px;box-shadow:0 2px 0 rgba(0,0,0,.04),0 18px 40px rgba(20,20,20,.10)}
.date .wd{font-size:28px;font-weight:700;letter-spacing:4px;color:${d.accent};text-transform:uppercase}
.date .num{font-size:176px;line-height:.95;font-weight:800;letter-spacing:-6px;margin-top:6px}
.date .my{font-size:32px;font-weight:500;color:#55554f;margin-top:8px}
.date .time{font-size:84px;font-weight:700;letter-spacing:-2px;text-align:right;padding-bottom:8px}
.rows{margin-top:46px;display:grid;gap:30px}
.row{display:flex;align-items:center;gap:26px}
.ico{flex:none;width:76px;height:76px;border-radius:50%;background:${d.accent}1a;display:flex;align-items:center;justify-content:center}
.ico svg{width:36px;height:36px;stroke:${d.accent};fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.k{font-size:22px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#8a8a82}
.v{font-size:40px;font-weight:600;margin-top:4px;overflow-wrap:anywhere}
.foot{position:absolute;left:72px;right:72px;bottom:56px;border-top:3px dashed #d9d6cd;padding-top:26px;display:flex;justify-content:space-between;align-items:center;font-size:24px;color:#8a8a82}
.foot b{color:#1d1d1b;font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#b9b6ab;margin:0 12px;vertical-align:middle}
.k{line-height:1.5}
.native .kicker,.native .wd,.native .pill,.native .k{letter-spacing:0;text-transform:none}
.native .kicker{font-size:34px}.native .k{font-size:28px}.native .wd{font-size:34px}.native .title{line-height:1.25}
</style></head><body class="${d.fam === 'en' || d.fam === 'hl' ? '' : 'native'}"><div class="card">
<div class="top"><svg class="deco" viewBox="0 0 1080 560">${circles}</svg>
<div class="brand"><div class="name">${LOGO('#fff')}<span>${escapeHtml(d.persona)}</span></div><div class="pill">${escapeHtml(d.status)}</div></div>
<div class="kicker">${escapeHtml(d.kicker)}</div>
<div class="title">${escapeHtml(d.title)}</div></div>
<div class="body">
${d.day ? `<div class="date"><div><div class="wd">${escapeHtml(d.weekday)}</div><div class="num">${escapeHtml(d.day)}</div><div class="my">${escapeHtml(d.monthYear)}</div></div><div class="time">${escapeHtml(d.time)}</div></div>` : ''}
<div class="rows">
${row(d.labels.where, d.place, '<svg viewBox="0 0 24 24"><path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/></svg>')}
${row(d.labels.who, d.contact, '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18.5 14a6 6 0 0 1 3 6"/></svg>')}
${row(d.hostLabel, d.owner, '<svg viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>')}
</div></div>
<div class="foot"><span>${escapeHtml(d.footer)}</span><span><b>${escapeHtml(d.ref)}</b>${dot}${escapeHtml(d.issued)}</span></div>
</div></body></html>`;
}
