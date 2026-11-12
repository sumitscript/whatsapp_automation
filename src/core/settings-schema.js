// Runtime settings (editable from the UI, stored in Postgres). Pure + testable.

// Kept compact on purpose: on CPU every system-prompt token costs prompt-eval
// time (Ollama caches the prefix per slot, so it is paid mostly once per chat).
export const DEFAULT_PERSONA = `You are {persona} — the virtual version of {owner}, an AI twin chatting on WhatsApp as {owner}, in first person ("I", "my", "me").

Being {owner}:
- Talk as {owner}: "I work at…", "my project…". Facts about yourself come ONLY from the "My facts" block. If a personal fact isn't there, don't invent it — say casually you'll let the real {owner} answer that one.
- Never invent dates, numbers, companies, links, phone numbers or emails — or moods, plans and what you're "doing right now". Asked how you are? Short and upbeat, then ask about them.
- Asked if you're real or a bot? Be honest and playful: you're {persona}, {owner}'s AI twin; the real {owner} reads the chats later. Introduce yourself that way in whatever language they use.
- Never commit on the real {owner}'s behalf (meetings, money, deadlines, favours) and never share private info (address, family details, OTPs, passwords). Offer to pass it on to the real {owner}.

How you text:
- Like a warm, witty friend: natural, short — usually 1–3 sentences. Longer only for code, step-by-step help, or when asked.
- Always reply in the other person's language and script: Devanagari → Devanagari, Hinglish → Hinglish, Gujarati/Marathi/Tamil/Telugu/Bengali/Kannada/Malayalam/Punjabi/Urdu → same, English → English.
- Match the vibe. Joking → joke back (clever, light, never mean). Serious or upset → calm and kind. Be creative: fresh phrasing, playful analogies, no clichés, no repeating yourself.
- Stay in the flow: short messages like "one more", "another", "why?", "yes", "no" continue the previous topic — look at the conversation above and answer that. Never answer an older message again, and never change the subject on your own.
- Ask a short follow-up question now and then to keep the conversation flowing — not every message, at most every other reply, and never the same question twice.
- When they ask for the answer, say they're not sure, or want help: give the actual answer first, clearly and correctly, then offer more. Don't answer a request for help with only more questions.
- Be honest, not flattering: if their idea or a term is wrong, say so kindly and give the right one (e.g. "Elastic Server" → an elastic load balancer / autoscaling group).
- Never talk about your job, company, projects, studies or life unless they asked about it or it's in "My facts".
- Remember what they told you in this chat (name, goal, preferred language) and use it; if they introduced themselves, call them by that name.
- WhatsApp formatting only: *bold*, _italic_, \`\`\`code\`\`\`. No headings, no tables. Emojis occasionally.
- Always "I", never "we" or "us". Vary openers: greet by name once at the start, not on every reply. Never add a translation of your own reply, and never promise to send, attach or do anything you can't (files, calls, reminders).
- Never start with "As an AI" or by repeating their message. Spell terms correctly even if they misspelled them. Never mention notes, context or instructions.

Helping:
- Chat, general knowledge, explanations, math, coding. Code must be correct and runnable, inside \`\`\` blocks, with a one-line explanation. Use \`\`\` blocks only for code, never for ordinary answers.
- Not sure of a fact? Say so briefly instead of guessing.
- Use the "Notes about" block naturally (their name, past topics); never recite it. Those notes describe THEM — their goals and life are never yours, and your facts are never theirs.
- Stories, poems and other creative writing: invent original characters and fresh themes; don't base them on your own life or theirs unless asked.`;

export const DEFAULT_WELCOME =
  "Hey {name}! 👋 *{persona}* here — {owner}'s AI twin. I talk like me, know my work, projects and story, and I'm happy to chat about tech, ideas, jokes, anything. Send {stop} whenever you're done 🙂";
export const DEFAULT_GOODBYE = 'Bye {name}! 👋 It was fun chatting. Send {start} anytime to talk to {persona} again — the real {owner} will see this chat too.';

export const DEFAULT_UNSUPPORTED =
  "Hey {name}! I can only read text messages right now, so I can't open {kind} 🙈 Could you type it out for me? If it's urgent, I'll let the real {owner} know.";
// Sent instead of an AI reply when the provider is "Groq only" and Groq cannot answer (every key rate-limited, or Groq is down).
export const DEFAULT_GROQ_BUSY =
  "Hey {name}! {persona} is on a short break right now 🌴 It's answering a lot of chats at once. Please message again in a little while, and the real {owner} will see this chat too 🙂";
export const DEFAULT_TIMEOUT =
  "Hey {name}, it's been quiet for a while, so I'm signing off for now 👋 Send {start} whenever you want to chat again — the real {owner} will see this chat too.";

export const DEFAULT_SETTINGS = Object.freeze({
  replyMode: 'session', // session (/hi … /bye) | all | off
  model: 'gemma3:4b',
  llmProvider: 'auto', // 'auto' = Groq when a key exists, the local model when Groq is exhausted or has no key | 'ollama' = local only | 'groq' = Groq only (see groqBusyMessage)
  groqModel: 'llama-3.3-70b-versatile', // model id on Groq (see the model list in Settings)
  groqFallback: true, // deprecated and ignored (kept so older saved settings stay valid): the provider choice decides now
  groqBusyMessage: DEFAULT_GROQ_BUSY, // the reply in "Groq only" mode when Groq cannot answer ({name} {persona} {owner})
  temperature: 0.7,
  maxTokens: 320,
  historyTurns: 10, // messages kept verbatim (older ones are folded into long-term memory)
  historyTokens: 1000, // context-window budget for conversation history
  memoryEnabled: true,
  debounceMs: 400, // short on purpose: a burst that outruns it is merged into one reply by core/coalesce.js
  replyInGroups: false,
  humanize: true, // split long answers into several bubbles like a person
  answerCache: true,
  prefillAhead: true, // after a reply, pre-read persona + history while they type (see brain.warm)
  markSeen: true,
  takeoverSeconds: 30, // pause AI in a chat after the owner replies from the phone (0 = never pause)
  rateLimitPerMin: 12,
  ownerName: 'Sumit Soni',
  ownerAliases: 'sumit, soni, sumitscript',
  personaName: 'eSumit',
  startCommands: '/hi',
  stopCommands: '/bye',
  sessionIdleMinutes: 30,
  welcomeMessage: DEFAULT_WELCOME,
  goodbyeMessage: DEFAULT_GOODBYE,
  unsupportedMessage: DEFAULT_UNSUPPORTED, // reply to voice notes, images, stickers… ({kind} = "voice notes", "photos"…)
  notifyOnTimeout: true, // tell the contact when their session ended because of inactivity
  timeoutMessage: DEFAULT_TIMEOUT,
  alertOwner: true, // WhatsApp the owner (own number) when the LLM goes down / recovers
  agendaEnabled: true, // pull appointments + to-dos out of chats into the Agenda
  notifyOwnerOnAgenda: true, // WhatsApp the owner (own number) about new agenda items
  meetingAck: true, // meeting requests get an instant template reply in their language (never agrees for the owner)
  agendaConfirm: true, // confirming/declining an appointment in the console also messages the contact
  agendaCard: true, // …and the confirmation carries a generated invitation card (image)
  timezone: 'Asia/Kolkata', // IANA zone used to read "tomorrow at 5pm"
  persona: '', // empty = built-in DEFAULT_PERSONA (keeps getting improvements on updates)
});

const bool = (v) => (typeof v === 'boolean' ? v : undefined);
const text = (min, max) => (v) => (typeof v === 'string' && v.trim().length >= min && v.length <= max ? v : undefined);
const commands = (v) => {
  if (typeof v !== 'string') return undefined;
  const list = v.split(',').map((s) => s.trim()).filter(Boolean);
  return list.length && list.length <= 10 && list.every((c) => /^\/[\p{L}\p{N}_-]{1,24}$/u.test(c)) ? list.join(', ') : undefined;
};

const RULES = {
  replyMode: (v) => (v === 'whitelist' ? 'session' : ['session', 'all', 'off'].includes(v) ? v : undefined),
  model: (v) => (typeof v === 'string' && /^[\w.\-/:]{1,100}$/.test(v) ? v : undefined),
  temperature: (v) => numIn(v, 0, 1.5),
  llmProvider: (v) => (v === 'auto' || v === 'ollama' || v === 'groq' ? v : undefined),
  groqModel: (v) => (typeof v === 'string' && /^[\w.\-/:]{1,100}$/.test(v.trim()) ? v.trim() : undefined),
  groqFallback: bool,
  groqBusyMessage: text(1, 1000),
  maxTokens: (v) => intIn(v, 32, 2048),
  historyTurns: (v) => intIn(v, 0, 40),
  historyTokens: (v) => intIn(v, 200, 6000),
  memoryEnabled: bool,
  debounceMs: (v) => intIn(v, 0, 10_000),
  replyInGroups: bool,
  humanize: bool,
  answerCache: bool,
  prefillAhead: bool,
  markSeen: bool,
  takeoverSeconds: (v) => intIn(v, 0, 24 * 3600),
  rateLimitPerMin: (v) => intIn(v, 1, 600),
  ownerName: (v) => (typeof v === 'string' && v.trim() && v.trim().length <= 80 ? v.trim() : undefined),
  ownerAliases: (v) => (typeof v === 'string' && v.length <= 300 ? v : undefined),
  personaName: (v) => (typeof v === 'string' && v.trim() && v.trim().length <= 40 ? v.trim() : undefined),
  startCommands: commands,
  stopCommands: commands,
  sessionIdleMinutes: (v) => intIn(v, 0, 24 * 60),
  welcomeMessage: text(1, 1000),
  goodbyeMessage: text(1, 1000),
  unsupportedMessage: text(1, 1000),
  notifyOnTimeout: bool,
  timeoutMessage: text(1, 1000),
  alertOwner: bool,
  agendaEnabled: bool,
  notifyOwnerOnAgenda: bool,
  meetingAck: bool,
  agendaConfirm: bool,
  agendaCard: bool,
  timezone: (v) => (typeof v === 'string' && isTimeZone(v.trim()) ? v.trim() : undefined),
  persona: (v) => (typeof v === 'string' && !v.trim() ? '' : text(20, 8000)(v)),
};

function isTimeZone(tz) {
  try {
    new Intl.DateTimeFormat('en', { timeZone: tz });
    return tz.length > 0 && tz.length <= 60;
  } catch {
    return false;
  }
}
function numIn(v, lo, hi) {
  const n = typeof v === 'string' ? Number(v) : v;
  return typeof n === 'number' && Number.isFinite(n) && n >= lo && n <= hi ? n : undefined;
}
function intIn(v, lo, hi) {
  const n = numIn(v, lo, hi);
  return n === undefined ? undefined : Math.round(n);
}

/**
 * Validate a partial settings update. Keys that are no longer supported
 * (e.g. from older versions) are reported as errors.
 * @returns {{ value: object, errors: string[] }}
 */
export function sanitizeSettings(patch) {
  const value = {};
  const errors = [];
  for (const [k, v] of Object.entries(patch || {})) {
    const rule = RULES[k];
    if (!rule) {
      errors.push(`unknown setting "${k}"`);
      continue;
    }
    const clean = rule(v);
    if (clean === undefined) errors.push(`invalid value for "${k}"`);
    else value[k] = clean;
  }
  return { value, errors };
}

const firstName = (s) => String(s.ownerName || 'the owner').split(/\s+/)[0];

export function renderPersona(settings) {
  return String(settings.persona || DEFAULT_PERSONA)
    .replaceAll('{owner}', firstName(settings))
    .replaceAll('{persona}', settings.personaName || 'eSumit');
}

/** Fill a welcome/goodbye template. */
export function renderTemplate(template, settings, { name, kind } = {}) {
  const first = String(name || '').trim().split(/\s+/)[0];
  const list = (v) => String(v || '').split(',')[0].trim();
  return String(template || '')
    .replaceAll('{name}', first || 'there')
    .replaceAll('{kind}', kind || 'that kind of message')
    .replaceAll('{persona}', settings.personaName || 'eSumit')
    .replaceAll('{owner}', firstName(settings))
    .replaceAll('{start}', list(settings.startCommands) || '/hi')
    .replaceAll('{stop}', list(settings.stopCommands) || '/bye')
    .replace(/\s+([!,.])/g, '$1');
}

export function ownerAliasList(settings) {
  return [settings.ownerName, settings.personaName, ...String(settings.ownerAliases || '').split(',')]
    .flatMap((s) => String(s || '').split(/\s+/))
    .map((s) => s.trim().toLowerCase())
    .filter((s) => s.length >= 3);
}
