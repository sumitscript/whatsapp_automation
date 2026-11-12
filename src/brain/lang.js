// Script-based language detection tuned for Indian users. Zero cost (regex only)
// and good enough to give the LLM an explicit "reply in X" hint — the single
// biggest accuracy win for small multilingual models.

const SCRIPTS = [
  { re: /[ऀ-ॿ]/g, code: 'hi', name: 'Hindi (Devanagari script)' }, // Hindi/Marathi/Nepali
  { re: /[ঀ-৿]/g, code: 'bn', name: 'Bengali' },
  { re: /[਀-੿]/g, code: 'pa', name: 'Punjabi (Gurmukhi)' },
  { re: /[઀-૿]/g, code: 'gu', name: 'Gujarati' },
  { re: /[଀-୿]/g, code: 'or', name: 'Odia' },
  { re: /[஀-௿]/g, code: 'ta', name: 'Tamil' },
  { re: /[ఀ-౿]/g, code: 'te', name: 'Telugu' },
  { re: /[ಀ-೿]/g, code: 'kn', name: 'Kannada' },
  { re: /[ഀ-ൿ]/g, code: 'ml', name: 'Malayalam' },
  { re: /[؀-ۿ]/g, code: 'ur', name: 'Urdu (Arabic script)' },
];

// Marathi-specific function words written in Devanagari.
const MARATHI = /(आहे|आहेत|नाही|काय|मला|तुम्ही|आणि|होते|कसा|कशी)/;

// High-signal romanised Hindi/Urdu words (Hinglish).
const HINGLISH = new Set(
  ('hai hain hoon tha thi kya kyu kyun kaise kaisa kaisi kab kahan kaha kidhar nahi nhi nai haan ' +
    'accha acha achha theek thik bhai yaar yar dost kar karo karna kiya kr rha raha rahi rhe hoga hogi ' +
    'mujhe mujhko mera meri mere tera teri tere tum tumhara tumhari aap aapka aapki apna hum humara ' +
    'batao bata bolo bol samjha samajh chal chalo matlab bahut bohot bhot kuch koi sab abhi kal aaj ' +
    'pata pta lagta gaya gya gayi diya liya wala wali kaam paisa khana suno dekho').split(' '),
);

const TAMIL_ROMAN = /\b(enna|epdi|eppadi|sollu|panna|irukku|illa|seri|da|machan)\b/;

const NAMED = {
  hinglish: { code: 'hi-Latn', name: 'Hinglish (Hindi written in English letters)', script: 'latin' },
  hindi: { code: 'hi', name: 'Hindi (Devanagari script)', script: 'native' },
  english: { code: 'en', name: 'English', script: 'latin' },
  marathi: { code: 'mr', name: 'Marathi (Devanagari script)', script: 'native' },
  gujarati: { code: 'gu', name: 'Gujarati', script: 'native' },
  tamil: { code: 'ta', name: 'Tamil', script: 'native' },
  telugu: { code: 'te', name: 'Telugu', script: 'native' },
  bengali: { code: 'bn', name: 'Bengali', script: 'native' },
  kannada: { code: 'kn', name: 'Kannada', script: 'native' },
  malayalam: { code: 'ml', name: 'Malayalam', script: 'native' },
  punjabi: { code: 'pa', name: 'Punjabi (Gurmukhi)', script: 'native' },
  urdu: { code: 'ur', name: 'Urdu (Arabic script)', script: 'native' },
};
const LANG_NAMES = Object.keys(NAMED).join('|');
// "I prefer hinglish", "reply in hindi", "hinglish mein baat karo", "talk to me in english please"
const PREFERENCE = new RegExp(
  // 1: "I prefer hinglish", "reply in hindi", "talk to me in English", "speak with me in hindi", "baat karo english"
  // 2: "hinglish mein baat karo", "english please, reply"
  // 3: a message that is only the language: "English", "in english please"
  `\\b(?:prefer(?:s|red)?|(?:reply|respond|answer|talk|speak|write|chat|text|message)(?:\\s+(?:to|with)\\s+me)?\\s+(?:in|using)|baat karo|bolo|likho)\\b[^.\\n]{0,20}?\\b(${LANG_NAMES})\\b|\\b(${LANG_NAMES})\\b[^.\\n]{0,12}?\\b(?:mein|me|mai|please|pls|only)\\b[^.\\n]{0,12}?\\b(?:baat|bolo|likho|reply|karo|talk)|^\\s*(?:in\\s+)?(${LANG_NAMES})\\s*(?:please|pls|plz|only)?\\s*[.!]*\\s*$`,
  'i',
);

/**
 * A language the person explicitly asked for, newest mention wins.
 * @param {string[]} texts their messages and/or memory notes, oldest first
 */
export function statedLanguagePreference(texts) {
  for (let i = texts.length - 1; i >= 0; i--) {
    const m = String(texts[i] ?? '').toLowerCase().match(PREFERENCE);
    if (m) return NAMED[m[1] || m[2] || m[3]] ?? null;
  }
  return null;
}

/**
 * The language to answer in: an explicit preference beats a romanised (Latin-letter) message;
 * otherwise a lone English line inside a Hinglish chat stays Hinglish (people switch
 * scripts mid-chat, the reply shouldn't).
 * @param {string} text current message
 * @param {string[]} previous their earlier messages (oldest first)
 * @param {string|null} notes long-term memory notes
 */
export function replyLanguage(text, previous = [], notes = null) {
  const current = detectLanguage(text);
  // "Talk to me in English" said in this chat is explicit and newest-wins. A preference that only comes from the long-term notes
  // ("Prefers Hinglish" from an old chat) gives way once the person has been writing plain English for a few messages.
  const fromChat = statedLanguagePreference([...previous, text]);
  const fromNotes = notes ? statedLanguagePreference([notes]) : null;
  let stated = fromChat || fromNotes;
  if (!fromChat && fromNotes && current.code === 'en') {
    const recent = previous.slice(-2).map(detectLanguage);
    if (recent.length >= 2 && recent.every((l) => l.code === 'en')) stated = null;
  }
  // Writing in a native script (Devanagari, Tamil…) is itself the strongest signal.
  if (stated && current.script !== 'native') return { ...stated, reason: 'preference' };
  if (current.code === 'en') {
    const recent = previous.slice(-3).map(detectLanguage);
    if (recent.length >= 2 && recent.filter((l) => l.code === 'hi-Latn').length >= 2) return { ...NAMED.hinglish, reason: 'recent' };
  }
  return { ...current, reason: 'message' };
}

export function detectLanguage(text) {
  const s = String(text ?? '');
  let best = null;
  let bestCount = 0;
  for (const sc of SCRIPTS) {
    const c = (s.match(sc.re) || []).length;
    if (c > bestCount) {
      best = sc;
      bestCount = c;
    }
  }
  if (best && bestCount >= 2) {
    if (best.code === 'hi' && MARATHI.test(s)) return { code: 'mr', name: 'Marathi (Devanagari script)', script: 'native' };
    return { code: best.code, name: best.name, script: 'native' };
  }

  const words = s.toLowerCase().match(/[a-z]+/g) || [];
  if (!words.length) return { code: 'und', name: 'the same language as the user', script: 'none' };
  const hits = words.filter((w) => HINGLISH.has(w)).length;
  if (hits >= 2 || (hits >= 1 && words.length <= 4)) {
    return { code: 'hi-Latn', name: 'Hinglish (Hindi written in English letters)', script: 'latin' };
  }
  if (TAMIL_ROMAN.test(s.toLowerCase())) return { code: 'ta-Latn', name: 'Tanglish (Tamil in English letters)', script: 'latin' };
  return { code: 'en', name: 'English', script: 'latin' };
}
