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
