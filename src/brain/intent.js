import { editDistance } from '../rag/text.js';

// Zero-latency intent routing. Picking the intent with regexes (instead of an
// extra LLM call) keeps CPU inference fast; the intent then tunes retrieval,
// temperature and the length budget of the reply.

const GREETING = /^(hi+|hello+|hey+|hii+|helo|hlo|yo|sup|namaste|namaskar|namaskaram|vanakkam|sat sri akal|salaam|assalam|good\s?(morning|afternoon|evening|night)|gm|gn|kaise ho|kaise hain|kaisa hai|kaisi ho|kya ha+l|kya chal raha|kya scene|how are (you|u)|how r u|hru|kem cho|kasa kay|wassup|whatsup|नमस्ते|हेलो)\b/i;
const THANKS = /\b(thanks|thank you|thx|ty|dhanyavaad|dhanyawad|shukriya|thnx|शुक्रिया|धन्यवाद)\b/i;
const JOKE = /(😂|🤣|😆|😜|😝|😹|😅|\bha(ha)+\b|\bhe(he)+\b|\blo+l\b|\blmao\b|\brofl\b|\bjoke\b|\bjokes\b|\bfunny\b|\bmeme\b|\bmazak\b|\bmajak\b|\bmasti\b|\bchutkula\b|\bpj\b|\broast\b|मज़ाक|चुटकुला)/i;
const CODE = /(```|\b(code|coding|function|bug|debug|error|exception|stack ?trace|compile|syntax|regex|sql|query|api|endpoint|python|javascript|typescript|java|c\+\+|cpp|golang|rust|react|node|express|django|flask|fastapi|docker|kubernetes|git|leetcode|algorithm|complexity|recursion|array|linked list|class|object|async|promise|npm|pip)\b|[{};]\s*$|=>|\bdef\s+\w+\(|\bconsole\.log\b)/i;
const MATH = /^[\d\s+\-*/().%^=x×÷]+$|\d\s*[+\-*/×÷^%]\s*\d|\b(calculate|solve|integral|derivative|equation|percent of|kitna hota)\b/i;
const OWNER_WORDS = /\b(you|your|yours|u|ur|tum|tumhara|tumhari|aap|aapka|aapki|tera|teri|he|his|him|uska|uski|unka|unki)\b/i;
const OWNER_POSSESSIVE = /\b(your|yours|ur|tumhara|tumhari|tumhare|aapka|aapki|aapke|tera|teri|tere|his|uska|uski|unka|unki)\b/i;
const USER_WORDS = /\b(my|mine|i|i'm|im|me|mera|meri|mere|mujhe|mujhko|main|apna|apni)\b/i;
const PROFILE = /\b(work|working|job|naukri|kaam|company|intern|internship|experience|project|projects|skill|skills|tech ?stack|college|university|degree|cgpa|gpa|study|padhai|education|resume|cv|portfolio|github|linkedin|website|contact|email|phone|number|hire|hiring|available|availability|open to work|achievement|certificate|certification|hackathon|blog|article|extension|location|city|rehte|from where|kahan se|age|about yourself|introduce|who are you|kaun ho|tum kaun)\b/i;

// Single profile words, matched with typo tolerance ("pojcet", "experiance", "certifcate").
// Only words ≥ 6 letters: shorter ones collide with everyday words.
const PROFILE_WORDS = [
  'project', 'projects', 'experience', 'company', 'internship', 'college', 'university', 'degree', 'skills',
  'education', 'resume', 'portfolio', 'github', 'linkedin', 'website', 'contact', 'achievement', 'achievements',
  'certificate', 'certificates', 'certification', 'certifications', 'hackathon', 'article', 'articles', 'working',
  'naukri', 'padhai', 'location',
];

function hasProfileWord(lower) {
  if (PROFILE.test(lower)) return true;
  for (const w of lower.match(/[a-z]{6,}/g) || []) {
    for (const p of PROFILE_WORDS) {
      if (p[0] === w[0] && Math.abs(p.length - w.length) <= 2 && editDistance(w, p, 2) <= 2) return true;
    }
  }
  return false;
}

/**
 * About-me = a profile topic (work, projects, college…, typos allowed) AND either a
 * "you / tum / aap" word or a short message ("projects?"). "tum meri help karo" has
 * no profile topic, so it stays general.
 * @param {string} text
 * @param {{ ownerAliases?: string[] }} [opts]
 * @returns {'greeting'|'thanks'|'joke'|'coding'|'math'|'about_owner'|'general'}
 */
export function classifyIntent(text, { ownerAliases = [] } = {}) {
  const s = String(text ?? '').trim();
  const lower = s.toLowerCase();
  const words = lower.split(/\s+/).filter(Boolean);

  const mentionsOwner = ownerAliases.some((a) => {
    if (!a) return false;
    const esc = a.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return new RegExp(`(^|[^\\p{L}])${esc}([^\\p{L}]|$)`, 'u').test(lower);
  });
  if (mentionsOwner) return 'about_owner';
  if (hasProfileWord(lower)) {
    // "your project" / "tumhara college" is about me; "my project" / "check my resume" is about them.
    const aboutThem = USER_WORDS.test(lower);
    if (OWNER_POSSESSIVE.test(lower) || (!aboutThem && (OWNER_WORDS.test(lower) || words.length <= 6))) return 'about_owner';
  }
  if (CODE.test(s)) return 'coding';
  if (MATH.test(lower) && /\d/.test(lower)) return 'math';
  if (JOKE.test(s)) return 'joke';
  if (words.length <= 6 && GREETING.test(lower)) return 'greeting';
  if (words.length <= 6 && THANKS.test(lower)) return 'thanks';
  return 'general';
}

/** Per-intent generation profile. */
export const INTENT_PROFILES = Object.freeze({
  greeting: { temperature: 0.7, tokenScale: 0.4, useKb: false },
  thanks: { temperature: 0.7, tokenScale: 0.3, useKb: false },
  joke: { temperature: 0.9, tokenScale: 0.6, useKb: false },
  coding: { temperature: 0.2, tokenScale: 2.5, useKb: false },
  math: { temperature: 0.1, tokenScale: 0.8, useKb: false },
  about_owner: { temperature: 0.3, tokenScale: 1, useKb: true },
  general: { temperature: null, tokenScale: 1, useKb: 'auto' }, // KB used only when retrieval is confident
});

/**
 * Short follow-ups ("one more", "another one", "why?", "yes") say nothing about
 * their topic, so classifying them alone yields 'general' and drags in random
 * knowledge-base facts. Inherit the topic of the previous user turn instead.
 * @param {string} text
 * @param {{role:string, content:string}[]} turns conversation so far (oldest first)
 * @param {{ ownerAliases?: string[] }} [opts]
 * @returns {{ intent: string, inherited: boolean }}
 */
export function resolveIntent(text, turns, opts = {}) {
  const own = classifyIntent(text, opts);
  if (own !== 'general' || !isFollowUp(text)) return { intent: own, inherited: false };
  const lastUser = [...(turns || [])].reverse().find((t) => t.role === 'user')?.content;
  if (!lastUser) return { intent: own, inherited: false };
  const prev = classifyIntent(lastUser, opts);
  // A greeting/thanks says nothing about what the next short message is about.
  if (prev === 'general' || prev === 'greeting' || prev === 'thanks') return { intent: own, inherited: false };
  return { intent: prev, inherited: true };
}

/** Heuristic: does the message lean on earlier turns ("what about his 2nd one?")? */
export function isFollowUp(text) {
  const s = String(text ?? '').toLowerCase().trim();
  const words = s.split(/\s+/).filter(Boolean);
  if (words.length <= 3 && !/\?$/.test(s)) return true;
  return /\b(it|that|this|those|these|he|his|him|she|her|they|them|more|aur|uska|uski|unka|woh|wo|ye|yeh|isme|usme|aage|phir|then|also|same|again|elaborate|explain more|detail)\b/.test(s);
}
