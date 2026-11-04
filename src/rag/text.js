// Text normalisation + tokenisation for the vectorless retriever.
// Unicode-aware so Devanagari / Gujarati / Tamil etc. tokens survive intact.

const STOPWORDS = new Set(
  (
    // English
    'a an and are as at be been but by can could did do does for from had has have he her hers him his how i if in into ' +
    'is it its just me my of on or our ours she so than that the their them then there these they this those to too ' +
    'us was we were what when where which who whom why will with would you your yours yourself about tell please ' +
    'know any some also very much more most like get got give let lets want need okay ok hey hi hello ' +
    'use uses used using does done make made thing things something anything ' +
    // Hinglish fillers (romanised Hindi)
    'hai hain ho tha thi the kya kyu kyun ka ki ke ko se me mein par bhi to toh na nahi ye yeh wo woh hi ' +
    'aur ya kuch koi kaun kab kaise kitna kitne mujhe mera meri mere tum tumhara aap apna apni bata batao ' +
    'bolo zara thoda yaar bhai bro please plz pls ' +
    // Hindi (Devanagari)
    'है हैं था थी थे का की के को से में पर भी तो ना नहीं ये यह वो वह और या कुछ कोई क्या कैसे मुझे मेरा आप'
  ).split(/\s+/),
);

// Symbolic tech names that would otherwise be shredded by the tokenizer.
const REWRITES = [
  [/c\+\+/gi, ' cpp '],
  [/c#/gi, ' csharp '],
  [/\.net\b/gi, ' dotnet '],
  [/node\.?js/gi, ' nodejs '],
  [/next\.?js/gi, ' nextjs '],
  [/react\.?js/gi, ' react '],
  [/vue\.?js/gi, ' vue '],
  [/express\.?js/gi, ' express '],
  [/gen\s?-?ai\b/gi, ' genai '],
];

export function normalize(text) {
  let t = String(text ?? '').normalize('NFKC').toLowerCase();
  for (const [re, rep] of REWRITES) t = t.replace(re, rep);
  return t;
}

/** Light English suffix stripping. Non-Latin tokens are returned untouched. */
export function stem(token) {
  if (!/^[a-z]+$/.test(token) || token.length <= 4 || token.endsWith('js')) return token;
  if (token.endsWith('ies') && token.length > 5) return token.slice(0, -3) + 'y';
  if (token.endsWith('ing') && token.length > 6) return token.slice(0, -3);
  if (token.endsWith('ed') && token.length > 5) return token.slice(0, -2);
  if (token.endsWith('es') && /(sses|xes|ches|shes)$/.test(token)) return token.slice(0, -2);
  if (token.endsWith('s') && !token.endsWith('ss') && !token.endsWith('us') && !token.endsWith('is')) return token.slice(0, -1);
  return token;
}

/** Raw tokens (normalised, no stopword removal, no stemming). */
export function rawTokens(text) {
  return normalize(text)
    .split(/[^\p{L}\p{N}\p{M}]+/u)
    .filter((t) => t.length > 0);
}

/** Index/query tokens: normalised, stopwords removed, stemmed. */
export function tokenize(text, { keepStopwords = false } = {}) {
  const out = [];
  for (const t of rawTokens(text)) {
    if (!keepStopwords && STOPWORDS.has(t)) continue;
    if (t.length === 1 && !/\p{N}/u.test(t) && /[a-z]/.test(t)) continue; // stray latin letters
    out.push(stem(t));
  }
  return out;
}

export function isStopword(t) {
  return STOPWORDS.has(t);
}

/** Bounded Levenshtein distance (returns max+1 as soon as it is exceeded). */
export function editDistance(a, b, max = 2) {
  if (a === b) return 0;
  if (Math.abs(a.length - b.length) > max) return max + 1;
  // Optimal-string-alignment distance: a swap of two neighbouring letters ("pojcet" → "projcet" → "project")
  // counts as one edit, which is how people actually mistype on phones.
  let prev2 = null;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (prev2 && i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, prev2[j - 2] + 1);
      cur.push(v);
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    prev2 = prev;
    prev = cur;
  }
  return prev[b.length];
}

/** Rough token estimate for budget control (~4 chars/token for Latin, ~2 for Indic). */
export function estimateTokens(text) {
  const s = String(text ?? '');
  const nonLatin = (s.match(/[^\u0000-ɏ]/g) || []).length;
  return Math.ceil((s.length - nonLatin) / 4 + nonLatin / 2);
}
