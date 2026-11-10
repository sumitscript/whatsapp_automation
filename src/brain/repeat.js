// Detects a draft bubble that says (nearly) what an earlier reply already said.
// Small models loop: "Arre Prachi! Pagal toh nahin hoon…" came back three times in one chat.
// Pure and dependency-free.

const words = (s) =>
  String(s || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean);

/** Split stored assistant turns (bubbles joined by a blank line) into single bubbles. */
export function recentBubbles(turns, { turnsBack = 6 } = {}) {
  return turns
    .filter((t) => t.role === 'assistant')
    .slice(-turnsBack)
    .flatMap((t) => String(t.content || '').split(/\n{2,}/))
    .map((b) => b.trim())
    .filter(Boolean);
}

/** Jaccard overlap of the word sets, 0..1. */
export function similarity(a, b) {
  const A = new Set(words(a));
  const B = new Set(words(b));
  if (!A.size || !B.size) return 0;
  let both = 0;
  for (const w of A) if (B.has(w)) both++;
  return both / (A.size + B.size - both);
}

/**
 * True when `draft` repeats one of `recent`. Drafts under `minWords` words ("Haan!", "Acme Corp.")
 * are never flagged: repeating them is often correct.
 */
export function isRepeat(draft, recent, { threshold = 0.8, minWords = 5 } = {}) {
  const dw = words(draft);
  if (!dw.length) return false;
  const norm = dw.join(' ');
  for (const r of recent) {
    const rw = words(r);
    // Short answers ("Acme Corp.", "Haan") may legitimately repeat, e.g. the same question asked twice.
    if (dw.length >= minWords && rw.length >= minWords && (rw.join(' ') === norm || similarity(draft, r) >= threshold)) return true;
  }
  return false;
}
