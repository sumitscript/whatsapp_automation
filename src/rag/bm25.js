// BM25F: field-weighted BM25 (title/tags/body) — the core of the vectorless
// retriever. Exact lexical scoring, microsecond lookups, no embeddings.
import { editDistance } from './text.js';

export class BM25FIndex {
  /**
   * @param {object} opts
   * @param {Record<string, number>} opts.weights field weights
   * @param {Record<string, number>} [opts.b] per-field length normalisation
   * @param {number} [opts.k1]
   */
  constructor({ weights = { title: 3, tags: 2, body: 1 }, b = {}, k1 = 1.2 } = {}) {
    this.weights = weights;
    this.fields = Object.keys(weights);
    this.b = Object.fromEntries(this.fields.map((f) => [f, b[f] ?? (f === 'body' ? 0.75 : 0.5)]));
    this.k1 = k1;
    this.docs = []; // { id, len: {field: n}, tf: {field: Map(term -> count)} }
    this.df = new Map();
    this.avgLen = {};
    this.vocab = [];
  }

  /** @param {string} id @param {Record<string, string[]>} fieldTokens */
  add(id, fieldTokens) {
    const doc = { id, len: {}, tf: {} };
    const seen = new Set();
    for (const f of this.fields) {
      const toks = fieldTokens[f] || [];
      const tf = new Map();
      for (const t of toks) {
        tf.set(t, (tf.get(t) || 0) + 1);
        seen.add(t);
      }
      doc.tf[f] = tf;
      doc.len[f] = toks.length;
    }
    for (const t of seen) this.df.set(t, (this.df.get(t) || 0) + 1);
    this.docs.push(doc);
  }

  build() {
    for (const f of this.fields) {
      const total = this.docs.reduce((s, d) => s + d.len[f], 0);
      this.avgLen[f] = this.docs.length ? total / this.docs.length || 1 : 1;
    }
    this.vocab = [...this.df.keys()];
    return this;
  }

  idf(term) {
    const n = this.docs.length;
    const df = this.df.get(term) || 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  has(term) {
    return this.df.has(term);
  }

  /** Closest vocabulary term within a small edit distance (typo tolerance). */
  fuzzy(term) {
    if (term.length < 5 || !/^[a-z]+$/.test(term)) return null;
    const max = term.length >= 8 ? 2 : 1;
    let best = null;
    let bestD = max + 1;
    for (const v of this.vocab) {
      if (Math.abs(v.length - term.length) > max || v[0] !== term[0]) continue;
      const d = editDistance(term, v, max);
      if (d < bestD || (d === bestD && best && (this.df.get(v) || 0) > (this.df.get(best) || 0))) {
        bestD = d;
        best = v;
        if (d === 1 && max === 1) break;
      }
    }
    return bestD <= max ? best : null;
  }

  /**
   * @param {{term: string, weight: number}[]} query
   * @param {number} limit
   * @returns {{id: string, score: number, matched: string[]}[]}
   */
  search(query, limit = 10) {
    const scores = new Map();
    for (const { term, weight } of query) {
      if (!this.df.has(term)) continue;
      const idf = this.idf(term);
      for (const doc of this.docs) {
        let tf = 0;
        for (const f of this.fields) {
          const c = doc.tf[f].get(term);
          if (!c) continue;
          const norm = 1 - this.b[f] + this.b[f] * (doc.len[f] / this.avgLen[f]);
          tf += (this.weights[f] * c) / norm;
        }
        if (tf === 0) continue;
        const s = weight * idf * ((tf * (this.k1 + 1)) / (tf + this.k1));
        const cur = scores.get(doc.id);
        if (cur) {
          cur.score += s;
          cur.matched.push(term);
        } else scores.set(doc.id, { id: doc.id, score: s, matched: [term] });
      }
    }
    // Coverage bonus: documents matching more distinct query terms rank higher.
    const distinct = new Set(query.filter((q) => q.weight >= 1).map((q) => q.term)).size || 1;
    for (const r of scores.values()) {
      const covered = new Set(r.matched).size;
      r.score *= 1 + 0.25 * Math.min(1, covered / distinct);
    }
    return [...scores.values()].sort((a, b) => b.score - a.score).slice(0, limit);
  }
}
