// Vectorless knowledge base.
//
// Markdown files are parsed into a heading tree (doc › H2 › H3). Every node is
// a self-contained "section" indexed with BM25F over three fields:
//   title (heading + parent heading) ×3, tags (doc title/tags/aliases) ×2, body ×1.
// Queries are expanded with synonyms.json (Hinglish/Hindi → English, tech
// abbreviations) and typo-corrected against the vocabulary. Rebuilding the
// whole index takes milliseconds, so edits hot-reload — no embedding step.
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { BM25FIndex } from './bm25.js';
import { estimateTokens, isStopword, normalize, rawTokens, stem, tokenize } from './text.js';
import { LRUCache, sha1 } from '../core/util.js';

const MAX_SECTION_TOKENS = 450;

export function parseFrontmatter(src) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(src);
  if (!m) return { meta: {}, body: src };
  const meta = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = /^([A-Za-z_][\w-]*)\s*:\s*(.*)$/.exec(line);
    if (kv) meta[kv[1].toLowerCase()] = kv[2].trim().replace(/^["']|["']$/g, '');
  }
  return { meta, body: src.slice(m[0].length) };
}

/** Split a markdown document into retrievable sections. */
export function parseMarkdown(file, src) {
  const { meta, body } = parseFrontmatter(src);
  const lines = body.split(/\r?\n/);
  const fallbackTitle = file.replace(/\.(md|txt)$/i, '').replace(/^\d+[_-]?/, '').replace(/[_-]+/g, ' ');
  let docTitle = meta.title || '';
  const sections = [];
  let h2 = null;
  let cur = { level: 1, heading: '', lines: [] };
  let inFence = false;

  const flush = () => {
    const text = cur.lines.join('\n').trim();
    if (text) {
      const heading = cur.level === 1 ? docTitle || fallbackTitle : cur.heading;
      const parent = cur.level === 3 && h2 ? h2 : null;
      sections.push({ level: cur.level, heading, parent, text });
    }
  };

  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) inFence = !inFence;
    const h = inFence ? null : /^(#{1,3})\s+(.+?)\s*#*\s*$/.exec(line);
    if (!h) {
      cur.lines.push(line);
      continue;
    }
    const level = h[1].length;
    const heading = h[2].trim();
    if (level === 1) {
      if (!docTitle) docTitle = heading;
      continue; // the H1 itself is the doc title; following text is the summary
    }
    flush();
    if (level === 2) h2 = heading;
    cur = { level, heading, lines: [] };
  }
  flush();

  docTitle = docTitle || fallbackTitle;
  return {
    file,
    title: docTitle,
    tags: meta.tags || '',
    aliases: meta.aliases || '',
    sections: sections.map((s, i) => ({
      id: `${file}#${i}`,
      file,
      docTitle,
      heading: s.heading,
      parent: s.parent,
      path: [docTitle, s.parent, s.level > 1 ? s.heading : null].filter(Boolean).join(' › '),
      text: s.text,
      tokens: estimateTokens(s.text),
    })),
  };
}

/** Plain-text fallback: group paragraphs into ~180-word sections. */
export function parseText(file, src) {
  const title = file.replace(/\.txt$/i, '').replace(/^\d+[_-]?/, '').replace(/[_-]+/g, ' ');
  const paras = src.split(/\r?\n\s*\r?\n/).map((p) => p.trim()).filter(Boolean);
  const sections = [];
  let buf = [];
  let words = 0;
  const push = () => {
    if (!buf.length) return;
    const text = buf.join('\n\n');
    const heading = buf[0].split('\n')[0].slice(0, 80);
    sections.push({ id: `${file}#${sections.length}`, file, docTitle: title, heading, parent: null, path: `${title} › ${heading}`, text, tokens: estimateTokens(text) });
    buf = [];
    words = 0;
  };
  for (const p of paras) {
    const w = p.split(/\s+/).length;
    if (words + w > 180) push();
    buf.push(p);
    words += w;
  }
  push();
  return { file, title, tags: '', aliases: '', sections };
}

export class KnowledgeBase {
  constructor({ dir, logger, topK = 4, maxContextTokens = 900, minScore = 2.2 }) {
    this.dir = dir;
    this.log = logger;
    this.topK = topK;
    this.maxContextTokens = maxContextTokens;
    this.minScore = minScore;
    this.docs = [];
    this.sections = new Map();
    this.synonyms = new Map();
    this.index = new BM25FIndex().build();
    this.version = 'empty';
    this.loadedAt = null;
    this.memo = new LRUCache({ max: 1000 });
    this.watcher = null;
  }

  async load() {
    const started = performance.now();
    let entries = [];
    try {
      entries = await fsp.readdir(this.dir, { withFileTypes: true });
    } catch (err) {
      this.log?.error('Knowledge base directory not readable', { dir: this.dir, err });
    }
    const files = entries
      .filter((e) => e.isFile() && /\.(md|txt)$/i.test(e.name) && !e.name.startsWith('_') && !/^readme\.md$/i.test(e.name))
      .map((e) => e.name)
      .sort();

    const docs = [];
    const hash = [];
    for (const name of files) {
      const src = await fsp.readFile(path.join(this.dir, name), 'utf8');
      hash.push(name, src);
      docs.push(name.toLowerCase().endsWith('.md') ? parseMarkdown(name, src) : parseText(name, src));
    }

    const synonyms = new Map();
    try {
      const raw = await fsp.readFile(path.join(this.dir, 'synonyms.json'), 'utf8');
      hash.push('synonyms', raw);
      for (const [k, v] of Object.entries(JSON.parse(raw))) {
        if (Array.isArray(v)) synonyms.set(normalize(k).trim(), v.map(String));
      }
    } catch (err) {
      if (err.code !== 'ENOENT') this.log?.warn('synonyms.json ignored', { err });
    }

    const index = new BM25FIndex({ weights: { title: 3, tags: 2, body: 1 } });
    const sections = new Map();
    for (const doc of docs) {
      const docTags = [doc.title, doc.tags, doc.aliases].join(' ');
      for (const s of doc.sections) {
        sections.set(s.id, s);
        index.add(s.id, {
          title: tokenize([s.heading, s.parent].filter(Boolean).join(' ')),
          tags: tokenize(docTags),
          body: tokenize(s.text),
        });
      }
    }
    index.build();

    this.docs = docs;
    this.sections = sections;
    this.synonyms = synonyms;
    this.index = index;
    this.version = docs.length ? sha1(hash.join('\u0000')).slice(0, 12) : 'empty';
    this.loadedAt = new Date();
    this.memo.clear();
    this.log?.info('Knowledge base indexed', {
      files: docs.length,
      sections: sections.size,
      vocab: index.vocab.length,
      synonyms: synonyms.size,
      version: this.version,
      ms: Math.round(performance.now() - started),
    });
    return this;
  }

  /** Watch the directory and re-index on change (debounced). */
  watch() {
    if (this.watcher) return;
    let timer = null;
    try {
      this.watcher = fs.watch(this.dir, () => {
        clearTimeout(timer);
        timer = setTimeout(() => this.load().catch((err) => this.log?.error('KB reload failed', { err })), 300);
      });
      this.watcher.on('error', (err) => this.log?.warn('KB watcher error', { err }));
    } catch (err) {
      this.log?.warn('KB watch unavailable; use the reload API instead', { err });
    }
  }

  close() {
    this.watcher?.close();
    this.watcher = null;
  }

  /** Build the weighted query: original terms, synonym expansions, typo fixes. */
  expandQuery(text) {
    const raw = rawTokens(text);
    const terms = new Map();
    const add = (t, w) => {
      if (!t || (terms.get(t) ?? 0) >= w) return;
      terms.set(t, w);
    };
    const expandWith = (list) => {
      for (const phrase of list) for (const t of tokenize(phrase)) add(t, 0.6);
    };

    for (let i = 0; i < raw.length; i++) {
      const tok = raw[i];
      if (i + 1 < raw.length) {
        const bigram = this.synonyms.get(`${tok} ${raw[i + 1]}`);
        if (bigram) expandWith(bigram);
      }
      const syn = this.synonyms.get(tok) || this.synonyms.get(stem(tok));
      if (syn) expandWith(syn);
      if (isStopword(tok)) continue;
      const s = stem(tok);
      if (this.index.has(s)) add(s, 1);
      else if (!syn) {
        const fixed = this.index.fuzzy(s);
        if (fixed) add(fixed, 0.8);
        else add(s, 1); // unknown term: harmless, keeps coverage honest
      }
    }
    return [...terms].map(([term, weight]) => ({ term, weight }));
  }

  /**
   * Retrieve the most relevant sections for a query.
   * @returns {{ sections: object[], context: string, topScore: number, expandedTerms: string[], tookMs: number }}
   */
  search(text, { topK = this.topK, maxTokens = this.maxContextTokens, minScore = this.minScore } = {}) {
    const key = `${this.version}|${topK}|${maxTokens}|${minScore}|${normalize(text).trim()}`;
    const hit = this.memo.get(key);
    if (hit) return { ...hit, tookMs: 0, memo: true };

    const started = performance.now();
    const query = this.expandQuery(text);
    const ranked = this.index.search(query, Math.max(topK * 3, 10));
    const topScore = ranked[0]?.score ?? 0;
    const floor = Math.max(minScore, topScore * 0.35);

    const picked = [];
    let budget = maxTokens;
    for (const r of ranked) {
      if (r.score < floor || picked.length >= topK) break;
      const s = this.sections.get(r.id);
      const cost = Math.min(s.tokens, MAX_SECTION_TOKENS);
      if (cost > budget && picked.length) continue;
      budget -= cost;
      picked.push({ ...s, score: Number(r.score.toFixed(3)), matched: r.matched });
    }

    const context = picked
      .map((s, i) => `[${i + 1}] ${s.path}\n${truncateTokens(s.text, MAX_SECTION_TOKENS)}`)
      .join('\n\n');
    const result = {
      sections: picked,
      context,
      topScore: Number(topScore.toFixed(3)),
      expandedTerms: query.map((q) => q.term),
      tookMs: Number((performance.now() - started).toFixed(2)),
    };
    this.memo.set(key, result);
    return result;
  }

  stats() {
    return {
      version: this.version,
      loadedAt: this.loadedAt,
      sectionCount: this.sections.size,
      files: this.docs.map((d) => ({ name: d.file, title: d.title, sections: d.sections.length })),
    };
  }
}

function truncateTokens(text, maxTokens) {
  if (estimateTokens(text) <= maxTokens) return text;
  const approxChars = maxTokens * 4;
  const cut = text.slice(0, approxChars);
  const lastBreak = Math.max(cut.lastIndexOf('\n'), cut.lastIndexOf('. '));
  return (lastBreak > approxChars * 0.6 ? cut.slice(0, lastBreak + 1) : cut) + ' …';
}
