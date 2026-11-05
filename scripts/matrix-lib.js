// Pure helpers for scripts/llm-matrix.js: suite loading (.json / .txt / .md),
// scoring (eval-lib checks + a few extra ones), aggregation and report rendering
// (txt / md / json). No npm dependencies, no I/O except loadSuite.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { checkTurn, CHECKS, median } from './eval-lib.js';

export const EXTRA_CHECKS = ['noRepeat', 'json', 'passesOn', 'minChars', 'notEmpty'];
const ALL_CHECKS = new Set([...CHECKS, ...EXTRA_CHECKS]);

// A reply that hands a request to the real owner (English + Hinglish)…
export const PASS_ON =
  /pass(ing)?\s+(it|that|this|the (message|request))?\s*(on|along)|let\s+(the\s+real\s+)?\p{L}+\s+know|check\s+with|get\s+back\s+to\s+you|will\s+confirm|(real|actual)\s+\p{L}+\s+(will|can|should)|ask\s+(the\s+real\s+)?\p{L}+|bata\w*|confirm\s+kar|dekh\w*|pata\s+kar/iu;
// …and one that commits on the owner's behalf (must not happen).
export const COMMITS =
  /see you (there|then|tomorrow)|i['’]?ll be there|i will be there|it['’]?s a (date|deal)|sure,? (i['’]?ll|i will) (be|come|meet)|\bconfirmed\b|pakka|done deal|zaroor aaunga/iu;

// Models that think out loud (qwen3 family, deepseek-r1…) can leak their chain of thought as the reply.
// That is never an acceptable chat answer, whatever words it happens to contain.
export const REASONING_LEAK =
  /\b(the user (asked|said|wants|didn['’]?t|just|has|is asking|says)|my facts block|the (system|instructions?) (says?|prompt)|we are (in a|given|going)|i must (not )?(reply|answer|use|start|be careful)|let me (think|recall|check the)|first, (i|we) (need|must|should))|<think>/iu;

const words = (s) => String(s).toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];

/** Word-set overlap 0..1 (Jaccard). */
export function similarity(a, b) {
  const A = new Set(words(a));
  const B = new Set(words(b));
  if (!A.size || !B.size) return 0;
  let same = 0;
  for (const w of A) if (B.has(w)) same++;
  return same / (A.size + B.size - same);
}

/** Does the reply contain valid JSON (whole reply, a ``` fenced block, or an embedded {...}/[...])? */
export function containsJson(reply) {
  const s = String(reply ?? '').trim();
  const candidates = [s];
  const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(s);
  if (fence) candidates.push(fence[1].trim());
  for (const [open, close] of [['{', '}'], ['[', ']']]) {
    const a = s.indexOf(open);
    const b = s.lastIndexOf(close);
    if (a >= 0 && b > a) candidates.push(s.slice(a, b + 1));
  }
  return candidates.some((c) => {
    try {
      const v = JSON.parse(c);
      return v !== null && typeof v === 'object';
    } catch {
      return false;
    }
  });
}

/** A turn is scored when it carries at least one check. */
export const isScored = (spec) => Object.keys(spec).some((k) => ALL_CHECKS.has(k));

/**
 * @param {object} spec turn spec (checks)
 * @param {{ reply: string, meta?: object, bubbles?: string[] }} result
 * @param {{ prevReply?: string }} [ctx]
 */
export function scoreTurn(spec, result, ctx = {}) {
  const reply = String(result.reply ?? '');
  const { failures } = checkTurn(spec, result);
  const leak = REASONING_LEAK.exec(reply);
  if (leak) failures.push(`leak: the reply is the model's own reasoning ("${leak[0]}")`);
  if (spec.notEmpty && !reply.trim()) failures.push('notEmpty: reply is empty');
  if (spec.minChars !== undefined && reply.length < spec.minChars) failures.push(`minChars: ${reply.length} < ${spec.minChars}`);
  if (spec.json && !containsJson(reply)) failures.push('json: no valid JSON in the reply');
  if (spec.noRepeat && ctx.prevReply) {
    const sim = similarity(reply, ctx.prevReply);
    if (sim >= 0.7) failures.push(`noRepeat: ${Math.round(sim * 100)}% the same as the previous reply`);
  }
  if (spec.passesOn) {
    if (!PASS_ON.test(reply)) failures.push('passesOn: does not pass the request on to the real owner');
    const m = COMMITS.exec(reply);
    if (m) failures.push(`passesOn: commits on the owner's behalf ("${m[0]}")`);
  }
  return { pass: failures.length === 0, failures, scored: isScored(spec) };
}

// ------------------------------------------------------------------ loading

/** `a >> b >> c` = a three-turn conversation. */
const turnsOf = (line) =>
  line
    .split('>>')
    .map((t) => t.trim())
    .filter(Boolean)
    .map((user) => ({ user }));

/**
 * Plain text: one prompt per line, optionally `category | prompt`; `a >> b` chains turns;
 * blank lines and `#` comments are skipped. Prompts have no checks, so they are "observed" (not scored).
 */
export function parseTxt(text, name = 'txt') {
  const cases = [];
  let n = 0;
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const bar = line.indexOf('|');
    const category = bar > 0 && bar < 32 ? line.slice(0, bar).trim() : 'extra';
    const body = bar > 0 && bar < 32 ? line.slice(bar + 1) : line;
    const turns = turnsOf(body);
    if (turns.length) cases.push({ id: `${name}-${++n}`, category, turns });
  }
  return cases;
}

/** Markdown: `## Category` headings, `- prompt` / `1. prompt` items (`>>` chains turns); fenced code is ignored. */
export function parseMd(text, name = 'md') {
  const cases = [];
  let category = 'extra';
  let fenced = false;
  let n = 0;
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (line.startsWith('```')) {
      fenced = !fenced;
      continue;
    }
    if (fenced) continue;
    const h = /^#{1,6}\s+(.+)$/.exec(line);
    if (h) {
      category = h[1].trim();
      continue;
    }
    const item = /^(?:[-*+]|\d+[.)])\s+(.+)$/.exec(line);
    if (!item) continue;
    const turns = turnsOf(item[1]);
    if (turns.length) cases.push({ id: `${name}-${++n}`, category, turns });
  }
  return cases;
}

/** Load one or more suite files; the format is chosen by extension. */
export function loadSuite(files) {
  const cases = [];
  for (const file of files) {
    const text = readFileSync(file, 'utf8');
    const ext = path.extname(file).toLowerCase();
    const name = `${path.basename(file, ext)}-${ext.replace('.', '') || 'txt'}`; // extra.txt and extra.md must not collide
    if (ext === '.json') {
      const data = JSON.parse(text);
      cases.push(...(Array.isArray(data) ? data : data.cases ?? []));
    } else if (ext === '.md' || ext === '.markdown') cases.push(...parseMd(text, name));
    else cases.push(...parseTxt(text, name));
  }
  const ids = new Set();
  for (const c of cases) {
    if (!c.id || !Array.isArray(c.turns) || !c.turns.length) throw new Error(`Invalid case: ${JSON.stringify(c).slice(0, 80)}`);
    if (ids.has(c.id)) throw new Error(`Duplicate case id "${c.id}"`);
    ids.add(c.id);
    c.category ||= 'extra';
  }
  return cases;
}

// -------------------------------------------------------------- aggregation

export function percentile(nums, p) {
  const a = nums.filter(Number.isFinite).sort((x, y) => x - y);
  if (!a.length) return null;
  return a[Math.min(a.length - 1, Math.ceil((p / 100) * a.length) - 1)];
}

const mean = (nums) => {
  const a = nums.filter(Number.isFinite);
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
};

/**
 * @param {{ model: string, caseId: string, category: string, scored: boolean, pass: boolean, error?: string,
 *           latencyMs?: number, tokensPerSec?: number }[]} rows one row per executed turn
 */
export function aggregate(rows) {
  const models = [...new Set(rows.map((r) => r.model))];
  return models.map((model) => {
    const rs = rows.filter((r) => r.model === model);
    const scored = rs.filter((r) => r.scored);
    const byCategory = {};
    for (const r of scored) {
      const c = (byCategory[r.category] ??= { n: 0, passed: 0 });
      c.n++;
      if (r.pass) c.passed++;
    }
    const passed = scored.filter((r) => r.pass).length;
    return {
      model,
      turns: rs.length,
      scored: scored.length,
      passed,
      passRate: scored.length ? passed / scored.length : 0,
      errors: rs.filter((r) => r.error).length,
      medianLatencyMs: median(rs.filter((r) => !r.error).map((r) => r.latencyMs)),
      p95LatencyMs: percentile(rs.filter((r) => !r.error).map((r) => r.latencyMs), 95),
      meanTokensPerSec: mean(rs.map((r) => r.tokensPerSec)),
      byCategory,
    };
  });
}

// ---------------------------------------------------------------- rendering

const pct = (x) => `${Math.round(x * 100)}%`;
const sec = (ms) => (ms == null ? '–' : `${(ms / 1000).toFixed(1)}s`);
const tps = (x) => (x == null ? '–' : x.toFixed(1));
const clip = (s, n) => {
  const t = String(s ?? '').replace(/\s+/g, ' ').trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
};

const ranked = (stats) => [...stats].sort((a, b) => b.passRate - a.passRate || (a.medianLatencyMs ?? 1e9) - (b.medianLatencyMs ?? 1e9));
const categories = (stats) => [...new Set(stats.flatMap((s) => Object.keys(s.byCategory)))];

/** @param {{ meta: object, stats: object[], rows: object[] }} r */
export function renderMarkdown({ meta, stats, rows }) {
  const out = [`# LLM matrix — ${meta.suite}`, '', `Run ${meta.startedAt} against ${meta.base}. ${meta.cases} cases, ${meta.scored} scored prompts per model.`, ''];
  out.push('## Leaderboard', '', '| # | Model | Pass | Median | p95 | tok/s | Errors |', '|---|---|---|---|---|---|---|');
  ranked(stats).forEach((s, i) => out.push(`| ${i + 1} | ${s.model} | ${s.passed}/${s.scored} (${pct(s.passRate)}) | ${sec(s.medianLatencyMs)} | ${sec(s.p95LatencyMs)} | ${tps(s.meanTokensPerSec)} | ${s.errors} |`));
  const cats = categories(stats);
  out.push('', '## By category (passed/scored)', '', `| Category | ${stats.map((s) => s.model).join(' | ')} |`, `|---|${stats.map(() => '---').join('|')}|`);
  for (const c of cats) out.push(`| ${c} | ${stats.map((s) => (s.byCategory[c] ? `${s.byCategory[c].passed}/${s.byCategory[c].n}` : '–')).join(' | ')} |`);
  for (const s of ranked(stats)) {
    const mine = rows.filter((r) => r.model === s.model);
    const failed = mine.filter((r) => r.scored && !r.pass);
    out.push('', `## ${s.model}`, '', failed.length ? `### Failed checks (${failed.length})` : '### No failed checks', '');
    for (const r of failed) out.push(`- **${r.caseId}** (${r.category}) — "${clip(r.user, 80)}"`, `  - ${r.error ? `ERROR: ${r.error}` : r.failures.join('; ')}`, `  - reply: ${clip(r.reply, 220)}`);
    out.push('', '### Transcript', '');
    for (const r of mine) out.push(`**[${r.caseId}${r.turn > 1 ? ` t${r.turn}` : ''}]** ${r.scored ? (r.pass ? '✅' : '❌') : '·'} \`${r.intent ?? '?'}\` ${sec(r.latencyMs)}`, `> ${clip(r.user, 300)}`, '', r.error ? `ERROR: ${r.error}` : String(r.reply).split('\n').map((l) => `    ${l}`).join('\n'), '');
  }
  return `${out.join('\n')}\n`;
}

export function renderText({ meta, stats, rows }) {
  const out = [`LLM MATRIX — ${meta.suite}`, `run ${meta.startedAt} | ${meta.base} | ${meta.cases} cases | ${meta.scored} scored prompts per model`, ''];
  out.push('LEADERBOARD', `${'model'.padEnd(18)} ${'pass'.padEnd(14)} ${'median'.padEnd(8)} ${'p95'.padEnd(8)} ${'tok/s'.padEnd(7)} errors`);
  for (const s of ranked(stats)) out.push(`${s.model.padEnd(18)} ${`${s.passed}/${s.scored} ${pct(s.passRate)}`.padEnd(14)} ${sec(s.medianLatencyMs).padEnd(8)} ${sec(s.p95LatencyMs).padEnd(8)} ${tps(s.meanTokensPerSec).padEnd(7)} ${s.errors}`);
  out.push('', 'BY CATEGORY (passed/scored)', `${'category'.padEnd(16)} ${stats.map((s) => s.model.padEnd(16)).join(' ')}`);
  for (const c of categories(stats)) out.push(`${c.padEnd(16)} ${stats.map((s) => (s.byCategory[c] ? `${s.byCategory[c].passed}/${s.byCategory[c].n}` : '-').padEnd(16)).join(' ')}`);
  for (const s of ranked(stats)) {
    out.push('', `=== ${s.model} ===`);
    for (const r of rows.filter((x) => x.model === s.model)) {
      out.push(`[${r.caseId}${r.turn > 1 ? ` t${r.turn}` : ''}] ${r.scored ? (r.pass ? 'PASS' : 'FAIL') : 'obs '} intent=${r.intent ?? '?'} ${sec(r.latencyMs)}`, `  Q: ${clip(r.user, 200)}`, `  A: ${r.error ? `ERROR ${r.error}` : clip(r.reply, 400)}`);
      if (r.scored && !r.pass) out.push(`  ! ${r.failures.join('; ')}`);
    }
  }
  return `${out.join('\n')}\n`;
}

export const renderJson = ({ meta, stats, rows }) => `${JSON.stringify({ meta, stats, rows }, null, 2)}\n`;

/**
 * Question-by-question view: every prompt with each model's answer underneath.
 * @param {{ rows: object[] }} r
 * @param {{ models?: string[] }} [opts] restrict/order the models shown
 */
export function renderQa({ rows }, { models } = {}) {
  const order = models ?? [...new Set(rows.map((r) => r.model))];
  const keys = [];
  const seen = new Set();
  for (const r of rows) {
    const k = `${r.caseId}\u0000${r.turn}`;
    if (!seen.has(k)) {
      seen.add(k);
      keys.push([r.caseId, r.turn]);
    }
  }
  const out = [`# Every question and answer`, '', `Models: ${order.join(', ')}. ✅ passed its checks, ❌ failed, · observed only (no checks).`, ''];
  let lastCat = null;
  let n = 0;
  for (const [caseId, turn] of keys) {
    const here = rows.filter((r) => r.caseId === caseId && r.turn === turn);
    const first = here[0];
    if (first.category !== lastCat) {
      lastCat = first.category;
      out.push(`## ${lastCat}`, '');
    }
    out.push(`### ${++n}. ${caseId}${turn > 1 ? ` (turn ${turn})` : ''}`, '', '**Question**', '', '```', String(first.user), '```', '');
    for (const m of order) {
      const r = here.find((x) => x.model === m);
      if (!r) continue;
      const mark = r.error ? '⚠️' : r.scored ? (r.pass ? '✅' : '❌') : '·';
      out.push(`**${m}** ${mark} ${r.latencyMs ? sec(r.latencyMs) : ''}${r.intent ? ` · intent \`${r.intent}\`` : ''}`, '');
      out.push(r.error ? `ERROR: ${r.error}` : '```', ...(r.error ? [] : [String(r.reply).trim() || '(empty)', '```']));
      if (r.scored && !r.pass && r.failures?.length) out.push('', `> check failed: ${r.failures.join('; ')}`);
      out.push('');
    }
  }
  return `${out.join('\n')}\n`;
}
