// Pure helpers for scripts/eval.js (no I/O except loadCases, no npm dependencies).
import { readFileSync } from 'node:fs';

export const CHECKS = ['intent', 'matches', 'notMatches', 'maxChars', 'maxBubbles', 'usesKb', 'scriptOf', 'inheritedIntent'];

const SCRIPTS = {
  latin: /\p{Script=Latin}/u,
  devanagari: /\p{Script=Devanagari}/u,
  arabic: /\p{Script=Arabic}/u,
  bengali: /\p{Script=Bengali}/u,
  gurmukhi: /\p{Script=Gurmukhi}/u,
  gujarati: /\p{Script=Gujarati}/u,
  tamil: /\p{Script=Tamil}/u,
  telugu: /\p{Script=Telugu}/u,
  kannada: /\p{Script=Kannada}/u,
  malayalam: /\p{Script=Malayalam}/u,
  cyrillic: /\p{Script=Cyrillic}/u,
  han: /\p{Script=Han}/u,
};
export const SCRIPT_NAMES = Object.keys(SCRIPTS);

/** The script with the most letters in `text` (emoji, digits and punctuation ignored), or null. */
export function majorScript(text) {
  const counts = {};
  for (const ch of String(text ?? '')) {
    if (!/\p{L}/u.test(ch)) continue;
    for (const [name, re] of Object.entries(SCRIPTS)) {
      if (re.test(ch)) {
        counts[name] = (counts[name] || 0) + 1;
        break;
      }
    }
  }
  const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return best ? best[0] : null;
}

/**
 * Validate a parsed cases file. Returns an array of problems (empty = valid).
 * Every regex string is compiled so typos surface before any model runs.
 */
export function validateCases(data) {
  const problems = [];
  if (!data || !Array.isArray(data.cases)) return ['top-level "cases" array is missing'];
  const ids = new Set();
  data.cases.forEach((c, i) => {
    const where = `cases[${i}]${c?.id ? ` (${c.id})` : ''}`;
    if (!c || typeof c.id !== 'string' || !c.id) problems.push(`${where}: missing string id`);
    else if (ids.has(c.id)) problems.push(`${where}: duplicate id`);
    else ids.add(c.id);
    if (!Array.isArray(c?.turns) || !c.turns.length) {
      problems.push(`${where}: turns must be a non-empty array`);
      return;
    }
    c.turns.forEach((t, j) => {
      const tw = `${where} turn ${j + 1}`;
      if (typeof t?.user !== 'string' || !t.user.trim()) problems.push(`${tw}: missing "user" text`);
      for (const key of ['matches', 'notMatches']) {
        if (t[key] === undefined) continue;
        if (!Array.isArray(t[key])) {
          problems.push(`${tw}: ${key} must be an array of regex strings`);
          continue;
        }
        for (const src of t[key]) {
          try {
            new RegExp(src, 'iu');
          } catch (err) {
            problems.push(`${tw}: bad regex ${JSON.stringify(src)} in ${key} (${err.message})`);
          }
        }
      }
      for (const key of ['maxChars', 'maxBubbles']) if (t[key] !== undefined && !(Number.isFinite(t[key]) && t[key] > 0)) problems.push(`${tw}: ${key} must be a positive number`);
      for (const key of ['usesKb', 'inheritedIntent']) if (t[key] !== undefined && typeof t[key] !== 'boolean') problems.push(`${tw}: ${key} must be true or false`);
      if (t.intent !== undefined && typeof t.intent !== 'string') problems.push(`${tw}: intent must be a string`);
      if (t.scriptOf !== undefined && !SCRIPT_NAMES.includes(t.scriptOf)) problems.push(`${tw}: scriptOf must be one of ${SCRIPT_NAMES.join('|')}`);
    });
  });
  return problems;
}

/** Read, parse and validate a cases file; throws one Error listing every problem. */
export function loadCases(file) {
  let data;
  try {
    data = JSON.parse(readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`cannot read cases file ${file}: ${err.message}`);
  }
  const problems = validateCases(data);
  if (problems.length) throw new Error(`invalid cases file ${file}:\n  - ${problems.join('\n  - ')}`);
  return data;
}

/**
 * Check one turn.
 * @param {object} spec the turn from the cases file
 * @param {{ reply: string, bubbles?: string[], meta?: object }} result
 * @returns {{ pass: boolean, failures: string[] }}
 */
export function checkTurn(spec, result) {
  const reply = String(result.reply ?? '');
  const meta = result.meta || {};
  const bubbles = result.bubbles ?? (reply ? reply.split(/\n{2,}/) : []);
  const failures = [];

  if (spec.intent !== undefined && meta.intent !== spec.intent) failures.push(`intent: expected "${spec.intent}", got "${meta.intent}"`);
  if (spec.inheritedIntent !== undefined && !!meta.inheritedIntent !== spec.inheritedIntent) failures.push(`inheritedIntent: expected ${spec.inheritedIntent}, got ${!!meta.inheritedIntent}`);
  for (const src of spec.matches ?? []) if (!new RegExp(src, 'iu').test(reply)) failures.push(`matches: /${src}/ not found`);
  for (const src of spec.notMatches ?? []) {
    const m = new RegExp(src, 'iu').exec(reply);
    if (m) failures.push(`notMatches: /${src}/ found ("${m[0]}")`);
  }
  if (spec.maxChars !== undefined && reply.length > spec.maxChars) failures.push(`maxChars: ${reply.length} > ${spec.maxChars}`);
  if (spec.maxBubbles !== undefined && bubbles.length > spec.maxBubbles) failures.push(`maxBubbles: ${bubbles.length} > ${spec.maxBubbles}`);
  if (spec.usesKb !== undefined) {
    const used = (meta.sources?.length ?? 0) > 0;
    if (used !== spec.usesKb) failures.push(`usesKb: expected ${spec.usesKb}, got ${used} (${meta.sources?.length ?? 0} sources)`);
  }
  if (spec.scriptOf !== undefined) {
    const got = majorScript(reply);
    if (got !== spec.scriptOf) failures.push(`scriptOf: expected ${spec.scriptOf}, got ${got ?? 'none'}`);
  }
  return { pass: failures.length === 0, failures };
}

export function median(nums) {
  const a = nums.filter(Number.isFinite).sort((x, y) => x - y);
  if (!a.length) return null;
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : Math.round((a[mid - 1] + a[mid]) / 2);
}

/**
 * Summarise rows of { model, caseId, turn, pass, latencyMs } into per-model stats.
 * @returns {{ model: string, turns: number, passed: number, failed: number, passRate: number, casesFailed: number, medianLatencyMs: number|null }[]}
 */
export function summarize(rows) {
  const byModel = new Map();
  for (const r of rows) {
    if (!byModel.has(r.model)) byModel.set(r.model, []);
    byModel.get(r.model).push(r);
  }
  return [...byModel].map(([model, rs]) => {
    const passed = rs.filter((r) => r.pass).length;
    return {
      model,
      turns: rs.length,
      passed,
      failed: rs.length - passed,
      passRate: rs.length ? passed / rs.length : 0,
      casesFailed: new Set(rs.filter((r) => !r.pass).map((r) => r.caseId)).size,
      medianLatencyMs: median(rs.map((r) => r.latencyMs)),
    };
  });
}
