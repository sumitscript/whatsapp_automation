#!/usr/bin/env node
// Re-score a finished (or partial) llm-matrix run with the CURRENT checks, without calling any model,
// and write fresh reports + the question-by-question file.
//   node scripts/matrix-rescore.js eval-results/run1/results.json --suite test/eval/matrix50.json,... --out eval-results/run1-rescored [--models a,b]
import fs from 'node:fs';
import path from 'node:path';
import { aggregate, isScored, loadSuite, renderJson, renderMarkdown, renderQa, renderText, scoreTurn } from './matrix-lib.js';

const argv = process.argv.slice(2);
const file = argv.find((a) => !a.startsWith('--'));
const opt = (n) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 ? argv[i + 1] : undefined;
};
if (!file || !opt('suite') || !opt('out')) {
  console.error('Usage: node scripts/matrix-rescore.js <results.json> --suite a.json,b.txt --out <dir> [--models a,b]');
  process.exit(2);
}
const data = JSON.parse(fs.readFileSync(file, 'utf8'));
const cases = new Map(loadSuite(opt('suite').split(',').map((s) => s.trim())).map((c) => [c.id, c]));
const only = opt('models')?.split(',').map((s) => s.trim());

const rows = data.rows.filter((r) => !only || only.includes(r.model)).map((r) => ({ ...r }));
const lastReply = new Map(); // model|case → previous assistant reply
for (const r of rows) {
  const spec = cases.get(r.caseId)?.turns[r.turn - 1];
  const key = `${r.model}|${r.caseId}`;
  if (spec && !r.error) {
    const s = scoreTurn(spec, { reply: r.reply, meta: { intent: r.intent, inheritedIntent: r.inheritedIntent, sources: new Array(r.sources || 0).fill(0) } }, { prevReply: lastReply.get(key) ?? '' });
    r.pass = s.pass;
    r.failures = s.failures;
    r.scored = isScored(spec);
  }
  lastReply.set(key, r.reply ?? '');
}
const models = only ?? [...new Set(rows.map((r) => r.model))];
const meta = { ...data.meta, models, rescored: true };
const payload = { meta, stats: aggregate(rows), rows };
fs.mkdirSync(opt('out'), { recursive: true });
fs.writeFileSync(path.join(opt('out'), 'results.json'), renderJson(payload));
fs.writeFileSync(path.join(opt('out'), 'report.md'), renderMarkdown(payload));
fs.writeFileSync(path.join(opt('out'), 'report.txt'), renderText(payload));
fs.writeFileSync(path.join(opt('out'), 'questions-and-answers.md'), renderQa(payload, { models }));
for (const s of [...payload.stats].sort((a, b) => b.passRate - a.passRate)) console.log(`${s.model.padEnd(16)} ${s.passed}/${s.scored} (${Math.round(s.passRate * 100)}%) median ${s.medianLatencyMs ? (s.medianLatencyMs / 1000).toFixed(1) : '–'}s`);
console.log(`Wrote ${opt('out')}/{report.md,report.txt,results.json,questions-and-answers.md}`);
