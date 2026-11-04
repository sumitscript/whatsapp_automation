#!/usr/bin/env node
// Compare Ollama models on a prompt suite through a RUNNING console's playground
// (the real reply pipeline: intent, KB, history, persona), without touching the
// live model setting. Writes report.txt, report.md and results.json.
//
//   WA_COOKIE=<wa_session value> node scripts/llm-matrix.js --base https://host \
//       --models gemma3:4b,qwen3:4b          (or --models all)
//       [--suite test/eval/matrix50.json,extra.md,more.txt] [--out dir] [--only category]
//       [--limit N] [--timeout ms] [--dry-run]
//
// Needs the models to be installed already (it never pulls). The cookie is read from the
// environment only and is never written to the reports.
import fs from 'node:fs';
import path from 'node:path';
import { aggregate, isScored, loadSuite, renderJson, renderMarkdown, renderText, scoreTurn } from './matrix-lib.js';

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue;
    const k = argv[i].slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) a[k] = true;
    else {
      a[k] = next;
      i++;
    }
  }
  return a;
}

const args = parseArgs(process.argv.slice(2));
const fail = (msg, code = 2) => {
  console.error(msg);
  process.exit(code);
};

const base = String(args.base || process.env.WA_BASE || '').replace(/\/+$/, '');
const suiteFiles = String(args.suite || 'test/eval/matrix50.json').split(',').map((s) => s.trim()).filter(Boolean);
let cases;
try {
  cases = loadSuite(suiteFiles);
} catch (err) {
  fail(`Could not load the suite: ${err.message}`);
}
if (args.only) cases = cases.filter((c) => c.category.toLowerCase().includes(String(args.only).toLowerCase()) || c.id.includes(String(args.only)));
if (args.limit) cases = cases.slice(0, Number(args.limit));
const scoredPerModel = cases.reduce((n, c) => n + c.turns.filter(isScored).length, 0);

if (args['dry-run']) {
  const cats = {};
  for (const c of cases) cats[c.category] = (cats[c.category] || 0) + c.turns.filter(isScored).length;
  console.log(`${cases.length} cases, ${cases.reduce((n, c) => n + c.turns.length, 0)} turns, ${scoredPerModel} scored`);
  console.log(cats);
  process.exit(0);
}

if (!base) fail('Pass --base https://your-console-host (or set WA_BASE).');
const rawCookie = process.env.WA_COOKIE || '';
if (!rawCookie) fail('Set WA_COOKIE to your wa_session cookie value (browser DevTools → Application → Cookies).');
const cookie = rawCookie.startsWith('wa_session=') ? rawCookie : `wa_session=${rawCookie}`;
const timeoutMs = Number(args.timeout) || 300_000;

async function call(method, route, body, ms = timeoutMs) {
  const res = await fetch(base + route, {
    method,
    headers: { cookie, ...(body ? { 'content-type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(ms),
  });
  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    /* not JSON */
  }
  if (!res.ok) throw new Error(`${method} ${route} → ${res.status}: ${(json?.error || text).slice(0, 200)}`);
  return json;
}

let installed;
let liveModel;
try {
  installed = (await call('GET', '/api/models', undefined, 20_000)).installed.map((m) => m.name);
  liveModel = (await call('GET', '/api/settings', undefined, 20_000)).settings.model;
} catch (err) {
  fail(`Cannot reach the console: ${err.message}`);
}

const wanted = !args.models || args.models === 'all' ? installed : String(args.models).split(',').map((s) => s.trim()).filter(Boolean);
const missing = wanted.filter((m) => !installed.includes(m));
if (missing.length) fail(`Not installed on the server: ${missing.join(', ')}\nInstalled: ${installed.join(', ')}`);

const startedAt = new Date().toISOString();
const outDir = String(args.out || path.join('eval-results', startedAt.replace(/[:.]/g, '-')));
fs.mkdirSync(outDir, { recursive: true });
const meta = { suite: suiteFiles.map((f) => path.basename(f)).join(' + '), base, startedAt, cases: cases.length, scored: scoredPerModel, models: wanted, liveModel };
const rows = [];

function writeReports() {
  const payload = { meta, stats: aggregate(rows), rows };
  fs.writeFileSync(path.join(outDir, 'results.json'), renderJson(payload));
  fs.writeFileSync(path.join(outDir, 'report.md'), renderMarkdown(payload));
  fs.writeFileSync(path.join(outDir, 'report.txt'), renderText(payload));
}

console.log(`Matrix: ${wanted.length} models × ${scoredPerModel} scored prompts (${cases.length} cases). Output: ${outDir}`);
console.log(`Live model (restored at the end): ${liveModel}\n`);

for (const model of wanted) {
  console.log(`=== ${model} ===`);
  try {
    await call('POST', '/api/playground', { message: 'hi', model }); // load the model; this call is not timed or scored
  } catch (err) {
    console.log(`  warm-up failed: ${err.message}`);
  }
  let consecutiveErrors = 0;
  let done = 0;
  for (const c of cases) {
    const history = (c.seed || []).map((t) => ({ role: t.role, content: t.content }));
    let prevReply = [...history].reverse().find((t) => t.role === 'assistant')?.content ?? '';
    for (const [i, spec] of c.turns.entries()) {
      const scored = isScored(spec);
      const row = { model, caseId: c.id, category: c.category, turn: i + 1, user: spec.user, scored, pass: false, failures: [] };
      const started = Date.now();
      try {
        const r = await call('POST', '/api/playground', { message: spec.user, history, model, contactName: c.contactName, longTermMemory: c.longTermMemory });
        Object.assign(row, { reply: r.reply ?? '', intent: r.intent, inheritedIntent: !!r.inheritedIntent, sources: (r.sources || []).length, language: r.language, latencyMs: r.latencyMs ?? Date.now() - started, clientMs: Date.now() - started, tokensPerSec: r.tokensPerSec ?? null });
        const s = scoreTurn(spec, { reply: r.reply, meta: r, bubbles: r.bubbles }, { prevReply });
        row.pass = s.pass;
        row.failures = s.failures;
        history.push({ role: 'user', content: spec.user }, { role: 'assistant', content: r.reply ?? '' });
        prevReply = r.reply ?? '';
        consecutiveErrors = 0;
      } catch (err) {
        row.error = err.message;
        row.reply = '';
        row.clientMs = Date.now() - started;
        consecutiveErrors++;
        history.push({ role: 'user', content: spec.user }, { role: 'assistant', content: '' });
      }
      rows.push(row);
      if (scored) done++;
      const tag = row.error ? 'ERR ' : !scored ? 'obs ' : row.pass ? 'PASS' : 'FAIL';
      console.log(`  ${String(done).padStart(2)}/${scoredPerModel} ${tag} ${c.id}${i ? ` t${i + 1}` : ''} ${row.latencyMs ? `${(row.latencyMs / 1000).toFixed(1)}s` : ''}${row.failures[0] ? `  — ${row.failures[0]}` : ''}${row.error ? `  — ${row.error}` : ''}`);
      if (consecutiveErrors >= 4) break;
    }
    if (consecutiveErrors >= 4) {
      console.log('  too many consecutive errors — skipping the rest of this model');
      break;
    }
  }
  writeReports(); // progress survives a crash or Ctrl+C
}

// Reload the production model into Ollama (testing other models evicted it).
try {
  await call('POST', '/api/playground', { message: 'hi' });
  console.log(`\nRestored: ${liveModel} is loaded again.`);
} catch (err) {
  console.log(`\nCould not reload ${liveModel}: ${err.message} (the first real reply will load it)`);
}

writeReports();
const stats = aggregate(rows);
console.log('\nLEADERBOARD');
for (const s of [...stats].sort((a, b) => b.passRate - a.passRate)) {
  console.log(`  ${s.model.padEnd(18)} ${s.passed}/${s.scored} (${Math.round(s.passRate * 100)}%)  median ${s.medianLatencyMs ? (s.medianLatencyMs / 1000).toFixed(1) : '–'}s  ${s.errors} errors`);
}
console.log(`\nReports: ${path.join(outDir, 'report.md')}  ${path.join(outDir, 'report.txt')}  ${path.join(outDir, 'results.json')}`);
