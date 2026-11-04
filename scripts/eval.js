// Conversation-quality regression harness: replays multi-turn conversations against a LIVE
// Ollama model through the real reply pipeline and checks the answers.
//   npm run eval                                       -> default model (LLM_MODEL), all cases
//   npm run eval -- --model gemma3:4b,llama3.2:3b      -> compare models
//   npm run eval -- --only joke --json out.json        -> subset, machine-readable results
// Models must already be installed (never pulled). Writes only to stdout and the optional --json file.
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { config } from '../src/core/config.js';
import { OllamaClient } from '../src/llm/ollama.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { DEFAULT_SETTINGS } from '../src/core/settings-schema.js';
import { checkTurn, loadCases, summarize } from './eval-lib.js';

const USAGE = 'usage: npm run eval -- [--model a,b] [--cases file] [--only substring] [--json out.json] [--temperature N]';

function fail(msg) {
  console.error(msg);
  process.exit(2);
}

function parseArgs(argv) {
  const opts = { model: null, cases: 'test/eval/cases.json', only: null, json: null, temperature: null };
  for (let i = 0; i < argv.length; i++) {
    const [flag, inline] = argv[i].split(/=(.*)/s);
    const val = () => inline ?? argv[++i];
    if (flag === '--model' || flag === '--models') opts.model = val();
    else if (flag === '--cases') opts.cases = val();
    else if (flag === '--only') opts.only = val();
    else if (flag === '--json') opts.json = val();
    else if (flag === '--temperature') opts.temperature = Number(val());
    else if (flag === '--help' || flag === '-h') {
      console.log(USAGE);
      process.exit(0);
    } else fail(`unknown argument ${argv[i]}\n${USAGE}`);
  }
  if (opts.temperature !== null && !Number.isFinite(opts.temperature)) fail('--temperature must be a number');
  return opts;
}

const pad = (s, n) => String(s).padEnd(n).slice(0, n);
const oneLine = (s, n = 300) => String(s).replace(/\s+/g, ' ').trim().slice(0, n);

const opts = parseArgs(process.argv.slice(2));
let data;
try {
  data = loadCases(path.resolve(opts.cases));
} catch (err) {
  fail(err.message);
}
const cases = data.cases.filter((c) => !opts.only || c.id.includes(opts.only));
if (!cases.length) fail(`no cases match --only "${opts.only}"`);
const models = (opts.model || config.ollama.defaultModel).split(',').map((s) => s.trim()).filter(Boolean);

// Generous timeout: CPU inference of a long answer can take a while. Keep the model resident while we run.
const llm = new OllamaClient({ ...config.ollama, keepAlive: config.ollama.keepAlive || '30m', timeoutMs: 5 * 60_000 });
try {
  const v = await llm.health();
  console.log(`Ollama ${v.version ?? ''} at ${config.ollama.url}  (num_ctx ${llm.numCtx})`);
} catch (err) {
  fail(`Cannot reach Ollama at ${config.ollama.url} (${err.message}).\nStart it (or set OLLAMA_URL) and try again.`);
}
for (const m of models) {
  if (!(await llm.hasModel(m))) fail(`Model "${m}" is not installed on this Ollama. Install it yourself first, e.g.  ollama pull ${m}\n(the eval never pulls models).`);
}

const kb = await new KnowledgeBase({ ...config.kb }).load();
console.log(`${cases.length} case(s), ${cases.reduce((n, c) => n + c.turns.length, 0)} turn(s), KB ${config.kb.dir}`);

const rows = [];
for (const model of models) {
  const loadMs = await llm.warmup(model);
  console.log(`\n== ${model}  (loaded in ${loadMs} ms)`);
  console.log(`${pad('case', 30)} ${pad('turn', 4)} ${pad('result', 6)} ${pad('intent', 12)} ${pad('latency', 9)} tok/s`);
  const details = [];
  const settings = {
    ...DEFAULT_SETTINGS,
    ownerName: 'Sumit Soni',
    ownerAliases: 'sumit',
    personaName: 'eSumit',
    model,
    ...(opts.temperature !== null ? { temperature: opts.temperature } : {}),
  };
  for (const c of cases) {
    // Fresh memory per case, like a brand-new chat.
    const stores = createMemoryStores();
    const brain = createBrain({ kb, llm, memory: stores.memory, cache: stores.cache });
    const chatId = `eval-${c.id}@c.us`;
    if (c.seed?.length) await stores.memory.append(chatId, c.seed);
    for (const [i, turn] of c.turns.entries()) {
      let reply = '';
      let meta = {};
      const bubbles = [];
      let error = null;
      try {
        for await (const ev of brain.respond({ chatId, text: turn.user, settings, contactName: c.contactName, longTermMemory: c.longTermMemory })) {
          if (ev.type === 'chunk') bubbles.push(ev.text);
          else if (ev.type === 'done') {
            reply = ev.reply;
            meta = ev.meta;
          }
        }
      } catch (err) {
        error = err.message;
      }
      const { pass, failures } = error ? { pass: false, failures: [`error: ${error}`] } : checkTurn(turn, { reply, bubbles, meta });
      rows.push({
        model, caseId: c.id, turn: i + 1, user: turn.user, reply, bubbles: bubbles.length,
        intent: meta.intent ?? null, inheritedIntent: !!meta.inheritedIntent, sources: meta.sources?.length ?? 0,
        latencyMs: meta.latencyMs ?? null, tokensPerSec: meta.tokensPerSec ?? null, pass, failures,
      });
      console.log(`${pad(c.id, 30)} ${pad(i + 1, 4)} ${pad(pass ? 'PASS' : 'FAIL', 6)} ${pad(meta.intent ?? '-', 12)} ${pad(meta.latencyMs != null ? `${meta.latencyMs} ms` : '-', 9)} ${meta.tokensPerSec ?? '-'}`);
      if (!pass) details.push({ c, i, turn, reply, failures });
      if (error) break; // later turns depend on this one
    }
  }
  if (details.length) {
    console.log(`\n-- failures for ${model}`);
    for (const d of details) {
      console.log(`[${d.c.id} #${d.i + 1}] user: ${oneLine(d.turn.user, 120)}\n  reply: ${oneLine(d.reply)}`);
      for (const f of d.failures) console.log(`  x ${f}`);
    }
  }
}

console.log('\n== summary');
const summary = summarize(rows);
for (const s of summary) {
  console.log(`${pad(s.model, 24)} ${s.passed}/${s.turns} turns passed (${(s.passRate * 100).toFixed(1)}%), ${s.casesFailed} case(s) failing, median latency ${s.medianLatencyMs ?? '-'} ms`);
}
if (opts.json) {
  writeFileSync(opts.json, JSON.stringify({ ranAt: new Date().toISOString(), summary, rows }, null, 2));
  console.log(`results written to ${opts.json}`);
}
process.exit(rows.some((r) => !r.pass) ? 1 : 0);
