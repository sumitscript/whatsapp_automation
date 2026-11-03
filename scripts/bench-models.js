// Benchmark Ollama models on THIS machine: time-to-first-token and tokens/sec
// across English, Hinglish, Hindi and coding prompts.
//   npm run bench                         → every installed model
//   npm run bench -- gemma3:4b qwen3:4b   → specific models (pulled if missing)
import { config } from '../src/core/config.js';
import { OllamaClient } from '../src/llm/ollama.js';

const PROMPTS = [
  { label: 'english', text: 'In two sentences, what is retrieval-augmented generation?' },
  { label: 'hinglish', text: 'Bhai ek chhota sa joke sunao programmers pe' },
  { label: 'hindi', text: 'भारत की राजधानी क्या है? एक वाक्य में बताइए।' },
  { label: 'coding', text: 'Write a JavaScript function that debounces another function. Code only.' },
];

const llm = new OllamaClient({ ...config.ollama, timeoutMs: 10 * 60_000 });
let models = process.argv.slice(2);
if (!models.length) models = (await llm.listModels()).map((m) => m.name);
if (!models.length) {
  console.error('No models installed. Try: npm run bench -- gemma3:4b');
  process.exit(1);
}

const rows = [];
for (const model of models) {
  await llm.ensureModel(model, { autoPull: true, onProgress: (e) => e.total && process.stdout.write(`\rpulling ${model} ${Math.round((e.completed / e.total) * 100)}%   `) });
  const loadMs = await llm.warmup(model);
  console.log(`\n== ${model} (load ${loadMs} ms)`);
  for (const p of PROMPTS) {
    const t0 = performance.now();
    let ttft = null;
    let out = '';
    let stats = {};
    for await (const ev of llm.chat({ model, messages: [{ role: 'user', content: p.text }], temperature: 0.4, maxTokens: 200 })) {
      if (ev.type === 'token') {
        ttft ??= performance.now() - t0;
        out += ev.text;
      } else stats = ev.stats;
    }
    const tps = stats.evalMs ? (stats.evalTokens / stats.evalMs) * 1000 : 0;
    rows.push({ model, prompt: p.label, ttftMs: Math.round(ttft ?? 0), tokPerSec: Number(tps.toFixed(1)), totalMs: Math.round(performance.now() - t0) });
    console.log(`-- ${p.label}: ${out.trim().slice(0, 160).replace(/\s+/g, ' ')}`);
  }
}
console.log('\n');
console.table(rows);
