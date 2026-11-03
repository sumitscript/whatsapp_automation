// Query the vectorless knowledge base from the CLI:  npm run kb:search -- "where does sumit work?"
import { config } from '../src/core/config.js';
import { KnowledgeBase } from '../src/rag/kb.js';

const query = process.argv.slice(2).join(' ');
if (!query) {
  console.error('usage: npm run kb:search -- "<question>"');
  process.exit(1);
}
const kb = await new KnowledgeBase({ ...config.kb }).load();
const r = kb.search(query);
console.log(`query: ${query}\nterms: ${r.expandedTerms.join(', ')}\ntop score: ${r.topScore}  (min ${kb.minScore})  took ${r.tookMs}ms\n`);
for (const s of r.sections) console.log(`${s.score.toFixed(2).padStart(6)}  ${s.path}`);
if (!r.sections.length) console.log('(no section above threshold — the bot will answer without notes)');
