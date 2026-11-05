#!/usr/bin/env node
// Turn a llm-matrix results.json into a question-by-question markdown file.
//   node scripts/matrix-qa.js eval-results/run1/results.json [--models a,b] [--out file.md]
import fs from 'node:fs';
import { renderQa } from './matrix-lib.js';

const argv = process.argv.slice(2);
const file = argv.find((a) => !a.startsWith('--'));
if (!file) {
  console.error('Usage: node scripts/matrix-qa.js <results.json> [--models a,b] [--out file.md]');
  process.exit(2);
}
const opt = (name) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
};
const data = JSON.parse(fs.readFileSync(file, 'utf8'));
const models = opt('models')?.split(',').map((s) => s.trim());
const md = renderQa(data, { models });
if (opt('out')) {
  fs.writeFileSync(opt('out'), md);
  console.log(`Wrote ${opt('out')}`);
} else process.stdout.write(md);
