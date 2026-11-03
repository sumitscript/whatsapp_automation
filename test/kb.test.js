import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { KnowledgeBase, parseFrontmatter, parseMarkdown } from '../src/rag/kb.js';

const dir = path.join(import.meta.dirname, 'fixtures', 'kb');
const kb = await new KnowledgeBase({ dir, minScore: 0.5 }).load();

test('frontmatter is parsed', () => {
  const { meta, body } = parseFrontmatter('---\ntitle: Hi\ntags: a, b\n---\n# Body');
  assert.equal(meta.title, 'Hi');
  assert.equal(meta.tags, 'a, b');
  assert.equal(body, '# Body');
});

test('markdown becomes a heading tree; fenced headings are ignored', () => {
  const doc = parseMarkdown('x.md', '# Doc\nsummary\n## A\none\n### A1\ntwo\n```\n## not a heading\n```\n## B\nthree');
  assert.deepEqual(
    doc.sections.map((s) => s.path),
    ['Doc', 'Doc › A', 'Doc › A › A1', 'Doc › B'],
  );
  assert.match(doc.sections[2].text, /not a heading/);
});

test('underscore files are skipped and sections indexed', () => {
  assert.equal(kb.docs.length, 2);
  assert.ok(kb.sections.size >= 4);
  assert.notEqual(kb.version, 'empty');
});

test('retrieves the right section for an English question', () => {
  const r = kb.search('where does Asha work as backend engineer?');
  assert.match(r.sections[0].path, /Acme Corp/);
  assert.match(r.context, /payment APIs/);
});

test('Hinglish synonyms route to the right document', () => {
  const r = kb.search('padhai kahan hui');
  assert.match(r.sections[0].path, /Education/);
});

test('Devanagari aliases match', () => {
  const r = kb.search('नौकरी');
  assert.match(r.sections[0].path, /Work Experience/);
});

test('typos are corrected against the vocabulary', () => {
  const r = kb.search('univrsity cgpa');
  assert.match(r.sections[0].path, /State University/);
});

test('off-topic queries return no sections', () => {
  const r = kb.search('tell me a joke about cats');
  assert.equal(r.sections.length, 0);
});

test('results are memoized per KB version', () => {
  kb.search('globex intern');
  assert.equal(kb.search('globex intern').memo, true);
});
