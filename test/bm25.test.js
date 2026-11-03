import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BM25FIndex } from '../src/rag/bm25.js';

const q = (...terms) => terms.map((term) => ({ term, weight: 1 }));

test('title matches outrank body matches', () => {
  const idx = new BM25FIndex({ weights: { title: 3, tags: 2, body: 1 } });
  idx.add('a', { title: ['payment'], tags: [], body: ['api', 'go'] });
  idx.add('b', { title: ['misc'], tags: [], body: ['payment', 'note'] });
  idx.build();
  const r = idx.search(q('payment'));
  assert.equal(r[0].id, 'a');
  assert.equal(r.length, 2);
});

test('documents covering more query terms win', () => {
  const idx = new BM25FIndex();
  idx.add('a', { title: [], tags: [], body: ['go', 'payment'] });
  idx.add('b', { title: [], tags: [], body: ['go', 'go', 'go'] });
  idx.build();
  assert.equal(idx.search(q('go', 'payment'))[0].id, 'a');
});

test('fuzzy finds near-miss vocabulary terms', () => {
  const idx = new BM25FIndex();
  idx.add('a', { title: ['experience'], tags: [], body: [] });
  idx.build();
  assert.equal(idx.fuzzy('experiance'), 'experience');
  assert.equal(idx.fuzzy('zzzzzzz'), null);
});

test('unknown terms return nothing', () => {
  const idx = new BM25FIndex();
  idx.add('a', { title: ['x'], tags: [], body: [] });
  idx.build();
  assert.deepEqual(idx.search(q('nothing')), []);
});
