import { test } from 'node:test';
import assert from 'node:assert/strict';
import { editDistance, estimateTokens, stem, tokenize } from '../src/rag/text.js';

test('tokenize lowercases, drops stopwords and stems', () => {
  assert.deepEqual(tokenize('What are the Projects he BUILT?'), ['project', 'built']);
});

test('tokenize keeps Indic scripts intact', () => {
  assert.deepEqual(tokenize('सुमित की पढ़ाई'), ['सुमित', 'पढ़ाई']);
});

test('tech names survive tokenisation', () => {
  assert.deepEqual(tokenize('C++ and Node.js and C#'), ['cpp', 'nodejs', 'csharp']);
});

test('stem handles common suffixes', () => {
  assert.equal(stem('libraries'), 'library');
  assert.equal(stem('skills'), 'skill');
  assert.equal(stem('class'), 'class');
  assert.equal(stem('analysis'), 'analysis');
});

test('editDistance is bounded', () => {
  assert.equal(editDistance('project', 'projct'), 1);
  assert.equal(editDistance('kitten', 'sitting', 2), 3);
});

test('estimateTokens is roughly chars/4 for latin', () => {
  assert.equal(estimateTokens('a'.repeat(400)), 100);
});
