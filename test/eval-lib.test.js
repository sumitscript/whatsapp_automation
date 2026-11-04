import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { checkTurn, majorScript, summarize, validateCases, median } from '../scripts/eval-lib.js';

const casesFile = path.join(import.meta.dirname, 'eval', 'cases.json');

test('eval cases file parses, ids unique, every regex compiles', () => {
  const data = JSON.parse(readFileSync(casesFile, 'utf8'));
  assert.deepEqual(validateCases(data), []);
  assert.ok(data.cases.length >= 12);
  assert.equal(new Set(data.cases.map((c) => c.id)).size, data.cases.length);
  assert.equal(typeof data._readme, 'string');
});

test('validateCases reports problems', () => {
  const problems = validateCases({ cases: [{ id: 'a', turns: [{ user: 'x', matches: ['('], scriptOf: 'klingon' }] }, { id: 'a', turns: [] }] });
  assert.equal(problems.length, 4);
  assert.ok(problems.some((p) => /bad regex/.test(p)));
  assert.ok(problems.some((p) => /duplicate id/.test(p)));
});

test('checkTurn: intent / inheritedIntent', () => {
  const r = { reply: 'hi', meta: { intent: 'joke', inheritedIntent: true } };
  assert.equal(checkTurn({ intent: 'joke', inheritedIntent: true }, r).pass, true);
  assert.match(checkTurn({ intent: 'math' }, r).failures[0], /intent/);
  assert.match(checkTurn({ inheritedIntent: false }, r).failures[0], /inheritedIntent/);
});

test('checkTurn: matches / notMatches are case-insensitive over the whole reply', () => {
  const r = { reply: 'I am eSumit\n\nnice', meta: {} };
  assert.equal(checkTurn({ matches: ['esumit', 'NICE'] }, r).pass, true);
  assert.equal(checkTurn({ matches: ['bob'] }, r).pass, false);
  assert.equal(checkTurn({ notMatches: ['iamneo'] }, r).pass, true);
  const bad = checkTurn({ notMatches: ['ESUMIT'] }, r);
  assert.equal(bad.pass, false);
  assert.match(bad.failures[0], /notMatches/);
});

test('checkTurn: maxChars / maxBubbles', () => {
  const r = { reply: 'a'.repeat(50), bubbles: ['a', 'b', 'c'], meta: {} };
  assert.equal(checkTurn({ maxChars: 50, maxBubbles: 3 }, r).pass, true);
  assert.equal(checkTurn({ maxChars: 49 }, r).pass, false);
  assert.equal(checkTurn({ maxBubbles: 2 }, r).pass, false);
});

test('checkTurn: usesKb', () => {
  const withKb = { reply: 'x', meta: { sources: [{ path: 'a' }] } };
  const noKb = { reply: 'x', meta: { sources: [] } };
  assert.equal(checkTurn({ usesKb: true }, withKb).pass, true);
  assert.equal(checkTurn({ usesKb: true }, noKb).pass, false);
  assert.equal(checkTurn({ usesKb: false }, noKb).pass, true);
  assert.equal(checkTurn({ usesKb: false }, withKb).pass, false);
});

test('checkTurn: scriptOf ignores emoji and digits', () => {
  assert.equal(majorScript('hello 😀 123'), 'latin');
  assert.equal(majorScript('मैं ठीक हूँ, ok'), 'devanagari');
  assert.equal(majorScript('123 😀'), null);
  assert.equal(checkTurn({ scriptOf: 'latin' }, { reply: 'bhai sab badhiya 😎', meta: {} }).pass, true);
  assert.equal(checkTurn({ scriptOf: 'devanagari' }, { reply: 'bhai sab badhiya', meta: {} }).pass, false);
  assert.equal(checkTurn({ scriptOf: 'devanagari' }, { reply: 'मैं ठीक हूँ 🙂', meta: {} }).pass, true);
});

test('checkTurn collects every failure', () => {
  const { pass, failures } = checkTurn({ intent: 'math', matches: ['4'], maxChars: 1 }, { reply: 'five', meta: { intent: 'general' } });
  assert.equal(pass, false);
  assert.equal(failures.length, 3);
});

test('summarize: pass rate, failing cases, median latency per model', () => {
  const rows = [
    { model: 'a', caseId: 'x', pass: true, latencyMs: 100 },
    { model: 'a', caseId: 'x', pass: false, latencyMs: 300 },
    { model: 'a', caseId: 'y', pass: false, latencyMs: 200 },
    { model: 'b', caseId: 'x', pass: true, latencyMs: null },
  ];
  const [a, b] = summarize(rows);
  assert.deepEqual([a.model, a.turns, a.passed, a.failed, a.casesFailed, a.medianLatencyMs], ['a', 3, 1, 2, 2, 200]);
  assert.equal(b.passRate, 1);
  assert.equal(b.medianLatencyMs, null);
  assert.equal(median([1, 2, 3, 4]), 3);
});
