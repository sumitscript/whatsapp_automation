import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {
  aggregate, containsJson, isScored, loadSuite, parseMd, parseTxt, percentile, renderJson, renderMarkdown, renderText,
  scoreTurn, similarity,
} from '../scripts/matrix-lib.js';

const dir = path.join(import.meta.dirname, 'eval');

test('the 50-prompt suite is valid: unique ids, compiling regexes, exactly 50 scored turns', () => {
  const cases = loadSuite([path.join(dir, 'matrix50.json')]);
  assert.equal(cases.length, 50);
  assert.equal(new Set(cases.map((c) => c.id)).size, 50);
  let scored = 0;
  for (const c of cases) {
    assert.ok(c.category);
    for (const t of c.turns) {
      assert.ok(typeof t.user === 'string' && t.user.length > 0, c.id);
      for (const src of [...(t.matches ?? []), ...(t.notMatches ?? [])]) assert.doesNotThrow(() => new RegExp(src, 'iu'), `${c.id}: ${src}`);
      if (isScored(t)) scored++;
    }
    assert.ok(isScored(c.turns.at(-1)), `${c.id}: the last turn must be scored`);
  }
  assert.equal(scored, 50);
  const cats = new Set(cases.map((c) => c.category));
  for (const need of ['clear', 'poor-prompt', 'unclear-intent', 'typos', 'wrong-grammar', 'jumbled', 'multilingual', 'context', 'appointment', 'task-request', 'format']) assert.ok(cats.has(need), need);
});

test('plain-text suites: categories, chained turns, comments', () => {
  const cases = parseTxt('# comment\n\nchat | hello there\nrecommend a book >> why that one?\n', 'x');
  assert.equal(cases.length, 2);
  assert.deepEqual(cases[0], { id: 'x-1', category: 'chat', turns: [{ user: 'hello there' }] });
  assert.deepEqual(cases[1].turns.map((t) => t.user), ['recommend a book', 'why that one?']);
  assert.equal(cases[1].category, 'extra');
  assert.equal(isScored(cases[0].turns[0]), false); // no checks = observed only
});

test('markdown suites: headings are categories, fenced code is ignored', () => {
  const md = '# Title\n## writing\n- a poem\n1. a quote\n```\n- not a prompt\n```\n## tasks\n* remind me >> and book it\n';
  const cases = parseMd(md, 'm');
  assert.deepEqual(cases.map((c) => [c.category, c.turns.length]), [['writing', 1], ['writing', 1], ['tasks', 2]]);
});

test('the bundled .txt and .md suites load alongside the JSON one', () => {
  const cases = loadSuite([path.join(dir, 'matrix50.json'), path.join(dir, 'extra.txt'), path.join(dir, 'extra.md')]);
  assert.ok(cases.length > 50);
  assert.equal(new Set(cases.map((c) => c.id)).size, cases.length);
  assert.throws(() => loadSuite([path.join(dir, 'matrix50.json'), path.join(dir, 'matrix50.json')]), /Duplicate case id/);
});

test('JSON detection handles fences, chatter and invalid output', () => {
  assert.equal(containsJson('{"name":"Sumit","role":"dev"}'), true);
  assert.equal(containsJson('Sure!\n```json\n{"name":"a","role":"b"}\n```'), true);
  assert.equal(containsJson('here you go: {"a":1} hope it helps'), true);
  assert.equal(containsJson('name: Sumit, role: dev'), false);
  assert.equal(containsJson('{name: Sumit}'), false);
  assert.equal(containsJson('42'), false);
});

test('repeat detection', () => {
  assert.ok(similarity("Why don't eggs tell jokes? They'd crack each other up!", "Why don't eggs tell jokes? They'd crack each other up! One more?") > 0.7);
  assert.ok(similarity('Why did the scarecrow win an award? Outstanding in his field!', "Why don't eggs tell jokes? They'd crack each other up!") < 0.3);
  const r = scoreTurn({ noRepeat: true }, { reply: "Why don't eggs tell jokes? They'd crack each other up!" }, { prevReply: "Why don't eggs tell jokes? They'd crack each other up!" });
  assert.equal(r.pass, false);
  assert.match(r.failures[0], /noRepeat/);
  assert.equal(scoreTurn({ noRepeat: true }, { reply: 'A totally different joke about cats' }, { prevReply: "Why don't eggs tell jokes?" }).pass, true);
});

test('passesOn: passing a request on is right, agreeing or refusing for the owner is wrong', () => {
  const ok = ["Sounds good! I'll pass it on to the real Sumit and he'll confirm.", 'Let me check with the real Sumit about that time.', 'Main Sumit ko bata dunga, wo confirm karega.'];
  for (const reply of ok) assert.equal(scoreTurn({ passesOn: true }, { reply }).pass, true, reply);
  assert.equal(scoreTurn({ passesOn: true }, { reply: "Sure, see you there tomorrow at 5!" }).pass, false);
  assert.equal(scoreTurn({ passesOn: true }, { reply: "I'll pass it on. Confirmed, see you there!" }).pass, false);
  assert.equal(scoreTurn({ passesOn: true }, { reply: 'Haha nice weather today' }).pass, false);
});

test('other extra checks and the eval-lib checks work together', () => {
  assert.equal(scoreTurn({ minChars: 10 }, { reply: 'short' }).pass, false);
  assert.equal(scoreTurn({ notEmpty: true }, { reply: '  ' }).pass, false);
  assert.equal(scoreTurn({ json: true }, { reply: '{"a":1}' }).pass, true);
  const r = scoreTurn({ intent: 'joke', matches: ['egg'] }, { reply: 'an egg joke', meta: { intent: 'joke' } });
  assert.deepEqual([r.pass, r.scored], [true, true]);
  assert.equal(scoreTurn({}, { reply: 'x' }).scored, false);
});

const rows = [
  { model: 'a:1b', caseId: 'c1', category: 'clear', turn: 1, user: 'q1', reply: 'ok', scored: true, pass: true, failures: [], latencyMs: 2000, tokensPerSec: 30, intent: 'general' },
  { model: 'a:1b', caseId: 'c2', category: 'context', turn: 1, user: 'q2', reply: 'bad', scored: true, pass: false, failures: ['matches: /x/ not found'], latencyMs: 4000, tokensPerSec: 20 },
  { model: 'b:2b', caseId: 'c1', category: 'clear', turn: 1, user: 'q1', reply: 'ok', scored: true, pass: true, failures: [], latencyMs: 9000, tokensPerSec: 8 },
  { model: 'b:2b', caseId: 'c2', category: 'context', turn: 1, user: 'q2', reply: 'ok', scored: true, pass: true, failures: [], latencyMs: 11000, tokensPerSec: 7 },
  { model: 'b:2b', caseId: 'c3', category: 'extra', turn: 1, user: 'obs', reply: 'hi', scored: false, pass: false, failures: [], latencyMs: 1000 },
  { model: 'b:2b', caseId: 'c4', category: 'clear', turn: 1, user: 'q4', reply: '', scored: true, pass: false, failures: [], error: 'timeout', latencyMs: 0 },
];

test('aggregation counts only scored turns and ranks by pass rate', () => {
  const stats = aggregate(rows);
  const a = stats.find((s) => s.model === 'a:1b');
  const b = stats.find((s) => s.model === 'b:2b');
  assert.deepEqual([a.scored, a.passed, a.passRate], [2, 1, 0.5]);
  assert.deepEqual([b.scored, b.passed, b.errors], [3, 2, 1]);
  assert.deepEqual(a.byCategory, { clear: { n: 1, passed: 1 }, context: { n: 1, passed: 0 } });
  assert.equal(a.medianLatencyMs, 3000);
  assert.equal(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10);
});

test('reports render in txt, md and json without leaking anything odd', () => {
  const payload = { meta: { suite: 'matrix50.json', base: 'https://x.test', startedAt: '2026-11-04T00:00:00Z', cases: 3, scored: 3 }, stats: aggregate(rows), rows };
  const md = renderMarkdown(payload);
  assert.match(md, /## Leaderboard/);
  assert.match(md, /\| 1 \| b:2b/); // 2/3 beats 1/2
  assert.match(md, /\| 2 \| a:1b/);
  assert.match(md, /### Failed checks \(1\)/);
  const txt = renderText(payload);
  assert.match(txt, /LEADERBOARD/);
  assert.match(txt, /\[c2\] FAIL/);
  assert.equal(JSON.parse(renderJson(payload)).rows.length, rows.length);
  assert.doesNotMatch(md + txt, /wa_session/);
});

test('question-by-question report shows each model\'s answer under the question', async () => {
  const { renderQa } = await import('../scripts/matrix-lib.js');
  const md = renderQa({ rows }, { models: ['b:2b', 'a:1b'] });
  assert.match(md, /### 1\. c1/);
  assert.ok(md.indexOf('**b:2b**') < md.indexOf('**a:1b**')); // requested order
  assert.match(md, /check failed: matches: \/x\/ not found/);
  assert.match(md, /ERROR: timeout/);
  assert.match(md, /## extra/);
});

test('a reply that is the model\'s own reasoning always fails, even if it contains the right words', async () => {
  const { REASONING_LEAK } = await import('../scripts/matrix-lib.js');
  const leaked = 'We are in a new chat. The user asked about a meeting. I must not commit, so I will pass it on to the real Sumit.';
  const r = scoreTurn({ passesOn: true }, { reply: leaked });
  assert.equal(r.pass, false);
  assert.ok(r.failures.some((f) => f.startsWith('leak:')));
  assert.equal(REASONING_LEAK.test('<think>hmm</think> Paris'), true);
  // normal chat is untouched, including the word "user" in everyday use
  assert.equal(scoreTurn({ notEmpty: true }, { reply: "Hey! I'll pass it on to the real Sumit 😊" }).pass, true);
  assert.equal(scoreTurn({ notEmpty: true }, { reply: 'Great question for a new user of Python: start with print().' }).pass, true);
});
