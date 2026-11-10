import test from 'node:test';
import assert from 'node:assert/strict';
import { Coalescer } from '../src/core/coalesce.js';
import { isRepeat, recentBubbles, similarity } from '../src/brain/repeat.js';
import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { DEFAULT_SETTINGS } from '../src/core/settings-schema.js';

test('queued jobs: only the newest runs and it answers every message since the last reply', () => {
  const co = new Coalescer();
  co.add('c', 'j1', 'Sumit I want to talk to you');
  co.add('c', 'j2', 'Stop this');
  co.add('c', 'j3', 'Haa');
  assert.equal(co.claim('c', 'j1'), null, 'older job is folded into the newest');
  assert.equal(co.claim('c', 'j2'), null);
  const run = co.claim('c', 'j3');
  assert.equal(run.text, 'Sumit I want to talk to you\nStop this\nHaa');
  assert.equal(co.claim('c', 'j3'), null, 'nothing is answered twice');
});

test('a generation that has not spoken yet is aborted and re-run with the new message', () => {
  const co = new Coalescer();
  co.add('c', 'j1', 'let us meet tomorrow');
  const first = co.claim('c', 'j1');
  assert.equal(co.add('c', 'j2', 'at six in the morning'), true);
  assert.equal(first.signal.aborted, true);
  assert.equal(co.isSuperseded(first.signal), true);
  assert.equal(co.release('c', 'j1'), true);
  const second = co.claim('c', 'j2');
  assert.equal(second.text, 'let us meet tomorrow\nat six in the morning', 'both messages are answered once, in order');
});

test('once a bubble is out the reply is committed: the new message waits for its own reply', () => {
  const co = new Coalescer();
  co.add('c', 'j1', 'tell me a joke');
  const first = co.claim('c', 'j1');
  co.markEmitted('c');
  assert.equal(co.add('c', 'j2', 'one more'), false);
  assert.equal(first.signal.aborted, false);
  assert.equal(co.release('c', 'j1'), false);
  assert.equal(co.claim('c', 'j2').text, 'one more', 'the joke is not asked for twice');
});

test('chats are independent and the state is cleaned up', () => {
  const co = new Coalescer();
  co.add('a', 'a1', 'hi');
  co.add('b', 'b1', 'hello');
  assert.equal(co.claim('a', 'a1').text, 'hi');
  assert.equal(co.claim('b', 'b1').text, 'hello');
  co.release('a', 'a1');
  co.release('b', 'b1');
  assert.equal(co.size, 0);
});

test('repeat detection: same reply again is caught, a different one and short ones are not', () => {
  const earlier = ['Arre, Prachi! Pagal toh nahin hoon, lekin mujhe lagta hai ki humein kuch confusion ho rahi hai.'];
  assert.equal(isRepeat('Arre Prachi! Pagal toh nahin hoon, lekin mujhe lagta hai ki humein kuch confusion ho rahi hai', earlier), true);
  assert.equal(isRepeat('Chalo ek joke sunata hoon: ek aadmi doctor ke paas gaya.', earlier), false);
  assert.equal(isRepeat('Haan 😄', ['Nahi yaar']), false);
  assert.equal(isRepeat('Haan', ['Haan']), false, 'one-word answers may repeat');
  assert.equal(isRepeat('Sounds good, see you then', ['sounds good, see you then!']), true);
  assert.ok(similarity('a b c d', 'a b c d') === 1);
});

test('recentBubbles splits stored multi-bubble turns', () => {
  const b = recentBubbles([{ role: 'user', content: 'x' }, { role: 'assistant', content: 'one\n\ntwo' }]);
  assert.deepEqual(b, ['one', 'two']);
});

// The pipeline retries once when the first bubble repeats an earlier reply.
const kb = new KnowledgeBase({ dir: new URL('../knowledge_base', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1') });
const settings = { ...DEFAULT_SETTINGS, ownerName: 'Sumit', maxTokens: 300 };

test('pipeline: a repeated first bubble is dropped before sending and regenerated', async () => {
  await kb.load().catch(() => {});
  const calls = [];
  const llm = {
    numCtx: 4096,
    async *chat(req) {
      calls.push(req);
      const text = calls.length === 1 ? 'Arre Prachi! Pagal toh nahin hoon, lekin humein kuch confusion ho rahi hai.' : 'Chalo, fir se batao kya hua.';
      yield { type: 'token', text };
      yield { type: 'done', stats: { evalTokens: 5, evalMs: 50, promptTokens: 40, doneReason: 'stop' } };
    },
  };
  const { memory, cache } = createMemoryStores();
  await memory.append('c', [
    { role: 'user', content: 'real sumit h hi pagal' },
    { role: 'assistant', content: 'Arre, Prachi! Pagal toh nahin hoon, lekin humein kuch confusion ho rahi hai.' },
  ]);
  const brain = createBrain({ kb, llm, memory, cache });
  const sent = [];
  for await (const ev of brain.respond({ chatId: 'c', text: 'Haa', settings })) if (ev.type === 'chunk') sent.push(ev.text);
  assert.equal(calls.length, 2, 'regenerated once');
  assert.match(calls[1].messages.at(-1).content, /Say something new/);
  assert.ok(calls[1].temperature > calls[0].temperature);
  assert.deepEqual(sent, ['Chalo, fir se batao kya hua.'], 'only the new text reaches the contact');
});

test('pipeline: a fresh reply is sent as is, with a single generation', async () => {
  const calls = [];
  const llm = {
    numCtx: 4096,
    async *chat(req) {
      calls.push(req);
      yield { type: 'token', text: 'Sure, tell me more.' };
      yield { type: 'done', stats: { evalTokens: 4, evalMs: 50, promptTokens: 40, doneReason: 'stop' } };
    },
  };
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  const sent = [];
  for await (const ev of brain.respond({ chatId: 'd', text: 'I need help with a project', settings })) if (ev.type === 'chunk') sent.push(ev.text);
  assert.equal(calls.length, 1);
  assert.equal(sent.length, 1);
});
