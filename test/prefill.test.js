import test from 'node:test';
import assert from 'node:assert/strict';
import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { DEFAULT_SETTINGS, sanitizeSettings } from '../src/core/settings-schema.js';

// The think-ahead only helps if the pre-read prompt is a prefix of what the real reply sends.
// This pins that: same system text, same selected history, and a placeholder where the message goes.
function fakeLlm() {
  const chats = [];
  const prefills = [];
  return {
    chats,
    prefills,
    numCtx: 4096,
    async *chat(req) {
      chats.push(req);
      yield { type: 'token', text: 'Sure thing.' };
      yield { type: 'done', stats: { evalTokens: 3, evalMs: 100, promptTokens: 50, doneReason: 'stop' } };
    },
    async prefill(req) {
      prefills.push(req);
      return { promptTokens: 40, promptMs: 5, loadMs: 0, totalMs: 6 };
    },
  };
}

const kb = new KnowledgeBase({ dir: new URL('../knowledge_base', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1') });
const settings = { ...DEFAULT_SETTINGS, ownerName: 'Sumit', maxTokens: 300 };

async function brainWith(llm, memory) {
  await kb.load().catch(() => {});
  const { cache } = createMemoryStores();
  return createBrain({ kb, llm, memory, cache });
}

test('warm() reads persona + history, then a placeholder turn, with the model and abort signal', async () => {
  const llm = fakeLlm();
  const { memory } = createMemoryStores();
  const brain = await brainWith(llm, memory);
  const history = [
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hey! How is your day going?' },
  ];
  const ctl = new AbortController();
  const out = await brain.warm({ history, settings, signal: ctl.signal });
  assert.equal(out.promptTokens, 40);
  const req = llm.prefills[0];
  assert.equal(req.model, settings.model);
  assert.equal(req.signal, ctl.signal);
  assert.equal(req.messages[0].role, 'system');
  assert.match(req.messages[0].content, /You are eSumit/);
  assert.deepEqual(req.messages.slice(1, 3), history);
  assert.equal(req.messages.at(-1).role, 'user');
});

test('the real reply starts with exactly the messages that warm() pre-read', async () => {
  const llm = fakeLlm();
  const { memory } = createMemoryStores();
  const brain = await brainWith(llm, memory);
  const history = [
    { role: 'user', content: 'Hi there' },
    { role: 'assistant', content: 'Hello! What are you up to?' },
    { role: 'user', content: 'working on a project' },
    { role: 'assistant', content: 'Nice, what kind of project?' },
  ];
  await brain.warm({ history, settings });
  for await (const ev of brain.respond({ text: 'a small whatsapp bot', settings, history })) void ev;
  const warmed = llm.prefills[0].messages.slice(0, -1); // everything before the placeholder turn
  const real = llm.chats[0].messages.slice(0, warmed.length);
  assert.deepEqual(real, warmed, 'the cached prefix must match what the reply sends');
});

test('warm() with no chat reads the persona alone (the prefix every chat shares)', async () => {
  const llm = fakeLlm();
  const { memory } = createMemoryStores();
  const brain = await brainWith(llm, memory);
  await brain.warm({ history: [], settings });
  assert.equal(llm.prefills[0].messages.length, 2); // system + placeholder
});

test('prefillAhead is a validated boolean setting, on by default', () => {
  assert.equal(DEFAULT_SETTINGS.prefillAhead, true);
  assert.deepEqual(sanitizeSettings({ prefillAhead: false }).value, { prefillAhead: false });
  assert.equal(sanitizeSettings({ prefillAhead: 'maybe' }).errors.length, 1);
});
