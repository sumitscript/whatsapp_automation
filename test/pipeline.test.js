import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createBrain, createMemoryStores, isShortReply, selectHistory, trimDangling } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { DEFAULT_SETTINGS } from '../src/core/settings-schema.js';

const kb = await new KnowledgeBase({ dir: path.join(import.meta.dirname, 'fixtures', 'kb'), minScore: 0.5 }).load();
const settings = { ...DEFAULT_SETTINGS, ownerName: 'Asha Rao', ownerAliases: 'asha', personaName: 'eAsha', model: 'fake:1b' };

function fakeLlm(reply, { numCtx = 4096 } = {}) {
  const calls = [];
  return {
    calls,
    numCtx,
    async *chat(req) {
      calls.push(req);
      for (const piece of reply.match(/.{1,7}/gs)) yield { type: 'token', text: piece };
      yield { type: 'done', stats: { evalTokens: 10, evalMs: 500, promptTokens: 100, doneReason: 'stop' } };
    },
  };
}

async function run(brain, args) {
  const chunks = [];
  let done;
  for await (const ev of brain.respond(args)) {
    if (ev.type === 'chunk') chunks.push(ev.text);
    else done = ev;
  }
  return { chunks, done };
}

test('speaks as the owner in first person, with facts and a language hint', async () => {
  const llm = fakeLlm('I am a **backend engineer** at Acme Corp.');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  const { chunks, done } = await run(brain, { chatId: 'c1', text: 'Where does Asha work?', settings });
  assert.deepEqual(chunks, ['I am a *backend engineer* at Acme Corp.']);
  const req = llm.calls[0];
  assert.equal(req.temperature, 0.3);
  assert.match(req.messages[0].content, /You are eAsha — the virtual version of Asha/);
  assert.match(req.messages.at(-1).content, /\[My facts — about Asha, the one replying \(me\)\. Not about them\]/);
  assert.match(req.messages.at(-1).content, /Reply in English/);
  assert.equal(done.meta.intent, 'about_owner');
  assert.ok(done.meta.sources.length > 0);
  assert.equal((await memory.get('c1')).length, 2);
});

test('long-term memory and the contact name are injected', async () => {
  const llm = fakeLlm('Hey Ravi! How did the interview go?');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await run(brain, { chatId: 'm1', text: 'hey whats up', settings, contactName: 'Ravi', longTermMemory: '- Ravi had an interview at Globex' });
  const last = llm.calls[0].messages.at(-1).content;
  assert.match(last, /\[Notes about Ravi — the person I'm chatting with\. Their life, not mine\]\n\(from our earlier chats[^)]*\)\n- Ravi had an interview at Globex/);
  assert.match(last, /\[I'm chatting with\] Ravi/);
});

test('memory is ignored when memoryEnabled is off', async () => {
  const llm = fakeLlm('ok');
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  await run(brain, { chatId: 'm2', text: 'hello there friend', settings: { ...settings, memoryEnabled: false }, longTermMemory: 'secret' });
  assert.doesNotMatch(llm.calls[0].messages.at(-1).content, /secret/);
});

test('repeated factual questions are served from the answer cache', async () => {
  const llm = fakeLlm('Acme Corp.');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await run(brain, { chatId: 'x', text: 'Where does Asha work?', settings });
  const second = await run(brain, { chatId: 'y', text: 'where does asha work', settings });
  assert.equal(llm.calls.length, 1);
  assert.equal(second.done.meta.cached, true);
  assert.deepEqual(second.chunks, ['Acme Corp.']);
});

test('changing the persona invalidates cached answers', async () => {
  const llm = fakeLlm('Acme Corp.');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await run(brain, { chatId: 'x', text: 'Where does Asha work?', settings });
  await run(brain, { chatId: 'x', text: 'Where does Asha work?', settings: { ...settings, personaName: 'AshaBot' } });
  assert.equal(llm.calls.length, 2);
});

test('chit-chat skips retrieval and uses a playful temperature', async () => {
  const llm = fakeLlm('Haha, good one!');
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  const { done } = await run(brain, { chatId: 'c2', text: 'haha you are funny 😂', settings });
  assert.equal(done.meta.intent, 'joke');
  assert.equal(llm.calls[0].temperature, 0.9);
  assert.doesNotMatch(llm.calls[0].messages.at(-1).content, /My facts/);
});

test('history is passed through and long answers split into bubbles', async () => {
  const llm = fakeLlm('Recursion is when a function calls itself to solve a smaller piece of the problem.\n\nSecond part of the answer.');
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  const history = [
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: 'hello!' },
  ];
  const { chunks } = await run(brain, { text: 'explain recursion please', settings, history });
  assert.equal(chunks.length, 2);
  assert.deepEqual(llm.calls[0].messages.slice(1, 3), history);
});

test('history is trimmed to fit a small context window', async () => {
  const llm = fakeLlm('sure', { numCtx: 1800 }); // persona ≈ 750 tokens
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  const history = Array.from({ length: 30 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `turn ${i} ` + 'x'.repeat(200) }));
  await run(brain, { text: 'continue', settings: { ...settings, historyTurns: 40, historyTokens: 6000 }, history });
  const sent = llm.calls[0].messages.slice(1, -1);
  assert.ok(sent.length > 0 && sent.length < 30);
  assert.match(sent.at(-1).content, /turn 29/); // newest turns are kept
});

test('selectHistory caps, starts on a user turn and merges same-role runs', () => {
  const turns = [
    { role: 'assistant', content: 'welcome!' },
    { role: 'user', content: 'a' },
    { role: 'assistant', content: 'b' },
    { role: 'assistant', content: 'c (owner from phone)' },
    { role: 'user', content: 'd' },
  ];
  const { history } = selectHistory(turns, { maxMessages: 10, maxTokens: 1000 });
  assert.deepEqual(history.map((t) => t.role), ['user', 'assistant', 'user']);
  assert.equal(history[1].content, 'b\nc (owner from phone)');
  assert.equal(selectHistory(turns, { maxMessages: 2, maxTokens: 1000 }).history.length, 1);
});

test('summarize folds turns into memory notes', async () => {
  const llm = fakeLlm('- Ravi is preparing for interviews\n- Prefers Hinglish');
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  const notes = await brain.summarize({
    previous: '- Ravi is a college friend',
    turns: [
      { role: 'user', content: 'bhai interview prep kaise karu' },
      { role: 'assistant', content: 'DSA + projects pe focus kar' },
    ],
    contactName: 'Ravi',
    settings,
  });
  assert.match(notes, /interviews/);
  const prompt = llm.calls[0].messages.at(-1).content;
  assert.match(prompt, /Existing notes:\n- Ravi is a college friend/);
  assert.match(prompt, /Them: bhai interview prep/);
  assert.equal(llm.calls[0].temperature, 0.2);
});

test('empty model output falls back to a polite message', async () => {
  const llm = {
    async *chat() {
      yield { type: 'done', stats: {} };
    },
  };
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  const { chunks } = await run(brain, { text: 'hmm', settings });
  assert.equal(chunks.length, 1);
  assert.match(chunks[0], /say it another way/);
});

test('"one more" after a joke stays a joke: no knowledge-base facts, history kept', async () => {
  const llm = fakeLlm('Why did the scarecrow win an award? Outstanding in his field!');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await memory.append('j1', [
    { role: 'user', content: 'Tell me a joke' },
    { role: 'assistant', content: "Why don't eggs tell jokes? They'd crack each other up!" },
  ]);
  const { done } = await run(brain, { chatId: 'j1', text: 'One more', settings });
  const req = llm.calls[0];
  assert.equal(done.meta.intent, 'joke');
  assert.equal(done.meta.inheritedIntent, true);
  assert.equal(done.meta.sources.length, 0);
  assert.doesNotMatch(req.messages.at(-1).content, /\[My facts/);
  assert.equal(req.messages.at(-2).role, 'assistant'); // the previous joke is in the prompt
  assert.equal(req.messages.at(-3).content, 'Tell me a joke');
});

test('tiny general messages never pull knowledge-base facts', async () => {
  const llm = fakeLlm('Sure!');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  const { done } = await run(brain, { chatId: 't1', text: 'ok cool', settings });
  assert.equal(done.meta.sources.length, 0);
});

test('facts/notes come with a guard telling the model to answer the message', async () => {
  const llm = fakeLlm('ok');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await run(brain, { chatId: 'g1', text: 'hey whats up', settings, contactName: 'Ravi', longTermMemory: '- likes chess' });
  assert.match(llm.calls[0].messages.at(-1).content, /ignore them and never bring them up/);
});

test('"Yes" is anchored to my last message, not an older topic', async () => {
  const llm = fakeLlm('Here you go!');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await memory.append('y1', [
    { role: 'user', content: 'Explain Two Sum' },
    { role: 'assistant', content: 'Use a hash map of complements.' },
    { role: 'user', content: 'What is multilevel inheritance in Java' },
    { role: 'assistant', content: 'A → B → C chain of classes. Want an example? 😊' },
  ]);
  const { done } = await run(brain, { chatId: 'y1', text: 'Yes', settings });
  const last = llm.calls[0].messages.at(-1).content;
  assert.match(last, /\[What I said last\]\nA → B → C chain of classes\. Want an example\?/);
  assert.match(last, /Their message \(a reply to what I said last\): Yes/);
  assert.match(last, /Do not switch to another topic/);
  assert.equal(done.meta.intent, 'coding'); // inherited from the inheritance question
});

test('long-term notes are used at the start of a chat but skipped mid-conversation', async () => {
  const llm = fakeLlm('ok');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  const notes = '- Asked about Two Sum';
  await run(brain, { chatId: 'n1', text: 'hello there friend', settings, longTermMemory: notes });
  assert.match(llm.calls[0].messages.at(-1).content, /Asked about Two Sum/);

  const turns = Array.from({ length: 8 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `turn ${i}` }));
  await memory.append('n2', turns);
  await run(brain, { chatId: 'n2', text: 'tell me something interesting', settings, longTermMemory: notes });
  assert.doesNotMatch(llm.calls[1].messages.at(-1).content, /Two Sum/);
  await run(brain, { chatId: 'n2', text: 'what did we talk about last time', settings, longTermMemory: notes });
  assert.match(llm.calls[2].messages.at(-1).content, /Asked about Two Sum/);
});

test('a reply cut off by the token limit is trimmed to a full sentence', async () => {
  const llm = {
    numCtx: 4096,
    async *chat() {
      yield { type: 'token', text: 'Two Sum is easy. Use a hash map for complements. Abhi toh apne fav tech project ke baareme bhi ba' };
      yield { type: 'done', stats: { doneReason: 'length', evalTokens: 40, evalMs: 1000 } };
    },
  };
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  const { chunks, done } = await run(brain, { text: 'how are you doing today friend', settings: { ...settings, humanize: false } });
  assert.deepEqual(chunks, ['Two Sum is easy. Use a hash map for complements.']);
  assert.equal(done.meta.stats.doneReason, 'length');
});

test('trimDangling and isShortReply edge cases', () => {
  assert.equal(trimDangling('All done.'), 'All done.');
  assert.equal(trimDangling('Nice 😄'), 'Nice 😄');
  assert.equal(trimDangling('```\nconst a = 1'), '```\nconst a = 1\n```');
  assert.equal(trimDangling('just one long running thought that never ends somew'), 'just one long running thought that never ends…');
  for (const t of ['Yes', 'ok sure', 'haan bhai', 'go on']) assert.equal(isShortReply(t), true, t);
  for (const t of ['why?', 'explain the hash map approach please', '']) assert.equal(isShortReply(t), false, t);
});

test('"one more" after a joke asks for a NEW joke instead of anchoring to the last one', async () => {
  const llm = fakeLlm('Why did the scarecrow win an award? Outstanding in his field!');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await memory.append('j2', [
    { role: 'user', content: 'Tell me a joke' },
    { role: 'assistant', content: "Why don't eggs tell jokes? They'd crack each other up!" },
  ]);
  await run(brain, { chatId: 'j2', text: 'One more', settings });
  const last = llm.calls[0].messages.at(-1).content;
  assert.doesNotMatch(last, /\[What I said last\]/); // that anchor made the model repeat the old joke
  assert.doesNotMatch(last, /Continue exactly what I said last/);
  assert.match(last, /give a NEW joke, different from everything I already said/);
});

test('arithmetic is answered in plain text, not code', async () => {
  const llm = fakeLlm('2 + 2 = 4');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await run(brain, { chatId: 'a1', text: 'What is 2 + 2', settings });
  assert.match(llm.calls[0].messages.at(-1).content, /not in a code block/);
});

test('a plain "yes" still continues what I just offered', async () => {
  const llm = fakeLlm('Great, here it is!');
  const { memory, cache } = createMemoryStores();
  const brain = createBrain({ kb, llm, memory, cache });
  await memory.append('y1', [
    { role: 'user', content: 'what is your name' },
    { role: 'assistant', content: 'I am eAsha. Want a quick intro?' },
  ]);
  await run(brain, { chatId: 'y1', text: 'yes', settings });
  assert.match(llm.calls[0].messages.at(-1).content, /\[What I said last\]/);
});
