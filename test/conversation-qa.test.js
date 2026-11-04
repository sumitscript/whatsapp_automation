// Regression tests for the live-chat QA findings: invented facts, lost memory,
// ignored language preference, interrogation loops, name mix-ups, slow first bubble.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { classifyIntent } from '../src/brain/intent.js';
import { replyLanguage, statedLanguagePreference } from '../src/brain/lang.js';
import { MessageChunker } from '../src/brain/format.js';
import { editDistance } from '../src/rag/text.js';
import { compactTurn, createBrain, createMemoryStores, requestedWords, selectHistory } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';
import { DEFAULT_SETTINGS } from '../src/core/settings-schema.js';

const aliases = ['sumit', 'esumit'];
const kb = await new KnowledgeBase({ dir: path.join(import.meta.dirname, 'fixtures', 'kb'), minScore: 0.5 }).load();
const settings = { ...DEFAULT_SETTINGS, ownerName: 'Sumit Soni', ownerAliases: 'sumit', personaName: 'eSumit', model: 'fake:1b' };

function fakeLlm(reply = 'ok') {
  const calls = [];
  return {
    calls,
    numCtx: 4096,
    async *chat(req) {
      calls.push(req);
      yield { type: 'token', text: reply };
      yield { type: 'done', stats: { doneReason: 'stop' } };
    },
  };
}
async function lastPrompt(args, reply) {
  const llm = fakeLlm(reply);
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  for await (const _ of brain.respond({ settings, ...args }));
  return llm.calls[0].messages.at(-1).content;
}

test('typos count as one edit per swapped pair', () => {
  assert.equal(editDistance('pojcet', 'project', 2), 2);
  assert.equal(editDistance('wrok', 'work', 2), 1);
  assert.equal(editDistance('abc', 'abc', 2), 0);
});

test('about-me needs a profile topic; "your" means me, "my" means them', () => {
  for (const t of ['Tell me your pojcet', 'your porject list', 'tumhara college kaunsa tha', 'where do you work', 'projects?'])
    assert.equal(classifyIntent(t, { ownerAliases: aliases }), 'about_owner', t);
  for (const t of ['tum meri help kro', 'Help me to prep', 'help me with my project', 'can you check my resume', 'I like working out'])
    assert.notEqual(classifyIntent(t, { ownerAliases: aliases }), 'about_owner', t);
});

test('a stated language preference wins over one English message', () => {
  const said = ["I'm Ravi, preparing for backend interviews, i prefer hinglish"];
  assert.equal(replyLanguage('System design', said).code, 'hi-Latn');
  assert.equal(replyLanguage('Help me to prep', [], '- Name: Ravi\n- Prefers Hinglish').code, 'hi-Latn');
  assert.equal(replyLanguage('नमस्ते आप कैसे हो', said).code, 'hi'); // native script still wins
  assert.equal(statedLanguagePreference(['I use english at work']), null);
  assert.equal(replyLanguage('ok cool', ['bhai kya haal hai', 'acha theek hai yaar']).code, 'hi-Latn');
  assert.equal(replyLanguage('what is java', []).code, 'en');
});

test('greetings without facts are told not to invent my job or projects', async () => {
  const prompt = await lastPrompt({ text: 'Hello' });
  assert.match(prompt, /don't mention my job, company, projects/);
});

test('"I told you last time" uses the notes, or admits there are none', async () => {
  const turns = Array.from({ length: 8 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `turn ${i}` }));
  const withNotes = await lastPrompt({ text: 'I told you last time', history: turns, longTermMemory: '- Name: Ravi\n- Preparing for backend interviews' });
  assert.match(withNotes, /Preparing for backend interviews/);
  assert.match(withNotes, /referring to an earlier chat: answer from the notes about them/);
  const without = await lastPrompt({ text: 'I already told you last time', history: turns });
  assert.match(without, /I have no notes about it: say so honestly/);
});

test('asking for the answer and question loops are handled', async () => {
  const history = [
    { role: 'user', content: 'Endless also i can go with elastic server' },
    { role: 'assistant', content: 'Nice! How many requests per second do we need? 🤔' },
  ];
  const prompt = await lastPrompt({ text: "I'm not sure help me with correct answer", history });
  assert.match(prompt, /give it directly and correctly first/);
  assert.match(prompt, /don't end this one with a question/);
});

test('a contact named like the owner is marked as a different person', async () => {
  const prompt = await lastPrompt({ text: 'hey there friend', contactName: 'Sumit' });
  assert.match(prompt, /\[I'm chatting with\] Sumit \(their WhatsApp name — a different person from me/);
});

test('stories are fiction: no KB facts, no notes, creative temperature, word target honoured', async () => {
  const llm = fakeLlm('Once upon a time…');
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  for await (const _ of brain.respond({ settings, text: 'Tell me long story minimum 200 words', longTermMemory: '- Preparing for backend interviews' }));
  const req = llm.calls[0];
  const last = req.messages.at(-1).content;
  assert.equal(req.temperature, 0.95);
  assert.ok(req.maxTokens >= 360, `maxTokens ${req.maxTokens}`); // 200 words ≈ 320 tokens + margin
  assert.doesNotMatch(last, /My facts|Notes about|backend interviews/);
  assert.match(last, /creative writing: invent original characters/);
  assert.match(last, /about 200 words/);
  for (const t of ['I want an fictional story', 'Tell me a horror story', 'ek kahani sunao']) assert.equal(classifyIntent(t, { ownerAliases: aliases }), 'creative', t);
  // …but "your story" is my real life
  assert.equal(classifyIntent('tell me your story', { ownerAliases: aliases }), 'about_owner');
  assert.equal(classifyIntent('can you tell me a horror story', { ownerAliases: aliases }), 'creative');
});

test('labels say whose information each block is', async () => {
  const prompt = await lastPrompt({ text: 'what projects did you build', contactName: 'Ravi', longTermMemory: '- Name: Ravi\n- Preparing for backend interviews' });
  assert.match(prompt, /\[My facts — about Sumit, the one replying \(me\)\. Not about Ravi\]/);
  assert.match(prompt, /\[Notes about Ravi — the person I'm chatting with\. Their life, not mine\]/);
  assert.match(prompt, /never mix the two up/);
  assert.doesNotMatch(prompt, /\[Facts about me\]|\[What I remember/);
});

test('old long replies are shortened in the history; the newest stays whole', () => {
  const story = 'Once upon a time there was a server. '.repeat(60);
  const turns = [
    { role: 'user', content: 'tell me a story' },
    { role: 'assistant', content: story },
    { role: 'user', content: 'another' },
    { role: 'assistant', content: story },
  ];
  const { history } = selectHistory(turns, { maxMessages: 10, maxTokens: 5000 });
  assert.ok(history[1].content.length < 700 && history[1].content.includes('[…]'));
  assert.equal(history[3].content, story);
  assert.equal(compactTurn({ role: 'assistant', content: 'short' }), 'short');
});

test('a stated language preference is always kept in the memory notes', async () => {
  const llm = fakeLlm('- Name: Ravi\n- Preparing for backend interviews');
  const brain = createBrain({ kb, llm, ...createMemoryStores() });
  const notes = await brain.summarize({
    previous: null,
    turns: [{ role: 'user', content: "I'm Ravi, preparing for backend interviews, i prefer hinglish" }, { role: 'assistant', content: 'Nice!' }],
    contactName: 'Sumit',
    settings,
  });
  assert.match(notes, /- Prefers Hinglish$/);
  assert.match(llm.calls[0].messages[0].content, /Name: …/);
  assert.equal(requestedWords('write 300 words on rain'), 300);
  assert.equal(requestedWords('tell me a story'), null);
});

test('the first bubble goes out at the first full sentence', () => {
  const c = new MessageChunker();
  const text = 'Okay, let us tackle that properly now, step by step for you. Scaling is mostly about caching. Then sharding.';
  const out = [];
  for (const piece of text.match(/.{1,6}/gs)) out.push(...c.push(piece));
  assert.equal(out[0], 'Okay, let us tackle that properly now, step by step for you.');
  out.push(...c.flush());
  assert.equal(out.length, 2);
});
