import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { KeyPool, parseDuration } from '../src/llm/key-pool.js';
import { GroqClient, GroqRequestError, GroqUnavailableError } from '../src/llm/groq.js';
import { RoutedLlm } from '../src/llm/router.js';
import { KeyStore, isGroqKeyFormat, parseKeyList } from '../src/llm/key-store.js';
import { decryptSecret, encryptSecret, maskKey } from '../src/core/secrets.js';
import { Semaphore } from '../src/core/util.js';
import { DEFAULT_SETTINGS, sanitizeSettings } from '../src/core/settings-schema.js';

const keysOf = (...ids) => ids.map((id) => ({ id, label: id, secret: `gsk_secret_${id}_xxxxxxxxxxxx`, source: 'env' }));
const drain = async (gen) => {
  const out = [];
  for await (const ev of gen) out.push(ev);
  return out;
};

// ------------------------------------------------------------------ helpers

test('secrets: encrypt/decrypt round trip, wrong secret fails, mask hides the key', () => {
  const tok = encryptSecret('gsk_abcdefghijklmnop1234', 'server-secret');
  assert.notEqual(tok, 'gsk_abcdefghijklmnop1234');
  assert.equal(decryptSecret(tok, 'server-secret'), 'gsk_abcdefghijklmnop1234');
  assert.equal(decryptSecret(tok, 'other-secret'), null);
  assert.equal(decryptSecret(tok.slice(0, -3) + 'AAA', 'server-secret'), null, 'tampering is detected');
  assert.equal(encryptSecret('x', 's') === encryptSecret('x', 's'), false, 'a fresh IV every time');
  assert.equal(maskKey('gsk_abcdefghijklmnop1234'), 'gsk_…1234');
});

test('key format check accepts real-looking keys and rejects junk', () => {
  assert.equal(isGroqKeyFormat(`gsk_${'aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5hJ7kL9mN1pQ3rS'}`), true);
  assert.equal(isGroqKeyFormat('gsk_with space_in_the_middle_of_the_key'), false);
  assert.equal(isGroqKeyFormat('short'), false);
  assert.equal(isGroqKeyFormat(''), false);
  assert.equal(isGroqKeyFormat(null), false);
  assert.equal(isGroqKeyFormat(`gsk_${'a'.repeat(300)}`), false);
});

test('durations from Groq headers', () => {
  assert.equal(parseDuration('2m59.56s'), 179_560);
  assert.equal(parseDuration('7.66s'), 7660);
  assert.equal(parseDuration('250ms'), 250);
  assert.equal(parseDuration('1h2m3s'), 3_723_000);
  assert.equal(parseDuration('12'), 12_000);
  assert.equal(parseDuration(null), null);
  assert.equal(parseDuration('soon'), null);
});

test('settings: provider, model and fallback are validated', () => {
  assert.equal(DEFAULT_SETTINGS.llmProvider, 'ollama');
  assert.equal(DEFAULT_SETTINGS.groqFallback, true);
  assert.deepEqual(sanitizeSettings({ llmProvider: 'groq', groqModel: 'llama-3.1-8b-instant', groqFallback: false }).value, { llmProvider: 'groq', groqModel: 'llama-3.1-8b-instant', groqFallback: false });
  assert.equal(sanitizeSettings({ llmProvider: 'openai' }).errors.length, 1);
  assert.equal(sanitizeSettings({ groqModel: 'bad model!' }).errors.length, 1);
});

// ------------------------------------------------------------------ key pool

test('pool: keys are used round-robin', () => {
  const pool = new KeyPool();
  pool.setKeys(keysOf('a', 'b', 'c'));
  const seq = Array.from({ length: 6 }, () => pool.pick().id);
  assert.deepEqual(seq, ['a', 'b', 'c', 'a', 'b', 'c']);
});

test('pool: a rate-limited key rests until its reset, then comes back; the others keep serving', () => {
  let t = 1000;
  const pool = new KeyPool({ now: () => t });
  pool.setKeys(keysOf('a', 'b'));
  const first = pool.pick();
  pool.rateLimited(first.id, { retryAfterMs: 30_000 });
  assert.equal(pool.status().find((k) => k.id === first.id).state, 'cooling');
  assert.deepEqual([pool.pick().id, pool.pick().id], ['b', 'b']);
  pool.rateLimited('b', { retryAfterMs: 10_000 });
  assert.equal(pool.nextAvailableMs(), 10_000, 'the earliest reset among the cooling keys');
  t += 30_001;
  assert.equal(pool.status().find((k) => k.id === first.id).state, 'ready');
  assert.ok(new Set([pool.pick().id, pool.pick().id]).size === 2, 'the recovered key rejoins the rotation');
});

test('pool: nothing available when every key is cooling or disabled; exclude skips already-tried keys', () => {
  const pool = new KeyPool();
  pool.setKeys(keysOf('a', 'b'));
  assert.equal(pool.pick({ exclude: new Set(['a', 'b']) }), null);
  pool.rateLimited('a', { retryAfterMs: 60_000 });
  pool.authFailed('b');
  assert.equal(pool.hasAvailable(), false);
  assert.equal(pool.pick(), null);
  assert.equal(pool.status().find((k) => k.id === 'b').state, 'disabled');
  pool.clearDisabled();
  assert.equal(pool.pick().id, 'b');
});

test('pool: a key with no tokens left for this request is passed over while a roomier key exists', () => {
  const pool = new KeyPool();
  pool.setKeys(keysOf('a', 'b'));
  pool.success('a', { remainingTokens: 100 });
  pool.success('b', { remainingTokens: 50_000 });
  assert.equal(pool.pick({ needTokens: 2000 }).id, 'b');
  assert.equal(pool.pick({ needTokens: 2000 }).id, 'b');
  pool.success('b', { remainingTokens: 50 });
  assert.ok(['a', 'b'].includes(pool.pick({ needTokens: 2000 }).id), 'if none has room, any ready key is still tried');
});

test('pool: an exhausted window rests the key proactively (no wasted 429)', () => {
  let t = 0;
  const pool = new KeyPool({ now: () => t });
  pool.setKeys(keysOf('a', 'b'));
  pool.success('a', { remainingRequests: 0, resetRequestsMs: 8000 });
  assert.equal(pool.status().find((k) => k.id === 'a').state, 'cooling');
  t = 8001;
  assert.equal(pool.status().find((k) => k.id === 'a').state, 'ready');
});

test('pool: transient errors back off, a success resets them; reloading keeps state of unchanged keys', () => {
  let t = 0;
  const pool = new KeyPool({ now: () => t });
  pool.setKeys(keysOf('a'));
  pool.transient('a');
  assert.equal(pool.status()[0].cooldownMs, 2000);
  t = 2001;
  pool.transient('a');
  assert.equal(pool.status()[0].cooldownMs, 4000);
  pool.success('a');
  pool.rateLimited('a', { retryAfterMs: 5000 });
  pool.setKeys([...keysOf('a'), ...keysOf('b')]);
  assert.equal(pool.status().find((k) => k.id === 'a').state, 'cooling', 'state survives a reload');
  pool.setKeys([{ id: 'a', label: 'a', secret: 'gsk_new_secret_value_1234567', source: 'env' }]);
  assert.equal(pool.status()[0].state, 'ready', 'a replaced secret starts clean');
});

// ------------------------------------------------------------------ Groq client against a fake server

function sse(tokens, usage = { prompt_tokens: 30, completion_tokens: tokens.length, total_tokens: 30 + tokens.length, prompt_time: 0.01, completion_time: 0.05, total_time: 0.06 }, finish = 'stop') {
  const lines = tokens.map((t) => `data: ${JSON.stringify({ choices: [{ delta: { content: t } }] })}\n\n`);
  lines.push(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: finish }], x_groq: { usage } })}\n\n`, 'data: [DONE]\n\n');
  return lines.join('');
}

function fakeGroq(handler) {
  return new Promise((resolve) => {
    const calls = [];
    const server = http.createServer((req, res) => {
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', () => {
        const auth = req.headers.authorization || '';
        calls.push({ url: req.url, auth, body: body ? JSON.parse(body) : null });
        handler({ req, res, auth, n: calls.length });
      });
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, calls, url: `http://127.0.0.1:${server.address().port}/openai/v1` }));
  });
}
const closeServer = (server) => {
  server.closeAllConnections?.();
  server.close();
};

test('groq: streams tokens with stats and sends the key and model', async () => {
  const { server, calls, url } = await fakeGroq(({ res }) => {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'x-ratelimit-remaining-tokens': '9000', 'x-ratelimit-remaining-requests': '20' });
    res.end(sse(['Hel', 'lo ', 'there']));
  });
  try {
    const pool = new KeyPool();
    pool.setKeys(keysOf('a'));
    const g = new GroqClient({ pool, baseUrl: url, timeoutMs: 2000 });
    const evs = await drain(g.chat({ model: 'llama-3.3-70b-versatile', messages: [{ role: 'user', content: 'hi' }], temperature: 0.5, maxTokens: 50 }));
    assert.equal(evs.filter((e) => e.type === 'token').map((e) => e.text).join(''), 'Hello there');
    const done = evs.at(-1);
    assert.equal(done.type, 'done');
    assert.equal(done.stats.promptTokens, 30);
    assert.equal(done.stats.doneReason, 'stop');
    assert.equal(done.stats.provider, 'groq');
    assert.equal(calls[0].auth, 'Bearer gsk_secret_a_xxxxxxxxxxxx');
    assert.equal(calls[0].body.model, 'llama-3.3-70b-versatile');
    assert.equal(calls[0].body.stream, true);
    assert.equal(calls[0].body.max_tokens, 50);
    assert.equal(pool.status()[0].remainingTokens, 9000);
  } finally {
    closeServer(server);
  }
});

test('groq: a 429 on one key switches to the next key in the same call, instantly', async () => {
  const { server, calls, url } = await fakeGroq(({ res, auth }) => {
    if (auth.includes('_a_')) {
      res.writeHead(429, { 'retry-after': '45', 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'Rate limit reached for model on tokens per minute (TPM)' } }));
    }
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end(sse(['from b']));
  });
  try {
    const pool = new KeyPool();
    pool.setKeys(keysOf('a', 'b'));
    const g = new GroqClient({ pool, baseUrl: url, timeoutMs: 2000 });
    const t0 = Date.now();
    const evs = await drain(g.chat({ model: 'm', messages: [{ role: 'user', content: 'hi' }] }));
    assert.equal(evs.filter((e) => e.type === 'token').map((e) => e.text).join(''), 'from b');
    assert.ok(Date.now() - t0 < 500, 'no waiting between keys');
    assert.deepEqual(calls.map((c) => c.auth.slice(-15)), ['_a_xxxxxxxxxxxx', '_b_xxxxxxxxxxxx']);
    const a = pool.status().find((k) => k.id === 'a');
    assert.equal(a.state, 'cooling');
    assert.ok(a.cooldownMs > 40_000 && a.cooldownMs <= 45_000, 'cooldown follows retry-after');
    // the next request does not even try the limited key
    calls.length = 0;
    await drain(g.chat({ model: 'm', messages: [{ role: 'user', content: 'again' }] }));
    assert.equal(calls.length, 1);
    assert.match(calls[0].auth, /_b_/);
  } finally {
    closeServer(server);
  }
});

test('groq: every key limited → GroqUnavailableError with the retry time; an invalid key is disabled; a bad request is not retried', async () => {
  const { server, calls, url } = await fakeGroq(({ res, auth, n }) => {
    if (auth.includes('_bad_')) {
      res.writeHead(401, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'Invalid API Key' } }));
    }
    if (auth.includes('_req_')) {
      res.writeHead(400, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'The model `nope` does not exist' } }));
    }
    res.writeHead(429, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'Rate limit reached. Please try again in 1m30s.' } }));
  });
  try {
    const pool = new KeyPool();
    pool.setKeys(keysOf('a', 'bad', 'b'));
    const g = new GroqClient({ pool, baseUrl: url, timeoutMs: 2000 });
    await assert.rejects(drain(g.chat({ model: 'm', messages: [{ role: 'user', content: 'hi' }] })), (e) => e instanceof GroqUnavailableError && e.retryAfterMs > 60_000);
    assert.equal(calls.length, 3, 'each key was tried exactly once');
    assert.equal(pool.status().find((k) => k.id === 'bad').state, 'disabled');
    assert.equal(g.usable(), false);

    const pool2 = new KeyPool();
    pool2.setKeys(keysOf('req', 'b'));
    const g2 = new GroqClient({ pool: pool2, baseUrl: url, timeoutMs: 2000 });
    calls.length = 0;
    await assert.rejects(drain(g2.chat({ model: 'nope', messages: [{ role: 'user', content: 'hi' }] })), GroqRequestError);
    assert.equal(calls.length, 1, 'a request error is not retried on another key');
    assert.equal(pool2.status().find((k) => k.id === 'req').state, 'ready', 'and does not punish the key');
  } finally {
    closeServer(server);
  }
});

test('groq: server errors and dead connections rest the key and try the next', async () => {
  const { server, url } = await fakeGroq(({ res, auth }) => {
    if (auth.includes('_a_')) {
      res.writeHead(503);
      return res.end('overloaded');
    }
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end(sse(['ok']));
  });
  try {
    const pool = new KeyPool();
    pool.setKeys(keysOf('a', 'b'));
    const g = new GroqClient({ pool, baseUrl: url, timeoutMs: 2000 });
    const evs = await drain(g.chat({ model: 'm', messages: [{ role: 'user', content: 'hi' }] }));
    assert.equal(evs.filter((e) => e.type === 'token').length, 1);
    assert.equal(pool.status().find((k) => k.id === 'a').state, 'cooling');
    const dead = new GroqClient({ pool: Object.assign(new KeyPool(), {}), baseUrl: 'http://127.0.0.1:1', timeoutMs: 500 });
    dead.pool.setKeys(keysOf('x'));
    await assert.rejects(drain(dead.chat({ model: 'm', messages: [{ role: 'user', content: 'hi' }] })), GroqUnavailableError);
  } finally {
    closeServer(server);
  }
});

test('groq: probe() checks a key and lists chat models', async () => {
  const { server, url } = await fakeGroq(({ res, auth }) => {
    if (auth.includes('bad')) {
      res.writeHead(401);
      return res.end('{}');
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ data: [{ id: 'llama-3.3-70b-versatile', active: true }, { id: 'whisper-large-v3' }, { id: 'llama-3.1-8b-instant' }, { id: 'old', active: false }] }));
  });
  try {
    const ok = await GroqClient.probe('gsk_good', { baseUrl: url });
    assert.deepEqual(ok.models, ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile']);
    const bad = await GroqClient.probe('gsk_bad', { baseUrl: url });
    assert.equal(bad.ok, false);
    assert.match(bad.error, /rejected/);
  } finally {
    closeServer(server);
  }
});

// ------------------------------------------------------------------ router

function fakeOllama(text = 'local answer') {
  const calls = [];
  return {
    calls,
    numCtx: 4096,
    reusesPrefix: () => true,
    async *chat(req) {
      calls.push(req);
      yield { type: 'token', text };
      yield { type: 'done', stats: { evalTokens: 2 } };
    },
    async prefill() {
      return { promptTokens: 10 };
    },
  };
}
function fakeGroqClient({ pool, behave }) {
  const calls = [];
  return {
    calls,
    pool,
    usable: () => pool.hasAvailable(),
    async *chat(req) {
      calls.push(req);
      yield* behave(req);
    },
  };
}
function mkRouter({ settings = {}, behave, ollama = fakeOllama(), keys = ['a', 'b'] } = {}) {
  const pool = new KeyPool();
  pool.setKeys(keysOf(...keys));
  const groq = fakeGroqClient({ pool, behave: behave || (async function* () { yield { type: 'token', text: 'groq answer' }; yield { type: 'done', stats: { provider: 'groq' } }; }) });
  const s = { ...DEFAULT_SETTINGS, llmProvider: 'groq', groqModel: 'llama-3.3-70b-versatile', groqFallback: true, ...settings };
  const router = new RoutedLlm({ ollama, groq, getSettings: () => s, gates: { ollama: new Semaphore(1), groq: new Semaphore(4) }, localReady: () => true });
  return { router, ollama, groq, pool, s };
}
const text = (evs) => evs.filter((e) => e.type === 'token').map((e) => e.text).join('');

test('router: provider ollama never touches Groq', async () => {
  const { router, ollama, groq } = mkRouter({ settings: { llmProvider: 'ollama' } });
  const evs = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(evs), 'local answer');
  assert.equal(groq.calls.length, 0);
  assert.equal(ollama.calls.length, 1);
  assert.deepEqual(evs[0], { type: 'meta', provider: 'ollama', model: 'gemma3:4b' });
});

test('router: provider groq answers from Groq with the Groq model and reports it', async () => {
  const { router, ollama, groq } = mkRouter();
  const evs = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(evs), 'groq answer');
  assert.equal(groq.calls[0].model, 'llama-3.3-70b-versatile');
  assert.equal(ollama.calls.length, 0);
  assert.deepEqual(evs[0], { type: 'meta', provider: 'groq', model: 'llama-3.3-70b-versatile' });
  assert.equal(router.status().counters.groq, 1);
});

test('router: all keys exhausted → the local model answers; when keys recover Groq is used again', async () => {
  let fail = true;
  const { router, ollama, groq, pool } = mkRouter({
    behave: async function* () {
      if (fail) {
        pool.rateLimited('a', { retryAfterMs: 60_000 });
        pool.rateLimited('b', { retryAfterMs: 60_000 });
        throw new GroqUnavailableError('every Groq key is rate-limited right now', { kind: 'exhausted', retryAfterMs: 60_000 });
      }
      yield { type: 'token', text: 'groq again' };
      yield { type: 'done', stats: {} };
    },
  });
  const first = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(first), 'local answer');
  assert.equal(first[0].provider, 'groq', 'it announced Groq first…');
  assert.equal(first.find((e) => e.type === 'meta' && e.provider === 'ollama').model, 'gemma3:4b', '…then corrected itself before any text');
  assert.equal(router.status().counters.fallbacks, 1);
  // while every key is cooling, Groq is not even tried
  const callsBefore = groq.calls.length;
  const second = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(second), 'local answer');
  assert.equal(groq.calls.length, callsBefore, 'no wasted Groq call while keys are cooling');
  assert.equal(ollama.calls.length, 2);
  // keys come back
  pool.setKeys(keysOf('a', 'b'));
  fail = false;
  pool.clearDisabled();
  pool.setKeys(keysOf('c')); // fresh key
  const third = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(third), 'groq again');
});

test('router: with fallback off an exhausted Groq fails instead of switching; a request that already spoke is not switched', async () => {
  const noFallback = mkRouter({
    settings: { groqFallback: false },
    behave: async function* () {
      throw new GroqUnavailableError('limited', { kind: 'exhausted' });
    },
  });
  await assert.rejects(drain(noFallback.router.chat({ model: 'x', messages: [] })), GroqUnavailableError);
  assert.equal(noFallback.ollama.calls.length, 0);

  const partial = mkRouter({
    behave: async function* () {
      yield { type: 'token', text: 'half a sent' };
      throw new Error('connection reset');
    },
  });
  await assert.rejects(drain(partial.router.chat({ model: 'x', messages: [] })), /connection reset/);
  assert.equal(partial.ollama.calls.length, 0, 'never mixes two models inside one reply');
});

test('router: a bad request (unknown Groq model) also falls back locally; abort is honoured', async () => {
  const { router, ollama } = mkRouter({
    behave: async function* () {
      throw new GroqRequestError('model does not exist', 404);
    },
  });
  assert.equal(text(await drain(router.chat({ model: 'x', messages: [] }))), 'local answer');
  assert.equal(ollama.calls.length, 1);
  const ctl = new AbortController();
  ctl.abort(new Error('superseded'));
  await assert.rejects(drain(router.chat({ model: 'x', messages: [], signal: ctl.signal })), /superseded/);
});

test('router: background work stays on the local model while it is ready, and think-ahead is skipped for Groq', async () => {
  const { router, ollama, groq } = mkRouter();
  await drain(router.chat({ model: 'gemma3:4b', messages: [], priority: 3 }));
  assert.equal(groq.calls.length, 0);
  assert.equal(ollama.calls.length, 1);
  assert.equal(router.reusesPrefix('llama3.2:3b'), false, 'no prefix cache to warm on the hosted route');
  assert.equal((await router.prefill({ model: 'x', messages: [] })).skipped, true);
  const local = mkRouter({ settings: { llmProvider: 'ollama' } });
  assert.equal(local.router.reusesPrefix('llama3.2:3b'), true);
  assert.equal((await local.router.prefill({ model: 'x', messages: [] })).promptTokens, 10);
});

test('router: a request can choose its own provider (playground) without changing the live setting', async () => {
  const { router, groq, ollama } = mkRouter({ settings: { llmProvider: 'groq' } });
  await drain(router.chat({ model: 'gemma3:4b', messages: [], provider: 'ollama' }));
  assert.equal(groq.calls.length, 0);
  assert.equal(ollama.calls.length, 1);
});

test('router: no keys configured → local (or a clear error with fallback off)', async () => {
  const { router, ollama } = mkRouter({ keys: [] });
  assert.equal(text(await drain(router.chat({ model: 'x', messages: [] }))), 'local answer');
  assert.equal(ollama.calls.length, 1);
});

// ------------------------------------------------------------------ key store

test('key list parsing and store: env keys + stored keys, deduplicated, unreadable ones counted', async () => {
  assert.deepEqual(parseKeyList('gsk_a, gsk_b\ngsk_a  gsk_c;'), ['gsk_a', 'gsk_b', 'gsk_c']);
  const pool = new KeyPool();
  const rows = [
    { id: 1, label: 'stored one', keyEnc: encryptSecret('gsk_stored_key_111111111111', 'S') },
    { id: 2, label: 'dup of env', keyEnc: encryptSecret('gsk_env_key_2222222222222', 'S') },
    { id: 3, label: 'old secret', keyEnc: encryptSecret('gsk_lost_key_33333333333', 'DIFFERENT') },
  ];
  const store = new KeyStore({ pool, secret: 'S', envKeys: 'gsk_env_key_1111111111111, gsk_env_key_2222222222222', repo: { listLlmKeys: async () => rows } });
  assert.equal(await store.load(), 3);
  assert.deepEqual(pool.status().map((k) => [k.id, k.source]), [['env:1', 'env'], ['env:2', 'env'], ['db:1', 'console']]);
  assert.equal(store.unreadable, 1);
  assert.equal(pool.status().every((k) => !k.hint.includes('gsk_env_key_1111')), true, 'hints are masked');
});

// ------------------------------------------------------------------ through the real pipeline

import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';

test('pipeline: the reply meta names the provider and model that really answered, and falls back mid-request', async () => {
  const kb = new KnowledgeBase({ dir: new URL('../knowledge_base', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1') });
  await kb.load().catch(() => {});
  const run = async (router) => {
    const brain = createBrain({ kb, llm: router, ...createMemoryStores() });
    const chunks = [];
    let done;
    for await (const ev of brain.respond({ chatId: 'c', text: 'how does a closure work in javascript', settings: { ...DEFAULT_SETTINGS, ownerName: 'Sumit', llmProvider: router.getSettings().llmProvider, groqModel: 'llama-3.3-70b-versatile', model: 'gemma3:4b' } })) {
      if (ev.type === 'chunk') chunks.push(ev);
      else if (ev.type === 'done') done = ev;
    }
    return { chunks, done };
  };
  const viaGroq = await run(mkRouter().router);
  assert.equal(viaGroq.chunks[0].meta.model, 'llama-3.3-70b-versatile');
  assert.equal(viaGroq.chunks[0].meta.provider, 'groq');
  assert.equal(viaGroq.done.meta.provider, 'groq');

  const exhausted = mkRouter({
    behave: async function* () {
      throw new GroqUnavailableError('every key limited', { kind: 'exhausted' });
    },
  });
  const viaLocal = await run(exhausted.router);
  assert.equal(viaLocal.chunks[0].meta.model, 'gemma3:4b', 'the console shows the local model that answered, not Groq');
  assert.equal(viaLocal.chunks[0].meta.provider, 'ollama');
  assert.equal(viaLocal.chunks.map((c) => c.text).join(' '), 'local answer');
});
