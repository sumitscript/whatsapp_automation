import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { KeyPool, parseDuration } from '../src/llm/key-pool.js';
import { GroqClient, GroqRequestError, GroqUnavailableError, orgFromMessage } from '../src/llm/groq.js';
import { RoutedLlm, parseModelList } from '../src/llm/router.js';
import { KeyStore, isGroqKeyFormat, parseKeyList } from '../src/llm/key-store.js';
import { decryptSecret, encryptSecret, maskKey } from '../src/core/secrets.js';
import { Semaphore } from '../src/core/util.js';
import { DEFAULT_SETTINGS, sanitizeSettings } from '../src/core/settings-schema.js';
import { createBrain, createMemoryStores } from '../src/brain/pipeline.js';
import { KnowledgeBase } from '../src/rag/kb.js';

const keysOf = (...ids) => ids.map((id) => ({ id, label: id, secret: `gsk_secret_${id}_xxxxxxxxxxxx`, source: 'env' }));
const M = ['gpt120', 'gpt20', 'qwen']; // the model priority used in these tests
const drain = async (gen) => {
  const out = [];
  for await (const ev of gen) out.push(ev);
  return out;
};
const picks = (pool, n, opts = {}) => Array.from({ length: n }, () => {
  const p = pool.pick({ models: M, ...opts });
  return p && `${p.id}|${p.model}`;
});

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

test('durations from Groq headers and the organization in a rate-limit message', () => {
  assert.equal(parseDuration('2m59.56s'), 179_560);
  assert.equal(parseDuration('7.66s'), 7660);
  assert.equal(parseDuration('250ms'), 250);
  assert.equal(parseDuration('1h2m3s'), 3_723_000);
  assert.equal(parseDuration('12'), 12_000);
  assert.equal(parseDuration(null), null);
  assert.equal(parseDuration('soon'), null);
  assert.equal(orgFromMessage('Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abcDEF23` service tier `on_demand` on tokens per day (TPD)'), 'org_01abcDEF23');
  assert.equal(orgFromMessage('nothing here'), null);
});

test('settings: the three provider modes, the model priority list, the strategy and the break message are validated', () => {
  assert.equal(DEFAULT_SETTINGS.llmProvider, 'auto', 'adding a key is enough: no second switch to flip');
  assert.equal(DEFAULT_SETTINGS.groqStrategy, 'priority');
  assert.equal(DEFAULT_SETTINGS.groqModels, 'openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b');
  for (const p of ['auto', 'ollama', 'groq']) assert.deepEqual(sanitizeSettings({ llmProvider: p }).value, { llmProvider: p });
  assert.equal(sanitizeSettings({ llmProvider: 'openai' }).errors.length, 1);
  assert.equal(sanitizeSettings({ groqModel: 'bad model!' }).errors.length, 1);
  assert.deepEqual(sanitizeSettings({ groqModels: 'openai/gpt-oss-120b, openai/gpt-oss-20b ,openai/gpt-oss-120b' }).value, { groqModels: 'openai/gpt-oss-120b,openai/gpt-oss-20b' }, 'trimmed and de-duplicated');
  assert.deepEqual(sanitizeSettings({ groqModels: ['a/b', 'c'] }).value, { groqModels: 'a/b,c' });
  for (const bad of ['', 'bad model!', 'a,b,c,d,e,f,g,h,i']) assert.equal(sanitizeSettings({ groqModels: bad }).errors.length, 1, JSON.stringify(bad));
  assert.deepEqual(sanitizeSettings({ groqStrategy: 'spread' }).value, { groqStrategy: 'spread' });
  assert.equal(sanitizeSettings({ groqStrategy: 'random' }).errors.length, 1);
  assert.deepEqual(sanitizeSettings({ groqBusyMessage: 'On leave 🌴 {name}' }).value, { groqBusyMessage: 'On leave 🌴 {name}' });
  assert.equal(sanitizeSettings({ groqBusyMessage: '' }).errors.length, 1);
  assert.deepEqual(parseModelList('a, b  c,a'), ['a', 'b', 'c']);
});

// ------------------------------------------------------------------ the pool: accounts in priority order, limits per model

test('pool: the hierarchy: account 1 with all its models first, then account 2, then account 3', () => {
  const pool = new KeyPool();
  pool.setKeys(keysOf('acc1', 'acc2', 'acc3'));
  assert.equal(pool.pick({ models: M }).id, 'acc1');
  // account 1 runs out of gpt120: its OWN next model serves, not the next account
  pool.rateLimited('acc1', { model: 'gpt120', retryAfterMs: 60_000 });
  assert.deepEqual(picks(pool, 2), ['acc1|gpt20', 'acc1|gpt20']);
  pool.rateLimited('acc1', { model: 'gpt20', retryAfterMs: 60_000 });
  assert.equal(picks(pool, 1)[0], 'acc1|qwen');
  pool.rateLimited('acc1', { model: 'qwen', retryAfterMs: 60_000 });
  // every model of account 1 is used up: account 2 starts again at the TOP of the list (its own quotas)
  assert.equal(picks(pool, 1)[0], 'acc2|gpt120');
  pool.rateLimited('acc2', { model: 'gpt120', retryAfterMs: 60_000 });
  pool.rateLimited('acc2', { model: 'gpt20', retryAfterMs: 60_000 });
  assert.equal(picks(pool, 1)[0], 'acc2|qwen');
  pool.rateLimited('acc2', { model: 'qwen', retryAfterMs: 60_000 });
  assert.equal(picks(pool, 1)[0], 'acc3|gpt120');
  for (const m of M) pool.rateLimited('acc3', { model: m, retryAfterMs: 60_000 });
  assert.equal(picks(pool, 1)[0], null, 'everything is used up: the caller falls back to the local model');
  assert.equal(pool.hasAvailable(M), false);
});

test('pool: a limit on one model does not touch the other models of that account, and a rested pair comes back', () => {
  let t = 1000;
  const pool = new KeyPool({ now: () => t });
  pool.setKeys(keysOf('a', 'b'));
  pool.rateLimited('a', { model: 'gpt120', retryAfterMs: 30_000 });
  const s = pool.status(M)[0];
  assert.deepEqual(s.models.map((m) => [m.id, m.state]), [['gpt120', 'cooling'], ['gpt20', 'ready'], ['qwen', 'ready']]);
  assert.equal(s.state, 'ready', 'the account is still usable through its other models');
  assert.equal(pool.nextAvailableMs(M), 0);
  t += 30_001;
  assert.equal(picks(pool, 1)[0], 'a|gpt120', 'the preferred model is used again as soon as its window resets');
});

test('pool: spread strategy lets accounts take turns while models stay in priority order', () => {
  const pool = new KeyPool();
  pool.setStrategy('spread');
  pool.setKeys(keysOf('a', 'b', 'c'));
  assert.deepEqual(picks(pool, 6), ['a|gpt120', 'b|gpt120', 'c|gpt120', 'a|gpt120', 'b|gpt120', 'c|gpt120']);
  pool.rateLimited('b', { model: 'gpt120', retryAfterMs: 60_000 });
  assert.deepEqual(picks(pool, 3), ['a|gpt120', 'b|gpt20', 'c|gpt120'], 'b is on its next model, the others are untouched, and the turns continue');
});

test('pool: keys of one organization share their limits, once that is learned from a rate-limit message', () => {
  const pool = new KeyPool();
  pool.setKeys(keysOf('a', 'b', 'c'));
  pool.rateLimited('a', { model: 'gpt120', retryAfterMs: 60_000, org: 'org_01JXABCDEFGH' });
  assert.equal(pool.pick({ models: M }).id + '', 'a', 'b is not known to share the organization yet, a rests only on gpt120');
  // b is then rate-limited with the same organization named in the message: from now on they are one quota pool
  pool.rateLimited('b', { model: 'qwen', retryAfterMs: 60_000, org: 'org_01JXABCDEFGH' });
  const st = pool.status(M);
  assert.equal(st[1].sameOrgAs, 'a');
  assert.equal(st[2].sameOrgAs, null);
  assert.deepEqual(st[1].models.map((m) => m.state), ['cooling', 'ready', 'cooling'], 'b inherits a\'s gpt120 rest and a inherits b\'s qwen rest');
  assert.deepEqual(st[0].models.map((m) => m.state), ['cooling', 'ready', 'cooling']);
  assert.equal(picks(pool, 1, { exclude: new Set() })[0], 'a|gpt20');
  assert.match(st[0].org, /^org_…/, 'the organization id is masked');
});

test('pool: "model not found" retires the pair (the account may not have it); other models and accounts still work', () => {
  const pool = new KeyPool();
  pool.setKeys(keysOf('a', 'b'));
  pool.modelUnavailable('a', 'gpt120', 'does not exist');
  assert.equal(picks(pool, 1)[0], 'a|gpt20');
  const a = pool.status(M)[0].models[0];
  assert.equal(a.gone, true);
  assert.ok(a.cooldownMs > 5 * 3_600_000);
  pool.clearDisabled();
  assert.equal(picks(pool, 1)[0], 'a|gpt120', 'editing the model list or keys forgets the "gone" mark');
});

test('pool: nothing available when every account is disabled or resting; exclude skips tried pairs and whole keys', () => {
  const pool = new KeyPool();
  pool.setKeys(keysOf('a', 'b'));
  assert.equal(pool.pick({ models: M, exclude: new Set(['a', 'b']) }), null);
  assert.equal(pool.pick({ models: M, exclude: new Set(['a|gpt120', 'a|gpt20', 'a|qwen']) }).id, 'b');
  pool.authFailed('a');
  assert.equal(pool.status(M)[0].state, 'disabled');
  assert.equal(pool.pick({ models: M }).id, 'b');
  pool.authFailed('b');
  assert.equal(pool.hasAvailable(M), false);
  assert.equal(pool.nextAvailableMs(M), null, 'nothing will come back by itself');
  pool.clearDisabled();
  assert.equal(pool.pick({ models: M }).id, 'a');
});

test('pool: a pair with no tokens left for this request is passed over while a roomier one exists; an exhausted window rests the pair', () => {
  let t = 0;
  const pool = new KeyPool({ now: () => t });
  pool.setKeys(keysOf('a', 'b'));
  pool.success('a', { model: 'gpt120', remainingTokens: 100 });
  assert.equal(pool.pick({ models: M, needTokens: 2000 }).model, 'gpt20', 'a|gpt120 has 100 tokens left, a|gpt20 is unknown: take the roomy one');
  pool.success('a', { model: 'gpt20', remainingTokens: 50 });
  pool.success('a', { model: 'qwen', remainingTokens: 40 });
  assert.equal(pool.pick({ models: M, needTokens: 2000 }).id, 'b', 'every a pair is low: the next account has room');
  pool.success('b', { model: 'gpt120', remainingRequests: 0, resetRequestsMs: 8000 });
  assert.equal(pool.status(M)[1].models[0].state, 'cooling', 'out of requests: rested until the window resets, no wasted 429');
  t = 8001;
  assert.equal(pool.status(M)[1].models[0].state, 'ready');
});

test('pool: network trouble rests the whole account (all models) with a growing backoff; reloading keeps state of unchanged keys', () => {
  let t = 0;
  const pool = new KeyPool({ now: () => t });
  pool.setKeys(keysOf('a'));
  pool.transient('a');
  assert.equal(pool.status(M)[0].cooldownMs, 2000);
  assert.equal(pool.status(M)[0].state, 'cooling');
  t = 2001;
  pool.transient('a');
  assert.equal(pool.status(M)[0].cooldownMs, 4000);
  pool.success('a');
  pool.rateLimited('a', { model: 'gpt120', retryAfterMs: 5000 });
  pool.setKeys([...keysOf('a'), ...keysOf('b')]);
  assert.equal(pool.status(M)[0].models[0].state, 'cooling', 'state survives a reload');
  pool.setKeys([{ id: 'a', label: 'a', secret: 'gsk_new_secret_value_1234567', source: 'env' }]);
  assert.equal(pool.status(M)[0].state, 'ready', 'a replaced secret starts clean');
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
        const parsed = body ? JSON.parse(body) : null;
        calls.push({ url: req.url, auth, body: parsed, who: `${(auth.match(/secret_(\w+?)_/) || [])[1]}|${parsed?.model}` });
        handler({ req, res, auth, model: parsed?.model, who: calls.at(-1).who, n: calls.length });
      });
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, calls, url: `http://127.0.0.1:${server.address().port}/openai/v1` }));
  });
}
const closeServer = (server) => {
  server.closeAllConnections?.();
  server.close();
};
const ok = (res, tokens = ['ok'], headers = {}) => {
  res.writeHead(200, { 'content-type': 'text/event-stream', ...headers });
  res.end(sse(tokens));
};
const limit = (res, msg = 'Rate limit reached', retryAfter = '45') => {
  res.writeHead(429, { 'retry-after': retryAfter, 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: { message: msg } }));
};
const mkClient = (url, keys = ['a', 'b'], opts = {}) => {
  const pool = new KeyPool();
  pool.setKeys(keysOf(...keys));
  pool.setModels(M);
  return { pool, g: new GroqClient({ pool, baseUrl: url, timeoutMs: 2000, ...opts }) };
};
const ask = (g, extra = {}) => drain(g.chat({ messages: [{ role: 'user', content: 'hi' }], ...extra }));

test('groq: streams tokens with stats, names the model that answered, and sends the key', async () => {
  const { server, calls, url } = await fakeGroq(({ res }) => ok(res, ['Hel', 'lo ', 'there'], { 'x-ratelimit-remaining-tokens': '9000', 'x-ratelimit-remaining-requests': '20' }));
  try {
    const { pool, g } = mkClient(url, ['a']);
    const evs = await ask(g, { temperature: 0.5, maxTokens: 50 });
    assert.deepEqual(evs[0], { type: 'meta', provider: 'groq', model: 'gpt120', keyLabel: 'a' });
    assert.equal(evs.filter((e) => e.type === 'token').map((e) => e.text).join(''), 'Hello there');
    const done = evs.at(-1);
    assert.equal(done.stats.promptTokens, 30);
    assert.equal(done.stats.doneReason, 'stop');
    assert.equal(done.stats.model, 'gpt120');
    assert.equal(calls[0].auth, 'Bearer gsk_secret_a_xxxxxxxxxxxx');
    assert.equal(calls[0].body.model, 'gpt120');
    assert.equal(calls[0].body.max_tokens, 50);
    assert.equal(pool.status(M)[0].models[0].remainingTokens, 9000, 'the remaining budget is tracked per model');
  } finally {
    closeServer(server);
  }
});

test('groq: a 429 on one model moves the SAME call to that account\'s next model, then to the next account, with no pause', async () => {
  const { server, calls, url } = await fakeGroq(({ res, who }) => {
    // account a is out of gpt120 and gpt20; b is out of gpt120; qwen on a still has room
    if (['a|gpt120', 'a|gpt20', 'b|gpt120'].includes(who)) return limit(res);
    ok(res, [`from ${who}`]);
  });
  try {
    const { pool, g } = mkClient(url);
    const t0 = Date.now();
    const evs = await ask(g);
    assert.equal(evs.filter((e) => e.type === 'token').map((e) => e.text).join(''), 'from a|qwen');
    assert.ok(Date.now() - t0 < 500, 'no waiting between attempts');
    assert.deepEqual(calls.map((c) => c.who), ['a|gpt120', 'a|gpt20', 'a|qwen'], 'account a\'s models in priority order, b untouched');
    assert.deepEqual(pool.status(M)[0].models.map((m) => m.state), ['cooling', 'cooling', 'ready']);
    // the next request does not retry the pairs that are resting
    calls.length = 0;
    await ask(g);
    assert.deepEqual(calls.map((c) => c.who), ['a|qwen']);
    // qwen is limited as well: now account b, from the top of its list
    pool.rateLimited('a', { model: 'qwen', retryAfterMs: 60_000 });
    calls.length = 0;
    const evs2 = await ask(g);
    assert.deepEqual(calls.map((c) => c.who), ['b|gpt120', 'b|gpt20'], 'b|gpt120 was limited by the first request, so b continues at gpt20');
    assert.equal(evs2.find((e) => e.type === 'meta').model, 'gpt20');
  } finally {
    closeServer(server);
  }
});

test('groq: a rate limit naming the organization makes keys of that organization share the rest', async () => {
  const { server, calls, url } = await fakeGroq(({ res, who }) => {
    if (who.endsWith('|gpt120')) return limit(res, 'Rate limit reached for model `gpt120` in organization `org_SHARED` service tier `on_demand` on tokens per day (TPD)', '600');
    ok(res, [who]);
  });
  try {
    const { pool, g } = mkClient(url, ['a', 'b']);
    await ask(g); // a|gpt120 429 (org learned) → a|gpt20
    pool.rateLimited('b', { model: 'qwen', retryAfterMs: 1000, org: 'org_SHARED' }); // b is another key of the same org
    calls.length = 0;
    await ask(g);
    assert.deepEqual(calls.map((c) => c.who), ['a|gpt20'], 'the shared gpt120 limit is not retried on b either');
    assert.equal(pool.status(M)[1].sameOrgAs, 'a');
  } finally {
    closeServer(server);
  }
});

test('groq: "model not found" on one account skips just that model; an invalid key skips all of its models', async () => {
  const { server, calls, url } = await fakeGroq(({ res, who, auth }) => {
    if (auth.includes('_bad_')) {
      res.writeHead(401, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'Invalid API Key' } }));
    }
    if (who === 'a|gpt120') {
      res.writeHead(404, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'The model `gpt120` does not exist or you do not have access to it.' } }));
    }
    ok(res, [who]);
  });
  try {
    const { pool, g } = mkClient(url, ['bad', 'a']);
    const evs = await ask(g);
    assert.equal(evs.filter((e) => e.type === 'token').map((e) => e.text).join(''), 'a|gpt20');
    assert.deepEqual(calls.map((c) => c.who), ['bad|gpt120', 'a|gpt120', 'a|gpt20'], 'the bad key is tried once, not once per model');
    assert.equal(pool.status(M)[0].state, 'disabled');
    assert.equal(pool.status(M)[1].models[0].gone, true);
  } finally {
    closeServer(server);
  }
});

test('groq: every account and model used up → GroqUnavailableError with the retry time; a bad request is not retried elsewhere', async () => {
  const { server, calls, url } = await fakeGroq(({ res, who }) => {
    if (who.startsWith('req|')) {
      res.writeHead(400, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'bad parameter' } }));
    }
    limit(res, 'Rate limit reached. Please try again in 1m30s.', '90');
  });
  try {
    const { pool, g } = mkClient(url, ['a', 'b']);
    await assert.rejects(ask(g), (e) => e instanceof GroqUnavailableError && e.retryAfterMs > 60_000);
    assert.equal(calls.length, 6, 'each (account, model) pair was tried exactly once');
    assert.equal(g.usable(M), false);
    const c2 = mkClient(url, ['req', 'b']);
    calls.length = 0;
    await assert.rejects(ask(c2.g), GroqRequestError);
    assert.equal(calls.length, 1, 'a request error is not retried on another account');
    assert.equal(c2.pool.status(M)[0].state, 'ready', 'and does not punish the account');
  } finally {
    closeServer(server);
  }
});

test('groq: a request too large for one model (413) goes to the next model; server errors and dead connections rest the account', async () => {
  const { server, calls, url } = await fakeGroq(({ res, who }) => {
    if (who === 'a|gpt120') {
      res.writeHead(413, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: { message: 'Request too large for model on tokens per minute' } }));
    }
    if (who.startsWith('s|')) {
      res.writeHead(503);
      return res.end('overloaded');
    }
    ok(res, [who]);
  });
  try {
    const big = mkClient(url, ['a']);
    assert.equal((await ask(big.g)).find((e) => e.type === 'meta').model, 'gpt20');
    const flaky = mkClient(url, ['s', 'b']);
    calls.length = 0;
    const evs = await ask(flaky.g);
    assert.equal(evs.find((e) => e.type === 'meta').keyLabel, 'b');
    assert.deepEqual(calls.map((c) => c.who), ['s|gpt120', 'b|gpt120'], 'a 503 skips the account\'s other models too');
    assert.equal(flaky.pool.status(M)[0].state, 'cooling');
    const dead = new GroqClient({ pool: Object.assign(new KeyPool(), {}), baseUrl: 'http://127.0.0.1:1', timeoutMs: 500 });
    dead.pool.setKeys(keysOf('x'));
    await assert.rejects(drain(dead.chat({ models: M, messages: [{ role: 'user', content: 'hi' }] })), GroqUnavailableError);
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
    res.end(JSON.stringify({ data: [{ id: 'llama-3.3-70b-versatile', active: true }, { id: 'whisper-large-v3' }, { id: 'canopylabs/orpheus-v1-english' }, { id: 'llama-3.1-8b-instant' }, { id: 'old', active: false }] }));
  });
  try {
    const okR = await GroqClient.probe('gsk_good', { baseUrl: url });
    assert.deepEqual(okR.models, ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile']);
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
    usable: (models) => pool.hasAvailable(models?.length ? models : pool.models),
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
  const s = { ...DEFAULT_SETTINGS, llmProvider: 'auto', groqModels: M.join(','), ...settings };
  const router = new RoutedLlm({ ollama, groq, getSettings: () => s, gates: { ollama: new Semaphore(1), groq: new Semaphore(4) }, localReady: () => true });
  return { router, ollama, groq, pool, s };
}
const text = (evs) => evs.filter((e) => e.type === 'token').map((e) => e.text).join('');
const exhaust = (pool) => {
  for (const k of ['a', 'b']) for (const m of M) pool.rateLimited(k, { model: m, retryAfterMs: 60_000 });
};

test('router: provider ollama never touches Groq', async () => {
  const { router, ollama, groq } = mkRouter({ settings: { llmProvider: 'ollama' } });
  const evs = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(evs), 'local answer');
  assert.equal(groq.calls.length, 0);
  assert.equal(ollama.calls.length, 1);
  assert.deepEqual(evs[0], { type: 'meta', provider: 'ollama', model: 'gemma3:4b' });
});

test('router: Groq gets the whole model priority list and the strategy; the first model is announced until the client names the real one', async () => {
  const { router, ollama, groq, pool } = mkRouter({ settings: { groqStrategy: 'spread' } });
  const evs = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(evs), 'groq answer');
  assert.deepEqual(groq.calls[0].models, M);
  assert.equal(pool.strategy, 'spread', 'the live strategy setting is applied to the pool');
  assert.deepEqual(pool.models, M);
  assert.equal(ollama.calls.length, 0);
  assert.deepEqual(evs[0], { type: 'meta', provider: 'groq', model: 'gpt120' });
  assert.equal(router.status().counters.groq, 1);
  assert.deepEqual(router.status().groqModels, M);
  assert.equal(router.status().keys[0].models.length, 3, 'per-model state for the console');
});

test('router: every account and model used up → the local model answers; when limits reset Groq is used again', async () => {
  let fail = true;
  const { router, ollama, groq, pool } = mkRouter({
    behave: async function* () {
      if (fail) {
        exhaust(pool);
        throw new GroqUnavailableError('every Groq account is rate-limited right now', { kind: 'exhausted', retryAfterMs: 60_000 });
      }
      yield { type: 'token', text: 'groq again' };
      yield { type: 'done', stats: {} };
    },
  });
  const first = await drain(router.chat({ model: 'gemma3:4b', messages: [] }));
  assert.equal(text(first), 'local answer');
  assert.equal(first.find((e) => e.type === 'meta' && e.provider === 'ollama').model, 'gemma3:4b');
  assert.equal(router.status().counters.fallbacks, 1);
  const callsBefore = groq.calls.length;
  assert.equal(text(await drain(router.chat({ model: 'gemma3:4b', messages: [] }))), 'local answer');
  assert.equal(groq.calls.length, callsBefore, 'no wasted Groq call while everything is resting');
  assert.equal(ollama.calls.length, 2);
  pool.setKeys(keysOf('c')); // a fresh account arrives
  fail = false;
  assert.equal(text(await drain(router.chat({ model: 'gemma3:4b', messages: [] }))), 'groq again');
});

test('router: Groq-only mode never uses the local model: exhausted accounts fail with GroqUnavailableError (the break message)', async () => {
  const only = mkRouter({
    settings: { llmProvider: 'groq' },
    behave: async function* () {
      throw new GroqUnavailableError('limited', { kind: 'exhausted', retryAfterMs: 45_000 });
    },
  });
  await assert.rejects(drain(only.router.chat({ model: 'x', messages: [] })), (e) => e instanceof GroqUnavailableError && e.retryAfterMs === 45_000);
  assert.equal(only.ollama.calls.length, 0, 'the local model is never touched');
  assert.equal(only.router.status().counters.busy, 1);
  exhaust(only.pool);
  const callsBefore = only.groq.calls.length;
  await assert.rejects(drain(only.router.chat({ model: 'x', messages: [] })), GroqUnavailableError);
  assert.equal(only.groq.calls.length, callsBefore);
  assert.equal(only.ollama.calls.length, 0);
  const noKey = mkRouter({ settings: { llmProvider: 'groq' }, keys: [] });
  await assert.rejects(drain(noKey.router.chat({ model: 'x', messages: [] })), (e) => e instanceof GroqUnavailableError && e.kind === 'no_keys');
  const bad = mkRouter({
    settings: { llmProvider: 'groq' },
    behave: async function* () {
      throw new GroqRequestError('bad parameter', 400);
    },
  });
  await assert.rejects(drain(bad.router.chat({ model: 'x', messages: [] })), GroqUnavailableError);
  assert.equal(bad.ollama.calls.length, 0);
  const fine = mkRouter({ settings: { llmProvider: 'groq' } });
  assert.equal(text(await drain(fine.router.chat({ model: 'x', messages: [] }))), 'groq answer');
});

test('router: needsLocal says when the local model must be loaded', () => {
  const key = ['a'];
  assert.equal(mkRouter({ settings: { llmProvider: 'ollama' }, keys: key }).router.needsLocal(), true);
  assert.equal(mkRouter({ settings: { llmProvider: 'groq' }, keys: [] }).router.needsLocal(), false, 'Groq-only never needs it');
  assert.equal(mkRouter({ settings: { llmProvider: 'auto' }, keys: key }).router.needsLocal(), false, 'an account is ready');
  assert.equal(mkRouter({ settings: { llmProvider: 'auto' }, keys: [] }).router.needsLocal(), true, 'no key: auto is local');
  const cooling = mkRouter({ settings: { llmProvider: 'auto' }, keys: key });
  for (const m of M) cooling.pool.rateLimited('a', { model: m, retryAfterMs: 60_000 });
  assert.equal(cooling.router.needsLocal(), true, 'every model of every account resting: auto needs the local model');
  const partial = mkRouter({ settings: { llmProvider: 'auto' }, keys: key });
  partial.pool.rateLimited('a', { model: 'gpt120', retryAfterMs: 60_000 });
  assert.equal(partial.router.needsLocal(), false, 'one model resting does not push anything to the local model');
});

test('router: a half-sent reply is not switched to another model', async () => {
  const partial = mkRouter({
    behave: async function* () {
      yield { type: 'token', text: 'half a sent' };
      throw new Error('connection reset');
    },
  });
  await assert.rejects(drain(partial.router.chat({ model: 'x', messages: [] })), /connection reset/);
  assert.equal(partial.ollama.calls.length, 0, 'never mixes two models inside one reply');
});

test('router: a bad request in auto mode falls back locally; abort is honoured', async () => {
  const { router, ollama } = mkRouter({
    behave: async function* () {
      throw new GroqRequestError('bad parameter', 400);
    },
  });
  assert.equal(text(await drain(router.chat({ model: 'x', messages: [] }))), 'local answer');
  assert.equal(ollama.calls.length, 1);
  const ctl = new AbortController();
  ctl.abort(new Error('superseded'));
  await assert.rejects(drain(router.chat({ model: 'x', messages: [], signal: ctl.signal })), /superseded/);
});

test('router: background work stays local while it is ready, think-ahead is skipped for Groq, a request can pick its own provider/models', async () => {
  const { router, ollama, groq } = mkRouter();
  await drain(router.chat({ model: 'gemma3:4b', messages: [], priority: 3 }));
  assert.equal(groq.calls.length, 0);
  assert.equal(ollama.calls.length, 1);
  assert.equal(router.reusesPrefix('llama3.2:3b'), false);
  assert.equal((await router.prefill({ model: 'x', messages: [] })).skipped, true);
  const local = mkRouter({ settings: { llmProvider: 'ollama' } });
  assert.equal(local.router.reusesPrefix('llama3.2:3b'), true);
  assert.equal((await local.router.prefill({ model: 'x', messages: [] })).promptTokens, 10);
  const pg = mkRouter({ settings: { llmProvider: 'groq' } });
  await drain(pg.router.chat({ model: 'gemma3:4b', messages: [], provider: 'ollama' }));
  assert.equal(pg.groq.calls.length, 0);
  assert.equal(pg.ollama.calls.length, 1);
  const one = mkRouter();
  await drain(one.router.chat({ model: 'x', messages: [], groqModel: 'only-this' }));
  assert.deepEqual(one.groq.calls[0].models, ['only-this'], 'the playground can force one model');
  const legacy = mkRouter({ settings: { groqModels: '', groqModel: 'legacy-model' } });
  await drain(legacy.router.chat({ model: 'x', messages: [] }));
  assert.deepEqual(legacy.groq.calls[0].models, ['legacy-model']);
});

test('router: auto with no key configured is simply local', async () => {
  const { router, ollama } = mkRouter({ keys: [] });
  assert.equal(text(await drain(router.chat({ model: 'x', messages: [] }))), 'local answer');
  assert.equal(ollama.calls.length, 1);
});

// ------------------------------------------------------------------ key store

test('key list parsing and store: env keys then stored keys in their saved order, deduplicated, unreadable ones counted', async () => {
  assert.deepEqual(parseKeyList('gsk_a, gsk_b\ngsk_a  gsk_c;'), ['gsk_a', 'gsk_b', 'gsk_c']);
  const pool = new KeyPool();
  const rows = [
    { id: 9, label: 'stored one', keyEnc: encryptSecret('gsk_stored_key_111111111111', 'S') },
    { id: 2, label: 'dup of env', keyEnc: encryptSecret('gsk_env_key_2222222222222', 'S') },
    { id: 3, label: 'old secret', keyEnc: encryptSecret('gsk_lost_key_33333333333', 'DIFFERENT') },
  ];
  const store = new KeyStore({ pool, secret: 'S', envKeys: 'gsk_env_key_1111111111111, gsk_env_key_2222222222222', repo: { listLlmKeys: async () => rows } });
  assert.equal(await store.load(), 3);
  assert.deepEqual(pool.status().map((k) => [k.id, k.source, k.priority]), [['env:1', 'env', 1], ['env:2', 'env', 2], ['db:9', 'console', 3]], 'the list order is the priority order');
  assert.equal(store.unreadable, 1);
  assert.equal(pool.status().every((k) => !k.hint.includes('gsk_env_key_1111')), true, 'hints are masked');
});

// ------------------------------------------------------------------ through the real pipeline

test('pipeline: the reply meta names the provider and the model that really answered; fallback shows the local model', async () => {
  const kb = new KnowledgeBase({ dir: new URL('../knowledge_base', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1') });
  await kb.load().catch(() => {});
  const run = async (router) => {
    const brain = createBrain({ kb, llm: router, ...createMemoryStores() });
    const chunks = [];
    let done;
    for await (const ev of brain.respond({ chatId: 'c', text: 'how does a closure work in javascript', settings: { ...DEFAULT_SETTINGS, ownerName: 'Sumit', llmProvider: 'auto', groqModels: M.join(','), model: 'gemma3:4b' } })) {
      if (ev.type === 'chunk') chunks.push(ev);
      else if (ev.type === 'done') done = ev;
    }
    return { chunks, done };
  };
  // a client that, like the real one, says which model accepted the call (the second one in the list)
  const viaGroq = await run(mkRouter({
    behave: async function* () {
      yield { type: 'meta', provider: 'groq', model: 'gpt20', keyLabel: 'a' };
      yield { type: 'token', text: 'groq answer' };
      yield { type: 'done', stats: {} };
    },
  }).router);
  assert.equal(viaGroq.chunks[0].meta.model, 'gpt20');
  assert.equal(viaGroq.chunks[0].meta.provider, 'groq');
  const exhausted = mkRouter({
    behave: async function* () {
      throw new GroqUnavailableError('every Groq account is rate-limited right now', { kind: 'exhausted' });
    },
  });
  const viaLocal = await run(exhausted.router);
  assert.equal(viaLocal.chunks[0].meta.model, 'gemma3:4b');
  assert.equal(viaLocal.chunks[0].meta.provider, 'ollama');
  assert.equal(viaLocal.chunks.map((c) => c.text).join(' '), 'local answer');
});
