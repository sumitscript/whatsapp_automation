import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { KeyPool } from '../src/llm/key-pool.js';
import { GroqClient, GroqUnavailableError, limitFromMessage, msToUtcMidnight } from '../src/llm/groq.js';
import { CF_DAILY_NEURONS, createCloudflareClient, estimateNeurons, isCloudflareKeyFormat, parseCloudflareKey, probeCloudflare } from '../src/llm/cloudflare.js';
import { RoutedLlm } from '../src/llm/router.js';
import { UsageStore } from '../src/llm/usage.js';
import { Watchdog, withTimeout } from '../src/core/watchdog.js';
import { DEFAULT_SETTINGS, sanitizeSettings } from '../src/core/settings-schema.js';
import { Semaphore } from '../src/core/util.js';
import { formatLeft } from '../public/js/views/provider-limits.js';

const ACCT = 'a'.repeat(32);
const TOKEN = 'tok_' + 'x'.repeat(30);
const SECRET = `${ACCT}:${TOKEN}`;
const MODEL = '@cf/google/gemma-4-26b-a4b-it';
const drain = async (gen) => {
  const out = [];
  for await (const ev of gen) out.push(ev);
  return out;
};
const sse = (text, usage) => `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], ...(usage ? { usage } : {}) })}\n\ndata: [DONE]\n\n`;

function fakeCloudflare(handler) {
  const seen = [];
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      seen.push({ url: req.url, auth: req.headers.authorization, body: body ? JSON.parse(body) : null });
      handler(req, res, seen.length);
    });
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, seen, base: `http://127.0.0.1:${server.address().port}/client/v4` })));
}

test('a Cloudflare key is "ACCOUNT_ID:TOKEN" and the neuron estimate follows Cloudflare\'s per-model rates', () => {
  assert.deepEqual(parseCloudflareKey(SECRET), { accountId: ACCT, token: TOKEN });
  for (const bad of ['', 'nocolon', `${'a'.repeat(31)}:${TOKEN}`, `${ACCT}:short`, `${ACCT}: spaces in token here please ok`]) assert.equal(isCloudflareKeyFormat(bad), false, bad);
  // 1,500 prompt + 150 completion tokens on gemma-4-26b: 1500*9091 + 150*27273 per million
  assert.equal(estimateNeurons(MODEL, 1500, 150), 17.73);
  assert.ok(estimateNeurons('@cf/unknown/model', 1000, 100) > 0, 'an unknown model still counts (dear rate, errs on the safe side)');
  assert.ok(CF_DAILY_NEURONS / estimateNeurons(MODEL, 1500, 150) > 500, 'hundreds of replies a day on the free tier');
});

test('the Cloudflare client talks to the account\'s OpenAI-compatible endpoint, sends only the token, and counts neurons', async () => {
  const cf = await fakeCloudflare((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end(sse('namaste', { prompt_tokens: 1500, completion_tokens: 150 }));
  });
  try {
    const events = [];
    const pool = new KeyPool();
    pool.usageSink = (id, d) => events.push([id, d]);
    pool.setKeys([{ id: 'cf-env:1', label: 'cf', secret: SECRET }]);
    pool.setModels([MODEL]);
    const client = createCloudflareClient({ pool, baseUrl: cf.base });
    const evs = await drain(client.chat({ models: [MODEL], messages: [{ role: 'user', content: 'hi' }] }));
    assert.equal(cf.seen[0].url, `/client/v4/accounts/${ACCT}/ai/v1/chat/completions`);
    assert.equal(cf.seen[0].auth, `Bearer ${TOKEN}`, 'the account id is not part of the credential');
    assert.equal(cf.seen[0].body.model, MODEL);
    assert.equal(cf.seen[0].body.reasoning_effort, undefined, 'Groq-only flags are not sent');
    assert.deepEqual(cf.seen[0].body.stream_options, { include_usage: true });
    assert.equal(evs.find((e) => e.type === 'meta').provider, 'cloudflare');
    assert.equal(evs.filter((e) => e.type === 'token').map((e) => e.text).join(''), 'namaste');
    assert.equal(evs.at(-1).stats.provider, 'cloudflare');
    assert.deepEqual(events.find(([, d]) => d.ok)[1], { ok: 1, tokens: 1650, model: MODEL, neurons: 17.73 });
  } finally {
    cf.server.close();
  }
});

test('"daily free allocation used up" (429, code 3036) rests the account until 00:00 UTC and records what was reached', async () => {
  const cf = await fakeCloudflare((req, res) => {
    res.writeHead(429, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ errors: [{ code: 3036, message: "AiError: You have used up your daily free allocation of 10,000 neurons. Please upgrade to Cloudflare's Workers Paid plan if you would like to continue usage." }] }));
  });
  try {
    const pool = new KeyPool();
    pool.setKeys([{ id: 'cf-env:1', label: 'cf', secret: SECRET }]);
    pool.setModels([MODEL]);
    const client = createCloudflareClient({ pool, baseUrl: cf.base });
    await assert.rejects(drain(client.chat({ models: [MODEL], messages: [{ role: 'user', content: 'hi' }] })), (e) => e instanceof GroqUnavailableError && e.kind === 'rate_limit');
    const m = pool.status([MODEL])[0].models[0];
    assert.equal(m.state, 'cooling');
    assert.equal(m.hit.what, 'daily free allowance');
    const left = m.restUntil - Date.now();
    assert.ok(left > 0 && left <= 24 * 3_600_000 + 1000, 'rests until the next 00:00 UTC, not a fixed minute');
    assert.ok(Math.abs(left - msToUtcMidnight()) < 5000);
    assert.equal(client.usable([MODEL]), false);
  } finally {
    cf.server.close();
  }
});

test('probe: a good key, the used-up allowance (still a good key), a wrong token and a malformed key', async () => {
  let mode = 'ok';
  const cf = await fakeCloudflare((req, res) => {
    if (mode === 'ok') return res.writeHead(200, { 'content-type': 'application/json' }), res.end('{"choices":[]}');
    if (mode === 'spent') return res.writeHead(429), res.end('You have used up your daily free allocation of 10,000 neurons');
    res.writeHead(403), res.end('{"errors":[{"code":10000,"message":"Authentication error"}]}');
  });
  try {
    assert.equal((await probeCloudflare(SECRET, { baseUrl: cf.base })).ok, true);
    mode = 'spent';
    const spent = await probeCloudflare(SECRET, { baseUrl: cf.base });
    assert.equal(spent.ok, true);
    assert.match(spent.note, /already used up/);
    mode = 'bad';
    const bad = await probeCloudflare(SECRET, { baseUrl: cf.base });
    assert.equal(bad.ok, false);
    assert.match(bad.error, /rejected/);
    assert.match((await probeCloudflare('nope', { baseUrl: cf.base })).error, /ACCOUNT_ID:API_TOKEN/);
  } finally {
    cf.server.close();
  }
});

// ------------------------------------------------------------------ router: Groq → Cloudflare → local

function fakeOllama(text = 'local answer') {
  return {
    calls: [],
    numCtx: 4096,
    reusesPrefix: () => true,
    async *chat(req) {
      this.calls.push(req);
      yield { type: 'token', text };
      yield { type: 'done', stats: {} };
    },
    async prefill() {
      return {};
    },
  };
}
function fakeCloud(name, { fails = false } = {}) {
  const pool = new KeyPool();
  pool.setKeys([{ id: `${name}:1`, label: name, secret: `secret_${name}_xxxxxxxxxxxxxxxx` }]);
  return {
    pool,
    calls: 0,
    usable: (models) => pool.hasAvailable(models?.length ? models : pool.models),
    async *chat() {
      this.calls++;
      if (fails) {
        for (const m of pool.models) pool.rateLimited(`${name}:1`, { model: m, retryAfterMs: 60_000 });
        throw new GroqUnavailableError(`${name} is limited`, { kind: 'rate_limit', retryAfterMs: 60_000 });
      }
      yield { type: 'token', text: `${name} answer` };
      yield { type: 'done', stats: { provider: name } };
    },
  };
}
const mk = (settings, { groqFails = false, cfFails = false, noCf = false } = {}) => {
  const groq = fakeCloud('groq', { fails: groqFails });
  const cloudflare = noCf ? null : fakeCloud('cloudflare', { fails: cfFails });
  const ollama = fakeOllama();
  const s = { ...DEFAULT_SETTINGS, llmProvider: 'auto', groqModels: 'g1,g2', cloudflareModel: MODEL, ...settings };
  const router = new RoutedLlm({
    ollama,
    groq,
    cloudflare: cloudflare && { client: cloudflare, pool: cloudflare.pool },
    getSettings: () => s,
    gates: { ollama: new Semaphore(1), groq: new Semaphore(4), cloudflare: new Semaphore(4) },
    localReady: () => true,
  });
  return { router, groq, cloudflare, ollama };
};
const say = (evs) => evs.filter((e) => e.type === 'token').map((e) => e.text).join('');

test('router: Groq answers first; when Groq is used up Cloudflare answers; only when both are used up the local model does', async () => {
  const a = mk({});
  assert.equal(say(await drain(a.router.chat({ model: 'm', messages: [] }))), 'groq answer');
  assert.equal(a.cloudflare.calls, 0, 'Cloudflare quota is not spent while Groq can answer');

  const b = mk({}, { groqFails: true });
  const evs = await drain(b.router.chat({ model: 'm', messages: [] }));
  assert.equal(say(evs), 'cloudflare answer');
  assert.equal(b.router.status().counters.cloudflare, 1);
  assert.equal(b.ollama.calls.length, 0);
  assert.equal(b.cloudflare.pool.models[0], MODEL, 'the Cloudflare model setting is applied to its pool');

  const c = mk({}, { groqFails: true, cfFails: true });
  assert.equal(say(await drain(c.router.chat({ model: 'm', messages: [] }))), 'local answer');
  assert.equal(c.router.status().counters.fallbacks, 1);
  const before = c.groq.calls + c.cloudflare.calls;
  assert.equal(say(await drain(c.router.chat({ model: 'm', messages: [] }))), 'local answer');
  assert.equal(c.groq.calls + c.cloudflare.calls, before, 'no wasted calls while every account rests');
});

test('router: cloud-only mode tries both providers, then sends the break message instead of using the local model', async () => {
  const x = mk({ llmProvider: 'groq' }, { groqFails: true });
  assert.equal(say(await drain(x.router.chat({ model: 'm', messages: [] }))), 'cloudflare answer');
  const y = mk({ llmProvider: 'groq' }, { groqFails: true, cfFails: true });
  await assert.rejects(drain(y.router.chat({ model: 'm', messages: [] })), (e) => e instanceof GroqUnavailableError);
  assert.equal(y.ollama.calls.length, 0);
  assert.equal(y.router.status().counters.busy, 1);
});

test('router: needsLocal is false while either provider can answer; local-only never touches the cloud; status lists Cloudflare', async () => {
  const x = mk({}, { groqFails: true });
  await drain(x.router.chat({ model: 'm', messages: [] }));
  assert.equal(x.router.needsLocal(), false, 'Groq is resting but Cloudflare is ready');
  const y = mk({ llmProvider: 'ollama' });
  assert.equal(say(await drain(y.router.chat({ model: 'm', messages: [] }))), 'local answer');
  assert.equal(y.groq.calls + y.cloudflare.calls, 0);
  const st = x.router.status();
  assert.equal(st.cloudflare.model, MODEL);
  assert.equal(st.cloudflare.keys.length, 1);
  // without a Cloudflare setup everything works as before
  const z = mk({}, { noCf: true });
  assert.equal(say(await drain(z.router.chat({ model: 'm', messages: [] }))), 'groq answer');
  assert.equal(z.router.status().cloudflare, null);
});

test('settings: the Cloudflare model id may start with @cf/', () => {
  assert.equal(DEFAULT_SETTINGS.cloudflareModel, MODEL);
  assert.equal(sanitizeSettings({ cloudflareModel: '@cf/meta/llama-3.1-8b-instruct-fp8' }).value.cloudflareModel, '@cf/meta/llama-3.1-8b-instruct-fp8');
  assert.equal(sanitizeSettings({ cloudflareModel: 'bad model;rm' }).errors.length, 1);
});

// ------------------------------------------------------------------ limit bars

test('the pool remembers each model\'s limits, remaining amount and reset time for the usage bars; a passed reset counts as full', () => {
  let now = 1_000_000;
  const pool = new KeyPool({ now: () => now });
  pool.setKeys([{ id: 'a', label: 'a', secret: 'gsk_secret_a_xxxxxxxxxxxx' }]);
  pool.setModels(['m1']);
  pool.success('a', { model: 'm1', limitRequests: 14_400, remainingRequests: 14_000, resetRequestsMs: 3_600_000, limitTokens: 8000, remainingTokens: 2000, resetTokensMs: 20_000 });
  let l = pool.status(['m1'])[0].models[0].limits;
  assert.deepEqual(l.requests, { limit: 14_400, remaining: 14_000, resetAt: now + 3_600_000 });
  assert.deepEqual(l.tokens, { limit: 8000, remaining: 2000, resetAt: now + 20_000 });
  now += 25_000; // the per-minute window has reset, the daily one has not
  l = pool.status(['m1'])[0].models[0].limits;
  assert.deepEqual(l.tokens, { limit: 8000, remaining: 8000, resetAt: null });
  assert.equal(l.requests.remaining, 14_000);
});

test('a 429 keeps which limit was reached and when the model is back, so the bar can say "exhausted, resets at…"', () => {
  const now = 5_000_000;
  const pool = new KeyPool({ now: () => now });
  pool.setKeys([{ id: 'a', label: 'a', secret: 'gsk_secret_a_xxxxxxxxxxxx' }]);
  pool.setModels(['m1']);
  const hit = limitFromMessage('Rate limit reached for model `m1` in organization `org_1` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199961, Requested 1500. Please try again in 11m52s.');
  assert.deepEqual(hit, { what: 'tokens per day', limit: 200_000, used: 199_961 });
  pool.rateLimited('a', { model: 'm1', retryAfterMs: 712_000, hit });
  const m = pool.status(['m1'])[0].models[0];
  assert.equal(m.state, 'cooling');
  assert.equal(m.restUntil, now + 712_000);
  assert.equal(m.hit.what, 'tokens per day');
  assert.equal(limitFromMessage('plain text'), null);
});

test('usage keeps Cloudflare neurons (in hundredths, so a 17.73-neuron reply is not rounded away)', async () => {
  const data = new Map();
  const redis = {
    pipeline() {
      const ops = [];
      const p = {
        hincrby: (k, f, n) => (ops.push(() => { const h = data.get(k) || {}; h[f] = (h[f] || 0) + n; data.set(k, h); return [null, 1]; }), p),
        expire: () => p,
        hgetall: (k) => (ops.push(() => [null, Object.fromEntries(Object.entries(data.get(k) || {}).map(([f, n]) => [f, String(n)]))]), p),
        exec: async () => ops.map((o) => o()),
      };
      return p;
    },
  };
  const u = new UsageStore({ redis, keyFor: (k, d) => `${k}:${d}`, getTimezone: () => 'UTC', now: () => new Date('2026-11-13T10:00:00Z') });
  await u.record('cf-db:1', { ok: 1, tokens: 1650, neurons: 17.73, model: MODEL });
  await u.record('cf-db:1', { ok: 1, tokens: 1650, neurons: 17.73, model: MODEL });
  const s = await u.summary(['cf-db:1']);
  assert.equal(s['cf-db:1'].today.neurons, 35.46);
  assert.equal(s['cf-db:1'].today.models[MODEL].tokens, 3300);
});

test('time left is shown the way a person reads it', () => {
  assert.equal(formatLeft(45_000), '45s');
  assert.equal(formatLeft(10 * 60_000), '10m');
  assert.equal(formatLeft(2 * 3_600_000 + 3 * 60_000), '2h 03m');
  assert.equal(formatLeft(3 * 86_400_000), '3d 0h');
  assert.equal(formatLeft(-5), '0s');
});

// ------------------------------------------------------------------ WhatsApp watchdog

test('watchdog: one slow probe is forgiven, three failures in a row mean the session is stale, success resets the count', () => {
  const w = new Watchdog({ failLimit: 3 });
  assert.equal(w.record(false), false);
  assert.equal(w.record(false), false);
  assert.equal(w.record(true), false, 'recovered');
  assert.equal(w.record(false), false);
  assert.equal(w.record(false), false);
  assert.equal(w.record(false), true, 'third in a row');
  assert.equal(w.record(false), false, 'counting starts over after a restart verdict');
});

test('a hung browser call is cut off instead of blocking the health check forever', async () => {
  await assert.rejects(withTimeout(new Promise(() => {}), 20, 'state check'), /state check timed out/);
  assert.equal(await withTimeout(Promise.resolve('CONNECTED'), 1000), 'CONNECTED');
});
