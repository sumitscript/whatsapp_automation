// AI worker: consumes reply jobs from the bus, runs the RAG + LLM pipeline and
// streams WhatsApp-sized bubbles back to the gateway as soon as each is ready.
import './env.js';
import http from 'node:http';
import { config } from '../core/config.js';
import { logger } from '../core/logger.js';
import { attachRedisLogSink, channels, createRedis, createRedisAnswerCache, createRedisMemory, keys } from '../core/redis.js';
import { createSettingsService } from '../core/settings.js';
import { KeyedQueue, Semaphore, sleep } from '../core/util.js';
import { createPool, migrate } from '../db/pg.js';
import { createRepo } from '../db/repo.js';
import { createBus, TOPICS } from '../bus/index.js';
import { KnowledgeBase } from '../rag/kb.js';
import { OllamaClient } from '../llm/ollama.js';
import { createBrain } from '../brain/pipeline.js';

const log = logger.child({ component: 'worker' });
const STALE_JOB_MS = 5 * 60_000; // don't answer messages that sat in the queue during downtime

const redis = createRedis('main');
const sub = createRedis('sub');
attachRedisLogSink(logger, redis);

const pool = createPool();
await migrate(pool, log);
const repo = createRepo(pool);
const settings = await createSettingsService({ repo, redis, subscriber: sub, logger: log });

const kb = new KnowledgeBase({ ...config.kb, logger: log });
await kb.load();
kb.watch();
await sub.subscribe(channels.kb);
sub.on('message', (ch) => {
  if (ch === channels.kb) kb.load().catch((err) => log.error('KB reload failed', { err }));
});

const llm = new OllamaClient({ ...config.ollama, url: config.ollama.url, timeoutMs: config.ollama.requestTimeoutMs, logger: log });
const memory = createRedisMemory(redis);
const brain = createBrain({ kb, llm, memory, cache: createRedisAnswerCache(redis), logger: log });

let modelState = { model: settings.get().model, ready: false, error: null };
async function prepareModel(model) {
  modelState = { model, ready: false, error: null };
  for (let attempt = 1; ; attempt++) {
    try {
      await llm.health();
      break;
    } catch (err) {
      if (attempt % 10 === 1) log.warn('Waiting for Ollama…', { url: config.ollama.url, err: err.message });
      await sleep(Math.min(1000 * attempt, 5000));
    }
  }
  try {
    const ok = await llm.ensureModel(model, { autoPull: config.ollama.autoPull });
    if (!ok) throw new Error(`Model ${model} is not installed (pull it from the console or set LLM_AUTO_PULL=true)`);
    const ms = await llm.warmup(model);
    modelState = { model, ready: true, error: null };
    log.info('Model loaded and resident', { model, loadMs: ms });
  } catch (err) {
    modelState = { model, ready: false, error: err.message };
    log.error('Model preparation failed', { model, err });
  }
}
let modelReady = prepareModel(settings.get().model);
settings.onChange((next, prev) => {
  if (next.model !== prev.model) modelReady = prepareModel(next.model);
});

const bus = await createBus({ logger: log, redisFactory: createRedis });
const lanes = new KeyedQueue(); // one lane per chat → replies stay in order
const gate = new Semaphore(config.ollama.concurrency); // bounded parallel generations
let inFlight = 0;
let lastReply = null;
let lastError = null;

// Heartbeat for the console's "LLM status" card (expires if the worker dies).
const startedAt = Date.now();
const publishStatus = () =>
  redis
    .set(
      keys.workerStatus,
      JSON.stringify({ at: Date.now(), startedAt, model: modelState, inFlight, lanes: lanes.size, folding, queuedFolds: foldQueue.size, concurrency: config.ollama.concurrency, bus: bus.driver, kb: kb.version, lastReply, lastError }),
      'EX',
      20,
    )
    .catch(() => {});
setInterval(publishStatus, 5000).unref();

async function handleJob(job) {
  const out = (payload) => bus.publish(TOPICS.outbound, job.chatId, { jobId: job.jobId, chatId: job.chatId, ...payload });
  if (Date.now() - job.receivedAt > STALE_JOB_MS) {
    log.warn('Dropping stale job', { chatId: job.chatId, ageMs: Date.now() - job.receivedAt });
    await out({ type: 'error', error: 'stale', silent: true });
    return;
  }
  await modelReady;
  const s = settings.get();
  inFlight++;
  try {
    const longTerm = s.memoryEnabled ? await repo.getMemory(job.chatId).catch(() => null) : null;
    const stream = brain.respond({
      chatId: job.chatId,
      text: job.text,
      settings: s,
      contactName: job.contactName || longTerm?.name || null,
      longTermMemory: longTerm?.memory || null,
    });
    for await (const ev of stream) {
      if (ev.type === 'chunk') await out({ type: 'chunk', index: ev.index, text: ev.text, meta: ev.meta });
      else if (ev.type === 'done') {
        lastReply = { at: Date.now(), latencyMs: ev.meta.latencyMs, firstChunkMs: ev.meta.firstChunkMs, tokensPerSec: ev.meta.tokensPerSec ?? null, cached: !!ev.meta.cached, model: ev.meta.model };
        await out({ type: 'done', meta: ev.meta });
      }
    }
  } catch (err) {
    lastError = { at: Date.now(), message: err.message };
    log.error('Reply generation failed', { chatId: job.chatId, err });
    await out({ type: 'error', error: err.message });
  } finally {
    inFlight--;
  }
  // Conversation outgrew the verbatim window → fold the oldest turns into long-term memory later.
  const stored = (await memory.get(job.chatId).catch(() => [])).length;
  if (stored > s.historyTurns + 8) scheduleFold(job.chatId, false);
}

// ---- Long-term memory: folded only while no reply is being generated ----
const foldQueue = new Map(); // chatId → { all }
let folding = false;
function scheduleFold(chatId, all) {
  foldQueue.set(chatId, { all: all || foldQueue.get(chatId)?.all || false });
}
async function fold(chatId, { all }) {
  const s = settings.get();
  const turns = await memory.get(chatId);
  const n = turns.length - (all ? 0 : s.historyTurns);
  if (n <= 0) return;
  if (!s.memoryEnabled) {
    if (all) await memory.trimOldest(chatId, n);
    return;
  }
  await modelReady;
  const started = Date.now();
  const current = await repo.getMemory(chatId);
  const updated = await brain.summarize({ previous: current.memory, turns: turns.slice(0, n), contactName: current.name, settings: s });
  await repo.setMemory(chatId, updated);
  await memory.trimOldest(chatId, n);
  log.info('Long-term memory updated', { chatId, folded: n, ms: Date.now() - started, chars: updated?.length ?? 0 });
  await bus.publish(TOPICS.outbound, chatId, { type: 'memory', chatId });
}
setInterval(async () => {
  if (folding || inFlight > 0 || !foldQueue.size) return;
  const [chatId, opts] = foldQueue.entries().next().value;
  foldQueue.delete(chatId);
  folding = true;
  try {
    await gate.run(() => fold(chatId, opts));
  } catch (err) {
    log.warn('Memory fold failed', { chatId, err });
  } finally {
    folding = false;
  }
}, 1500).unref();

await bus.subscribe(TOPICS.inbound, 'ai-worker', async (job) => {
  if (!job?.chatId) return;
  if (job.type === 'summarize') return scheduleFold(job.chatId, true); // session ended
  if (!job.text) return;
  lanes.push(job.chatId, () => gate.run(() => handleJob(job))).catch((err) => log.error('Job failed', { err }));
});
publishStatus();
log.info('Worker ready', { bus: bus.driver, model: settings.get().model, concurrency: config.ollama.concurrency });

// Liveness/readiness for Docker healthchecks.
const health = http.createServer((req, res) => {
  const body = JSON.stringify({ ok: true, model: modelState, inFlight, lanes: lanes.size, kb: kb.version });
  res.writeHead(req.url === '/healthz' ? 200 : 404, { 'content-type': 'application/json' }).end(body);
});
health.listen(Number(process.env.WORKER_HEALTH_PORT || 3001), '0.0.0.0');

let stopping = false;
async function shutdown(signal) {
  if (stopping) return;
  stopping = true;
  log.info('Worker shutting down', { signal });
  health.close();
  kb.close();
  await bus.close().catch(() => {});
  await Promise.allSettled([pool.end(), redis.quit(), sub.quit()]);
  process.exit(0);
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('unhandledRejection', (err) => log.error('Unhandled rejection', { err }));
