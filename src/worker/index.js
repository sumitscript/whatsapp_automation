// AI worker: consumes reply jobs from the bus, runs the RAG + LLM pipeline and
// streams WhatsApp-sized bubbles back to the gateway as soon as each is ready.
import './env.js';
import http from 'node:http';
import { config } from '../core/config.js';
import { logger } from '../core/logger.js';
import { attachRedisLogSink, channels, createRedis, createRedisAnswerCache, createRedisMemory, keys } from '../core/redis.js';
import { createSettingsService } from '../core/settings.js';
import { KeyedQueue, Semaphore, sleep } from '../core/util.js';
import { Coalescer } from '../core/coalesce.js';
import { createPool, migrate } from '../db/pg.js';
import { createRepo } from '../db/repo.js';
import { createBus, TOPICS } from '../bus/index.js';
import { KnowledgeBase } from '../rag/kb.js';
import { OllamaClient } from '../llm/ollama.js';
import { createBrain } from '../brain/pipeline.js';
import { looksLikeCommitmentInContext, sameItem } from '../brain/agenda.js';

const log = logger.child({ component: 'worker' });
const STALE_JOB_MS = 5 * 60_000; // don't answer messages that sat in the queue during downtime
const MODEL_WAIT_MS = Number(process.env.WORKER_MODEL_WAIT_MS || 20_000); // how long a reply waits for a loading model before giving up

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

// The gate bounds parallel *model* calls, and nothing else. Template replies, answer-cache hits and
// commands never touch the model, so they no longer queue behind someone else's 30-second story.
// Waiting calls are served by priority: 1 normal reply, 2 long reply (story/code: big token budget),
// 3 background (memory folds, agenda), 4 think-ahead; aging keeps long jobs from starving.
const gate = new Semaphore(config.ollama.concurrency);
const llmGated = {
  get numCtx() {
    return llm.numCtx;
  },
  async *chat(req) {
    await gate.acquire({ priority: req.priority ?? (req.maxTokens > 450 ? 2 : 1), signal: req.signal });
    try {
      yield* llm.chat(req);
    } finally {
      gate.release();
    }
  },
  prefill: (req) => gate.run(() => llm.prefill(req), { priority: 4, signal: req.signal }),
};
const brain = createBrain({ kb, llm: llmGated, memory, cache: createRedisAnswerCache(redis), logger: log });

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
const co = new Coalescer(); // bursts of messages → one reply (see core/coalesce.js)
let inFlight = 0;
// Background work (agenda extraction, memory folds) waits for a quiet period after the last message or
// reply: people send bursts, and a background generation started in a gap would split the CPU with the
// next live reply (measured: 3x slower generation while two run at once).
const BG_QUIET_MS = Number(process.env.WORKER_BG_QUIET_MS || 8000);
let lastActivity = Date.now();
let lastReply = null;
let lastError = null;

// Heartbeat for the console's "LLM status" card (expires if the worker dies).
const startedAt = Date.now();
const publishStatus = () =>
  redis
    .set(
      keys.workerStatus,
      JSON.stringify({ at: Date.now(), startedAt, model: modelState, inFlight, lanes: lanes.size, folding, queuedFolds: foldQueue.size + agendaQueue.size, concurrency: config.ollama.concurrency, bus: bus.driver, kb: kb.version, lastReply, lastError }),
      'EX',
      20,
    )
    .catch(() => {});
setInterval(publishStatus, 5000).unref();

// ---- Think-ahead: while the person reads our reply and types the next one, the model pre-reads the part of
// the next prompt we already know (persona + history), so the real message only pays for its own tokens.
// Strictly lower priority than live work: a new message cancels it on arrival (the model stops mid-read).
let warmCtl = null;
function cancelWarm() {
  warmCtl?.abort();
  warmCtl = null;
}
// Background work (agenda extraction, memory folds) is pre-empted the same way: a live message aborts it and it is re-queued.
let bgCtl = null;
function cancelBg() {
  bgCtl?.abort(new Error('preempted'));
  bgCtl = null;
}
async function warmChat(chatId) {
  const s = settings.get();
  if (!s.prefillAhead || !modelState.ready || inFlight > 0 || !llm.reusesPrefix(s.model)) return;
  cancelWarm();
  const ctl = (warmCtl = new AbortController());
  try {
    // brain.warm → llm.prefill takes the gate at the lowest priority, so it only runs when nothing else wants the model
    const r = await brain.warm({ chatId, settings: settings.get(), signal: ctl.signal });
    log.debug('Think-ahead ready', { chatId, promptTokens: r.promptTokens, ms: r.promptMs });
  } catch (err) {
    if (!ctl.signal.aborted) log.warn('Think-ahead failed', { chatId, err });
  } finally {
    if (warmCtl === ctl) warmCtl = null;
  }
}

async function handleJob(job) {
  const out = (payload) => bus.publish(TOPICS.outbound, job.chatId, { jobId: job.jobId, chatId: job.chatId, ...payload });
  if (Date.now() - job.receivedAt > STALE_JOB_MS) {
    log.warn('Dropping stale job', { chatId: job.chatId, ageMs: Date.now() - job.receivedAt });
    co.claim(job.chatId, job.jobId); // discard it (and anything stuck with it) instead of leaving it pending
    co.release(job.chatId, job.jobId);
    await out({ type: 'error', error: 'stale', silent: true });
    return;
  }
  // Ollama down or the model still loading must not park every job (and the "typing…" indicator) forever.
  const ready = await Promise.race([modelReady.then(() => true), sleep(MODEL_WAIT_MS).then(() => false)]);
  if (!ready || !modelState.ready) {
    co.claim(job.chatId, job.jobId);
    co.release(job.chatId, job.jobId);
    log.warn('Model not ready; reply skipped', { chatId: job.chatId, model: modelState.model, error: modelState.error });
    await out({ type: 'error', error: modelState.error || 'model not ready' });
    return;
  }
  // One reply per burst: a newer job from this chat answers everything said since the last reply.
  const claim = co.claim(job.chatId, job.jobId);
  if (!claim) {
    log.info('Folded into a newer message', { chatId: job.chatId, jobId: job.jobId });
    // Every flushed job holds one "typing…" reference in the gateway; this one will never speak, so release it.
    await out({ type: 'error', error: 'folded', silent: true });
    return;
  }
  const text = claim.text;
  const s = settings.get();
  inFlight++;
  let superseded = false;
  try {
    const longTerm = s.memoryEnabled ? await repo.getMemory(job.chatId).catch(() => null) : null;
    const stream = brain.respond({
      chatId: job.chatId,
      text,
      settings: s,
      contactName: job.contactName || longTerm?.name || null,
      longTermMemory: longTerm?.memory || null,
      signal: claim.signal,
    });
    for await (const ev of stream) {
      if (ev.type === 'chunk') {
        co.markEmitted(job.chatId);
        await out({ type: 'chunk', index: ev.index, text: ev.text, meta: ev.meta });
      } else if (ev.type === 'done') {
        lastReply = { at: Date.now(), latencyMs: ev.meta.latencyMs, firstChunkMs: ev.meta.firstChunkMs, tokensPerSec: ev.meta.tokensPerSec ?? null, cached: !!ev.meta.cached, model: ev.meta.model };
        await out({ type: 'done', meta: ev.meta });
      }
    }
  } catch (err) {
    if (co.isSuperseded(claim.signal)) {
      superseded = true; // a newer message arrived before we said anything: its job re-answers both
      log.info('Reply dropped, newer message arrived', { chatId: job.chatId, jobId: job.jobId });
      await out({ type: 'error', error: 'superseded', silent: true }).catch(() => {}); // release this job's typing reference
    } else {
      lastError = { at: Date.now(), message: err.message };
      log.error('Reply generation failed', { chatId: job.chatId, err });
      await out({ type: 'error', error: err.message });
    }
  } finally {
    inFlight--;
    lastActivity = Date.now();
    co.release(job.chatId, job.jobId);
  }
  if (superseded) return;
  warmChat(job.chatId).catch(() => {}); // fire and forget: the next message cancels it if it arrives first
  // Conversation outgrew the verbatim window → fold the oldest turns into long-term memory later.
  const turnsNow = await memory.get(job.chatId).catch(() => []);
  if (turnsNow.length > s.historyTurns + 8) scheduleFold(job.chatId, false);
  // "let's meet tomorrow at 5" → Agenda (off the hot path, like memory folds). The request and its
  // details often come as separate messages ("I want to meet" … "tomorrow at six near the office").
  const earlierUserTexts = turnsNow.filter((t) => t.role === 'user').map((t) => t.content).slice(0, -1);
  if (s.agendaEnabled && looksLikeCommitmentInContext(text, earlierUserTexts)) agendaQueue.set(job.chatId, { text, contactName: job.contactName || null, at: Date.now() });
}

// ---- Agenda: appointments / to-dos spotted in chats ----
const agendaQueue = new Map(); // chatId → latest message that looks like a commitment
async function extractAgendaFor(chatId, { text, contactName }, signal) {
  const s = settings.get();
  if (!s.agendaEnabled) return;
  await modelReady;
  const turns = await memory.get(chatId);
  const found = await brain.extractAgenda({ turns, text, contactName, settings: s, signal });
  if (!found.length) return log.debug('No agenda items found', { chatId });
  const existing = await repo.listAgenda({ view: 'open', chatId });
  for (const it of found) {
    const draft = { ...it, chatId, sourceMessage: text.slice(0, 500), source: 'ai' };
    const same = existing.find((e) => sameItem(e, draft));
    if (same) {
      // The person is refining an earlier proposal ("make it 6pm"): keep the item, update what changed.
      const saved = await repo.updateAgendaItem(same.id, { title: draft.title, details: draft.details ?? same.details, dueAt: draft.dueAt ?? same.dueAt, allDay: draft.dueAt ? draft.allDay : same.allDay, dueText: draft.dueText ?? same.dueText, location: draft.location ?? same.location });
      log.info('Agenda item updated', { chatId, id: same.id });
      await bus.publish(TOPICS.outbound, chatId, { type: 'agenda', chatId, itemId: saved.id, created: false });
    } else {
      const saved = await repo.createAgendaItem(draft);
      log.info('Agenda item created', { chatId, id: saved.id, kind: saved.kind, dueAt: saved.dueAt });
      await bus.publish(TOPICS.outbound, chatId, { type: 'agenda', chatId, itemId: saved.id, created: true });
    }
  }
}

// ---- Long-term memory: folded only while no reply is being generated ----
const foldQueue = new Map(); // chatId → { all }
let folding = false;
function scheduleFold(chatId, all) {
  foldQueue.set(chatId, { all: all || foldQueue.get(chatId)?.all || false });
}
async function fold(chatId, { all }, signal) {
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
  const updated = await brain.summarize({ previous: current.memory, turns: turns.slice(0, n), contactName: current.name, settings: s, signal });
  await repo.setMemory(chatId, updated);
  await memory.trimOldest(chatId, n);
  log.info('Long-term memory updated', { chatId, folded: n, ms: Date.now() - started, chars: updated?.length ?? 0 });
  await bus.publish(TOPICS.outbound, chatId, { type: 'memory', chatId });
}
setInterval(async () => {
  if (folding || inFlight > 0 || (!foldQueue.size && !agendaQueue.size)) return;
  if (Date.now() - lastActivity < BG_QUIET_MS) return; // someone is mid-conversation: stay out of the way
  folding = true;
  const ctl = (bgCtl = new AbortController());
  let preempted = false;
  try {
    if (agendaQueue.size) {
      // Agenda first: a meeting request is more time-sensitive than a memory summary.
      const [chatId, req] = agendaQueue.entries().next().value;
      agendaQueue.delete(chatId);
      try {
        await extractAgendaFor(chatId, req, ctl.signal);
      } catch (err) {
        if (ctl.signal.aborted) {
          preempted = true;
          if (!agendaQueue.has(chatId)) agendaQueue.set(chatId, req); // a live message came in: do it later
          log.debug('Agenda extraction pre-empted by a live message; re-queued', { chatId });
        } else log.warn('Agenda extraction failed', { chatId, err });
      }
      return;
    }
    const [chatId, opts] = foldQueue.entries().next().value;
    foldQueue.delete(chatId);
    try {
      await fold(chatId, opts, ctl.signal);
    } catch (err) {
      if (ctl.signal.aborted) {
        preempted = true;
        scheduleFold(chatId, opts.all);
        log.debug('Memory fold pre-empted by a live message; re-queued', { chatId });
      } else log.warn('Memory fold failed', { chatId, err });
    }
  } finally {
    if (bgCtl === ctl) bgCtl = null;
    folding = false;
    // Background prompts overwrite Ollama's cached persona prefix: read it back so the next live reply starts warm.
    if (!preempted) warmChat(null).catch(() => {});
  }
}, 1500).unref();

await bus.subscribe(TOPICS.inbound, 'ai-worker', async (job) => {
  if (!job?.chatId) return;
  if (job.type === 'summarize') return scheduleFold(job.chatId, true); // session ended
  if (!job.text) return;
  lastActivity = Date.now();
  cancelWarm(); // a real message beats any think-ahead…
  cancelBg(); // …and any background work (re-queued afterwards)
  co.add(job.chatId, job.jobId, job.text); // may abort a reply that has not started speaking
  lanes.push(job.chatId, () => handleJob(job)).catch((err) => log.error('Job failed', { err }));
});
// After a restart or model switch, pre-read the persona alone: the one prefix every chat shares.
modelReady.then(() => warmChat(null)).catch(() => {});
settings.onChange((next, prev) => {
  if (next.model !== prev.model) modelReady.then(() => warmChat(null)).catch(() => {});
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
