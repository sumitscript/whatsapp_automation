// Gateway: owns the WhatsApp session, serves the web console + API + Socket.IO,
// applies reply policy and delivers AI replies. Run exactly ONE gateway per
// WhatsApp account (a session can't be shared between browsers).
import './env.js';
import http from 'node:http';
import { config } from '../core/config.js';
import { logger } from '../core/logger.js';
import { attachRedisLogSink, channels, createRedis, createRedisAnswerCache, createRedisMemory, keys } from '../core/redis.js';
import { createSettingsService } from '../core/settings.js';
import { createPool, migrate } from '../db/pg.js';
import { createRepo } from '../db/repo.js';
import { createAuth } from '../auth/service.js';
import { createBus, TOPICS } from '../bus/index.js';
import { KnowledgeBase } from '../rag/kb.js';
import { OllamaClient } from '../llm/ollama.js';
import { KeyPool } from '../llm/key-pool.js';
import { KeyStore } from '../llm/key-store.js';
import { GroqClient } from '../llm/groq.js';
import { createCloudflareClient } from '../llm/cloudflare.js';
import { RoutedLlm } from '../llm/router.js';
import { UsageStore } from '../llm/usage.js';
import { Semaphore } from '../core/util.js';
import { createBrain } from '../brain/pipeline.js';
import { createAvailability } from '../brain/availability.js';
import { WhatsAppManager } from './whatsapp.js';
import { Inbox } from './inbox.js';
import { SessionManager } from './sessions.js';
import { createApp, probeLlmStatus } from './http.js';
import { LlmMonitor } from './llm-monitor.js';
import { createRealtime } from './realtime.js';

const log = logger.child({ component: 'gateway' });

const redis = createRedis('main');
const sub = createRedis('sub');
attachRedisLogSink(logger, redis);

const pool = createPool();
await migrate(pool, log);
const repo = createRepo(pool);
const auth = createAuth({ repo, logger: log });
await auth.bootstrapAdmin();
const settings = await createSettingsService({ repo, redis, subscriber: sub, logger: log });

const kb = new KnowledgeBase({ ...config.kb, logger: log });
await kb.load();
kb.watch();
await sub.subscribe(channels.kb);
sub.on('message', (ch, version) => {
  if (ch === channels.kb && version !== kb.version) kb.load().catch((err) => log.error('KB reload failed', { err }));
});

const llm = new OllamaClient({ ...config.ollama, timeoutMs: config.ollama.requestTimeoutMs, logger: log });
const memory = createRedisMemory(redis);
// Groq keys + router for the console's playground (the worker has its own pool for live chats; both read the same key list).
const groqPool = new KeyPool();
const usage = new UsageStore({ redis, keyFor: keys.llmUsage, getTimezone: () => settings.get().timezone, onError: (err) => log.debug('Could not record Groq usage', { err: err.message }) });
groqPool.usageSink = (id, d) => usage.record(id, d);
const keyStore = new KeyStore({ pool: groqPool, repo, secret: config.auth.jwtSecret, envKeys: config.groq.keys, logger: log });
await keyStore.start();
const cfPool = new KeyPool();
cfPool.usageSink = (id, d) => usage.record(id, d);
const cfStore = new KeyStore({ pool: cfPool, repo, secret: config.auth.jwtSecret, envKeys: config.cloudflare.keys, logger: log, provider: 'cloudflare', idPrefix: 'cf-' });
await cfStore.start();
const cf = createCloudflareClient({ pool: cfPool, baseUrl: config.cloudflare.baseUrl, timeoutMs: config.cloudflare.timeoutMs, logger: log });
await sub.subscribe(channels.llmKeys);
sub.on('message', (ch) => {
  if (ch === channels.llmKeys) {
    keyStore.onChange();
    cfStore.onChange();
  }
});
const groq = new GroqClient({ pool: groqPool, baseUrl: config.groq.baseUrl, timeoutMs: config.groq.timeoutMs, logger: log });
const router = new RoutedLlm({
  ollama: llm,
  groq,
  cloudflare: { client: cf, pool: cfPool },
  getSettings: () => settings.get(),
  gates: { ollama: new Semaphore(config.ollama.concurrency), groq: new Semaphore(config.groq.concurrency), cloudflare: new Semaphore(config.cloudflare.concurrency) },
  logger: log,
});
const availability = createAvailability({ repo, getSettings: () => settings.get() });
const brain = createBrain({ kb, llm: router, memory, cache: createRedisAnswerCache(redis), logger: log, availability });
const bus = await createBus({ logger: log, redisFactory: createRedis });

const wa = new WhatsAppManager({ ...config.wa, logger: log });
const hub = { emit: () => {} }; // replaced once Socket.IO is up
const sessions = new SessionManager({ repo, settings, logger: log });
sessions.startSweeper(); // ends idle sessions (sessionIdleMinutes)
const inbox = new Inbox({ wa, repo, redis, bus, settings, memory, sessions, hub, availability, logger: log });

const app = createApp({ auth, repo, redis, wa, inbox, settings, kb, llm, brain, memory, hub, availability, groq: { pool: groqPool, client: groq, store: keyStore, router, usage }, cloudflare: { pool: cfPool, client: cf, store: cfStore }, logger: log });
const server = http.createServer(app);
const realtime = await createRealtime(server, { auth, wa, subscriber: sub, logger: log });
hub.emit = realtime.emit;

wa.on('status', (s) => realtime.emit('wa:status', s));
wa.on('message', (msg) => inbox.handleIncoming(msg).catch((err) => log.error('Inbound handling failed', { err })));
wa.on('own_message', (msg) => inbox.handleOwnMessage(msg).catch((err) => log.error('Own message handling failed', { err })));
// Not awaited: per-chat lanes keep order, and one slow chat must not block a partition.
await bus.subscribe(TOPICS.outbound, 'gateway', (ev) => {
  inbox.handleOutbound(ev);
});

const llmMonitor = new LlmMonitor({ probe: () => probeLlmStatus({ llm, redis, settings }), settings, inbox, hub, logger: log });
llmMonitor.start();

server.listen(config.http.port, config.http.host, () => {
  log.info('Gateway listening', { port: config.http.port, bus: bus.driver, kb: kb.version, model: settings.get().model });
});

if (config.wa.autoConnect) wa.connect().catch((err) => log.error('WhatsApp connect failed', { err }));

let stopping = false;
async function shutdown(signal) {
  if (stopping) return;
  stopping = true;
  log.info('Gateway shutting down', { signal });
  inbox.shutdown();
  sessions.stop();
  llmMonitor.stop();
  server.close();
  await realtime.close().catch(() => {});
  await wa.shutdown().catch(() => {});
  await bus.close().catch(() => {});
  kb.close();
  await Promise.allSettled([pool.end(), redis.quit(), sub.quit()]);
  process.exit(0);
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('unhandledRejection', (err) => log.error('Unhandled rejection', { err }));
