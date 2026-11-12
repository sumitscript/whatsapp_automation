// Central configuration. Every tunable comes from the environment so the same
// image runs locally and on the KVM. Keep this file dependency-free.
import { randomBytes } from 'node:crypto';
import path from 'node:path';

const env = process.env;

const str = (v, d) => (v === undefined || v === '' ? d : String(v));
const int = (v, d) => {
  if (v === undefined || v === '') return d;
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : d;
};
const num = (v, d) => {
  if (v === undefined || v === '') return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};
const bool = (v, d) =>
  v === undefined || v === '' ? d : ['1', 'true', 'yes', 'on'].includes(String(v).toLowerCase());
const list = (v, d) => (v ? String(v).split(',').map((s) => s.trim()).filter(Boolean) : d);

const isProd = env.NODE_ENV === 'production';
const root = path.resolve(import.meta.dirname, '..', '..');

let jwtSecret = env.JWT_SECRET || '';
if (!jwtSecret) {
  if (isProd) {
    throw new Error('JWT_SECRET is required in production (generate one with: openssl rand -hex 32)');
  }
  jwtSecret = randomBytes(32).toString('hex'); // dev only: sessions reset on restart
}

export const config = Object.freeze({
  isProd,
  root,
  service: str(env.SERVICE_NAME, 'app'),

  http: Object.freeze({
    host: str(env.HOST, '0.0.0.0'),
    port: int(env.PORT, 3000),
    trustProxy: bool(env.TRUST_PROXY, true),
    // e.g. https://wa.example.com — when set, mutating requests must come from it.
    publicOrigin: str(env.PUBLIC_ORIGIN, ''),
  }),

  auth: Object.freeze({
    jwtSecret,
    sessionTtlHours: int(env.SESSION_TTL_HOURS, 168),
    cookieName: 'wa_session',
    cookieSecure: bool(env.COOKIE_SECURE, isProd),
    adminUsername: str(env.ADMIN_USERNAME, 'admin'),
    adminPassword: str(env.ADMIN_PASSWORD, ''),
  }),

  pg: Object.freeze({
    url: str(env.DATABASE_URL, 'postgres://wa:wa@localhost:5432/wa'),
    poolMax: int(env.PG_POOL_MAX, 10),
  }),

  redis: Object.freeze({
    url: str(env.REDIS_URL, 'redis://localhost:6379'),
    prefix: str(env.REDIS_PREFIX, 'wa:'),
  }),

  bus: Object.freeze({
    driver: str(env.BUS_DRIVER, 'redis'), // redis (default, lighter) | kafka
    kafkaBrokers: list(env.KAFKA_BROKERS, ['localhost:9092']),
    kafkaClientId: str(env.KAFKA_CLIENT_ID, 'wa-assistant'),
    partitions: int(env.KAFKA_PARTITIONS, 3),
  }),

  ollama: Object.freeze({
    url: str(env.OLLAMA_URL, 'http://localhost:11434').replace(/\/+$/, ''),
    defaultModel: str(env.LLM_MODEL, 'gemma3:4b'),
    numCtx: int(env.LLM_NUM_CTX, 4096),
    numThread: int(env.LLM_NUM_THREAD, 0), // 0 = let Ollama decide
    keepAlive: str(env.LLM_KEEP_ALIVE, '-1'), // keep the model resident in RAM
    autoPull: bool(env.LLM_AUTO_PULL, true),
    concurrency: int(env.LLM_CONCURRENCY, 1), // match OLLAMA_NUM_PARALLEL; 1 on a CPU box: parallel generations just slow each other down
    requestTimeoutMs: int(env.LLM_TIMEOUT_MS, 120_000),
  }),

  // Groq: hosted LLM with several API keys that are rotated when one hits its rate limit (see src/llm/router.js).
  groq: Object.freeze({
    keys: str(env.GROQ_API_KEYS || env.GROQ_API_KEY, ''), // comma/space/newline separated; more keys can be added in the console
    baseUrl: str(env.GROQ_BASE_URL, 'https://api.groq.com/openai/v1').replace(/\/+$/, ''),
    concurrency: int(env.GROQ_CONCURRENCY, 32), // parallel Groq calls (the hosted model is not CPU-bound like the local one); each key's own rate limit is what really caps it
    timeoutMs: int(env.GROQ_TIMEOUT_MS, 30_000), // "no data for N ms"
  }),

  // Cloudflare Workers AI: a second hosted provider after Groq (free tier 10,000 neurons/day per account, reset 00:00 UTC).
  // Keys are "ACCOUNT_ID:API_TOKEN", separated by commas/spaces; more can be added in the console.
  cloudflare: Object.freeze({
    keys: str(env.CLOUDFLARE_API_KEYS, ''),
    baseUrl: str(env.CLOUDFLARE_BASE_URL, 'https://api.cloudflare.com/client/v4').replace(/\/+$/, ''),
    concurrency: int(env.CLOUDFLARE_CONCURRENCY, 8),
    timeoutMs: int(env.CLOUDFLARE_TIMEOUT_MS, 30_000),
  }),

  kb: Object.freeze({
    dir: path.resolve(root, str(env.KB_DIR, 'knowledge_base')),
    topK: int(env.KB_TOP_K, 4),
    maxContextTokens: int(env.KB_MAX_CONTEXT_TOKENS, 900),
    minScore: num(env.KB_MIN_SCORE, 2.2),
  }),

  memory: Object.freeze({
    contextTtlSec: int(env.CONTEXT_TTL_SEC, 6 * 3600),
    answerCacheTtlSec: int(env.ANSWER_CACHE_TTL_SEC, 12 * 3600),
  }),

  wa: Object.freeze({
    sessionDir: path.resolve(root, str(env.WA_SESSION_DIR, '.wwebjs_auth')),
    clientId: str(env.WA_CLIENT_ID, 'main'),
    autoConnect: bool(env.WA_AUTO_CONNECT, true),
    chromiumPath: str(env.PUPPETEER_EXECUTABLE_PATH, undefined),
    // Pin a WhatsApp Web build if the latest one breaks whatsapp-web.js, e.g. 2.3000.1023204200
    webVersion: str(env.WA_WEB_VERSION, ''),
  }),

  logs: Object.freeze({
    level: str(env.LOG_LEVEL, 'info'),
    bufferSize: int(env.LOG_BUFFER_SIZE, 2000),
  }),
});
