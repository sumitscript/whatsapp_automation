# Architecture

eSumit is Sumit's AI twin on WhatsApp. It is a self-hosted system with two Node processes, three datastores and a local LLM, all running on one CPU-only KVM with Docker Compose.

## Components

| Component | Code | Responsibility |
|---|---|---|
| **Gateway** | `src/gateway/` | Runs the WhatsApp Web session, applies reply policy, delivers replies, and serves the REST API, Socket.IO and the web console |
| **Worker** | `src/worker/index.js` | Consumes reply jobs, runs the reply pipeline against Ollama, folds conversation memory |
| **Brain** | `src/brain/` | Intent, language, commands, prompt assembly, context budgeting, streaming to WhatsApp bubbles |
| **RAG** | `src/rag/` | Vectorless retrieval: markdown section tree + BM25F |
| **LLM client** | `src/llm/ollama.js`, `src/llm/catalog.js` | Streaming chat, model pull/warmup, curated model catalog |
| **Bus** | `src/bus/index.js` | Kafka (`kafkajs`) or Redis Streams, chosen by `BUS_DRIVER` |
| **Data** | `src/db/`, `db/migrations/` | Postgres pool, migrations, repository (rows → camelCase) |
| **Core** | `src/core/` | Config, logger, Redis helpers, settings service + schema, utilities |
| **Auth** | `src/auth/` | scrypt passwords, HS256 JWT cookies |
| **Console** | `public/` | WhatsApp-style admin UI. No build step and no frontend libraries: just vanilla ES modules |
| **Edge** | `deploy/Caddyfile` | TLS (Let's Encrypt), gzip/zstd and the reverse proxy to the gateway. It's the only public service |

## Message flow

```
 WhatsApp phone
      │  (WhatsApp Web, whatsapp-web.js + Chromium)
      ▼
┌──────────────────────── gateway ────────────────────────┐
│ whatsapp.js  state machine, QR → data URL                │
│ inbox.js     save to Postgres → policy → commands →      │
│              session check → debounce → seen + typing    │
│ sessions.js  /hi … /bye, idle sweeper                    │
└───────────────┬───────────────────────────▲──────────────┘
   wa.inbound   │ (key = chatId)            │ wa.outbound (chunk / done / error / memory)
                ▼                           │
┌──────────────────────── worker ─────────────────────────┐
│ KeyedQueue per chat (order) + Semaphore (LLM_CONCURRENCY)│
│ brain/pipeline.js: intent → language → retrieval →       │
│   answer cache → context budget → Ollama stream →        │
│   MessageChunker bubbles                                 │
│ fold queue: long-term memory (only when idle)            │
└───────┬───────────────┬──────────────────┬──────────────┘
        ▼               ▼                  ▼
     Ollama          Redis              Postgres
  (gemma3:4b…)   ctx, cache, logs    messages, contacts,
                                     sessions, settings, audit
```

1. `gateway/whatsapp.js` receives the message.
2. `gateway/inbox.js` stores it. It then checks, in order: `replyMode` (`session` | `all` | `off`), blocked contacts, groups (replies only when mentioned, if `replyInGroups`), and empty or unsupported messages.
3. Commands are parsed by `brain/commands.js`. The rate limit (`rateLimitPerMin`) applies to commands, and to normal messages when a session is open.
   - Start and stop commands change session state even if the chat is paused.
   - The welcome, goodbye and `/help` replies are instant templates and make no LLM call.
4. In `session` mode an open session is required, and each message touches it. Per-chat `aiPaused` and owner takeover (Redis `takeover:<chatId>`) keep the AI quiet.
5. Bursts are debounced (`debounceMs`). The gateway sends seen and typing, then publishes a `{type:'reply'}` job to `wa.inbound`.
6. The worker (consumer group `ai-worker`) runs the pipeline and streams bubbles back on `wa.outbound`.
7. The gateway (consumer group `gateway`) delivers them in order per chat:
   - It refreshes typing every 20 s and saves each bubble.
   - It drops bubbles if the session ended or a takeover started mid-generation.

## Session lifecycle

```
           /hi (any case, "/hi!" ok, "/hi <question>")      console "Start"
 (closed) ───────────────────────────────────────────────►  (open) ──┐
     ▲                                                        │  each message touches
     │  /bye · idle > sessionIdleMinutes · console "End"      │  last_activity_at
     └────────────────────────────────────────────────────────┘◄─┘
```

- **Commands** come from settings `startCommands` and `stopCommands` (comma lists, validated as `/word`). `/reset` (clears short-term memory) and `/help` are built in.
- **Storage:** the Postgres `sessions` table has a unique partial index, so there is at most one open session per chat. `SessionManager` keeps a cache in front of it.
- **Idle timeout:** the sweeper runs every 60 s and ends idle sessions with reason `timeout`. Other end reasons are `command` and `console`.
- **On every session end** (one `sessions.on('end')` listener):
  - cancel the pending debounce;
  - stop typing;
  - publish a `{type:'summarize'}` job;
  - emit `chat:update`.
- **From the console** you can start or end a session with optional `notify`, which sends the welcome or goodbye template.

## Memory (two levels)

| Level | Where | Contents | Lifetime |
|---|---|---|---|
| Short-term | Redis list `wa:ctx:<chatId>` | Raw user/assistant turns, capped at 80 (`MAX_MEMORY_MESSAGES`) | `CONTEXT_TTL_SEC` (default 6 h) |
| Long-term | Postgres `contacts.memory`, `memory_updated_at` | Bullet notes about the person, at most 120 words | Permanent; editable or erasable in the console |

**Folding:**
- **When:**
  - after a reply, if the stored turns exceed `historyTurns + 8`;
  - or on a `summarize` job at session end.
- **How:**
  1. The worker queues a fold.
  2. A 1.5 s timer drains one fold only when `inFlight === 0`, so summarizing never delays a live reply.
  3. `brain.summarize` merges the oldest turns into the existing notes (temperature 0.2).
  4. `repo.setMemory` saves the notes, `memory.trimOldest` drops the folded turns, and a `memory` event goes out on `wa.outbound`.

Owner messages sent from the phone are added to short-term memory as assistant turns. They also set the takeover key for `takeoverSeconds` (default 30).

## Reply pipeline (`brain/pipeline.js`)

1. **Intent:** `classifyIntent` (regex, about 0 ms) returns `greeting`, `thanks`, `joke`, `coding`, `math`, `about_owner` or `general`. Each maps to an `INTENT_PROFILES` entry:
   - temperature;
   - token scale;
   - whether to use the KB: `true`, `false`, or `'auto'` (only if `topScore ≥ 1.6 × minScore`).
2. **Language:** `detectLanguage` uses Unicode script ranges plus Hinglish word cues. It yields a `(Reply in X.)` hint for Hindi, Hinglish, Marathi, Gujarati, Tamil, Telugu, Bengali, Kannada, Malayalam, Punjabi, Urdu and Odia.
3. **Retrieval:** `kb.search`. A follow-up (`isFollowUp`) reuses the previous user turn as the query.
4. **Answer cache:** only for standalone `about_owner` questions with KB hits.
   - Key: `ans:sha1(model | kb.version | rendered persona | language | normalized text)`.
   - KB edits, persona edits and model switches therefore invalidate it.
5. **Context budget:**
   - Formula: `historyBudget = min(historyTokens, numCtx − system − finalTurn − maxTokens − 256)`.
   - `selectHistory` keeps the newest turns within that budget, capped at `historyTurns` messages. It starts on a user turn and merges same-role runs.
6. **Prompt layout:**
   - The system prompt is the persona (first person, as Sumit). It stays stable so Ollama can reuse its KV-cache prefix.
   - Then the history.
   - Then a final user turn holding `[Facts about me]` (KB sections, written in first person), `[What I remember about <name>]`, `[Chatting with] <name>`, `---`, the message, and the language hint.
7. **Generation:** Ollama `/api/chat` streams with `keep_alive: -1`, `num_ctx` (`LLM_NUM_CTX`), and `think:false` for qwen3-family models.
8. **Bubbles:** `MessageChunker` (`brain/format.js`) converts Markdown to WhatsApp formatting and splits at paragraph breaks (`minChars 60`, `maxChars 700`). It never splits inside a code fence and caps output at `maxBubbles 4`.
9. **Record:** the raw user text and reply are appended to short-term memory (never the KB notes), and `done` meta (intent, language, latency, sources, cached) is saved with the message.

## Vectorless RAG (`src/rag/`)

- **Corpus:** `knowledge_base/*.md`. Files starting with `_` and `README.md` are ignored.
  - Each `##` / `###` heading is one section, with a breadcrumb path; an H1 summary becomes a section too.
  - Headings inside code fences are ignored.
  - Frontmatter supplies `title`, `tags` and `aliases`.
  - Sections are written in first person as Sumit.
- **Index:** `BM25FIndex` scores three fields: title (heading + parent) ×3, tags (doc title/tags/aliases) ×2, body ×1. Sections that match more distinct query terms get a coverage bonus.
- **Text processing** (`text.js`):
  - normalization and rewrites (`c++ → cpp`, …);
  - English, Hinglish and Devanagari stopwords;
  - a light stemmer (tokens ending in `js` are not stemmed);
  - token estimate.
- **Query expansion:** `knowledge_base/synonyms.json`, including bigram keys, at weight 0.6. Unknown words are typo-matched by bounded edit distance at weight 0.8.
- **Selection:** score ≥ `max(KB_MIN_SCORE, 0.35 × top)`, at most `KB_TOP_K` sections, within `KB_MAX_CONTEXT_TOKENS`.
- **Reloading:** `fs.watch` hot reload plus the Redis `ch:kb` channel. Results are memoized in an LRU keyed by KB version.

## Data model (Postgres)

Migrations live in `db/migrations/`. `db/pg.js` applies them in order under an advisory lock.

| Table | Purpose / key columns |
|---|---|
| `users` | Console users: `username` (unique), `password_hash` (scrypt), `role`, `last_login_at` |
| `contacts` | One row per chat: `chat_id` (unique), `phone` (unique), `name`, `is_group`, `is_blocked`, `ai_paused`, `notes`, `memory`, `memory_updated_at`, `last_message_at` |
| `messages` | Permanent log: `wa_message_id` (unique), `chat_id`, `direction` (`in`/`out`), `body`, `type`, `is_ai`, `sent_by`, `model`, `intent`, `language`, `latency_ms`, `cached`, `sources` (jsonb), `created_at` |
| `sessions` | AI sessions: `chat_id`, `started_at`, `last_activity_at`, `ended_at`, `ended_reason` (`command`/`timeout`/`console`), `message_count`. Unique open session per chat |
| `settings` | Key/value `jsonb`, validated by `core/settings-schema.js` |
| `audit_log` | Console actions: `username`, `action`, `details` |

## Redis

The key prefix comes from `REDIS_PREFIX` (default `wa:`).

| Key / channel | Use |
|---|---|
| `wa:ctx:<chatId>` | Short-term memory list |
| `wa:ans:<sha1>` | Memoized answers (`ANSWER_CACHE_TTL_SEC`, default 12 h) |
| `wa:rl:<scope>:<id>:<minute>` | Per-minute rate limits (chat replies, logins) |
| `wa:takeover:<chatId>` | Owner takeover (TTL = `takeoverSeconds`) |
| `wa:logs` | Capped log ring buffer (`LOG_BUFFER_SIZE`, default 2000) |
| `wa:stats:cache_hits:<day>` | Cache-hit counter for the dashboard |
| `wa:ch:logs` | Pub/sub: live log lines → Socket.IO room `logs` |
| `wa:ch:settings` | Pub/sub: every process reloads settings |
| `wa:ch:kb` | Pub/sub: every process reloads the KB |

Redis runs with AOF on, `maxmemory 512mb` and `volatile-lru`. Only keys that have a TTL can be evicted.

## Bus

| Topic | Producer → consumer group | Payloads |
|---|---|---|
| `wa.inbound` | gateway → `ai-worker` | `reply` jobs, `summarize` jobs |
| `wa.outbound` | worker → `gateway` | `chunk`, `done`, `error`, `memory` events |

Messages are keyed by `chatId`, so one chat stays in order on one partition.

**Drivers:**
- **`BUS_DRIVER=kafka`:** `kafkajs` with a single-node KRaft broker. Topics are created with `KAFKA_PARTITIONS` (default 3).
- **`BUS_DRIVER=redis`:** Redis Streams with consumer groups (`XADD MAXLEN ~10000`, `XREADGROUP`). Uses less memory, and the compose `kafka` profile can be turned off.

## Auth and security

- **Passwords:** scrypt via `node:crypto`.
- **Sessions:** an HS256 JWT in the `wa_session` cookie (`HttpOnly`, `SameSite=Strict`, `Secure` when `COOKIE_SECURE`). TTL is `SESSION_TTL_HOURS`.
  - The token embeds a fingerprint of the password hash, so a password change revokes every existing session.
- **First admin:** comes from `ADMIN_USERNAME` / `ADMIN_PASSWORD`. If the password is empty, a one-time password is generated and logged. `npm run create-admin` creates or resets a user.
- **Login limit:** 10 attempts per minute per IP.
- **Same-origin writes:** mutating requests must come from `PUBLIC_ORIGIN`, or from the same host when it's unset.
- **Headers:** strict CSP (`script-src 'self'`, so no inline scripts), `X-Frame-Options: DENY`, `nosniff`, and HSTS in production.
- **Audit:** console actions are recorded in `audit_log`.
- **Network:** only Caddy publishes ports (80/443). Ollama, Postgres, Redis and Kafka stay on the private compose network.

## Realtime (Socket.IO)

- **Connection:** authenticated by the same cookie. Admins join the room `admins`.
- **Logs:** `logs:subscribe` / `logs:unsubscribe` join and leave the room `logs`.
- **Events:**
  - `wa:status` (state and QR data URL)
  - `message:new`
  - `chat:update`
  - `typing`
  - `log`
  - `model:pull` (pull progress)

## Failure handling

- **WhatsApp:**
  - State machine: `DISCONNECTED → QR → AUTHENTICATED → READY`.
  - It reconnects with exponential backoff (5 s doubling, capped at 120 s), except after `LOGOUT`, "max qrcode retries" or unpair.
  - QR codes stop after 8 retries (about 3 minutes) to save CPU.
  - `auth_failure` wipes the saved session so the next connect shows a fresh QR.
  - Stale Chromium `Singleton*` locks are removed before launch.
- **Redis:** the client retries with backoff (up to 5 s). Redis Streams read errors are logged and retried.
- **Kafka:** the topic-creation race between processes is tolerated. The outbound handler isn't awaited per message, so one slow chat can't block a partition.
- **LLM:** if the model returns empty output, the reply is a polite fallback. Request timeout is `LLM_TIMEOUT_MS` (default 120 s). Errors are sent to the chat as an error notice.
- **Deploy:** `deploy/deploy.sh` waits for `/readyz` (Postgres and Redis reachable). If the gateway isn't healthy within about 3 minutes, it rolls back to the previous image.
- **Health endpoints:**
  - gateway: `/healthz` (liveness) and `/readyz` (readiness);
  - worker: `/healthz` on port 3001;
  - Docker: a `HEALTHCHECK` in the image.
