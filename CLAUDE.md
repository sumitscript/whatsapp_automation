# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A self-hosted WhatsApp auto-responder (v2):
- **Gateway** (Node, `whatsapp-web.js`): runs the WhatsApp session and the web console.
- **Worker** (Node): answers messages using **vectorless RAG** (BM25F over markdown sections) and a local Ollama LLM.

The bot is **eSumit**, the owner's AI twin, speaking in first person. It only replies inside a **session**: `/hi` starts one and `/bye` ends it. Both are case-insensitive and configurable via `startCommands`/`stopCommands`. Idle sessions time out. There is no number whitelisting.

The two talk over Kafka or Redis Streams. Postgres stores everything permanently. Redis holds chat memory, the cache and the live logs. The deploy target is a single CPU-only 8 vCPU / 32 GB KVM running Docker Compose.

**Owner preference:** do not install or download dependencies on the dev machine (`npm install`, `pip install`, model pulls). Declare them in `package.json` and let CI/Docker install them. Everything that can be verified without deps uses only Node built-ins.

Detailed docs: `docs/ARCHITECTURE.md` (design), `docs/DEPLOYMENT.md` (step-by-step deploy and ops), `docs/RESOURCES.md` (resource budget and dependencies). Knowledge-base sections are written in first person as Sumit.

## Commands

```bash
npm test                          # node:test (built-in runner), all test/*.test.js
node --test test/kb.test.js       # single file
node --test --test-name-pattern="cache" test/pipeline.test.js   # single test
npm run lint                      # scripts/check-syntax.js → node --check on src, scripts, test, public/js
npm run kb:search -- "question"   # inspect retrieval scores against knowledge_base/
npm run dev:gateway / dev:worker  # node --watch with --env-file-if-exists=.env (needs Postgres, Redis, Ollama)
npm run bench -- gemma3:4b        # TTFT / tok/s per model against a live Ollama
npm run eval -- --model gemma3:4b,llama3.2:3b   # replay test/eval/cases.json through the real pipeline against a LIVE Ollama (needs the models installed; exit 1 on any failed check; --only <id>, --json out.json, --temperature N)
```

Tests, lint and `kb:search` run **without `node_modules`**. The modules they touch (`src/rag`, `src/brain`, `src/auth/{jwt,password}`, `src/core/{util,settings-schema,config,logger}`, `src/llm/status.js`, `scripts/eval-lib.js`) must stay free of npm dependencies. Anything importing `pg`, `ioredis`, `kafkajs`, `express`, `socket.io` or `whatsapp-web.js` belongs elsewhere.

## Architecture

**Message flow:**
1. `gateway/whatsapp.js` (state machine; the QR is pushed as a data URL) receives the message.
2. `gateway/inbox.js` saves it to Postgres and applies reply policy:
   - checks `replyMode`, block, groups (reply only when mentioned) and rate limit;
   - handles commands via `brain/commands.js`. Start/stop change session state even when the chat is paused; welcome, goodbye and help are instant templates;
   - in `session` mode requires an open session (`gateway/sessions.js`: Postgres `sessions` table plus a cache; one open session per chat; idle sweeper);
   - then checks per-chat pause and the owner takeover key.
   Session side effects (summarize job, `chat:update`) live in the `sessions.on('end')` listener, so `/bye`, console and timeout endings behave the same.
3. Inbox debounces bursts per chat, sends seen + typing, and publishes a job to `wa.inbound` (keyed by chatId).
4. `worker/index.js` runs jobs in a per-chat `KeyedQueue` (keeps order) behind a `Semaphore` (`LLM_CONCURRENCY`), then calls `brain/pipeline.js`.
5. The pipeline streams bubbles back on `wa.outbound`. Inbox delivers them in order through its own per-chat lanes, keeps typing alive and saves each bubble. It drops bubbles whose session ended mid-generation.

**Memory (two levels):**
- **Short-term:** raw turns in a Redis list (`createRedisMemory`, capped at 80).
- **Long-term:** `contacts.memory` notes in Postgres.
- **Folding:** after a reply, if stored turns exceed `historyTurns + 8`, or when a session ends (`summarize` job on `wa.inbound`), the worker queues a fold: `brain.summarize` merges the oldest turns into the notes, then `trimOldest`. The fold queue only drains when `inFlight === 0`, so summaries never delay a live reply.

**Reply pipeline (`brain/pipeline.js`):**
- `resolveIntent` (regex `classifyIntent` plus inheritance) picks an `INTENT_PROFILES` entry. Short follow-ups ("one more", "why?") inherit the previous user turn's intent, and tiny `general` messages never use the KB. Without this a 3B model drifts onto random KB facts. The entry sets: temperature, token scale, and whether to use the KB (`true`, `false`, or `'auto'` = only if `topScore ≥ 1.6×minScore`).
- `detectLanguage` produces the "(Reply in X.)" hint. Follow-ups reuse the previous user turn as the retrieval query.
- **Context window:** `selectHistory` keeps the newest turns within `min(historyTokens, numCtx − system − final turn − maxTokens − 256)` and `historyTurns` messages. It starts on a user turn and merges same-role runs.
- The final user turn carries `[My facts — about <owner>, the one replying (me). Not about <name>]`, `[Notes about <name> — the person I'm chatting with. Their life, not mine]`, `[I'm chatting with]`, then `Their message:` and the language hint. These blocks sit in the *user* role, where "me/my" reads as the user, so every label states whose information it is (with "[Facts about me]" the model credited the owner's projects to the contact and retold the contact's life as its own). The persona prompt refers to "My facts" and "Notes about", so keep them in sync.
- `creative` intent (story, kahani, poem, shayari, horror…) gets no KB facts and no notes, temperature 0.95 and a larger budget; "your story" / "how you built X" stays `about_owner`. "N words" requests raise `maxTokens` (`requestedWords`).
- `selectHistory` shortens old long turns with `compactTurn` (head + tail, deterministic so Ollama's prefix cache survives); the newest turn stays whole. This keeps prompt evaluation fast after long stories or code.
- `summarize` always keeps a stated language preference as `- Prefers <Language>` in the notes.
- Short replies ("yes", "ok", "haan") get `[What I said last]` (the previous assistant turn) next to them so a small model continues that offer instead of an older topic. Long-term notes are only injected when the chat is fresh (< 6 stored turns) or the message refers to the past; mid-conversation they pulled models back to old topics. If generation stops on `doneReason: 'length'`, the last bubble is cut back to a full sentence (`trimDangling`).
- The final user turn ends with a `(…)` rules line built per message: no facts → "don't mention my job/projects/life" (stops invented employers); past reference ("last time", "I told you") → use the notes or admit there are none; "not sure / correct answer" → answer first; previous reply ended in "?" → don't end with another. `test/conversation-qa.test.js` pins these.
- Reply language comes from `replyLanguage` (`brain/lang.js`): a stated preference ("I prefer Hinglish", or "Prefers Hinglish" in the notes) beats a Latin-script message; native scripts always win; an English line inside a Hinglish chat stays Hinglish.
- About-me intent needs a profile topic (typo-tolerant, `PROFILE_WORDS` + OSA `editDistance`) plus "your/tumhara" or a short/neutral message; "my/mera/I" means it's about them ("tum meri help karo" stays general).
- `settings.persona` empty means the built-in `DEFAULT_PERSONA`, so prompt improvements ship with updates. A saved custom persona overrides it.
- KB notes go into the **last user turn**, not the system prompt, so the system prompt + history prefix stays stable for Ollama's prompt cache.
- Chat memory in Redis stores raw text only, never the notes.
- Answer cache only for standalone `about_owner` questions that had KB hits. Its key includes `kb.version` and a hash of the rendered persona, so KB or persona edits invalidate it.
- `MessageChunker` (`brain/format.js`) turns the stream into WhatsApp-formatted bubbles. It never splits inside a code fence and caps at `maxBubbles`.

**Retrieval (`rag/kb.js`, `rag/bm25.js`, `rag/text.js`):**
- Each H2/H3 is a section (an H1 summary becomes a section too). Fields: title = heading + parent ×3, tags = doc title/tags/aliases ×2, body ×1.
- Files starting with `_`, `README.md`, and fenced-code headings are ignored.
- Query expansion via `knowledge_base/synonyms.json` (weight 0.6, bigram keys supported). Typo fixes via bounded edit distance (weight 0.8).
- Selection: score ≥ max(`KB_MIN_SCORE`, 0.35×top), up to `KB_TOP_K`, within the `KB_MAX_CONTEXT_TOKENS` budget.
- When changing tokenizer or stopword rules, rerun `kb:search` on a few real questions. The thresholds were calibrated on the current KB.

**Cross-process state:**
- **Settings:** stored in Postgres (`settings` key/value jsonb), validated by `core/settings-schema.js`. Every process caches them and reloads on the Redis `ch:settings` pub/sub. A new setting must be added to `DEFAULT_SETTINGS` **and** `RULES` there, or it is rejected.
- **KB edits:** made through the API, which writes the file, reloads it and publishes `ch:kb`. Both processes also `fs.watch` the directory.
- **LLM status:** the worker writes a heartbeat to Redis `worker:status` (every 5 s, 20 s TTL). `GET /api/llm/status` merges it with live Ollama `/api/version`, `/api/tags` and `/api/ps` for the dashboard card.
- **Agenda:** the worker spots commitments ("let's meet tomorrow at 5", "remind me to…") and stores them in `agenda_items`. `brain/agenda.js` (pure) gates messages with `looksLikeCommitment`, and `brain.extractAgenda` asks the LLM for kind/title/details as JSON. **Dates are read by `parseWhen` with Intl in `settings.timezone`, never by the model** (a 3B model is unreliable at date math). Extraction runs on the worker's idle queue, ahead of memory folds, and never delays a live reply. If the model fails, `heuristicItem` still records a plain "let's meet tomorrow at 5pm". `sameItem` merges refinements ("make it 6pm") into the existing item. The worker publishes `{type:'agenda'}` on `wa.outbound`; inbox emits `agenda:update` and WhatsApps the owner (`notifyOwner`). The reply prompt also gets a rule not to agree to meetings for the owner. Console: `/api/agenda` + `public/js/views/agenda.js`; confirming/declining can message the contact (`notify: true`).
- **Speed (docs/SPEED.md):** one generation at a time (`LLM_CONCURRENCY=1`: parallel generations share the CPU and each runs ~3x slower); background work (agenda, folds) waits `WORKER_BG_QUIET_MS` after the last message; and "think-ahead": after a reply the worker has the model pre-read persona + history (`brain.warm` → `OllamaClient.prefill`, cancelled when a message arrives) so the next reply reads only its own tokens. gemma3 cannot reuse a partial prefix (`reusesPrefix()` is false, think-ahead is skipped); llama/qwen can (−25% to −31% prompt time measured). Setting `prefillAhead`. `warm()` must build exactly the prefix `respond()` sends (tested in `test/prefill.test.js`).
- **Owner alerts:** `gateway/llm-monitor.js` probes the LLM every 30 s (`probeLlmStatus` → pure `deriveLlmState`), emits `llm:status`, and after 90 s of "down" (not in the first 5 min after boot) messages the owner's own number once, then once on recovery (`DownAlerter`). `inbox.notifyOwner` sends to `wa.me.wid`; inbound `fromMe` messages are ignored so these never become chat input.
- **Notices to contacts:** session timeout (`timeoutMessage`, from the `end` listener with reason `timeout`) and unreadable media — voice notes, photos, stickers (`unsupportedMessage`, 60 s cooldown per chat, only types in `MEDIA_KINDS`). Both are templates rendered by `renderTemplate` ({name} {persona} {owner} {start} {stop} {kind}).
- **Logs:** go to stdout **and** to a capped Redis list + pub/sub (`core/redis.js` `attachRedisLogSink`). `gateway/realtime.js` relays them to Socket.IO room `logs`. Never log inside that relay handler: it would loop.
- **Own-message echoes:** inbox tracks bodies it is sending (`echo` map). `message_create` events from the phone that aren't ours count as owner takeover (Redis `takeover:<chatId>`, TTL = `takeoverSeconds`, default 30) and are added to memory as assistant turns.

**Web console (`public/`):** vanilla ES modules, no build step, served statically by the gateway.
- The API contract is the routes in `gateway/http.js`. Socket events are `wa:status`, `message:new`, `chat:update`, `typing`, `log` and `model:pull`.
- The CSP is `script-src 'self'`, so no inline scripts.
- Render with the escaping `html` tagged template from `public/js/util.js`. Use `raw()` only for trusted markup.
- `public/js/fx.js` holds library-free ports of the Aceternity effects (Evervault, ContainerScroll, MacbookScroll; FocusCards is the CSS class `.focus-cards`). They write CSS variables once per frame and fall back to a static state under `prefers-reduced-motion`. The Socket.IO client is loaded lazily after sign-in.

**Schema:** `db/migrations/*.sql`, applied in order by `db/pg.js` under an advisory lock (gateway and worker can both boot it). Add new files; never edit applied ones. Rows are mapped to camelCase in `db/repo.js`.

**Deploy:**
- One image with two commands (`docker-compose.yml`). The bus defaults to Redis Streams; Kafka is opt-in (`BUS_DRIVER=kafka` + compose profile `kafka`, ~1 GB RAM).
- CI (`.github/workflows/ci.yml`) runs a boot smoke test with `BUS_DRIVER=redis` and `WA_AUTO_CONNECT=false`.
- `deploy.yml` builds to GHCR and runs `deploy/deploy.sh` over SSH, which health-checks and rolls back.
- The server's `knowledge_base/` is a bind mount seeded once from the image; it is the live source of truth.
