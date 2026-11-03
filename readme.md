# eSumit — WhatsApp AI twin (v2.1)

**eSumit** is Sumit's virtual version: a self-hosted AI that chats on WhatsApp *as Sumit*, in first person.

People talk to it by opening a session:
- `/hi` (any case, e.g. `/Hi`, `/HI`, `/hi tell me about your projects`) starts a chat with eSumit. The welcome message says up front that it's an AI twin.
- `/bye` (`/Bye`, `/BYE`…) ends it. Sessions also end after 30 idle minutes.
- `/reset` gives a fresh start; `/help` lists the commands.
- Outside a session eSumit stays silent, so normal chats with the real Sumit are untouched.

It's a self-hosted WhatsApp assistant that replies for you. It uses a local LLM through Ollama and a **vectorless RAG** knowledge base, and comes with a WhatsApp-style web console:
- link WhatsApp by QR code
- see every conversation
- see who has an open eSumit session, and what it remembers about each person
- tail logs live
- switch models
- edit the knowledge base

It is built for one CPU-only KVM (8 vCPU / 32 GB), deployed with Docker Compose and GitHub Actions.

```
 WhatsApp ──► gateway (Node) ──Kafka/Redis──► worker (Node) ──► Ollama (gemma3:4b, CPU)
               │  whatsapp-web.js + Chromium          │  intent + language (regex, 0 ms)
               │  policy · debounce · typing          │  BM25F retrieval (~1 ms)
               │  REST + Socket.IO + console          │  memoized answers (Redis)
               ▼                                      ▼
           Postgres (all messages, contacts,      Redis (chat memory, cache,
           settings, audit)                        rate limits, live logs)
```

## Why vectorless RAG

The knowledge base is ~100 markdown sections. Embeddings + a vector DB add a model call per query, a re-indexing step, and ~2 GB of Python ML dependencies. None of that buys accuracy at this size. Instead:

- **Heading-tree sections.** Each `##` / `###` in `knowledge_base/*.md` is one self-contained unit, with its breadcrumb path (`Projects › NeoAI — Architecture`).
- **BM25F scoring.** Headings are weighted ×3, file tags/aliases ×2 and body ×1, with a bonus for sections that match more of the query.
- **Query expansion** from `knowledge_base/synonyms.json`: Hinglish and Hindi → English (`naukri → job`, `पढ़ाई → education`) and abbreviations (`js → javascript`).
- **Typo tolerance.** Unknown words are matched against the vocabulary by edit distance (`projct → project`).
- **Hot reload.** Re-indexing takes milliseconds, so edits (from the console or on disk) are live immediately.

Measured on this KB: **0.2–1.1 ms per query**. Try it with `npm run kb:search -- "Sumit kahan kaam karta hai"`.

## What makes replies fast and natural

| Technique | Effect |
|---|---|
| Model kept resident in RAM (`keep_alive -1`) and warmed up at boot | No cold start |
| Intent routing without an LLM call (greeting, joke, coding, math, about-owner, general) | Picks retrieval on/off, temperature (0.1 for math → 0.9 for jokes) and reply length |
| Explicit language hint (Devanagari Hindi, Hinglish, Marathi, Gujarati, Tamil, Telugu, Bengali, Kannada, Malayalam, Punjabi, Urdu, Odia) | Small models reliably answer in the user's language and script |
| Streaming + bubble splitting at paragraph breaks (never inside code) | The first WhatsApp bubble goes out while the rest is still being written, like a person texting |
| Memoized answers in Redis, keyed by model + KB version + language + normalized question | Repeated factual questions answer in ~0 ms; invalidated automatically when the KB changes |
| Debouncing (900 ms default) | "hi" + "u there?" + "need help" becomes one turn, not three replies |
| Typing indicator refreshed every 20 s, blue ticks, ordered delivery per chat | Feels like a person on the other end |
| Owner takeover | If you reply from your phone, the AI pauses in that chat for 30 s (configurable, in seconds) |
| Instant command replies | Welcome/goodbye/help are templates (0 ms, no LLM); `/hi <question>` answers the question right after the welcome |
| Context-window budgeting | System prompt + facts + reply budget are reserved first; the newest history that fits `historyTokens` and `num_ctx` is sent, never an overflowing prompt |
| Conversation memory, 2 levels | **Short-term**: raw turns in Redis. **Long-term**: per-contact notes in Postgres, summarized from older turns and at session end — only while no reply is generating, so it never slows a chat |
| Small `num_ctx` (4096), q8 KV cache, flash attention, stable system-prompt prefix | Faster prompt processing on CPU |

## Choosing a model (CPU, 8 vCPU)

Switch models live in **Settings → Models** (one click pulls them). Benchmark on your own box with `npm run bench -- gemma3:4b qwen3:4b llama3.2:3b`.

| Model | Tier | RAM | Indian languages | Rough speed* | Use it when |
|---|---|---|---|---|---|
| **gemma3:4b** (default) | balanced | ~4 GB | Strong Hindi/Hinglish, good across major Indic scripts | ~10–15 tok/s | Best all-rounder: natural tone, jokes, multilingual |
| qwen3:4b | balanced | ~3.5 GB | Good Hindi | ~10–15 tok/s | More coding/reasoning weight (thinking auto-disabled) |
| llama3.2:3b | fast | ~2.5 GB | Hindi OK, weaker on other scripts | ~15–22 tok/s | Maximum speed, mostly English/Hinglish chats |
| qwen2.5:3b | fast | ~2.3 GB | Hindi OK | ~15–22 tok/s | Short factual answers |
| gemma3:12b | quality | ~9 GB | Best Indic quality here | ~4–6 tok/s | Quality over speed |
| qwen3:8b | quality | ~6 GB | Good | ~6–9 tok/s | Heavier coding help |
| aya-expanse:8b | quality | ~6 GB | Very natural Hindi chat | ~6–9 tok/s | Hindi-first audiences |

\*Estimates for Q4 quantization on 8 shared vCPUs. Your numbers depend on the CPU generation; `npm run bench` gives the real ones.

## Console features

- **Connection**: QR code shown in the browser, live state (DISCONNECTED → QR → AUTHENTICATED → READY), and Connect / Disconnect (keeps the session) / Log out (unlinks the device) / Restart. Auto-reconnects with backoff.
- **Chats**: every conversation with timestamps; AI bubbles tagged with model, latency and cache hits; live "typing…"; send manual messages; per-chat start/end session, pause, block, and a memory drawer.
- **Sessions**: chats with an open eSumit session are marked "AI on". You can start/end a session from the chat header (optionally telling the contact). Reply modes are `session` (default: only after `/hi`), `all` or `off`.
- **Memory**: per chat, view or edit the long-term notes eSumit keeps about the person, see the short-term turns, or make it forget.
- **Logs**: live tail across gateway and worker, with level, service and text filters, plus pause.
- **Knowledge**: edit KB files in the browser (Ctrl+S saves and re-indexes) and test retrieval with scores.
- **Settings**: eSumit persona name, start/stop commands, welcome/goodbye templates, idle timeout, history messages/tokens, memory on/off, persona prompt, temperature, length, history depth, debounce, groups, humanize, cache, rate limit, model catalog and pull, a playground, and password change.
- **Dashboard**: messages today, AI replies, average/p95 latency, cache hits, active chats, and a 24 h chart.

Security: scrypt passwords, HS256 session cookies (httpOnly, SameSite=Strict, Secure), sessions revoked when the password changes, login rate limit, same-origin checks on writes, strict CSP, audit log. Only Caddy (80/443) is exposed; Ollama, Postgres, Redis and Kafka stay private.

## Deploy on the KVM

Short version:
1. Run `bash deploy/server-setup.sh deploy` on the server, as root.
2. Fill in `/opt/whatsapp-ai/.env`.
3. Point DNS at the server.
4. Add the GitHub secrets.
5. Push to `main`.

CI tests and builds the image, and Deploy rolls it out over SSH, with automatic rollback if the gateway isn't healthy. Then open the console and scan the QR under **Connection**.

The full walkthrough is in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**: env values, secrets, verification, backups and restore, model changes and troubleshooting.

## Local development

You need Node 22.9+, plus Postgres, Redis and Ollama running locally (or `docker compose up -d postgres redis ollama`).

```bash
cp .env.example .env          # set BUS_DRIVER=redis to skip Kafka locally
npm install
npm run dev:gateway           # http://localhost:3000
npm run dev:worker
npm test                      # node:test, no extra deps
npm run lint                  # node --check on every file
node --test test/kb.test.js   # a single test file
npm run kb:search -- "what is his cgpa"
npm run bench -- gemma3:4b
npm run create-admin -- alice 'a-long-password'
```

## Knowledge base format

See `knowledge_base/_GUIDE.md`. In short:
- Each file has frontmatter with `title`, `tags` (English keywords) and `aliases` (Hinglish, Devanagari and common misspellings).
- Each `##` / `###` section must stand alone, so name the person, project and dates in it.
- `10_faq.md` uses `## Q: …` headings, which match questions very well.
- Sections are written in first person, as Sumit ("I work at…"), so retrieved facts drop straight into eSumit's replies.

## Notes and limits

- `whatsapp-web.js` drives WhatsApp Web through an unofficial client. Keep reply volume human-like: session mode, rate limits and debouncing are on by default. Automated accounts can be restricted by WhatsApp.
- Run **one gateway** per WhatsApp account. Workers are stateless and can be scaled (`docker compose up -d --scale worker=2`), but on a CPU-only box Ollama is the bottleneck.
- Voice notes and images are logged but not answered.
- eSumit answers personal questions only from the knowledge base. It won't make commitments, share private details or invent facts about Sumit; it offers to pass those to the real Sumit instead.


## Docs

- [Architecture](docs/ARCHITECTURE.md): components, message flow, sessions, memory, the reply pipeline, RAG internals, data model, security.
- [Deployment](docs/DEPLOYMENT.md): step by step from a fresh KVM to a scanned QR, plus operations and troubleshooting.
- [Resources](docs/RESOURCES.md): RAM/CPU/disk/ports per service, model sizes, and every dependency and external service used.

## License

MIT
