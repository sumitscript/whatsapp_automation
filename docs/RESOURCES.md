# Resources

This page lists what the stack uses on the target KVM (8 vCPU / 32 GB RAM, CPU only) and which software it depends on.
- Limits and ports come from `docker-compose.yml`.
- Figures marked *est.* are rough estimates. Measure on your own box with `docker stats` and `npm run bench`.

## Services, limits and ports

| Service | Image | Memory limit | CPU | Ports | Notes |
|---|---|---|---|---|---|
| `caddy` | `caddy:2.10-alpine` | 256 MB | shared | **80, 443/tcp, 443/udp (public)** | TLS, compression, reverse proxy to `gateway:3000` |
| `gateway` | app image | 2 GB | shared | 3000 (internal) | Node and Chromium (whatsapp-web.js). `shm_size: 1gb` |
| `worker` | app image | 768 MB | shared | 3001 (internal, `/healthz`) | Node only. The heavy work happens in Ollama |
| `ollama` | `ollama/ollama:${OLLAMA_VERSION:-0.12.6}` | 20 GB | most of the CPU while generating | 11434 (internal) | `KEEP_ALIVE=-1`, `NUM_PARALLEL=${LLM_CONCURRENCY:-2}`, `MAX_LOADED_MODELS=1`, flash attention, `q8_0` KV cache |
| `postgres` | `postgres:17-alpine` | 1 GB | shared | 5432 (internal) | `shared_buffers=256MB`, `max_connections=50` |
| `redis` | `redis:7.4-alpine` | 768 MB | shared | 6379 (internal) | AOF, `maxmemory 512mb`, `volatile-lru` |
| `kafka` | `apache/kafka:3.9.1` | 1 GB | shared | 9092/9093 (internal) | Profile `kafka`, KRaft single node, heap 256–512 MB, 72 h retention |

Only Caddy publishes ports. With `ufw` enabled by `deploy/server-setup.sh`, the host allows only SSH, 80 and 443.

## RAM budget (32 GB)

| | With Kafka | With `BUS_DRIVER=redis` |
|---|---|---|
| Sum of compose limits | 25.75 GB | 24.75 GB |
| Typical use with `gemma3:4b`, 2 parallel slots *(est.)* | ~7–9 GB | ~6–8 GB |
| Headroom for the OS, page cache and Docker | ~6.5 GB+ | ~7.5 GB+ |

Ollama's 20 GB limit fits any model in the catalog plus its KV cache. Only one model is loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`).

## CPU

- **Generation is CPU-bound in Ollama.** `LLM_NUM_THREAD` (0 = auto; 8 suggested on 8 vCPU) and `LLM_CONCURRENCY` (default 2, the same as `OLLAMA_NUM_PARALLEL`) set the trade-off: two parallel replies share the cores, so each one is slower.
- **The other services are light:**
  - Retrieval takes 0.2–1.1 ms per query (measured on the bundled KB).
  - Intent and language detection are regex.
  - Chromium idles after WhatsApp Web has loaded.
- **Waiting for a QR scan costs CPU,** so QR generation stops after 8 codes.

## Models (`src/llm/catalog.js`)

| Model | Tier | Params | RAM | Speed on 8 vCPU *(est., Q4)* |
|---|---|---|---|---|
| `gemma3:4b` (default) | balanced | 4B | ~4 GB | ≈10–15 tok/s |
| `qwen3:4b` | balanced | 4B | ~3.5 GB | ≈10–15 tok/s |
| `llama3.2:3b` | fast | 3B | ~2.5 GB | ≈15–22 tok/s |
| `qwen2.5:3b` | fast | 3B | ~2.3 GB | ≈15–22 tok/s |
| `gemma3:12b` | quality | 12B | ~9 GB | ≈4–6 tok/s |
| `qwen3:8b` | quality | 8B | ~6 GB | ≈6–9 tok/s |
| `aya-expanse:8b` | quality | 8B | ~6 GB | ≈6–9 tok/s |

Each model takes roughly its RAM figure in disk space as well, in the `ollama_models` volume.

## Disk

| Volume / path | Contents | Size *(est.)* |
|---|---|---|
| `ollama_models` | Pulled models | ~2.3–9 GB per model |
| App image | Node 22 slim, Chromium, Noto fonts, npm deps | ~1–1.5 GB per tag. Old tags are pruned after a healthy deploy |
| `pg_data` | Messages, contacts, sessions, settings, audit | Grows with traffic; small (MBs) for personal use |
| `redis_data` | AOF: memory, cache, logs | Under 512 MB |
| `kafka_data` | Bus logs, 72 h retention | Small at personal volume |
| `wa_session`, `wa_cache` | WhatsApp Web login and Chromium cache | ~100–500 MB |
| `caddy_data`, `caddy_config` | TLS certificates | KBs |
| `/opt/whatsapp-ai/knowledge_base` | Live KB (bind mount) | KBs |
| `/opt/whatsapp-ai/backups` | 14 days of `pg_dump` + KB tarballs | Depends on DB size |
| Container logs | `json-file`, 20 MB × 5 files per app container | ≤100 MB each |

40 GB of free disk is comfortable for one or two models.

## Network

**Inbound:** 80/443 (Caddy) and 22 (SSH).

**Outbound:**
- WhatsApp Web (`web.whatsapp.com` and WhatsApp's media/CDN hosts) for the linked session.
- `registry.ollama.ai` for model pulls.
- `ghcr.io` for image pulls during deploys.
- Let's Encrypt ACME for certificates.
- Docker Hub for the base images.

Message text and LLM inference never leave the box.

## Software used

### npm dependencies (`package.json`)

The dev machine installs none of these; CI and the Docker build install them.

| Package | Range | Why |
|---|---|---|
| `whatsapp-web.js` | ^1.31.0 | WhatsApp Web client (drives system Chromium via puppeteer) |
| `express` | ^5.1.0 | REST API and static console |
| `socket.io` | ^4.8.1 | Realtime console updates. Also serves the browser client at `/socket.io/socket.io.js` |
| `pg` | ^8.16.0 | Postgres driver |
| `ioredis` | ^5.6.1 | Redis: memory, cache, rate limits, logs, pub/sub, Streams bus |
| `kafkajs` | ^2.2.4 | Kafka bus driver |
| `qrcode` | ^1.5.4 | Renders the WhatsApp QR as a data URL for the console |

There are no dev dependencies:
- Tests use `node:test`.
- Lint is `node --check` (`scripts/check-syntax.js`).
- The console has no build step and no frontend libraries: vanilla ES modules and CSS.

### Node built-ins the dependency-free core relies on

`src/rag`, `src/brain`, `src/auth/{jwt,password}` and `src/core/{util,settings-schema,config,logger}` use only these:

| Built-in | Used for |
|---|---|
| `node:crypto` | scrypt passwords, HMAC-SHA256 JWTs, `timingSafeEqual`, SHA-1 cache keys, random secrets |
| `node:fs` / `node:fs/promises` | KB loading, `fs.watch` hot reload, file writes |
| `node:path` | Paths |
| `node:util` | `promisify` for scrypt |
| `node:test`, `node:assert` | Test runner |
| `node:child_process` | `scripts/check-syntax.js` (`node --check`) |

Outside the core, the services also use `node:http` (worker health server) and `node:events` (`EventEmitter` for sessions and the WhatsApp state machine).

### Container images

`caddy:2.10-alpine`, `node:22-bookworm-slim` (app base, plus Debian `chromium`, `fonts-noto-core`, `fonts-noto-color-emoji`, `dumb-init`, `curl`), `ollama/ollama:0.12.6` (pinned; override with `OLLAMA_VERSION`), `postgres:17-alpine`, `redis:7.4-alpine`, `apache/kafka:3.9.1`.

### External services

| Service | Use |
|---|---|
| GitHub Actions | CI (lint, tests, boot smoke test, image build) and deploys over SSH |
| GitHub Container Registry (`ghcr.io`) | Stores app images (`:<sha12>`, `:latest`) |
| Let's Encrypt (via Caddy) | Automatic TLS certificates |
| Ollama registry | Model downloads |
| GitHub Actions marketplace | `actions/checkout@v4`, `actions/setup-node@v4`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, `docker/build-push-action@v6`, `appleboy/scp-action@v0.1.7`, `appleboy/ssh-action@v1.2.0` |
