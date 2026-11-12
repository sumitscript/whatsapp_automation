# Deployment: step by step

This guide goes from a fresh Ubuntu KVM (8 vCPU / 32 GB, CPU only) to eSumit answering on WhatsApp. Every command matches the scripts in `deploy/` and `.github/workflows/`.

## 0. What you need

- **A KVM** running Ubuntu 22.04/24.04 or Debian 12, with root SSH access and at least 40 GB of free disk (see [RESOURCES.md](RESOURCES.md)).
- **A domain** where you can create an A record, e.g. `wa.example.com`. Without one you can test over plain HTTP on the IP (see step 3).
- **A GitHub repository** containing this code. `main` is the deploy branch.
- **A phone** with the WhatsApp account eSumit will speak for.

## 1. Point DNS at the server

Create an **A record** `wa.example.com → <server IP>`. Caddy requests the Let's Encrypt certificate on first start, so DNS must resolve before step 7.

```bash
dig +short wa.example.com   # should print the server IP
```

## 2. Bootstrap the server (once, as root)

Copy `deploy/server-setup.sh` to the server, or fetch it from your repo's raw URL, and run it:

```bash
bash server-setup.sh deploy
```

It does the following:
- installs Docker (via get.docker.com) if it's missing;
- creates the `deploy` user in the `docker` group, with `~/.ssh/authorized_keys`;
- creates `/opt/whatsapp-ai/{deploy,knowledge_base,backups}` and gives `knowledge_base` to UID 1000 (the container's `node` user);
- sets setgid + default ACLs on `/opt/whatsapp-ai`, so files a root shell creates there stay writable by `deploy`;

**Root vs deploy ownership.** CI failing with `tar: …: Cannot open: Permission denied` means something on the server was created as root. Three layers prevent it: the Deploy workflow's *Heal file ownership* step re-owns anything not owned by the deploy user before uploading (using a throw-away `alpine` container, so no sudo is needed); `deploy/deploy.sh` re-runs itself as the directory owner if started as root; and `server-setup.sh` sets ACLs. When working on the server, use `sudo -iu deploy` rather than a root shell. `SSH_USER` must never be `root`.
- if `ufw` is present, allows only OpenSSH, 80/tcp, 443/tcp and 443/udp, then enables the firewall.

## 3. Create `/opt/whatsapp-ai/.env`

Every deploy copies the latest `.env.example` to the server for reference, but the first deploy needs a filled-in `.env` before it can start, so create it once by hand:

```bash
scp .env.example deploy@<server>:/opt/whatsapp-ai/.env
ssh deploy@<server>
cd /opt/whatsapp-ai && nano .env
```

**Required values:**

| Variable | Value |
|---|---|
| `DOMAIN` | `wa.example.com`. Use `:80` to serve plain HTTP on the IP (testing only) |
| `PUBLIC_ORIGIN` | `https://wa.example.com`. Must match exactly, or every save and button in the console fails with 403 |
| `JWT_SECRET` | `openssl rand -hex 32`. The gateway refuses to start in production without it |
| `POSTGRES_PASSWORD` | `openssl rand -hex 16`. Compose refuses to start without it |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | Your console login. Used only on first boot. If the password is empty, a one-time password is printed in the gateway log |

**Message bus.** Pick one pair, and keep the two lines consistent:

| Setup | `.env` |
|---|---|
| Redis Streams (default, saves ~1 GB RAM) | `BUS_DRIVER=redis`, no `COMPOSE_PROFILES` |
| Kafka | `COMPOSE_PROFILES=kafka` and `BUS_DRIVER=kafka` |

Moving an existing server from Kafka to Redis: edit `.env` on the server (`BUS_DRIVER=redis`, delete the `COMPOSE_PROFILES=kafka` line), then `docker compose up -d --remove-orphans`. Jobs in flight at that moment are dropped, which is harmless for a chat bot.

**Testing over plain HTTP** (`DOMAIN=:80`): also set `COOKIE_SECURE=false` and `PUBLIC_ORIGIN=http://<server IP>`. Otherwise the browser drops the login cookie.

**Tuning (optional):**
- `LLM_MODEL` (default `gemma3:4b`)
- `LLM_NUM_THREAD=8` on an 8 vCPU KVM
- `LLM_CONCURRENCY` (default 1: on CPU, two parallel generations each run ~3x slower, so a second user waits instead)
- `LLM_NUM_CTX` (default 4096)
- `KB_*` retrieval thresholds
- `LOG_LEVEL`

**Leave alone:**
- `DATABASE_URL`, `REDIS_URL`, `KAFKA_BROKERS` and `OLLAMA_URL` are local-dev values; compose overrides them inside containers.
- `APP_IMAGE` is managed by `deploy.sh`.

## 4. SSH key for GitHub Actions

On your machine, create a key pair for CI and authorize it for the `deploy` user:

```bash
ssh-keygen -t ed25519 -f deploy_key -N '' -C github-actions
ssh-copy-id -i deploy_key.pub deploy@<server>    # or append it to /home/deploy/.ssh/authorized_keys
```

## 5. GitHub repository settings

**Secrets:** Settings → Secrets and variables → Actions → **Repository secrets**.

| Secret | Value |
|---|---|
| `SSH_HOST` | Server IP or hostname |
| `SSH_USER` | `deploy` |
| `SSH_KEY` | Contents of the private key `deploy_key` |
| `SSH_PORT` | Optional, default 22 |
| `DEPLOY_PATH` | `/opt/whatsapp-ai` |
| `GHCR_TOKEN` | A personal access token with `read:packages`, which the server uses to pull the image |

**Environment:** Settings → Environments → create **`production`**. Add required reviewers there if you want each deploy to wait for manual approval.

**Package visibility:** none needed. The image is pushed to `ghcr.io/<owner>/<repo>` with the workflow's `GITHUB_TOKEN`, and the server logs in with `GHCR_TOKEN`.

## 6. Push to `main`

```bash
git push origin main
```

**CI** (`.github/workflows/ci.yml`) runs three jobs:
1. **Lint + unit tests:** `npm run lint`, `npm test`, and a knowledge-base indexing check.
2. **Boot smoke test:** starts the gateway against real Postgres 17 and Redis 7.4 (`BUS_DRIVER=redis`, `WA_AUTO_CONNECT=false`), then exercises login, the API, the 401 check and the static console.
3. **Docker build** plus `docker compose config` validation.

**Deploy** (`.github/workflows/deploy.yml`) runs after CI succeeds on `main`, or by hand from **Actions → Deploy → Run workflow**:
1. It builds and pushes `ghcr.io/<owner>/<repo>:<sha12>` and `:latest`.
2. It copies `docker-compose.yml`, `deploy/Caddyfile`, `deploy/deploy.sh` and `deploy/backup.sh` over SCP.
3. Over SSH, it logs in to GHCR and runs `APP_IMAGE=<image> bash deploy/deploy.sh`.

`deploy/deploy.sh` then:
1. pulls the image;
2. seeds `knowledge_base/` from the image if the folder is empty (or if `SYNC_KB=1`);
3. writes `APP_IMAGE` into `.env` and runs `docker compose up -d --remove-orphans`;
4. polls the gateway's `/readyz` for about 3 minutes;
5. rolls back to the previous `APP_IMAGE` if the gateway never becomes healthy.

## 7. First boot

- **Model download.** The worker checks Ollama for `LLM_MODEL`, pulls it if `LLM_AUTO_PULL=true`, and warms it up. `gemma3:4b` is about 3.3 GB, so expect a few minutes on the first boot depending on bandwidth. Follow it with:
  ```bash
  cd /opt/whatsapp-ai
  docker compose logs -f worker
  ```
- **TLS.** Caddy gets its certificate on the first HTTPS request. Check it with `docker compose logs caddy`.

## 8. Sign in and link WhatsApp

1. Open `https://wa.example.com` and sign in with `ADMIN_USERNAME` / `ADMIN_PASSWORD`. If you left the password empty, get the one-time password from `docker compose logs gateway | grep one-time`.
2. If the password was generated, change it in **Settings → Change password**.
3. Go to **Connection**. With `WA_AUTO_CONNECT=true` the QR appears by itself; otherwise click **Connect**.
4. On the phone, open WhatsApp → **Settings → Linked devices → Link a device** and scan the QR.
5. The state moves through `QR → AUTHENTICATED → READY`.
6. From another phone, send `/hi` to the linked number. You should get the welcome message, then answers. `/bye` ends the session.

## 9. Verify

```bash
cd /opt/whatsapp-ai
docker compose ps                                   # all services Up / healthy
docker compose exec -T gateway curl -fsS localhost:3000/healthz
docker compose exec -T gateway curl -fsS localhost:3000/readyz   # {"ok":true,"wa":"READY"}
docker compose exec -T worker  curl -fsS localhost:3001/healthz
docker compose logs -f --tail=100 gateway worker
```

The console's **Logs** view tails both services live. The **Dashboard** shows message counts, latency, cache hits and active sessions.

## 10. Day-2 operations

**Updating.** Push to `main`. To roll out a specific image by hand:

```bash
cd /opt/whatsapp-ai
APP_IMAGE=ghcr.io/<owner>/<repo>:<sha12> bash deploy/deploy.sh
```

**Rolling back.** Run the same command with an older tag. The previous image is also restored automatically when a deploy fails its health check.

**Changing the model.** Go to **Settings → Models** in the console. A click pulls the model, with progress shown live, and switches to it with no restart. To change the boot default, set `LLM_MODEL` in `.env` and redeploy. Compare speeds on the box with:

```bash
docker compose exec worker npm run bench -- gemma3:4b llama3.2:3b
```

**Knowledge base.** The server's `knowledge_base/` is the source of truth once it has been seeded:
- Edits made in the console (**Knowledge**, Ctrl+S) or on disk reload immediately.
- To overwrite the server copy with the version in git, redeploy with:
  ```bash
  SYNC_KB=1 APP_IMAGE=<image> bash deploy/deploy.sh
  ```

**Adding or resetting a console user:**

```bash
docker compose exec gateway npm run create-admin -- alice 'a-long-password'
```

**Backups** (`deploy/backup.sh`). It writes a `pg_dump` (custom format) and a KB tarball to `backups/` and keeps 14 days. Install it in cron:

```bash
crontab -e
30 3 * * * cd /opt/whatsapp-ai && bash deploy/backup.sh >> backups/backup.log 2>&1
```

**Restore:**

```bash
cd /opt/whatsapp-ai
docker compose stop gateway worker
docker compose exec -T postgres pg_restore -U wa -d wa --clean --if-exists < backups/db-YYYYMMDD-HHMMSS.dump
tar -xzf backups/kb-YYYYMMDD-HHMMSS.tar.gz        # restores knowledge_base/
docker compose start gateway worker
```

Use your `POSTGRES_USER` / `POSTGRES_DB` values if you changed them. The WhatsApp link lives in the `wa_session` volume and isn't part of these backups. If that volume is lost, re-scan the QR.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **No QR on Connection** | Check `docker compose logs gateway`. Chromium needs `shm_size: 1gb`, which compose already sets. After 8 QR codes (about 3 minutes) with no scan it stops on purpose: click **Connect** again |
| **"profile appears to be in use" / Chromium won't start** | Stale `Singleton*` locks are removed automatically on connect. If it persists, run `docker compose restart gateway` |
| **Linked, then logged out / `auth_failure`** | The session is wiped automatically. Click **Connect** and scan again. Avoid running two gateways for one account |
| **WhatsApp Web update broke the client** | Pin a known build with `WA_WEB_VERSION=` in `.env`, then redeploy |
| **Slow replies** | Use a smaller model (`llama3.2:3b`), set `LLM_NUM_THREAD=8`, lower `LLM_CONCURRENCY` to 1, or lower `maxTokens` in Settings. Measure with `npm run bench` |
| **Ollama OOM / killed** | The model plus KV cache must fit in the 20 GB `ollama` limit. Drop to a 4B model or reduce `LLM_NUM_CTX` and `LLM_CONCURRENCY` |
| **Worker waits forever on first boot** | The model pull is still running (`docker compose logs worker`), or there's no egress to `registry.ollama.ai` |
| **401 right after login** | The cookie was rejected: on plain HTTP set `COOKIE_SECURE=false`. A changed `JWT_SECRET` or password also signs everyone out (expected) |
| **403 "Cross-origin request blocked"** | `PUBLIC_ORIGIN` doesn't exactly match the URL in your browser (scheme, host, port) |
| **Caddy TLS errors** | DNS doesn't point at the server yet, or ports 80/443 are blocked by a cloud firewall. Check `docker compose logs caddy` |
| **Worker errors about Kafka** | `BUS_DRIVER=kafka` but `COMPOSE_PROFILES` doesn't include `kafka`, or the other way round. The default is Redis (`BUS_DRIVER=redis`, no profile) |
| **Deploy rolled back** | The script prints the last 80 lines of gateway and worker logs. Usually a missing `.env` value (`JWT_SECRET`, `POSTGRES_PASSWORD`) |

## Evaluating conversation quality

`npm run eval` replays the multi-turn conversations in `test/eval/cases.json` (real failures seen in production) against a **live** Ollama model through the real reply pipeline, then checks the answers (intent, KB use, script, forbidden phrases, length). The unit tests use a fake LLM, so this is the only check of actual model behaviour. It never pulls models and writes nothing except an optional `--json` file.

The image includes `scripts/`, `knowledge_base/` and `test/eval/`, so on the server run:

```bash
docker compose exec worker node scripts/eval.js --model gemma3:4b,llama3.2:3b
```

Run it from a checkout (with `npm run eval -- --model gemma3:4b`, `OLLAMA_URL` pointing at the server's Ollama) if you prefer. Options: `--only <id-substring>`, `--json out.json`, `--temperature N`. Models must already be installed. The exit code is 1 if any check fails. Run it before switching `LLM_MODEL` or after editing the persona, and note it competes for CPU with live traffic, so prefer a quiet moment.

## Groq (optional hosted LLM, several keys)

1. Create one or more API keys at console.groq.com (a separate key per account/project gives separate rate limits).
2. Either put them in the server `.env` (`GROQ_API_KEYS=gsk_a,gsk_b`, then `docker compose up -d`), or add them in the console under **Settings → AI provider → Groq API keys** (stored encrypted; the console verifies each key before saving).
3. In the same card pick the mode and a model and save (the model box lists what your key can use; the default is `openai/gpt-oss-120b`; which models exist depends on the Groq account). It applies to the next message.
   - **Auto** (default): Groq as soon as a key is added; the local model takes over when every key is used up or Groq is down.
   - **Local only**: the model on this server, Groq is never used.
   - **Groq only**: never the local model. When no key can answer, the contact gets the "short break" message you set in the same card (once per 5 minutes per chat).

Behaviour: keys are used in turn; when one reports a rate limit the next key serves the same request immediately; when every key is rate-limited or Groq is unreachable, Auto mode answers with the local Ollama model and Groq-only mode sends the short-break message. Keep the local model installed: it is the fallback and also does memory summaries and agenda extraction. Stored keys are encrypted with a key derived from `JWT_SECRET`; if you rotate `JWT_SECRET`, add the console keys again. The Settings card shows each key's live state (ready, rate-limited with a countdown, rejected) and how many calls each served.

## Answering agenda requests from your own WhatsApp

When someone asks you for a meeting or a favour, the bot sends a message to your own WhatsApp chat ("Message yourself") with the request and its number, e.g. `Meeting request #12 from Prachi`. Reply there:

- `ok 12`, `12 yes`, `confirm 12`, `haan 12`: confirm. The requester gets the confirmation message with an invitation card.
- `no 12`, `decline 12`: decline (the requester is told).
- `done 12`: mark done. `ok 3 5` or `ok all`: several at once. `pending`: list what is waiting. `help`.
- Replying to a request message with just `ok` or `no` also works; so does a bare `ok` when only one request is waiting.

Concurrency: `GROQ_CONCURRENCY` (default 32) caps parallel Groq calls; the real limit is each key's own rate limit, which the key rotation handles. `LLM_CONCURRENCY` is only for the local CPU model (keep 1 to 2).

## Invitation cards and the whatsapp-web.js media bug

whatsapp-web.js 1.34.7 cannot send any image (error `Data passed to getter must include an id property`), which broke the invitation cards. The image build applies the upstream one-line fix (`scripts/patch-wwebjs.js`, wwebjs issue #201922) after `npm ci`; the build log line `[patch-wwebjs] patched…` shows it ran (a `WARNING` line means the library changed and the patch no longer fits). To check on the server, Settings → or `POST /api/wa/test-image` sends a small "Card test" picture to your own number and reports which sending method worked. Remove the patch step once a whatsapp-web.js release newer than 1.34.7 contains the fix.

## Availability: no double-booking

Settings → Agenda & alerts → **Availability** sets your working hours, working days, meeting length and the gap kept around meetings (defaults: 09:00-21:00, every day, 60 min, 15 min gap). Confirmed meetings and the time you block (Agenda → Availability: lunch, travel, leave) are "busy"; everything else inside working hours is a free slot, calculated on demand.

- Someone asks for a taken, blocked or out-of-hours time: they get "not available" plus up to four free slots (they are not told why), and the request is not filed.
- Two people ask for the same free time: both are accepted as pending and you are told on WhatsApp that they overlap. Confirming the first makes the second a clash.
- Confirming a clashing request needs `force N` on WhatsApp ("confirm anyway" in the console). Declining one sends the person your free slots.
- On WhatsApp, `slots` lists your next free times and what is booked. Turn it all off with "Check availability" in Settings.
