#!/usr/bin/env bash
# Zero-touch rollout on the KVM. Called by the Deploy workflow, or by hand:
#   APP_IMAGE=ghcr.io/<owner>/<repo>:<sha> bash deploy/deploy.sh
#   SYNC_KB=1 APP_IMAGE=... bash deploy/deploy.sh   # overwrite server KB with the one in the image
# Rolls back to the previous image automatically if the gateway is not healthy.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# Readable failures: every phase is timestamped, and any unexpected error names the line and command.
phase="init"
step() { phase="$*"; echo "[$(date -u +%H:%M:%S)] → $phase"; }
trap 'rc=$?; echo "✗ deploy failed in phase \"$phase\" at line $LINENO (exit $rc): $BASH_COMMAND" >&2' ERR

: "${APP_IMAGE:?APP_IMAGE is required}"

# Running this as root is what leaves root-owned files that later break CI deploys
# ("tar: Cannot open: Permission denied"). Run it as the deploy user instead.
if [ "$(id -u)" = 0 ] && [ "${ALLOW_ROOT:-0}" != 1 ]; then
  owner="$(stat -c %U .)"
  if [ "$owner" != root ] && command -v sudo >/dev/null; then
    echo "! running as root would create root-owned files; re-running as $owner"
    exec sudo -u "$owner" -E env APP_IMAGE="$APP_IMAGE" SYNC_KB="${SYNC_KB:-0}" bash "$0" "$@"
  fi
  echo "✗ refusing to run as root (set ALLOW_ROOT=1 to override)"; exit 1
fi
# Self-heal files someone else created (needs docker access, not sudo). knowledge_base/ is uid 1000's.
if [ -n "$(find . -path ./knowledge_base -prune -o ! -user "$(id -u)" -print -quit 2>/dev/null)" ]; then
  echo "! repairing file ownership"
  docker run --rm -v "$PWD:/w" alpine:3.20 sh -c "find /w -path /w/knowledge_base -prune -o ! -user $(id -u) -exec chown $(id -u):$(id -g) {} +" || echo "! could not repair ownership (continuing)"
fi
[ -f .env ] || { echo "Missing $(pwd)/.env — copy .env.example and fill it in"; exit 1; }

# BUS_DRIVER=kafka needs the kafka compose profile, or the apps can't reach a broker.
env_val() { grep -E "^$1=" .env | tail -n1 | cut -d= -f2- | tr -d "\"'" || true; }
bus="$(env_val BUS_DRIVER)"; bus="${bus:-redis}"
profiles="${COMPOSE_PROFILES:-$(env_val COMPOSE_PROFILES)}"
case ",$profiles," in *,kafka,*) has_kafka=1 ;; *) has_kafka=0 ;; esac
if [ "$bus" = kafka ] && [ "$has_kafka" = 0 ]; then
  echo "✗ BUS_DRIVER=kafka but COMPOSE_PROFILES does not include 'kafka'. Add COMPOSE_PROFILES=kafka or set BUS_DRIVER=redis in .env"
  exit 1
fi
if [ "$bus" = redis ] && [ "$has_kafka" = 1 ]; then
  echo "! BUS_DRIVER=redis but the kafka profile is on: Kafka runs unused (~1 GB RAM). Remove COMPOSE_PROFILES=kafka to save it."
fi

previous="$(grep -E '^APP_IMAGE=' .env | cut -d= -f2- || true)"
set_image() {
  if grep -qE '^APP_IMAGE=' .env; then sed -i "s|^APP_IMAGE=.*|APP_IMAGE=$1|" .env; else echo "APP_IMAGE=$1" >> .env; fi
}

step "pulling $APP_IMAGE"
docker pull "$APP_IMAGE" || { echo "✗ pull failed: wrong image tag (did the build job succeed?), GHCR login/token (read:packages), or no disk space ($(df -h . | awk 'NR==2{print $4" free"}'))"; exit 1; }

# The knowledge base lives on the server (editable from the console). Seed it
# from the image on first deploy, or when SYNC_KB=1.
mkdir -p knowledge_base
if [ -z "$(ls -A knowledge_base)" ] || [ "${SYNC_KB:-0}" = "1" ]; then
  echo "→ seeding knowledge_base/ from the image"
  docker run --rm --user 0 --entrypoint sh -v "$PWD/knowledge_base:/out" "$APP_IMAGE" \
    -c 'cp -r /app/knowledge_base/. /out/ && chown -R 1000:1000 /out'
fi

set_image "$APP_IMAGE"
step "starting stack (previous image: ${previous:-none})"
docker compose up -d --remove-orphans
docker compose ps

step "waiting for gateway health (up to 180 s)"
for i in $(seq 1 60); do
  if docker compose exec -T gateway curl -fsS http://127.0.0.1:3000/readyz >/dev/null 2>&1; then
    echo "✓ healthy after ~$((i * 3)) s: $APP_IMAGE"
    docker image prune -f >/dev/null || true
    exit 0
  fi
  # Surface a crash loop early instead of waiting out the full timeout.
  if [ $((i % 5)) -eq 0 ]; then echo "  … $((i * 3)) s: $(docker compose ps --format '{{.Service}}={{.State}}' | tr '\n' ' ')"; fi
  sleep 3
done

trap - ERR
echo "✗ gateway did not become healthy within 180 s"
echo "--- /readyz response ---"; docker compose exec -T gateway curl -sS http://127.0.0.1:3000/readyz 2>&1 | head -c 600 || true; echo
echo "--- container state ---"; docker compose ps -a || true
echo "--- gateway + worker logs (last 80) ---"
docker compose logs --tail=80 gateway worker || true
if [ -n "$previous" ] && [ "$previous" != "$APP_IMAGE" ]; then
  echo "↩ rolling back to $previous"
  set_image "$previous"
  docker compose up -d --remove-orphans
fi
exit 1
