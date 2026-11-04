#!/usr/bin/env bash
# Zero-touch rollout on the KVM. Called by the Deploy workflow, or by hand:
#   APP_IMAGE=ghcr.io/<owner>/<repo>:<sha> bash deploy/deploy.sh
#   SYNC_KB=1 APP_IMAGE=... bash deploy/deploy.sh   # overwrite server KB with the one in the image
# Rolls back to the previous image automatically if the gateway is not healthy.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${APP_IMAGE:?APP_IMAGE is required}"
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

echo "→ pulling $APP_IMAGE"
docker pull "$APP_IMAGE"

# The knowledge base lives on the server (editable from the console). Seed it
# from the image on first deploy, or when SYNC_KB=1.
mkdir -p knowledge_base
if [ -z "$(ls -A knowledge_base)" ] || [ "${SYNC_KB:-0}" = "1" ]; then
  echo "→ seeding knowledge_base/ from the image"
  docker run --rm --user 0 --entrypoint sh -v "$PWD/knowledge_base:/out" "$APP_IMAGE" \
    -c 'cp -r /app/knowledge_base/. /out/ && chown -R 1000:1000 /out'
fi

set_image "$APP_IMAGE"
echo "→ starting stack"
docker compose up -d --remove-orphans

echo "→ waiting for gateway health"
for _ in $(seq 1 60); do
  if docker compose exec -T gateway curl -fsS http://127.0.0.1:3000/readyz >/dev/null 2>&1; then
    echo "✓ healthy: $APP_IMAGE"
    docker image prune -f >/dev/null || true
    exit 0
  fi
  sleep 3
done

echo "✗ gateway did not become healthy"
docker compose logs --tail=80 gateway worker || true
if [ -n "$previous" ] && [ "$previous" != "$APP_IMAGE" ]; then
  echo "↩ rolling back to $previous"
  set_image "$previous"
  docker compose up -d --remove-orphans
fi
exit 1
