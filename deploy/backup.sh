#!/usr/bin/env bash
# Nightly Postgres + knowledge base backup, keeps 14 days.
# crontab -e →  30 3 * * * cd /opt/whatsapp-ai && bash deploy/backup.sh >> backups/backup.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

mkdir -p backups
stamp="$(date +%Y%m%d-%H%M%S)"
docker compose exec -T postgres pg_dump -U "${POSTGRES_USER:-wa}" -d "${POSTGRES_DB:-wa}" --format=custom > "backups/db-$stamp.dump"
tar -czf "backups/kb-$stamp.tar.gz" knowledge_base
find backups -type f \( -name 'db-*.dump' -o -name 'kb-*.tar.gz' \) -mtime +14 -delete
echo "$(date -Is) backup ok: $stamp"
