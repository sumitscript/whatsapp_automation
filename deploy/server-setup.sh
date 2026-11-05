#!/usr/bin/env bash
# One-time KVM bootstrap (Ubuntu/Debian). Run as root:
#   curl -fsSL <raw url>/deploy/server-setup.sh | bash -s -- deploy
# Installs Docker, opens only SSH/HTTP/HTTPS, creates the deploy user and /opt/whatsapp-ai.
set -euo pipefail
DEPLOY_USER="${1:-deploy}"
APP_DIR=/opt/whatsapp-ai

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi

id "$DEPLOY_USER" >/dev/null 2>&1 || adduser --disabled-password --gecos '' "$DEPLOY_USER"
usermod -aG docker "$DEPLOY_USER"
mkdir -p "/home/$DEPLOY_USER/.ssh" && chmod 700 "/home/$DEPLOY_USER/.ssh"
touch "/home/$DEPLOY_USER/.ssh/authorized_keys" && chmod 600 "/home/$DEPLOY_USER/.ssh/authorized_keys"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh"

mkdir -p "$APP_DIR/deploy" "$APP_DIR/knowledge_base" "$APP_DIR/backups"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$APP_DIR"
chown -R 1000:1000 "$APP_DIR/knowledge_base" # container user "node"

# Make the root/deploy mix-up harmless: new files inherit the deploy group (setgid) and, with ACLs,
# stay writable by the deploy user even when root creates them. (CI also heals ownership on every run.)
chmod g+rwXs "$APP_DIR" "$APP_DIR/deploy" "$APP_DIR/backups"
if ! command -v setfacl >/dev/null; then apt-get install -y acl >/dev/null 2>&1 || true; fi
if command -v setfacl >/dev/null; then
  setfacl -R -m "u:$DEPLOY_USER:rwX" "$APP_DIR" 2>/dev/null || true
  setfacl -R -d -m "u:$DEPLOY_USER:rwX" "$APP_DIR" 2>/dev/null || true
fi
# Root shells should work as the deploy user here: use `sudo -iu $DEPLOY_USER` instead of copying files as root.

if command -v ufw >/dev/null; then
  ufw allow OpenSSH
  ufw allow 80/tcp
  ufw allow 443/tcp
  ufw allow 443/udp
  ufw --force enable
fi

cat <<EOF

Done. Next:
  1. Add the CI public key to /home/$DEPLOY_USER/.ssh/authorized_keys
  2. Create $APP_DIR/.env from .env.example (JWT_SECRET, POSTGRES_PASSWORD, ADMIN_PASSWORD, DOMAIN)
  3. Point your domain's A record at this server
  4. Set the GitHub secrets listed in .github/workflows/deploy.yml and push to main
EOF
