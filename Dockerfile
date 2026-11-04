# One image, two roles: gateway (default CMD) and worker (`node src/worker/index.js`).
FROM node:22-bookworm-slim

ENV NODE_ENV=production \
    PUPPETEER_SKIP_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium \
    NPM_CONFIG_UPDATE_NOTIFIER=false

# Chromium for whatsapp-web.js (system package = no 170 MB puppeteer download),
# fonts so emoji/Indic glyphs render, dumb-init for clean signal handling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends chromium fonts-noto-core fonts-noto-color-emoji dumb-init ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; else npm install --omit=dev --no-audit --no-fund; fi \
 && npm cache clean --force

COPY src ./src
COPY public ./public
COPY db ./db
COPY scripts ./scripts
COPY test/eval ./test/eval
COPY knowledge_base ./knowledge_base

RUN mkdir -p /app/.wwebjs_auth /app/.wwebjs_cache && chown -R node:node /app
USER node

EXPOSE 3000 3001
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-3000}/healthz" || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "src/gateway/index.js"]
