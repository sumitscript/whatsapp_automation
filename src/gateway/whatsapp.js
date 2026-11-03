// WhatsApp connection lifecycle (whatsapp-web.js + headless Chromium).
// State machine: DISCONNECTED → INITIALIZING → QR → AUTHENTICATED → READY.
// The QR is rendered to a PNG data URL and pushed to the web console.
import { EventEmitter } from 'node:events';
import fsp from 'node:fs/promises';
import path from 'node:path';
import QRCode from 'qrcode';
import wweb from 'whatsapp-web.js';

const { Client, LocalAuth } = wweb;

const NO_RECONNECT = /LOGOUT|max qrcode retries|unpaired/i;

export class WhatsAppManager extends EventEmitter {
  constructor({ sessionDir, clientId, chromiumPath, webVersion, logger }) {
    super();
    this.opts = { sessionDir, clientId, chromiumPath, webVersion };
    this.log = logger.child({ component: 'whatsapp' });
    this.client = null;
    this.state = 'DISCONNECTED';
    this.qr = null;
    this.me = null;
    this.lastError = null;
    this.since = new Date().toISOString();
    this.manualStop = false;
    this.reconnectAttempts = 0;
    this.reconnectTimer = null;
  }

  status() {
    return { state: this.state, me: this.me, qr: this.state === 'QR' ? this.qr : null, since: this.since, lastError: this.lastError };
  }

  get ready() {
    return this.state === 'READY' && this.client;
  }

  #set(state, extra = {}) {
    Object.assign(this, extra);
    if (state !== this.state) {
      this.state = state;
      this.since = new Date().toISOString();
      this.log.info('WhatsApp state', { state, lastError: this.lastError || undefined });
    }
    this.emit('status', this.status());
  }

  get #sessionPath() {
    return path.join(this.opts.sessionDir, `session-${this.opts.clientId}`);
  }

  /** Chromium leaves Singleton* locks behind after a crash/kill; they block the next launch. */
  async #clearChromiumLocks() {
    await Promise.all(
      ['SingletonLock', 'SingletonSocket', 'SingletonCookie'].map((f) => fsp.rm(path.join(this.#sessionPath, f), { force: true }).catch(() => {})),
    );
  }

  async connect() {
    if (this.client) return this.status();
    clearTimeout(this.reconnectTimer);
    this.manualStop = false;
    await fsp.mkdir(this.opts.sessionDir, { recursive: true });
    await this.#clearChromiumLocks();

    const client = new Client({
      authStrategy: new LocalAuth({ clientId: this.opts.clientId, dataPath: this.opts.sessionDir }),
      qrMaxRetries: 8, // ~3 minutes of QR codes, then stop instead of burning CPU forever
      takeoverOnConflict: true,
      takeoverTimeoutMs: 10_000,
      ...(this.opts.webVersion
        ? {
            webVersion: this.opts.webVersion,
            webVersionCache: {
              type: 'remote',
              remotePath: `https://raw.githubusercontent.com/wppconnect-team/wa-version/main/html/${this.opts.webVersion}.html`,
            },
          }
        : {}),
      puppeteer: {
        headless: true,
        executablePath: this.opts.chromiumPath,
        args: [
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--disable-dev-shm-usage',
          '--disable-gpu',
          '--disable-extensions',
          '--no-first-run',
          '--no-default-browser-check',
          '--disable-background-networking',
          '--mute-audio',
        ],
      },
    });
    this.client = client;
    this.#set('INITIALIZING', { qr: null, lastError: null });

    client.on('qr', async (qr) => {
      try {
        const dataUrl = await QRCode.toDataURL(qr, { margin: 1, width: 320, errorCorrectionLevel: 'M' });
        this.#set('QR', { qr: dataUrl });
      } catch (err) {
        this.log.error('QR render failed', { err });
      }
    });
    client.on('loading_screen', (percent, message) => this.log.debug('WhatsApp loading', { percent, message }));
    client.on('authenticated', () => this.#set('AUTHENTICATED', { qr: null }));
    client.on('auth_failure', async (msg) => {
      // Saved session is no longer valid: drop it and come back with a fresh QR.
      this.log.warn('WhatsApp auth failure; clearing saved session', { msg });
      await this.#onDisconnected(client, `auth failure: ${msg}`); // destroys the browser first
      await fsp.rm(this.#sessionPath, { recursive: true, force: true }).catch(() => {});
    });
    client.on('ready', () => {
      const info = client.info || {};
      this.reconnectAttempts = 0;
      this.#set('READY', {
        qr: null,
        lastError: null,
        me: { wid: info.wid?._serialized ?? null, pushname: info.pushname ?? null, phone: info.wid?.user ?? null },
      });
    });
    client.on('change_state', (s) => this.log.debug('WhatsApp connection state', { s }));
    client.on('disconnected', (reason) => this.#onDisconnected(client, reason));
    client.on('message', (msg) => this.emit('message', msg));
    client.on('message_create', (msg) => {
      if (msg.fromMe) this.emit('own_message', msg);
    });

    client.initialize().catch((err) => {
      if (this.client !== client) return; // superseded by disconnect/restart
      this.log.error('WhatsApp initialisation failed', { err });
      this.#onDisconnected(client, `init failed: ${err.message}`);
    });
    return this.status();
  }

  async #onDisconnected(client, reason) {
    if (this.client !== client) return;
    this.client = null;
    await client.destroy().catch(() => {});
    this.#set('DISCONNECTED', { qr: null, me: null, lastError: String(reason || 'disconnected') });
    if (this.manualStop || NO_RECONNECT.test(String(reason))) return;
    const delay = Math.min(5000 * 2 ** this.reconnectAttempts++, 120_000);
    this.log.warn('WhatsApp disconnected; reconnecting', { reason, delayMs: delay });
    this.reconnectTimer = setTimeout(() => this.connect().catch((err) => this.log.error('Reconnect failed', { err })), delay);
  }

  /** Stop the browser but keep the saved session (no QR needed next time). */
  async disconnect() {
    this.manualStop = true;
    clearTimeout(this.reconnectTimer);
    const client = this.client;
    this.client = null;
    if (client) await client.destroy().catch((err) => this.log.warn('destroy failed', { err }));
    this.#set('DISCONNECTED', { qr: null, me: null, lastError: null });
    return this.status();
  }

  /** Unlink this device from WhatsApp and wipe the local session. */
  async logout() {
    this.manualStop = true;
    clearTimeout(this.reconnectTimer);
    const client = this.client;
    this.client = null;
    if (client) {
      await client.logout().catch((err) => this.log.warn('logout failed (continuing)', { err }));
      await client.destroy().catch(() => {});
    }
    await fsp.rm(this.#sessionPath, { recursive: true, force: true }).catch(() => {});
    this.#set('DISCONNECTED', { qr: null, me: null, lastError: null });
    return this.status();
  }

  async restart() {
    await this.disconnect();
    return this.connect();
  }

  #requireReady() {
    if (!this.ready) {
      const err = new Error(`WhatsApp is not connected (state: ${this.state})`);
      err.status = 409;
      throw err;
    }
  }

  async sendText(chatId, text, { quotedMessageId } = {}) {
    this.#requireReady();
    return this.client.sendMessage(chatId, text, quotedMessageId ? { quotedMessageId } : {});
  }

  async setTyping(chatId, on) {
    if (!this.ready) return;
    const chat = await this.client.getChatById(chatId);
    if (on) await chat.sendStateTyping();
    else await chat.clearState();
  }

  async markSeen(chatId) {
    if (!this.ready) return;
    await this.client.sendSeen(chatId);
  }

  async shutdown() {
    // Keep the session on process stop so restarts don't need a new QR.
    this.manualStop = true;
    clearTimeout(this.reconnectTimer);
    const client = this.client;
    this.client = null;
    if (client) await client.destroy().catch(() => {});
  }
}
