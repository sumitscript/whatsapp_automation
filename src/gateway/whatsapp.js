// WhatsApp connection lifecycle (whatsapp-web.js + headless Chromium).
// State machine: DISCONNECTED → INITIALIZING → QR → AUTHENTICATED → READY.
// The QR is rendered to a PNG data URL and pushed to the web console.
import { EventEmitter } from 'node:events';
import fsp from 'node:fs/promises';
import path from 'node:path';
import QRCode from 'qrcode';
import wweb from 'whatsapp-web.js';
import { SeenIds } from '../core/util.js';
import { trySendStrategies } from '../core/image-send.js';

const { Client, LocalAuth, MessageMedia } = wweb;

const NO_RECONNECT = /LOGOUT|max qrcode retries|unpaired/i;
const READY_RESTART_COOLDOWN_MS = 60_000; // never restart more than once a minute because of a repeated 'ready'

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
    this.connecting = false; // true while connect() prepares the profile (before this.client exists)
    this.connectToken = 0;
    this.seenEvents = new SeenIds({ ttlMs: 10 * 60_000, max: 2000 });
    this.lastReadyRestartAt = 0;
    this.lastImageStrategy = null; // the image-send way that worked last (tried first next time)
    this.imageWarned = new Set(); // "strategy:error" already logged
    this.ownChatCache = new Map(); // chatId → is it the owner's own chat (Contact.isMe)
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
    // `connecting` is set before the first await: a console "Connect" and the reconnect timer used to both pass
    // the `this.client` check while the other was still preparing, launching two Chromiums on one profile.
    if (this.client || this.connecting) return this.status();
    this.connecting = true;
    const token = ++this.connectToken;
    let client;
    try {
      clearTimeout(this.reconnectTimer);
      this.manualStop = false;
      await fsp.mkdir(this.opts.sessionDir, { recursive: true });
      await this.#clearChromiumLocks();
      if (token !== this.connectToken) return this.status(); // disconnect/logout/restart came in while we were preparing
      client = this.#newClient();
      this.client = client;
    } finally {
      this.connecting = false;
    }
    this.#wire(client);
    return this.status();
  }

  #newClient() {
    return new Client({
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
  }

  /** Attach the event handlers to a freshly created client and start it. */
  #wire(client) {
    this.#set('INITIALIZING', { qr: null, lastError: null });
    let readyCount = 0;

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
      if (this.client !== client) return; // a client we already dropped
      // A second 'ready' on the same client, with no disconnect in between, means WhatsApp Web reloaded inside
      // the browser and whatsapp-web.js injected its listeners again: from then on every event fires twice
      // (duplicate messages, our own replies mistaken for the owner typing). Start over from the saved session.
      if (++readyCount > 1) return this.#onReadyAgain(client);
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
    client.on('message', (msg) => {
      if (this.#isDuplicate('message', msg)) return;
      this.emit('message', msg);
    });
    client.on('message_create', (msg) => {
      if (!msg.fromMe || this.#isDuplicate('own', msg)) return;
      this.emit('own_message', msg);
    });

    client.initialize().catch((err) => {
      if (this.client !== client) return; // superseded by disconnect/restart
      this.log.error('WhatsApp initialisation failed', { err });
      this.#onDisconnected(client, `init failed: ${err.message}`);
    });
  }

  /** The same WhatsApp message id delivered twice (re-injected listeners, a second page): handle it once. */
  #isDuplicate(kind, msg) {
    const id = msg?.id?._serialized;
    if (!id) return false;
    const dup = this.seenEvents.seen(`${kind}:${id}`);
    if (dup) this.log.debug('Dropped a duplicate WhatsApp event', { kind, id });
    return dup;
  }

  #onReadyAgain(client) {
    const now = Date.now();
    if (now - this.lastReadyRestartAt < READY_RESTART_COOLDOWN_MS) {
      this.log.warn('WhatsApp reported ready again too soon after a restart; not restarting again');
      return;
    }
    this.lastReadyRestartAt = now;
    this.log.warn('WhatsApp Web reloaded under us (second ready on the same client); restarting the client to drop duplicate listeners');
    this.restart().catch((err) => this.log.error('Restart after a repeated ready failed', { err }));
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
    this.connectToken++; // cancels a connect() that is still preparing
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
    this.connectToken++;
    this.ownChatCache.clear(); // another account may be linked next
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

  /** The ways to send one image, in the order they are tried (see core/image-send.js for why there is more than one). */
  #imageStrategies(chatId, media, caption) {
    const opts = caption ? { caption } : {};
    return [
      { name: 'direct', run: () => this.client.sendMessage(chatId, media, opts) },
      { name: 'document', run: () => this.client.sendMessage(chatId, media, { ...opts, sendMediaAsDocument: true }) },
      {
        name: 'chat',
        run: async () => {
          const chat = await this.client.getChatById(chatId);
          return chat.sendMessage(media, opts);
        },
      },
      {
        name: 'phone-jid',
        run: async () => {
          if (!String(chatId).endsWith('@lid')) throw new Error('not a LID chat');
          const number = await this.#contactNumber(chatId);
          if (!number) throw new Error('no phone number known for this contact');
          return this.client.sendMessage(`${number}@c.us`, media, opts);
        },
      },
    ];
  }

  /**
   * Send a PNG (a Buffer) with an optional caption. Tries the ways in #imageStrategies, the one that worked last
   * first; throws only when all of them fail.
   */
  async sendImage(chatId, png, { caption, filename = 'invitation.png' } = {}) {
    this.#requireReady();
    const media = new MessageMedia('image/png', Buffer.from(png).toString('base64'), filename);
    const { sent, strategy, results } = await trySendStrategies(this.#imageStrategies(chatId, media, caption), { preferred: this.lastImageStrategy });
    for (const r of results.filter((x) => !x.ok)) {
      const key = `${r.strategy}:${r.error}`;
      if (this.imageWarned.has(key)) continue; // once per way and error, not on every card
      this.imageWarned.add(key);
      this.log.warn('Image send way failed', { strategy: r.strategy, error: r.error });
    }
    if (!sent) throw new Error(`Every image send way failed (${results.map((r) => `${r.strategy}: ${r.error}`).join(' | ')})`);
    if (strategy !== this.lastImageStrategy) this.log.info('Image sent', { strategy, previously: this.lastImageStrategy });
    this.lastImageStrategy = strategy;
    return sent;
  }

  /**
   * Diagnostics: send `png` to `chatId` with each way in turn (or until one works) and report what happened,
   * without throwing. Returns [{ strategy, ok, error?, id? }].
   */
  async testImageStrategies(chatId, png, caption, { all = false } = {}) {
    this.#requireReady();
    const media = new MessageMedia('image/png', Buffer.from(png).toString('base64'), 'card-test.png');
    const { results } = await trySendStrategies(this.#imageStrategies(chatId, media, caption), { all });
    return results.map((r) => ({ strategy: r.strategy, ok: r.ok, ...(r.ok ? { id: r.message?.id?._serialized ?? null } : { error: r.error }) }));
  }

  async #contact(chatId) {
    try {
      return await this.client.getContactById(chatId);
    } catch {
      return null;
    }
  }

  /** Is this chat id the owner's own account (the "Message yourself" chat)? Cached per id. */
  async isOwnChat(chatId) {
    if (!this.ready || !chatId) return false;
    if (this.ownChatCache.has(chatId)) return this.ownChatCache.get(chatId);
    const c = await this.#contact(chatId);
    const v = !!c?.isMe;
    if (c) {
      this.ownChatCache.set(chatId, v);
      if (this.ownChatCache.size > 500) this.ownChatCache.delete(this.ownChatCache.keys().next().value);
    }
    return v;
  }

  /** The real phone number behind a LID chat, when WhatsApp tells us. */
  async #contactNumber(chatId) {
    const c = await this.#contact(chatId);
    const n = String(c?.number || '').replace(/\D/g, '');
    const lid = String(chatId).split('@')[0].split(':')[0];
    return /^\d{6,15}$/.test(n) && n !== lid ? n : null;
  }

  /**
   * Render an HTML document to a PNG with the Chromium WhatsApp already runs (a throw-away tab, no
   * JavaScript, no network). One render at a time so a burst of invitations cannot starve the session.
   */
  async renderHtml(html, { width = 1080, height = 1350, scale = 1 } = {}) {
    const browser = this.client?.pupPage?.browser?.();
    if (!browser) {
      const err = new Error('The WhatsApp browser is not running yet');
      err.status = 409;
      throw err;
    }
    const run = async () => {
      const page = await browser.newPage();
      try {
        await page.setJavaScriptEnabled(false);
        await page.setViewport({ width, height, deviceScaleFactor: scale });
        await page.setContent(html, { waitUntil: 'load', timeout: 15_000 });
        return Buffer.from(await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width, height } }));
      } finally {
        await page.close().catch(() => {});
      }
    };
    const next = (this.renderChain || Promise.resolve()).then(run, run);
    this.renderChain = next.catch(() => {});
    return next;
  }

  /**
   * Show/hide "typing…" (the three dots) on the contact's phone. Never throws:
   * callers fire-and-forget it, so failures are logged (throttled) instead of lost.
   */
  async setTyping(chatId, on) {
    if (!this.ready) return;
    try {
      // WhatsApp only shows chat states from an account that is "available".
      if (on && Date.now() - (this.lastPresenceAt || 0) > 60_000) {
        this.lastPresenceAt = Date.now();
        await this.client.sendPresenceAvailable().catch(() => {});
      }
      try {
        const chat = await this.client.getChatById(chatId);
        if (on) await chat.sendStateTyping();
        else await chat.clearState();
      } catch (err) {
        // Library helper broke (WhatsApp Web updates do this) — talk to the page directly.
        if (!this.client?.pupPage) throw err;
        await this.client.pupPage.evaluate((id, state) => window.WWebJS.sendChatstate(state, id), chatId, on ? 'typing' : 'stop');
      }
    } catch (err) {
      if (Date.now() - (this.lastTypingWarnAt || 0) > 60_000) {
        this.lastTypingWarnAt = Date.now();
        this.log.warn('Could not send typing state', { chatId, on, err });
      }
    }
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
