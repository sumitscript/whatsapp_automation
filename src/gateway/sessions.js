// AI sessions: a person opens one with a start command (/hi) and closes it with
// a stop command (/bye); idle sessions end automatically. Postgres is the
// source of truth, this process keeps a hot cache (the gateway is the only writer).
import { EventEmitter } from 'node:events';

export class SessionManager extends EventEmitter {
  constructor({ repo, settings, logger }) {
    super();
    this.repo = repo;
    this.settings = settings;
    this.log = logger.child({ component: 'sessions' });
    this.cache = new Map(); // chatId → session | null
    this.sweeper = null;
  }

  #idleMs() {
    return (this.settings.get().sessionIdleMinutes || 0) * 60_000;
  }

  #expired(s) {
    const idle = this.#idleMs();
    return idle > 0 && Date.now() - new Date(s.lastActivityAt).getTime() > idle;
  }

  async #load(chatId) {
    if (this.cache.has(chatId)) return this.cache.get(chatId);
    const s = await this.repo.getOpenSession(chatId);
    this.cache.set(chatId, s);
    return s;
  }

  /** The open, non-expired session for a chat (ends it if it went idle). */
  async active(chatId) {
    const s = await this.#load(chatId);
    if (s && this.#expired(s)) {
      await this.end(chatId, 'timeout');
      return null;
    }
    return s;
  }

  async start(chatId, { by = 'command' } = {}) {
    const existing = await this.active(chatId);
    if (existing) {
      await this.touch(chatId);
      return { session: existing, created: false };
    }
    const session = await this.repo.startSession(chatId);
    this.cache.set(chatId, session);
    this.log.info('AI session started', { chatId, by });
    this.emit('start', { chatId, session, by });
    return { session, created: true };
  }

  async end(chatId, reason = 'command') {
    const ended = await this.repo.endSession(chatId, reason);
    this.cache.set(chatId, null);
    if (ended) {
      this.log.info('AI session ended', { chatId, reason, messages: ended.messageCount });
      this.emit('end', { chatId, session: ended, reason });
    }
    return ended;
  }

  /** Record activity (keeps the session alive). */
  async touch(chatId) {
    const s = this.cache.get(chatId);
    if (!s) return;
    s.lastActivityAt = new Date().toISOString();
    s.messageCount = (s.messageCount || 0) + 1;
    await this.repo.touchSession(s.id).catch((err) => this.log.warn('session touch failed', { err }));
  }

  startSweeper() {
    this.sweeper = setInterval(async () => {
      const minutes = this.settings.get().sessionIdleMinutes;
      if (!minutes) return;
      try {
        for (const chatId of await this.repo.idleSessionChats(minutes)) await this.end(chatId, 'timeout');
      } catch (err) {
        this.log.warn('session sweep failed', { err });
      }
    }, 60_000);
    this.sweeper.unref();
  }

  stop() {
    clearInterval(this.sweeper);
  }
}
