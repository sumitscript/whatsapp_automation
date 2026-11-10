// Inbound policy + outbound delivery.
//
// Inbound:  log every message → commands (/hi starts an AI session, /bye ends
//           it, /reset, /help) → in "session" mode only chats with an open
//           session get replies → pause / owner takeover / rate limit →
//           debounce bursts into one turn → mark seen + typing → publish job.
// Outbound: bubbles stream back from the worker and are sent in order with
//           "typing…" kept alive; bubbles for a session that ended meanwhile
//           are dropped.
import { chatKeyPatterns, keys, withinRate } from '../core/redis.js';
import { deleteKeysMatching, KeyedQueue, normalizePhone, phoneFromWid, sleep, uid } from '../core/util.js';
import { renderTemplate } from '../core/settings-schema.js';
import { helpText, parseCommand } from '../brain/commands.js';
import { TOPICS } from '../bus/index.js';
import { formatWhen } from '../brain/agenda.js';
import { replyLanguage } from '../brain/lang.js';
import { confirmationText, declineText, invitationData, invitationHtml } from '../brain/invite.js';

const TEXT_TYPES = new Set(['chat', 'text']);
const TYPING_REFRESH_MS = 8_000; // the contact's phone drops "typing…" after ~10–25s, so keep re-sending it
const TYPING_MAX_MS = 180_000;
const ERROR_NOTICE_COOLDOWN_MS = 10 * 60_000;
const UNSUPPORTED_COOLDOWN_MS = 60_000;
// Message types a person sends on purpose but that we can't read (system notifications are ignored silently).
const MEDIA_KINDS = {
  ptt: 'voice notes',
  audio: 'audio messages',
  image: 'photos',
  video: 'videos',
  sticker: 'stickers',
  document: 'documents',
  location: 'locations',
  vcard: 'contact cards',
  multi_vcard: 'contact cards',
  poll_creation: 'polls',
};

export class Inbox {
  constructor({ wa, repo, redis, bus, settings, memory, sessions, hub, logger }) {
    Object.assign(this, { wa, repo, redis, bus, settings, memory, sessions, hub });
    this.log = logger.child({ component: 'inbox' });
    this.pending = new Map(); // chatId → debounce buffer
    this.recentOut = new Map(); // chatId → last sent bubbles (duplicate guard)
    this.jobs = new Map(); // jobId → job
    this.typing = new Map(); // chatId → { refs, interval, timeout }
    this.lanes = new KeyedQueue(); // per-chat ordered delivery
    this.echo = new Map(); // `${chatId}|${body}` → count of messages we are sending (to ignore our own echoes)
    this.lastErrorNotice = new Map();
    this.lastUnsupported = new Map();

    // Session side effects live here so command, console and idle-timeout endings behave the same.
    sessions.on('start', ({ chatId }) => this.#emitChat(chatId));
    sessions.on('end', async ({ chatId, reason }) => {
      this.#cancelPending(chatId);
      this.#typingStop(chatId, true);
      await this.bus.publish(TOPICS.inbound, chatId, { type: 'summarize', chatId, reason, jobId: uid() }).catch((err) => this.log.warn('summarize job failed to publish', { err }));
      this.#emitChat(chatId);
      if (reason === 'timeout') this.#timeoutNotice(chatId).catch((err) => this.log.debug('timeout notice not sent', { chatId, err: err.message }));
    });
  }

  // ------------------------------------------------------------------ inbound

  async handleIncoming(msg) {
    if (msg.isStatus || msg.from === 'status@broadcast' || msg.broadcast) return;
    if (msg.fromMe) return; // our own alerts to the owner's number, or the owner texting themselves
    const chatId = msg.from;
    const isGroup = chatId.endsWith('@g.us');
    const waContact = await msg.getContact().catch(() => null);
    let name = waContact?.pushname || waContact?.name || waContact?.shortName || null;
    if (isGroup) name = (await msg.getChat().catch(() => null))?.name || name;
    const phone = isGroup ? null : normalizePhone(waContact?.number) || phoneFromWid(chatId);

    const contact = await this.repo.upsertContactForChat({ chatId, phone: phone || null, name, isGroup });
    const type = msg.type || 'chat';
    const body = msg.body || '';
    const saved = await this.repo.insertMessage({
      waMessageId: msg.id?._serialized,
      chatId,
      direction: 'in',
      body,
      type,
      sentBy: isGroup ? msg.author || null : null,
    });
    if (!saved) return; // duplicate delivery
    this.#broadcastMessage(saved, chatId);

    const s = this.settings.get();
    const skip = (reason) => this.log.debug('Not replying', { chatId, reason });
    if (s.replyMode === 'off') return skip('reply mode off');
    if (contact.isBlocked) return skip('blocked');
    if (isGroup) {
      if (!s.replyInGroups) return skip('groups disabled');
      const myId = this.wa.me?.wid;
      if (!myId || !(msg.mentionedIds || []).some((id) => String(id?._serialized ?? id) === myId)) return skip('group: not mentioned');
    }
    const unsupported = !TEXT_TYPES.has(type) && !body.trim();
    if (unsupported && !MEDIA_KINDS[type]) return skip(`unsupported type ${type}`);
    if (!unsupported && !body.trim()) return skip('empty');

    const cmd = parseCommand(body, s);
    // Count only messages we might answer: commands, or anything inside a session.
    if ((cmd || s.replyMode !== 'session' || (await this.sessions.active(chatId))) && !(await withinRate(this.redis, 'chat', chatId, s.rateLimitPerMin))) {
      this.log.warn('Rate limit hit; not replying', { chatId });
      return;
    }
    const canSpeak = !contact.aiPaused && !(await this.redis.exists(keys.takeover(chatId)));

    // --- session commands (state changes even when the AI is muted) ---
    if (cmd?.type === 'start') return this.#onStart(contact, cmd.rest, msg, canSpeak);
    if (cmd?.type === 'stop') return this.#onStop(contact, canSpeak);

    if (s.replyMode === 'session') {
      const session = await this.sessions.active(chatId);
      if (!session) return skip('no active session');
      await this.sessions.touch(chatId);
    }
    if (!canSpeak) return skip(contact.aiPaused ? 'ai paused for chat' : 'owner takeover active');

    if (unsupported) return this.#unsupportedNotice(contact, type);

    if (cmd?.type === 'reset') {
      await this.memory.clear(chatId);
      return this.#sendSystem(chatId, 'Fresh start ✨ — I forgot this conversation. What’s up?', 'command');
    }
    if (cmd?.type === 'help') return this.#sendSystem(chatId, helpText(s), 'command');

    this.#enqueue(contact, body, msg.id?._serialized, isGroup);
  }

  async #onStart(contact, rest, msg, canSpeak) {
    const s = this.settings.get();
    if (s.replyMode === 'session') await this.sessions.start(contact.chatId, { by: 'command' });
    if (!canSpeak) return;
    await this.#sendSystem(contact.chatId, renderTemplate(s.welcomeMessage, s, { name: contact.name }), 'welcome');
    if (rest) this.#enqueue(contact, rest, msg.id?._serialized, contact.isGroup); // "/hi what do you do?"
  }

  async #onStop(contact, canSpeak) {
    const s = this.settings.get();
    const wasActive = s.replyMode !== 'session' || !!(await this.sessions.active(contact.chatId));
    if (s.replyMode === 'session') await this.sessions.end(contact.chatId, 'command');
    if (canSpeak && wasActive) await this.#sendSystem(contact.chatId, renderTemplate(s.goodbyeMessage, s, { name: contact.name }), 'goodbye');
  }

  /** Start/end a session from the console (optionally telling the contact). */
  async consoleSession(chatId, action, { notify = false } = {}) {
    const s = this.settings.get();
    const contact = await this.repo.getContactByChatId(chatId);
    if (!contact) return null;
    if (action === 'start') {
      await this.sessions.start(chatId, { by: 'console' });
      if (notify) await this.#sendSystem(chatId, renderTemplate(s.welcomeMessage, s, { name: contact.name }), 'welcome');
    } else {
      const ended = await this.sessions.end(chatId, 'console');
      if (ended && notify) await this.#sendSystem(chatId, renderTemplate(s.goodbyeMessage, s, { name: contact.name }), 'goodbye');
    }
    return this.repo.getChat(chatId);
  }

  /** Voice note / photo / sticker: say what we can't read instead of staying silent. */
  async #unsupportedNotice(contact, type) {
    const chatId = contact.chatId;
    if (Date.now() - (this.lastUnsupported.get(chatId) || 0) < UNSUPPORTED_COOLDOWN_MS) return;
    this.lastUnsupported.set(chatId, Date.now());
    const s = this.settings.get();
    await this.#sendSystem(chatId, renderTemplate(s.unsupportedMessage, s, { name: contact.name, kind: MEDIA_KINDS[type] }), 'unsupported');
  }

  /** The session ended because the person went quiet: tell them, with a way back in. */
  async #timeoutNotice(chatId) {
    const s = this.settings.get();
    if (!s.notifyOnTimeout || s.replyMode !== 'session') return;
    const contact = await this.repo.getContactByChatId(chatId);
    if (!contact || contact.isGroup || contact.isBlocked || contact.aiPaused) return;
    if (await this.redis.exists(keys.takeover(chatId))) return; // the owner is talking to them
    await this.#sendSystem(chatId, renderTemplate(s.timeoutMessage, s, { name: contact.name }), 'timeout');
  }

  /** Message the owner on their own WhatsApp number (alerts). Returns false when WhatsApp is not connected. */
  async notifyOwner(text) {
    const me = this.wa.me?.wid;
    if (!me || !this.wa.ready) return false;
    try {
      await this.#send(me, text);
      return true;
    } catch (err) {
      this.log.warn('Could not message the owner', { err });
      return false;
    }
  }

  /** The owner confirmed or declined an appointment in the console and chose to tell the contact. */
  async notifyAgendaDecision(item, status) {
    if (!item.chatId || !['confirmed', 'declined'].includes(status)) return false;
    const s = this.settings.get();
    // Same language the contact wrote the request in (Hindi, Gujarati, Marathi, Hinglish; English otherwise).
    const lang = replyLanguage(item.sourceMessage || item.title || '', [], null);
    const p = { code: lang.code, name: item.contactName, owner: s.ownerName, title: item.title, dueAt: item.dueAt, allDay: item.allDay, place: item.location, timeZone: s.timezone };
    if (status === 'declined') {
      await this.#sendSystem(item.chatId, declineText(p), 'agenda');
      return true;
    }
    const text = confirmationText(p);
    // A freshly generated invitation card goes with the confirmation; if it cannot be drawn the text still goes out.
    if (s.agendaCard && item.kind === 'appointment') {
      try {
        const png = await this.renderInvitation(item, lang.code);
        await this.#sendImageSystem(item.chatId, png, text, 'agenda');
        return true;
      } catch (err) {
        this.log.warn('Invitation card failed; sending the confirmation as text', { id: item.id, err });
      }
    }
    await this.#sendSystem(item.chatId, text, 'agenda');
    return true;
  }

  /**
   * "Log out and delete data": forget every chat this server holds. Postgres (contacts + AI memory notes,
   * messages, sessions, agenda), the Redis short-term memory / takeover / rate keys, and this process's
   * in-memory per-chat state. Users, settings, the audit log and the knowledge base are kept.
   */
  async wipeChatData() {
    for (const buf of this.pending.values()) clearTimeout(buf.timer);
    for (const chatId of [...this.typing.keys()]) this.#typingStop(chatId, true);
    for (const m of [this.pending, this.recentOut, this.jobs, this.echo, this.lastErrorNotice, this.lastUnsupported]) m.clear();
    this.sessions.reset();
    const counts = await this.repo.wipeChatData();
    let redisKeys = 0;
    for (const pattern of chatKeyPatterns) redisKeys += await deleteKeysMatching(this.redis, pattern);
    this.hub.emit('chats:wiped', {});
    this.log.warn('All chat data wiped', { ...counts, redisKeys });
    return { ...counts, redisKeys };
  }

  /** Draw the invitation card for an agenda item (PNG). */
  async renderInvitation(item, code = 'en') {
    return this.wa.renderHtml(invitationHtml(invitationData(item, this.settings.get(), { code })));
  }

  /** Send a card + caption to the contact (used by the console's "Send invitation" button too). */
  async sendInvitation(item) {
    if (!item.chatId) return false;
    const lang = replyLanguage(item.sourceMessage || item.title || '', [], null);
    const s = this.settings.get();
    const text = confirmationText({ code: lang.code, name: item.contactName, owner: s.ownerName, title: item.title, dueAt: item.dueAt, allDay: item.allDay, place: item.location, timeZone: s.timezone });
    const png = await this.renderInvitation(item, lang.code);
    await this.#sendImageSystem(item.chatId, png, text, 'agenda');
    return true;
  }

  async #sendImageSystem(chatId, png, caption, intent) {
    const k = `${chatId}|${caption}`; // a media message's body is its caption: recognise the echo like any text
    this.echo.set(k, (this.echo.get(k) || 0) + 1);
    setTimeout(() => this.#forgetEcho(k), 30_000).unref();
    const sent = await this.wa.sendImage(chatId, png, { caption });
    const saved = await this.repo.insertMessage({ waMessageId: sent?.id?._serialized, chatId, direction: 'out', body: caption, type: 'image', isAi: true, sentBy: 'ai', intent, latencyMs: 0 });
    if (saved) this.#broadcastMessage(saved, chatId);
    await this.memory.append(chatId, [{ role: 'assistant', content: caption }]).catch(() => {});
  }

  async #onAgenda(ev) {
    this.hub.emit('agenda:update', { id: ev.itemId, created: !!ev.created });
    const s = this.settings.get();
    if (!ev.created || !s.notifyOwnerOnAgenda) return;
    const item = await this.repo.getAgendaItem(ev.itemId);
    if (!item) return;
    const who = item.contactName || item.phone || 'a contact';
    const when = item.dueAt ? ` — ${formatWhen(item.dueAt, s.timezone, item.allDay)}` : '';
    const head = item.kind === 'appointment' ? '📅 Meeting request' : '📝 New to-do';
    await this.notifyOwner(`${head} from ${who}: ${item.title}${when}\nReview it in the console → Agenda.`);
  }

  #enqueue(contact, body, messageId, isGroup) {
    const chatId = contact.chatId;
    const buf = this.pending.get(chatId) || { texts: [], lastMessageId: null, timer: null, isGroup, name: contact.name };
    buf.texts.push(body.trim());
    buf.lastMessageId = messageId;
    buf.lastAt = Date.now();
    clearTimeout(buf.timer);
    this.pending.set(chatId, buf);
    buf.timer = setTimeout(() => this.#flush(chatId).catch((err) => this.log.error('Flush failed', { chatId, err })), this.settings.get().debounceMs);
  }

  #cancelPending(chatId) {
    const buf = this.pending.get(chatId);
    if (buf) clearTimeout(buf.timer);
    this.pending.delete(chatId);
  }

  async #flush(chatId) {
    const buf = this.pending.get(chatId);
    if (!buf) return;
    this.pending.delete(chatId);
    const job = {
      type: 'reply',
      jobId: uid(),
      chatId,
      text: buf.texts.join('\n'),
      contactName: buf.name || null,
      receivedAt: buf.lastAt,
      quotedMessageId: buf.isGroup ? buf.lastMessageId : null,
    };
    this.jobs.set(job.jobId, job);
    setTimeout(() => this.jobs.delete(job.jobId), TYPING_MAX_MS).unref();
    if (this.settings.get().markSeen) this.wa.markSeen(chatId).catch(() => {});
    this.#typingStart(chatId);
    try {
      await this.bus.publish(TOPICS.inbound, chatId, job);
    } catch (err) {
      // The job never reached the worker: don't leave "typing…" on for three minutes, and tell the contact.
      this.jobs.delete(job.jobId);
      this.#typingStop(chatId);
      this.log.error('Could not queue the reply job', { chatId, err });
      await this.#errorNotice(chatId).catch(() => {});
      return;
    }
    this.log.info('Reply job queued', { chatId, jobId: job.jobId, parts: buf.texts.length });
  }

  // ----------------------------------------------------------------- outbound

  handleOutbound(ev) {
    if (!ev?.chatId) return;
    if (ev.type === 'memory') return this.#emitChat(ev.chatId);
    if (ev.type === 'agenda') return this.#onAgenda(ev).catch((err) => this.log.warn('agenda event failed', { err }));
    return this.lanes.push(ev.chatId, () => this.#deliver(ev)).catch((err) => this.log.error('Delivery failed', { chatId: ev.chatId, err }));
  }

  async #deliver(ev) {
    const job = this.jobs.get(ev.jobId) || { jobId: ev.jobId, chatId: ev.chatId, receivedAt: null, quotedMessageId: null };
    const { chatId } = ev;

    if (ev.type === 'done' || ev.type === 'error') {
      this.jobs.delete(ev.jobId);
      this.#typingStop(chatId);
      if (ev.type === 'error' && !ev.silent) await this.#errorNotice(chatId);
      return;
    }
    if (ev.type !== 'chunk' || !ev.text) return;

    // Session ended (/bye, console, timeout) or the owner jumped in while we were generating.
    if (this.settings.get().replyMode === 'session' && !(await this.sessions.active(chatId))) return;
    if (await this.redis.exists(keys.takeover(chatId))) {
      this.#typingStop(chatId, true);
      return;
    }

    // Safety net: never send the same sentence twice within 3 minutes (a repeated model loop or a
    // redelivered event). Very short bubbles ("ok", an emoji) may legitimately repeat.
    const key = ev.text.trim().toLowerCase();
    if (key.length >= 12) {
      const recent = (this.recentOut.get(chatId) || []).filter((r) => Date.now() - r.at < 180_000);
      if (recent.some((r) => r.key === key)) {
        this.log.warn('Dropped duplicate bubble', { chatId, jobId: ev.jobId });
        return;
      }
      recent.push({ key, at: Date.now() });
      this.recentOut.set(chatId, recent.slice(-8));
    }

    if (this.settings.get().humanize && job.lastSentAt) {
      const gap = Math.min(1200, 250 + ev.text.length * 6);
      const since = Date.now() - job.lastSentAt;
      if (since < gap) await sleep(gap - since);
    }

    const sent = await this.#send(chatId, ev.text, { quotedMessageId: ev.index === 0 ? job.quotedMessageId : null });
    job.lastSentAt = Date.now();
    const meta = ev.meta || {};
    const saved = await this.repo.insertMessage({
      waMessageId: sent?.id?._serialized,
      chatId,
      direction: 'out',
      body: ev.text,
      isAi: true,
      sentBy: 'ai',
      model: meta.model,
      intent: meta.intent,
      language: meta.language,
      latencyMs: job.receivedAt ? Date.now() - job.receivedAt : null,
      cached: !!meta.cached,
      sources: meta.sources?.length ? meta.sources : null,
    });
    if (saved) this.#broadcastMessage(saved, chatId);
    // Sending a message clears "typing…"; re-arm it while more bubbles are coming.
    if (this.typing.has(chatId)) this.wa.setTyping(chatId, true).catch(() => {});
  }

  /** Instant templated message (welcome, goodbye, help) — no LLM round-trip. */
  async #sendSystem(chatId, text, intent) {
    const sent = await this.#send(chatId, text);
    const saved = await this.repo.insertMessage({ waMessageId: sent?.id?._serialized, chatId, direction: 'out', body: text, isAi: true, sentBy: 'ai', intent, latencyMs: 0 });
    if (saved) this.#broadcastMessage(saved, chatId);
    // Keep welcome/goodbye in the conversation so the model knows what "it" already said.
    await this.memory.append(chatId, [{ role: 'assistant', content: text }]).catch(() => {});
  }

  async #errorNotice(chatId) {
    const last = this.lastErrorNotice.get(chatId) || 0;
    if (Date.now() - last < ERROR_NOTICE_COOLDOWN_MS) return;
    this.lastErrorNotice.set(chatId, Date.now());
    const owner = this.settings.get().ownerName?.split(' ')[0] || 'the owner';
    const text = `Oops, my brain glitched for a second 😅 — try again in a bit, or the real ${owner} will get back to you.`;
    const sent = await this.#send(chatId, text).catch(() => null);
    if (sent) {
      const saved = await this.repo.insertMessage({ waMessageId: sent.id?._serialized, chatId, direction: 'out', body: text, isAi: true, sentBy: 'ai', intent: 'error' });
      if (saved) this.#broadcastMessage(saved, chatId);
    }
  }

  /** Send while remembering the body so the message_create echo is recognised as ours. */
  async #send(chatId, text, opts) {
    const k = `${chatId}|${text}`;
    this.echo.set(k, (this.echo.get(k) || 0) + 1);
    setTimeout(() => this.#forgetEcho(k), 30_000).unref();
    return this.wa.sendText(chatId, text, opts);
  }

  #forgetEcho(k) {
    const n = (this.echo.get(k) || 0) - 1;
    if (n > 0) this.echo.set(k, n);
    else this.echo.delete(k);
  }

  // ---------------------------------------------------- owner / console sends

  /** Message the owner typed on the phone (or another linked device). */
  async handleOwnMessage(msg) {
    const chatId = msg.to;
    if (!chatId || chatId === 'status@broadcast') return;
    const k = `${chatId}|${msg.body}`;
    if (this.echo.has(k)) {
      this.#forgetEcho(k);
      return; // our own AI/console message
    }
    if (!(await this.repo.getContactByChatId(chatId))) {
      await this.repo.upsertContactForChat({ chatId, phone: phoneFromWid(chatId), name: null, isGroup: chatId.endsWith('@g.us') });
    }
    const saved = await this.repo.insertMessage({ waMessageId: msg.id?._serialized, chatId, direction: 'out', body: msg.body || '', type: msg.type || 'chat', sentBy: 'phone' });
    if (!saved) return;
    this.#broadcastMessage(saved, chatId);
    const s = this.settings.get();
    if (msg.body) await this.memory.append(chatId, [{ role: 'assistant', content: msg.body }]).catch(() => {});
    if (s.takeoverSeconds > 0 && (s.replyMode !== 'session' || (await this.sessions.active(chatId)))) {
      await this.redis.set(keys.takeover(chatId), '1', 'EX', s.takeoverSeconds);
      this.#cancelPending(chatId);
      this.#typingStop(chatId, true);
      this.log.info('Owner replied from phone; AI paused for this chat', { chatId, seconds: s.takeoverSeconds });
    }
  }

  /** Manual message sent from the web console. */
  async sendManual(chatId, body, username) {
    const sent = await this.#send(chatId, body);
    const saved = await this.repo.insertMessage({ waMessageId: sent?.id?._serialized, chatId, direction: 'out', body, sentBy: `console:${username}` });
    await this.memory.append(chatId, [{ role: 'assistant', content: body }]).catch(() => {});
    if (saved) this.#broadcastMessage(saved, chatId);
    return saved;
  }

  // ------------------------------------------------------------------ typing

  #typingStart(chatId) {
    const t = this.typing.get(chatId);
    if (t) {
      t.refs++;
      return;
    }
    const send = () => this.wa.setTyping(chatId, true).catch(() => {});
    send();
    this.typing.set(chatId, {
      refs: 1,
      interval: setInterval(send, TYPING_REFRESH_MS),
      timeout: setTimeout(() => this.#typingStop(chatId, true), TYPING_MAX_MS),
    });
    this.hub.emit('typing', { chatId, on: true });
  }

  #typingStop(chatId, force = false) {
    const t = this.typing.get(chatId);
    if (!t) return;
    if (!force && --t.refs > 0) return;
    clearInterval(t.interval);
    clearTimeout(t.timeout);
    this.typing.delete(chatId);
    this.wa.setTyping(chatId, false).catch(() => {});
    this.hub.emit('typing', { chatId, on: false });
  }

  async #broadcastMessage(message, chatId) {
    try {
      this.hub.emit('message:new', { message, chat: await this.repo.getChat(chatId) });
    } catch (err) {
      this.log.warn('broadcast failed', { err });
    }
  }

  async #emitChat(chatId) {
    try {
      const chat = await this.repo.getChat(chatId);
      if (chat) this.hub.emit('chat:update', { chat });
    } catch (err) {
      this.log.warn('chat:update failed', { err });
    }
  }

  shutdown() {
    for (const chatId of [...this.typing.keys()]) this.#typingStop(chatId, true);
    for (const buf of this.pending.values()) clearTimeout(buf.timer);
  }
}
