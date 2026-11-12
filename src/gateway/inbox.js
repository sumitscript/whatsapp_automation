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
import { ownerAsk, ownerDigest, ownerHelp, ownerNotice, ownerPendingList, ownerResultLine, ownerSlotsText, parseOwnerCommand, resolveDecision } from '../brain/owner-commands.js';
import { formatWhen } from '../brain/agenda.js';
import { clashLines, slotsText } from '../brain/slots.js';
import { replyLanguage } from '../brain/lang.js';
import { EchoTracker } from '../core/echo.js';
import { SelfChatTracker } from '../core/selfchat.js';
import { isSystemMessageType } from '../core/msgtypes.js';
import { confirmationText, declineText, invitationData, invitationHtml } from '../brain/invite.js';

const TEXT_TYPES = new Set(['chat', 'text']);
const TYPING_REFRESH_MS = 8_000; // the contact's phone drops "typing…" after ~10–25s, so keep re-sending it
const TYPING_MAX_MS = 180_000;
const ERROR_NOTICE_COOLDOWN_MS = 10 * 60_000;
const UNSUPPORTED_COOLDOWN_MS = 60_000;
const BUSY_NOTICE_COOLDOWN_MS = 5 * 60_000; // "on a short break" (Groq-only mode) at most once per chat per 5 minutes
// A 1x1 PNG: the fallback picture for the image-send test when the card cannot be drawn.
const TINY_PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==', 'base64');
const TEST_CARD_HTML = '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;width:1080px;height:1350px;background:#0f9d8a;color:#fff;font:700 110px sans-serif;display:flex;align-items:center;justify-content:center">Card test ✓</body></html>';
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
  constructor({ wa, repo, redis, bus, settings, memory, sessions, hub, availability = null, logger }) {
    Object.assign(this, { wa, repo, redis, bus, settings, memory, sessions, hub, availability });
    this.log = logger.child({ component: 'inbox' });
    this.pending = new Map(); // chatId → debounce buffer
    this.recentOut = new Map(); // chatId → last sent bubbles (duplicate guard)
    this.jobs = new Map(); // jobId → job
    this.typing = new Map(); // chatId → { refs, interval, timeout }
    this.lanes = new KeyedQueue(); // per-chat ordered delivery
    this.echo = new EchoTracker(); // ignores the message_create echoes of what we send ourselves (by message id, then by text)
    this.lastErrorNotice = new Map();
    this.lastUnsupported = new Map();
    this.lastBusyNotice = new Map();
    this.noticeTimes = []; // when the owner was last pinged about a request (burst control)
    this.digest = []; // agenda ids waiting to be sent as one digest
    this.digestTimer = null;
    this.selfChat = new SelfChatTracker(); // finds the owner's own "Message yourself" chat (a LID, not the phone id)
    this.selfChatsLoaded = null;

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
    if (isSystemMessageType(msg.type)) return; // encryption/group/call notices are not messages: no row, no contact
    if (msg.fromMe) {
      // our own alerts to the owner's number, or the owner texting themselves ("ok 12" answers an agenda request)
      if (await this.#isSelfChat(msg).catch(() => false)) await this.#ownerCommand(msg).catch((err) => this.log.warn('Owner command failed', { err }));
      return;
    }
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
    return !!(await this.#sendToOwner(text));
  }

  /** Send to the owner's own chat; returns the sent message (or null when WhatsApp is not connected / sending failed). */
  async #sendToOwner(text) {
    const me = this.wa.me?.wid;
    if (!me || !this.wa.ready) return null;
    try {
      return await this.#sendToSelf(me, text);
    } catch (err) {
      this.log.warn('Could not message the owner', { err });
      return null;
    }
  }

  /**
   * Send into the owner's own chat. The echo of this message arrives addressed to the chat's LID, not to `chatId`, and can
   * beat sendMessage: remember the text and the id so the echo is recognised as ours (and teaches us which chat it is).
   */
  async #sendToSelf(chatId, text) {
    this.selfChat.rememberNotice(text);
    const sent = await this.#send(chatId, text);
    this.selfChat.rememberNotice(text, sent?.id?._serialized);
    return sent;
  }

  /**
   * The owner confirmed or declined an agenda item (console or their own WhatsApp) and the contact is told.
   * Confirming sends a localized confirmation with a freshly drawn invitation card, for meetings and to-dos alike;
   * if the card cannot be drawn or sent, the confirmation still goes out as text.
   * @returns {Promise<{sent: boolean, card: boolean}>}
   */
  async notifyAgendaDecision(item, status, { rescheduled = false } = {}) {
    if (!item.chatId || !['confirmed', 'declined'].includes(status)) return { sent: false, card: false };
    const s = this.settings.get();
    // Same language the contact wrote the request in (Hindi, Gujarati, Marathi, Hinglish; English otherwise).
    const lang = replyLanguage(item.sourceMessage || item.title || '', [], null);
    const p = { code: lang.code, name: item.contactName, owner: s.ownerName, title: item.title, dueAt: item.dueAt, allDay: item.allDay, place: item.location, timeZone: s.timezone };
    if (status === 'declined') {
      // A declined meeting offers the owner's free times, so the conversation can continue instead of ending.
      let slots = '';
      if (this.availability && item.kind === 'appointment') {
        const free = await this.availability.freeSlots({ count: 4, perDay: 2 }).catch(() => []);
        slots = slotsText(free, s.timezone, ['hi', 'mr', 'gu'].includes(lang.code) ? lang.code : 'en');
      }
      await this.#sendSystem(item.chatId, declineText({ ...p, slots }), 'agenda');
      return { sent: true, card: false };
    }
    // Confirmed before with another time/place? Then this message says "updated", not "confirmed" a second time.
    const told = `${item.dueAt ? new Date(item.dueAt).toISOString() : ''}|${item.location || ''}`;
    const before = await this.redis.get(keys.agendaTold(item.id)).catch(() => null);
    const text = confirmationText({ ...p, rescheduled: rescheduled || (!!before && before !== told) });
    await this.redis.set(keys.agendaTold(item.id), told, 'EX', 30 * 86_400).catch(() => {});
    if (s.agendaCard) {
      try {
        const png = await this.renderInvitation(item, lang.code);
        await this.#sendImageSystem(item.chatId, png, text, 'agenda');
        return { sent: true, card: true };
      } catch (err) {
        this.log.warn('Invitation card failed; sending the confirmation as text', { id: item.id, err });
      }
    }
    await this.#sendSystem(item.chatId, text, 'agenda');
    return { sent: true, card: false };
  }

  /**
   * Apply an owner decision to an agenda item: set the status, tell the contact (confirm/decline), refresh the console.
   * Used by the owner's WhatsApp replies; the console's PATCH does the same steps with its own field edits.
   */
  async decideAgenda(item, action, { force = false } = {}) {
    const status = { confirm: 'confirmed', decline: 'declined', done: 'done' }[action];
    if (!status) return { item, action, error: 'unknown action' };
    if (item.status === status) return { item, action, skipped: true };
    // Never double-book by accident: a confirmation that clashes (a confirmed meeting, a block, outside hours) waits for "force".
    if (action === 'confirm' && !force && this.availability && item.kind === 'appointment' && item.dueAt && !item.allDay) {
      const av = await this.availability.assess(item.dueAt, { ignoreId: item.id }).catch(() => null);
      if (av && !av.ok) {
        const tz = this.settings.get().timezone;
        const free = await this.availability.freeSlots({ count: 4, perDay: 2 }).catch(() => []);
        return { item, action, clash: clashLines(av, { timeZone: tz, formatWhen }), freeText: slotsText(free, tz) };
      }
    }
    const updated = await this.repo.updateAgendaItem(item.id, { status });
    let r = { sent: false, card: false };
    if (action !== 'done' && this.settings.get().agendaConfirm) {
      r = await this.notifyAgendaDecision(updated, status).catch((err) => {
        this.log.warn('Could not tell the contact about the agenda decision', { id: item.id, err });
        return { sent: false, card: false };
      });
    }
    this.hub.emit('agenda:update', { id: item.id });
    return { item: updated, action, notified: r.sent, card: r.card };
  }

  // -------------------------------------------------- the owner answers from their own WhatsApp

  async #loadSelfChats() {
    if (!this.selfChatsLoaded) {
      this.selfChatsLoaded = this.redis
        .smembers(keys.selfChats)
        .then((ids) => this.selfChat.load(ids))
        .catch((err) => {
          this.log.warn('Could not load the owner chat ids', { err });
          this.selfChatsLoaded = null; // try again next time
        });
    }
    await this.selfChatsLoaded;
  }

  async #learnSelfChat(chatId) {
    if (!this.selfChat.learn(chatId)) return;
    this.log.info("Learned the owner's own chat", { chatId });
    await this.redis.sadd(keys.selfChats, chatId).catch(() => {});
  }

  /**
   * Is this fromMe message in the owner's own "Message yourself" chat? Only the owner's account can send fromMe, so a
   * yes here means "the owner writing to themselves" (or us writing to the owner).
   * That chat is addressed by a LID while our account id is the phone JID, so the id alone is not enough: the echo
   * of a notice we sent, a quote-reply to one, or WhatsApp's own Contact.isMe teaches us the id (persisted in Redis).
   */
  async #isSelfChat(msg) {
    if (!msg.fromMe || !msg.to) return false;
    await this.#loadSelfChats();
    const id = msg.id?._serialized;
    if (this.selfChat.matchesNotice(msg.body, id)) {
      await this.#learnSelfChat(msg.to);
      return true;
    }
    if (this.selfChat.isSelf({ fromMe: true, from: msg.from, to: msg.to, me: this.wa.me?.wid })) return true;
    if (msg.hasQuotedMsg && (await this.#quotedAgendaId(msg))) {
      await this.#learnSelfChat(msg.to);
      return true;
    }
    if (!String(msg.to).endsWith('@g.us') && (await this.wa.isOwnChat(msg.to).catch(() => false))) {
      await this.#learnSelfChat(msg.to);
      return true;
    }
    return false;
  }

  async #quotedAgendaId(msg) {
    if (!msg.hasQuotedMsg) return null;
    try {
      const q = await msg.getQuotedMessage();
      const v = q?.id?._serialized ? await this.redis.get(keys.agendaMsg(q.id._serialized)) : null;
      return v ? Number(v) : null;
    } catch {
      return null;
    }
  }

  /**
   * "ok 12", "no 7", "confirm 3 5", "ok" (quote-reply or the only pending request), "pending": handled here, answered in
   * the same chat. Returns true when the message was an owner command (so it is not logged as a chat message).
   */
  async #ownerCommand(msg) {
    // The caller already established this is the owner's own chat. Our own notices and answers are never commands.
    if (this.selfChat.matchesNotice(msg.body, msg.id?._serialized)) return false;
    const cmd = parseOwnerCommand(msg.body);
    if (!cmd) return false;
    const id = msg.id?._serialized;
    // The same message can arrive as a 'message' and as a 'message_create' event: act once.
    if (id && !(await this.redis.set(keys.ownerCmd(id), '1', 'EX', 300, 'NX'))) return true;
    const tz = this.settings.get().timezone;
    const reply = (text) => this.#sendToSelf(msg.to, text).catch((err) => this.log.warn('Could not answer the owner', { err }));
    if (cmd.type === 'help') {
      await reply(ownerHelp());
      return true;
    }
    const open = await this.repo.listAgenda({ view: 'open', limit: 300 });
    const pending = open.filter((i) => i.status === 'pending');
    if (cmd.type === 'list') {
      await reply(ownerPendingList(pending, { timeZone: tz }));
      return true;
    }
    if (cmd.type === 'slots') {
      if (!this.availability) return false;
      const o = await this.availability.overview({ days: 7, count: 8, perDay: 3 });
      await reply(ownerSlotsText({ slotsLine: slotsText(o.slots, tz), meetings: o.meetings, timeZone: tz }));
      return true;
    }
    const quotedId = await this.#quotedAgendaId(msg);
    const lookup = new Map(open.map((i) => [i.id, i]));
    for (const wanted of [...(cmd.ids || []), quotedId].filter(Boolean)) if (!lookup.has(wanted)) lookup.set(wanted, await this.repo.getAgendaItem(wanted));
    const d = resolveDecision(cmd, { pending, byId: (n) => lookup.get(n) || null, quotedId });
    if (d.kind === 'ignore') return false; // "ok" with nothing waiting is just a note to self
    if (d.kind === 'ask') {
      await reply(ownerAsk(d.pending, { timeZone: tz }));
      return true;
    }
    const lines = [];
    for (const item of d.targets) {
      try {
        lines.push(ownerResultLine(await this.decideAgenda(item, d.action, { force: !!cmd.force })));
      } catch (err) {
        this.log.error('Owner command failed', { id: item.id, err });
        lines.push(ownerResultLine({ item, error: 'something went wrong, try the console' }));
      }
    }
    for (const n of d.unknown) lines.push(`⚠️ #${n}: no such request. Send "pending" to see what is waiting.`);
    this.log.info('Agenda decided from the owner\'s WhatsApp', { action: d.action, ids: d.targets.map((t) => t.id) });
    await reply(lines.join('\n'));
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
    for (const m of [this.pending, this.recentOut, this.jobs, this.echo, this.lastErrorNotice, this.lastUnsupported]) m.clear(); // EchoTracker has clear() too
    this.sessions.reset();
    this.selfChat.clear(); // the next account has another own-chat id
    this.selfChatsLoaded = null;
    await this.redis.del(keys.selfChats).catch(() => {});
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
    this.echo.expect(chatId, caption); // a media message's body is its caption: recognise the echo like any text
    const sent = await this.wa.sendImage(chatId, png, { caption });
    this.echo.sent(sent?.id?._serialized);
    const saved = await this.repo.insertMessage({ waMessageId: sent?.id?._serialized, chatId, direction: 'out', body: caption, type: 'image', isAi: true, sentBy: 'ai', intent, latencyMs: 0 });
    if (saved) this.#broadcastMessage(saved, chatId);
    // Not appended to the chat memory: confirmation notices in the history made the model repeat them ("Where are you?" →
    // "Sorry, Mehul can't make this one…").
  }

  /**
   * The first few requests of a burst are sent to the owner one by one; once three have gone out within a minute the rest are
   * gathered into a single digest (30 requests in two minutes used to mean 30 WhatsApp pings). Returns true when this one
   * should be sent now.
   */
  #noticeNow(itemId) {
    const now = Date.now();
    this.noticeTimes = this.noticeTimes.filter((t) => now - t < 60_000);
    if (this.noticeTimes.length < 3 && !this.digest.length) {
      this.noticeTimes.push(now);
      return true;
    }
    if (!this.digest.includes(itemId)) this.digest.push(itemId);
    clearTimeout(this.digestTimer);
    this.digestTimer = setTimeout(() => this.#flushDigest().catch((err) => this.log.warn('Owner digest failed', { err })), this.digest.length >= 15 ? 0 : 25_000);
    this.digestTimer.unref?.();
    return false;
  }

  async #flushDigest() {
    const ids = this.digest.splice(0);
    this.noticeTimes = [];
    const items = (await Promise.all(ids.map((id) => this.repo.getAgendaItem(id)))).filter((i) => i && i.status === 'pending');
    if (items.length) await this.#sendToOwner(ownerDigest(items, { timeZone: this.settings.get().timezone }));
  }

  async #onAgenda(ev) {
    this.hub.emit('agenda:update', { id: ev.itemId, created: !!ev.created });
    const s = this.settings.get();
    // New request → tell the owner. A refinement of a request still waiting ("make it 6pm") → tell them what changed.
    if (!s.notifyOwnerOnAgenda || (!ev.created && !ev.changed)) return;
    const item = await this.repo.getAgendaItem(ev.itemId);
    if (!item || (!ev.created && item.status !== 'pending')) return;
    if (!this.#noticeNow(item.id)) return; // part of a burst: it will be in the digest
    // A request that lands on a confirmed meeting, a block or outside working hours says so, with free times to offer instead.
    let clash = [];
    let blocking = false;
    if (this.availability && item.kind === 'appointment' && item.dueAt && !item.allDay) {
      const av = await this.availability.assess(item.dueAt, { ignoreId: item.id }).catch(() => null);
      if (av && !av.skipped) {
        clash = clashLines(av, { timeZone: s.timezone, formatWhen });
        blocking = !av.ok;
        if (blocking) {
          const free = await this.availability.freeSlots({ count: 4, perDay: 2 }).catch(() => []);
          if (free.length) clash.push(`Free: ${slotsText(free, s.timezone)}`);
        }
      }
    }
    // Every request carries its number (#id); the owner answers "ok 12" from their own WhatsApp, or quote-replies "ok".
    const sent = await this.#sendToOwner(ownerNotice(item, { timeZone: s.timezone, updated: !ev.created, clash, blocking }));
    const sentId = sent?.id?._serialized;
    if (sentId) await this.redis.set(keys.agendaMsg(sentId), String(item.id), 'EX', 14 * 86_400).catch(() => {});
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
      if (ev.type === 'error' && !ev.silent) await (ev.busy ? this.#busyNotice(chatId) : this.#errorNotice(chatId));
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
    // Template replies (meeting acknowledgements) are exempt: asking the same question again is a legitimate answer.
    if (key.length >= 12 && ev.meta?.model !== 'template') {
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
    // Keep welcome/goodbye in the conversation so the model knows what "it" already said. Agenda confirmations/declines are
    // not kept: the model parroted them as if they answered the next message.
    if (intent !== 'agenda') await this.memory.append(chatId, [{ role: 'assistant', content: text }]).catch(() => {});
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

  /**
   * "Groq only" mode and Groq cannot answer (every key rate-limited, or Groq is down): say the AI is on a short break
   * instead of leaving the person hanging or sending a generic error. Once per chat per 5 minutes; not stored in the
   * chat memory (the model must not learn it as something it said).
   */
  async #busyNotice(chatId) {
    if (Date.now() - (this.lastBusyNotice.get(chatId) || 0) < BUSY_NOTICE_COOLDOWN_MS) return;
    const s = this.settings.get();
    if (s.replyMode === 'session' && !(await this.sessions.active(chatId))) return; // they ended the session meanwhile
    if (await this.redis.exists(keys.takeover(chatId))) return; // the owner is talking to them
    this.lastBusyNotice.set(chatId, Date.now());
    const contact = await this.repo.getContactByChatId(chatId).catch(() => null);
    const text = renderTemplate(s.groqBusyMessage || 'Hey {name}! {persona} is on a short break right now. Please message again in a little while 🙂', s, { name: contact?.name });
    const sent = await this.#send(chatId, text).catch(() => null);
    if (sent) {
      const saved = await this.repo.insertMessage({ waMessageId: sent.id?._serialized, chatId, direction: 'out', body: text, isAi: true, sentBy: 'ai', intent: 'busy' });
      if (saved) this.#broadcastMessage(saved, chatId);
    }
  }

  /**
   * Diagnostics for the image-send problem: draw a small picture and send it to the owner's own chat with each way in
   * turn. Returns { chat, rendered, results: [{ strategy, ok, error?, id? }] } and never throws on a failed way.
   */
  async testImage({ all = false } = {}) {
    const me = this.wa.me?.wid;
    if (!me || !this.wa.ready) throw Object.assign(new Error('WhatsApp is not connected'), { status: 409 });
    let png = TINY_PNG;
    let rendered = false;
    try {
      png = await this.wa.renderHtml(TEST_CARD_HTML);
      rendered = true;
    } catch (err) {
      this.log.warn('Could not draw the test picture; using a 1x1 image', { err: err.message });
    }
    const caption = 'Card test';
    this.selfChat.rememberNotice(caption);
    this.echo.expect(me, caption);
    const results = await this.wa.testImageStrategies(me, png, caption, { all });
    for (const r of results) {
      if (!r.id) continue;
      this.echo.sent(r.id);
      this.selfChat.rememberNotice(null, r.id);
    }
    return { chat: me, rendered, patch: this.wa.mediaPatch ?? null, results };
  }

  /** Send while remembering the body so the message_create echo is recognised as ours. */
  async #send(chatId, text, opts) {
    this.echo.expect(chatId, text); // fallback for an echo that arrives before sendMessage resolves
    const sent = await this.wa.sendText(chatId, text, opts);
    this.echo.sent(sent?.id?._serialized); // exact: the echo of this very message id is ours
    return sent;
  }

  // ---------------------------------------------------- owner / console sends

  /** Message the owner typed on the phone (or another linked device). */
  async handleOwnMessage(msg) {
    const chatId = msg.to;
    if (!chatId || chatId === 'status@broadcast') return;
    // Our own AI/console message echoing back (by WhatsApp id, or by text if the event beat sendMessage): not the owner typing.
    if (isSystemMessageType(msg.type)) return; // encryption/group/call notices: not a message
    if (this.echo.isOurs(chatId, msg.body, msg.id?._serialized)) return;
    // The owner's own chat ("Message yourself"): our notices and answers, the owner's "ok 12", their notes to self. Never stored as a
    // chat (it showed up as a phantom chat full of "phone" messages) and never a takeover; a command is acted on.
    if (await this.#isSelfChat(msg).catch(() => false)) {
      await this.#ownerCommand(msg).catch((err) => this.log.warn('Owner command failed', { err }));
      return;
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
