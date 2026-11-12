// Finds the owner's own "Message yourself" chat. Pure and dependency-free (unit-tested).
//
// WhatsApp addresses that chat by a LID ("203976637206652@lid") while our own account id is the phone JID
// ("918756365339@c.us"), so comparing ids never matches. Instead the chat is learned from evidence:
//   • the echo of a notice we just sent to the owner arrives with `to` = the self chat id (matched by the exact
//     message id, or by the text we sent, whichever is known when the event arrives: it can beat sendMessage);
//   • a reply that quotes one of those notices is, by definition, in that chat;
//   • the usual cheap checks (from === to, same phone digits) and a Contact.isMe lookup done by the caller.
// Learned ids are persisted by the caller (Redis) so a restart does not forget them.

const digitsOf = (wid) => String(wid || '').split('@')[0].split(':')[0].replace(/\D/g, '');

export class SelfChatTracker {
  constructor({ noticeTtlMs = 5 * 60_000, maxNotices = 200, now = () => Date.now() } = {}) {
    this.noticeTtlMs = noticeTtlMs;
    this.maxNotices = maxNotices;
    this.now = now;
    this.ids = new Set(); // learned self-chat ids
    this.texts = new Map(); // text we sent to the owner → expiry
    this.msgIds = new Map(); // WhatsApp id of a message we sent to the owner → expiry
  }

  /** Add ids loaded from storage. */
  load(ids) {
    for (const id of ids || []) if (id) this.ids.add(String(id));
  }

  /** Remember `chatId` as the owner's own chat. Returns true when it was new (the caller persists it). */
  learn(chatId) {
    if (!chatId) return false;
    const id = String(chatId);
    if (this.ids.has(id)) return false;
    this.ids.add(id);
    return true;
  }

  isLearned(chatId) {
    return this.ids.has(String(chatId));
  }

  /** We sent `text` to the owner (and, once sendMessage resolved, it got WhatsApp id `id`). */
  rememberNotice(text, id) {
    const exp = this.now() + this.noticeTtlMs;
    if (text) this.texts.set(String(text), exp);
    if (id) this.msgIds.set(String(id), exp);
    this.#prune();
  }

  /** Is this message the echo of something we sent to the owner? */
  matchesNotice(body, id) {
    const t = this.now();
    if (id && (this.msgIds.get(String(id)) ?? 0) > t) return true;
    return !!body && (this.texts.get(String(body)) ?? 0) > t;
  }

  /**
   * Cheap, synchronous check that an own message is in the owner's own chat.
   * @param {{ fromMe: boolean, from?: string, to?: string, me?: string }} m
   */
  isSelf({ fromMe, from, to, me }) {
    if (!fromMe || !to) return false;
    if (from && from === to) return true;
    if (this.ids.has(String(to))) return true;
    const mine = digitsOf(me);
    return !!mine && digitsOf(to) === mine && String(to).endsWith('@c.us');
  }

  clear() {
    this.ids.clear();
    this.texts.clear();
    this.msgIds.clear();
  }

  #prune() {
    const t = this.now();
    for (const map of [this.texts, this.msgIds]) {
      if (map.size <= this.maxNotices) continue;
      for (const [k, exp] of map) if (exp <= t) map.delete(k);
      while (map.size > this.maxNotices) map.delete(map.keys().next().value);
    }
  }
}
