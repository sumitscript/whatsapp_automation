// Recognises the echoes of messages this server sent itself.
//
// WhatsApp reports every message we send back as a "message_create" event with fromMe=true, exactly like a
// message the owner types on the phone. The inbox must tell the two apart: our own echo is ignored, the
// owner's typing pauses the AI (takeover). Two signals are used:
//   • the WhatsApp message id returned by sendMessage (exact, but the event can arrive before the promise
//     resolves), and
//   • the text we are sending (counted per chat+text, valid for ttlMs) as the fallback for that race.
// Once an echo is recognised its id is remembered too, so a repeat of the same event (duplicated listeners)
// is also treated as ours instead of as the owner typing. Pure and dependency-free.
import { SeenIds } from './util.js';

export class EchoTracker {
  constructor({ ttlMs = 30_000, idTtlMs = 10 * 60_000, maxIds = 1000 } = {}) {
    this.ttlMs = ttlMs;
    this.counts = new Map(); // `${chatId}|${text}` → messages with that text we are still waiting to see echoed
    this.ids = new SeenIds({ ttlMs: idTtlMs, max: maxIds });
  }

  /** We are about to send `text` to `chatId`. */
  expect(chatId, text) {
    const k = `${chatId}|${text}`;
    this.counts.set(k, (this.counts.get(k) || 0) + 1);
    setTimeout(() => this.#release(k), this.ttlMs).unref?.();
  }

  /** sendMessage resolved: remember the id WhatsApp gave the message. */
  sent(id) {
    if (id) this.ids.add(String(id));
  }

  /** Is this fromMe event (to `chatId`, with `body`, WhatsApp id `id`) one of our own sends? */
  isOurs(chatId, body, id) {
    if (id && this.ids.has(String(id))) return true;
    const k = `${chatId}|${body}`;
    if (this.counts.has(k)) {
      this.#release(k);
      if (id) this.ids.add(String(id)); // a repeat of this same event is ours as well
      return true;
    }
    return false;
  }

  clear() {
    this.counts.clear();
  }

  #release(k) {
    const n = (this.counts.get(k) || 0) - 1;
    if (n > 0) this.counts.set(k, n);
    else this.counts.delete(k);
  }
}
