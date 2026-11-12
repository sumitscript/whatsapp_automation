// The owner answers agenda requests from their own WhatsApp ("Message yourself"): every request gets a number,
// and the owner replies "ok 12", "12 yes", "confirm 3 and 5", "no 7", "done 4", "ok all", "pending", or quote-replies
// to the notice with just "ok". Pure and dependency-free (unit-tested); the inbox executes what this decides.
import { formatWhen } from './agenda.js';

const CONFIRM = new Set(['confirm', 'confirmed', 'ok', 'okay', 'okk', 'okkk', 'yes', 'yess', 'yep', 'yup', 'sure', 'accept', 'accepted', 'approve', 'approved', 'fine', 'agreed', 'haan', 'han', 'ha', 'haa', 'hn', 'theek', 'thik', 'pakka', 'bilkul', 'zaroor', 'kardo', '👍', '✅', 'हाँ', 'हां', 'ठीक', 'पक्का', 'ज़रूर', 'जरूर', 'हा', 'હા', 'ઠીક', 'પાક્કું']);
const DECLINE = new Set(['decline', 'declined', 'reject', 'rejected', 'cancel', 'cancelled', 'canceled', 'no', 'nope', 'nah', 'nahi', 'nahin', 'na', 'mana', 'skip', 'deny', 'denied', '👎', '❌', 'नहीं', 'ना', 'नही', 'ના', 'નહીં']);
const DONE = new Set(['done', 'complete', 'completed', 'finished', 'finish']);
const FILLER = new Set(['the', 'it', 'this', 'that', 'please', 'pls', 'plz', 'number', 'no', 'num', 'item', 'request', 'agenda', 'meeting', 'to', 'for', 'hai', 'hain', 'kar', 'do', 'de', 'dena', 'wala', 'wali']);
const LIST = /^(?:list|pending|agenda|status|show|show all|requests|what'?s pending|kya pending(?: hai)?|\?+)$/;
const HELP = /^(?:help|commands|how|menu)$/;
const SLOTS = /^(?:slots|free|free slots|my slots|availability|available|free time|khali|khali time)$/;
const FORCE = new Set(['force', 'forced', 'override', 'anyway']);

const clean = (t) =>
  String(t ?? '')
    .toLowerCase()
    .replace(/[.!,;:"'`()[\]]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

function verbOf(tok) {
  if (CONFIRM.has(tok)) return 'confirm';
  if (DECLINE.has(tok)) return 'decline';
  if (DONE.has(tok)) return 'done';
  return null;
}

/**
 * Read an owner message. Returns null when it is not a command (a normal note to self).
 * @returns {{type:'decide', action:'confirm'|'decline'|'done', ids:number[]|null, all:boolean} | {type:'list'} | {type:'help'} | null}
 */
export function parseOwnerCommand(text) {
  const raw = String(text ?? '').trim();
  if (!raw || raw.length > 90) return null;
  const s = clean(raw);
  if (!s) return null;
  if (LIST.test(s)) return { type: 'list' };
  if (HELP.test(s)) return { type: 'help' };
  if (SLOTS.test(s)) return { type: 'slots' };
  // "theek hai" / "kar do" are two words: join the common pairs before splitting
  const joined = s.replace(/\btheek hai\b/g, 'theek').replace(/\bthik hai\b/g, 'thik').replace(/\bkar ?do\b/g, 'kardo').replace(/\bgo ahead\b/g, 'ok').replace(/\bmark done\b/g, 'done');
  const toks = joined.split(' ').filter(Boolean);
  let action = null;
  let force = false;
  const ids = [];
  let all = false;
  for (const raw of toks) {
    const t = raw.replace(/^#/, '');
    if (/^\d{1,6}$/.test(t)) {
      ids.push(Number(t));
      continue;
    }
    if (t === 'all' || t === 'sab' || t === 'sabhi' || t === 'everything') {
      all = true;
      continue;
    }
    if (t === 'and' || t === '&' || t === 'aur' || t === ',') continue;
    if (FORCE.has(t)) {
      force = true; // "force 7" / "ok 7 anyway": confirm even though it clashes
      continue;
    }
    const v = verbOf(t);
    if (v) {
      if (action && action !== v) return null; // "yes no": ambiguous
      action = v;
      continue;
    }
    if (FILLER.has(t)) continue;
    return null; // any other word: this is a normal sentence, not a command
  }
  if (force && !action) action = 'confirm';
  if (!action) return null; // a bare number is not an answer
  if (force && action !== 'confirm') return null;
  return { type: 'decide', action, ids: ids.length ? [...new Set(ids)] : null, all, force };
}

/**
 * Decide what to do with a parsed command.
 * @param {object} cmd from parseOwnerCommand (type 'decide')
 * @param {{ pending: object[], byId: (id:number)=>object|null, quotedId?: number|null }} ctx
 * @returns {{kind:'apply', action:string, targets:object[], unknown:number[]} | {kind:'ask', pending:object[]} | {kind:'ignore'}}
 */
export function resolveDecision(cmd, { pending, byId, quotedId = null }) {
  const action = cmd.action;
  if (cmd.all) return pending.length ? { kind: 'apply', action, targets: pending, unknown: [] } : { kind: 'ignore', reason: 'nothing pending' };
  if (cmd.ids?.length) {
    const targets = [];
    const unknown = [];
    for (const id of cmd.ids) {
      const it = byId(id);
      if (it) targets.push(it);
      else unknown.push(id);
    }
    return { kind: 'apply', action, targets, unknown };
  }
  // No number: a quote-reply to a notice, or the only pending request, is unambiguous.
  if (quotedId) {
    const it = byId(quotedId);
    if (it) return { kind: 'apply', action, targets: [it], unknown: [] };
  }
  if (pending.length === 1) return { kind: 'apply', action, targets: pending, unknown: [] };
  if (pending.length === 0) return { kind: 'ignore', reason: 'nothing pending' };
  return { kind: 'ask', pending };
}

// ---------------------------------------------------------------- texts to the owner

const who = (it) => it.contactName || it.phone || 'a contact';
const whenOf = (it, tz) => (it.dueAt ? formatWhen(it.dueAt, tz, it.allDay) : 'no time set');
const KIND_LABEL = { appointment: 'Meeting request', todo: 'Request' };

/** The WhatsApp message the owner gets when a request is spotted (or changed). */
export function ownerNotice(item, { timeZone, updated = false, clash = [], blocking = false } = {}) {
  const icon = item.kind === 'appointment' ? '📅' : '📝';
  const head = `${updated ? '✏️ Updated' : icon} ${KIND_LABEL[item.kind] || 'Request'} *#${item.id}* from ${who(item)}`;
  const lines = [head, `${item.title}`, `🗓 ${whenOf(item, timeZone)}${item.location ? `  ·  📍 ${item.location}` : ''}`];
  if (item.dueText && item.dueText !== whenOf(item, timeZone)) lines.push(`They said: “${item.dueText}”`);
  if (clash.length) lines.push('', ...clash);
  lines.push('', `Reply *ok ${item.id}* to confirm (I'll send ${who(item).split(/\s+/)[0]} the confirmation and an invitation card) or *no ${item.id}* to decline${blocking ? ` (it clashes: *force ${item.id}* confirms it anyway)` : ''}.`, 'You can also reply to this message with just "ok". Send "pending" to see everything waiting, "slots" for your free times.');
  return lines.join('\n');
}

export function ownerPendingList(pending, { timeZone }) {
  if (!pending.length) return '✅ Nothing is waiting for you.';
  const lines = pending.slice(0, 12).map((it) => `*#${it.id}* ${it.kind === 'appointment' ? '📅' : '📝'} ${who(it)}: ${it.title} — ${whenOf(it, timeZone)}${it.location ? ` · ${it.location}` : ''}`);
  const more = pending.length > 12 ? `\n…and ${pending.length - 12} more in the console.` : '';
  return `Waiting for you (${pending.length}):\n${lines.join('\n')}${more}\n\nReply "ok <number>" to confirm, "no <number>" to decline, or "ok all".`;
}

export function ownerAsk(pending, { timeZone }) {
  return `You have ${pending.length} requests waiting, so I need the number.\n\n${ownerPendingList(pending, { timeZone }).split('\n').slice(1, -2).join('\n')}\n\nReply like "ok ${pending[0].id}" or "no ${pending[0].id}".`;
}

export function ownerHelp() {
  return 'Agenda by WhatsApp:\n• ok 12 / yes 12 / confirm 12: accept #12 (the person gets a confirmation and an invitation card)\n• no 12 / decline 12: decline it\n• done 12: mark it done\n• ok 3 5 or ok all: several at once\n• pending: list what is waiting\n• slots: your next free times\n• force 12: confirm #12 even if it clashes with another meeting\nYou can also reply to a request message with just "ok" or "no".';
}

/** One line per result, sent back to the owner after their command. */
export function ownerResultLine(res) {
  const n = `#${res.item.id}`;
  const name = who(res.item);
  if (res.error) return `⚠️ ${n}: ${res.error}`;
  if (res.skipped) return `ℹ️ ${n} is already ${res.item.status}.`;
  if (res.clash) return `⚠️ ${n} not confirmed:\n${res.clash.join('\n')}\n${res.freeText ? `Free: ${res.freeText}\n` : ''}Reply *force ${res.item.id}* to confirm it anyway, or *no ${res.item.id}* to decline (the person is offered free slots).`;
  const sent = res.notified ? `${name.split(/\s+/)[0]} got ${res.card ? 'the confirmation and invitation card' : 'the message'}` : res.item.chatId ? `couldn't message ${name.split(/\s+/)[0]} (is WhatsApp connected?)` : 'no chat to message';
  if (res.action === 'confirm') return `✅ ${n} confirmed: ${res.item.title}. ${sent}.`;
  if (res.action === 'decline') return `🚫 ${n} declined. ${res.notified ? `${name.split(/\s+/)[0]} was told.` : sent + '.'}`;
  return `☑️ ${n} marked done.`;
}

/** "slots": the owner's next free times and what is already booked. */
export function ownerSlotsText({ slotsLine, meetings, timeZone }) {
  const booked = meetings.slice(0, 6).map((m) => `• ${formatWhen(m.startsAt, timeZone, false)}: ${m.title}${m.contactName ? ` (${m.contactName})` : ''}${m.status === 'pending' ? ' (waiting)' : ''}`);
  return `🗓 Your free times:\n${slotsLine || 'Nothing free in the next 7 days.'}${booked.length ? `\n\nBooked:\n${booked.join('\n')}` : ''}`;
}

/**
 * One message for a burst of requests (the first few are sent one by one, the rest are gathered): every request keeps its
 * number, so "ok 12" still works for each, and "ok all" takes them all.
 */
export function ownerDigest(items, { timeZone }) {
  const lines = items.slice(0, 25).map((it) => `*#${it.id}* ${it.kind === 'appointment' ? '📅' : '📝'} ${who(it)}: ${whenOf(it, timeZone)}${it.location ? ` · ${it.location}` : ''}`);
  const more = items.length > 25 ? `\n…and ${items.length - 25} more (send "pending")` : '';
  return `📥 *${items.length} new request${items.length === 1 ? '' : 's'}* came in a burst:\n${lines.join('\n')}${more}\n\nReply "ok <number>" to confirm one, "ok all" for all of them, "no <number>" to decline, "slots" for your free times. A clash is checked when you confirm.`;
}
