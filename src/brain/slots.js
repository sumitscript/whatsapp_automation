// Availability: is a time free, and which times are? Pure and dependency-free (unit-tested).
//
// Nothing is stored as "slots": free time is derived every time from three facts that are already in the database, so it
// can never drift out of sync with the agenda:
//   working hours (settings)  minus  confirmed meetings (agenda_items)  minus  blocks the owner marked busy (time_blocks)
// A request that overlaps a *confirmed* meeting or a block is a hard clash. One that overlaps another *pending* request
// is a soft clash: it is accepted (the owner decides) but the owner is told.
import { localParts, zonedToUtc } from './agenda.js';

const DAY_KEYS = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'];
const MIN = 60_000;

export const parseHHMM = (s, fallback) => {
  const m = /^(\d{1,2}):(\d{2})$/.exec(String(s ?? '').trim());
  return m && +m[1] < 24 && +m[2] < 60 ? +m[1] * 60 + +m[2] : fallback;
};

/** Settings → the numbers the functions below use. */
export function availabilityConfig(settings = {}) {
  const startMin = parseHHMM(settings.workStart, 9 * 60);
  let endMin = parseHHMM(settings.workEnd, 21 * 60);
  if (endMin <= startMin) endMin = Math.min(startMin + 12 * 60, 24 * 60 - 1);
  const days = new Set(
    String(settings.workDays || 'mon,tue,wed,thu,fri,sat,sun')
      .split(/[\s,]+/)
      .map((d) => DAY_KEYS.indexOf(d.trim().toLowerCase()))
      .filter((i) => i >= 0),
  );
  const meetingMin = Number(settings.meetingMinutes) > 0 ? Number(settings.meetingMinutes) : 60;
  const bufferMin = Number(settings.meetingBufferMinutes) >= 0 ? Number(settings.meetingBufferMinutes) : 15;
  return {
    tz: settings.timezone || 'Asia/Kolkata',
    startMin,
    endMin,
    days: days.size ? days : new Set([0, 1, 2, 3, 4, 5, 6]),
    meetingMs: meetingMin * MIN,
    bufferMs: bufferMin * MIN,
    stepMs: 30 * MIN,
    enforce: settings.enforceAvailability !== false,
  };
}

/** Working-hours window (ms) on the local calendar day y-m-d. */
function windowFor(y, m, d, cfg) {
  return {
    start: zonedToUtc(y, m, d, Math.floor(cfg.startMin / 60), cfg.startMin % 60, cfg.tz).getTime(),
    end: zonedToUtc(y, m, d, Math.floor(cfg.endMin / 60), cfg.endMin % 60, cfg.tz).getTime(),
  };
}

/** The time a meeting occupies, or null for items that do not occupy time (to-dos, date-only, no date). */
export function itemInterval(item, cfg) {
  if (item.kind !== 'appointment' || !item.dueAt || item.allDay) return null;
  const start = new Date(item.dueAt).getTime();
  if (Number.isNaN(start)) return null;
  return { start, end: start + cfg.meetingMs };
}

const overlaps = (a, b, buffer = 0) => a.start < b.end + buffer && b.start < a.end + buffer;
const brief = (it) => ({ id: it.id, title: it.title, contactName: it.contactName ?? null, startsAt: it.dueAt, status: it.status });

/**
 * Can a meeting start at `dueAt`?
 * @returns {{ ok: boolean, reasons: string[], hard: object[], soft: object[], blocked: object[] }}
 *   reasons: 'past' | 'day_off' | 'outside_hours'; hard = confirmed meetings in the way; soft = pending requests at the
 *   same time; blocked = the owner's busy blocks. ok means none of past/day_off/outside_hours/hard/blocked applies.
 */
export function assess(dueAt, { items = [], blocks = [], cfg, now = new Date(), ignoreId = null }) {
  const start = new Date(dueAt).getTime();
  const iv = { start, end: start + cfg.meetingMs };
  const reasons = [];
  if (start < now.getTime()) reasons.push('past');
  const p = localParts(new Date(start), cfg.tz);
  if (!cfg.days.has(p.wd)) reasons.push('day_off');
  else {
    const w = windowFor(p.y, p.m, p.d, cfg);
    if (start < w.start || iv.end > w.end) reasons.push('outside_hours');
  }
  const hard = [];
  const soft = [];
  for (const it of items) {
    if (it.id === ignoreId) continue;
    const o = itemInterval(it, cfg);
    if (!o || !overlaps(iv, o, cfg.bufferMs)) continue;
    if (it.status === 'confirmed') hard.push(brief(it));
    else if (it.status === 'pending') soft.push(brief(it));
  }
  const blocked = blocks
    .filter((b) => overlaps(iv, { start: new Date(b.startsAt).getTime(), end: new Date(b.endsAt).getTime() }, 0))
    .map((b) => ({ id: b.id, label: b.label, startsAt: b.startsAt, endsAt: b.endsAt }));
  return { ok: !reasons.length && !hard.length && !blocked.length, reasons, hard, soft, blocked };
}

/**
 * The next free start times (30-minute steps inside working hours, never within `leadMin` of now).
 * `perDay` spreads suggestions over several days instead of listing one morning.
 * @returns {Date[]}
 */
export function freeSlots({ items = [], blocks = [], cfg, now = new Date(), from = null, days = 7, count = 6, perDay = Infinity, ignoreId = null, leadMin = 30 }) {
  const out = [];
  const earliest = Math.max(from ? new Date(from).getTime() : now.getTime(), now.getTime() + leadMin * MIN);
  const base = localParts(new Date(earliest), cfg.tz);
  for (let i = 0; i <= days && out.length < count; i++) {
    const day = new Date(Date.UTC(base.y, base.m - 1, base.d + i));
    if (!cfg.days.has(day.getUTCDay())) continue;
    const w = windowFor(day.getUTCFullYear(), day.getUTCMonth() + 1, day.getUTCDate(), cfg);
    let t = w.start + Math.max(0, Math.ceil((earliest - w.start) / cfg.stepMs)) * cfg.stepMs;
    let today = 0;
    for (; t + cfg.meetingMs <= w.end && out.length < count && today < perDay; t += cfg.stepMs) {
      if (assess(new Date(t), { items, blocks, cfg, now, ignoreId }).ok) {
        out.push(new Date(t));
        today++;
      }
    }
  }
  return out;
}

const LOCALES = { hi: 'hi-IN', mr: 'mr-IN', gu: 'gu-IN' };

/** "Fri, 13 Nov: 11:00 am, 12:00 pm · Sat, 14 Nov: 10:00 am" */
export function slotsText(slots, tz, code = 'en') {
  const loc = LOCALES[code] || 'en-IN';
  const dayFmt = new Intl.DateTimeFormat(loc, { timeZone: tz, weekday: 'short', day: 'numeric', month: 'short' });
  const timeFmt = new Intl.DateTimeFormat(loc, { timeZone: tz, hour: 'numeric', minute: '2-digit', hour12: true });
  const byDay = new Map();
  for (const s of slots) {
    const k = dayFmt.format(s);
    byDay.set(k, [...(byDay.get(k) || []), timeFmt.format(s)]);
  }
  return [...byDay].map(([day, times]) => `${day}: ${times.join(', ')}`).join(' · ');
}

/** One line telling the owner what a request clashes with. Empty when there is nothing to say. */
export function clashLines(av, { timeZone, formatWhen }) {
  const lines = [];
  const when = (iso) => formatWhen(iso, timeZone, false);
  for (const h of av.hard) lines.push(`⚠️ Clashes with *#${h.id}* (${h.title}${h.contactName ? `, ${h.contactName}` : ''}) already confirmed for ${when(h.startsAt)}.`);
  for (const b of av.blocked) lines.push(`⚠️ You marked ${when(b.startsAt)} – ${when(b.endsAt)} as busy (${b.label}).`);
  if (av.reasons.includes('day_off')) lines.push('⚠️ That day is outside your working days.');
  else if (av.reasons.includes('outside_hours')) lines.push('⚠️ That time is outside your working hours.');
  if (av.reasons.includes('past')) lines.push('⚠️ That time has already passed.');
  for (const s of av.soft) lines.push(`ℹ️ *#${s.id}* (${s.contactName || s.title}) also asked for about this time and is still waiting.`);
  return lines;
}
