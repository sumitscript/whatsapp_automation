// Availability service: loads what the pure functions in slots.js need from the repo. No heavy dependencies, so the
// worker, the gateway (playground, console, owner commands) and the tests all use the same object.
import { assess, availabilityConfig, freeSlots, itemInterval } from './slots.js';

/**
 * @param {{ repo: { listAgenda: Function, listTimeBlocks: Function }, getSettings: () => object, now?: () => Date }} o
 */
export function createAvailability({ repo, getSettings, now = () => new Date() }) {
  async function snapshot() {
    const cfg = availabilityConfig(getSettings());
    const t = now();
    const [items, blocks] = await Promise.all([
      repo.listAgenda({ view: 'open', kind: 'appointment', limit: 500 }),
      repo.listTimeBlocks({ from: new Date(t.getTime() - 86_400_000) }),
    ]);
    return { cfg, items, blocks, now: t };
  }
  return {
    snapshot,
    /** Is `dueAt` free? With enforceAvailability off every time is ok (nothing is checked). */
    async assess(dueAt, { ignoreId = null } = {}) {
      const s = await snapshot();
      if (!s.cfg.enforce) return { ok: true, reasons: [], hard: [], soft: [], blocked: [], skipped: true };
      return assess(dueAt, { ...s, ignoreId });
    },
    /** What the owner sees: free slots plus the meetings and blocks coming up. */
    async overview({ days = 7, count = 8, perDay = 3 } = {}) {
      const s = await snapshot();
      const upcoming = (iso) => new Date(iso).getTime() >= s.now.getTime();
      return {
        cfg: s.cfg,
        slots: freeSlots({ ...s, days, count, perDay }),
        meetings: s.items
          .filter((it) => itemInterval(it, s.cfg) && upcoming(it.dueAt))
          .sort((a, b) => new Date(a.dueAt) - new Date(b.dueAt))
          .map((it) => ({ id: it.id, title: it.title, contactName: it.contactName ?? null, startsAt: it.dueAt, status: it.status })),
        blocks: s.blocks.filter((b) => new Date(b.endsAt).getTime() > s.now.getTime()),
      };
    },
    /** The next free start times (Date[]). */
    async freeSlots(opts = {}) {
      const s = await snapshot();
      return freeSlots({ ...s, ...opts });
    },
  };
}
