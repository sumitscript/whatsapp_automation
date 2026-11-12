// Tries several ways to send one image until one works, and reports what each did. Pure (the ways are passed in),
// so it is unit-tested without WhatsApp.
//
// Why: media sends through whatsapp-web.js fail on some WhatsApp Web builds with errors from inside WhatsApp's own
// code ("Data passed to getter must include an id property…") while text sends work. Which alternative still works
// differs per build, so the manager tries them in order, remembers the one that worked, and exposes the per-way
// result so the owner can see it.

/**
 * @param {{ name: string, run: () => Promise<any> }[]} strategies in the default order
 * @param {{ preferred?: string|null, all?: boolean }} [opts] `preferred` goes first; `all` keeps going after a success (diagnostics)
 * @returns {Promise<{ sent: any, strategy: string|null, results: {strategy: string, ok: boolean, error?: string, message?: any}[] }>}
 */
export async function trySendStrategies(strategies, { preferred = null, all = false } = {}) {
  const first = preferred ? strategies.find((s) => s.name === preferred) : null;
  const ordered = first ? [first, ...strategies.filter((s) => s !== first)] : strategies;
  const results = [];
  let sent = null;
  let strategy = null;
  for (const s of ordered) {
    try {
      const message = await s.run();
      results.push({ strategy: s.name, ok: true, message });
      if (!strategy) {
        sent = message;
        strategy = s.name;
      }
      if (!all) break;
    } catch (err) {
      results.push({ strategy: s.name, ok: false, error: String(err?.message || err).split('\n')[0].slice(0, 300) });
    }
  }
  return { sent, strategy, results };
}
