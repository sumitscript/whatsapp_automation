// Live settings: Postgres is the source of truth, every process keeps an
// in-memory copy and reloads when another process publishes a change.
import { config } from './config.js';
import { channels } from './redis.js';
import { DEFAULT_SETTINGS, sanitizeSettings } from './settings-schema.js';

export async function createSettingsService({ repo, redis, subscriber, logger }) {
  const defaults = { ...DEFAULT_SETTINGS, model: config.ollama.defaultModel };
  let current = { ...defaults };
  const listeners = new Set();

  async function reload() {
    const stored = await repo.getSettings();
    const { value } = sanitizeSettings(Object.fromEntries(Object.entries(stored).filter(([k]) => k in defaults)));
    const prev = current;
    current = { ...defaults, ...value };
    for (const fn of listeners) {
      try {
        fn(current, prev);
      } catch (err) {
        logger.warn('settings listener failed', { err });
      }
    }
    return current;
  }

  await reload();
  await subscriber.subscribe(channels.settings);
  subscriber.on('message', (ch) => {
    if (ch === channels.settings) reload().catch((err) => logger.error('settings reload failed', { err }));
  });

  return {
    get: () => current,
    defaults: () => defaults,
    onChange(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    /** Validate + persist a partial update, then notify every process. */
    async update(patch) {
      const { value, errors } = sanitizeSettings(patch);
      if (errors.length) {
        const err = new Error(errors.join('; '));
        err.status = 400;
        throw err;
      }
      await repo.saveSettings(value);
      await reload();
      await redis.publish(channels.settings, '1');
      return current;
    },
  };
}
