// Structured JSON logger. Writes one JSON line per entry to stdout and fans the
// same entry out to registered sinks (e.g. the Redis ring buffer that powers the
// live log tail in the UI). Dependency-free.

export const LEVELS = Object.freeze({ debug: 10, info: 20, warn: 30, error: 40 });

function serializeError(err) {
  if (!(err instanceof Error)) return err;
  return { name: err.name, message: err.message, stack: err.stack?.split('\n').slice(0, 6).join('\n'), code: err.code };
}

function normalizeMeta(meta) {
  if (!meta) return {};
  if (meta instanceof Error) return { err: serializeError(meta) };
  const out = {};
  for (const [k, v] of Object.entries(meta)) out[k] = v instanceof Error ? serializeError(v) : v;
  return out;
}

export function createLogger({ service = 'app', level = 'info', write = (line) => process.stdout.write(line + '\n') } = {}) {
  const sinks = new Set();
  let threshold = LEVELS[level] ?? LEVELS.info;

  function emit(lvl, bindings, msg, meta) {
    if (LEVELS[lvl] < threshold) return;
    const entry = { ts: new Date().toISOString(), level: lvl, service, ...bindings, msg: String(msg), ...normalizeMeta(meta) };
    let line;
    try {
      line = JSON.stringify(entry);
    } catch {
      line = JSON.stringify({ ts: entry.ts, level: lvl, service, msg: entry.msg, note: 'unserializable meta' });
    }
    write(line);
    for (const sink of sinks) {
      try {
        sink(entry, line);
      } catch {
        // A broken sink must never take logging (or the process) down.
      }
    }
  }

  function make(bindings) {
    return {
      debug: (msg, meta) => emit('debug', bindings, msg, meta),
      info: (msg, meta) => emit('info', bindings, msg, meta),
      warn: (msg, meta) => emit('warn', bindings, msg, meta),
      error: (msg, meta) => emit('error', bindings, msg, meta),
      child: (extra) => make({ ...bindings, ...extra }),
      addSink: (fn) => {
        sinks.add(fn);
        return () => sinks.delete(fn);
      },
      setLevel: (lvl) => {
        if (LEVELS[lvl]) threshold = LEVELS[lvl];
      },
    };
  }

  return make({});
}

export const logger = createLogger({
  service: process.env.SERVICE_NAME || 'app',
  level: process.env.LOG_LEVEL || 'info',
});
