// Postgres pool + tiny migration runner (db/migrations/*.sql, applied in name
// order once, guarded by an advisory lock so gateway and worker can both boot).
import fsp from 'node:fs/promises';
import path from 'node:path';
import pg from 'pg';
import { config } from '../core/config.js';

// BIGINT ids/counts → JS numbers (safe for this app's volumes)
pg.types.setTypeParser(20, (v) => Number(v));

export function createPool() {
  return new pg.Pool({ connectionString: config.pg.url, max: config.pg.poolMax, idleTimeoutMillis: 30_000 });
}

const MIGRATION_LOCK = 7_340_001;

export async function migrate(pool, logger) {
  const dir = path.join(config.root, 'db', 'migrations');
  const files = (await fsp.readdir(dir)).filter((f) => f.endsWith('.sql')).sort();
  const client = await pool.connect();
  try {
    await client.query('SELECT pg_advisory_lock($1)', [MIGRATION_LOCK]);
    await client.query('CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())');
    const { rows } = await client.query('SELECT name FROM schema_migrations');
    const done = new Set(rows.map((r) => r.name));
    for (const f of files) {
      if (done.has(f)) continue;
      const sql = await fsp.readFile(path.join(dir, f), 'utf8');
      await client.query('BEGIN');
      try {
        await client.query(sql);
        await client.query('INSERT INTO schema_migrations (name) VALUES ($1)', [f]);
        await client.query('COMMIT');
        logger?.info('Migration applied', { migration: f });
      } catch (err) {
        await client.query('ROLLBACK');
        throw err;
      }
    }
  } finally {
    await client.query('SELECT pg_advisory_unlock($1)', [MIGRATION_LOCK]).catch(() => {});
    client.release();
  }
}
