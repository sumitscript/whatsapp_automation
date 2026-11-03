// Create or reset a console user:  npm run create-admin -- <username> <password>
import { hashPassword } from '../src/auth/password.js';
import { createPool, migrate } from '../src/db/pg.js';

const [username, password] = process.argv.slice(2);
if (!username || !password || password.length < 10) {
  console.error('usage: npm run create-admin -- <username> <password (10+ chars)>');
  process.exit(1);
}
const pool = createPool();
await migrate(pool);
const hash = await hashPassword(password);
await pool.query(
  `INSERT INTO users (username, password_hash) VALUES ($1, $2)
   ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash`,
  [username, hash],
);
console.log(`User "${username}" saved. Existing sessions for this user are revoked.`);
await pool.end();
