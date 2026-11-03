// Session auth for the console: scrypt passwords, HS256 JWT in an httpOnly
// SameSite=Strict cookie. Tokens embed a fingerprint of the password hash, so
// changing the password revokes every existing session.
import { randomBytes } from 'node:crypto';
import { config } from '../core/config.js';
import { LRUCache, sha1 } from '../core/util.js';
import { hashPassword, verifyPassword } from './password.js';
import { parseCookies, signJwt, verifyJwt } from './jwt.js';

const fingerprint = (hash) => sha1(hash).slice(0, 12);

export function createAuth({ repo, logger }) {
  const userCache = new LRUCache({ max: 100, ttlMs: 30_000 });
  const ttlSec = config.auth.sessionTtlHours * 3600;

  async function loadUser(id) {
    let u = userCache.get(id);
    if (!u) {
      u = await repo.findUserById(id);
      if (u) userCache.set(id, u);
    }
    return u;
  }

  return {
    async bootstrapAdmin() {
      if ((await repo.countUsers()) > 0) return;
      let password = config.auth.adminPassword;
      if (!password) {
        password = randomBytes(9).toString('base64url');
        logger.warn('No ADMIN_PASSWORD set — generated a one-time admin password. Change it in Settings.', {
          username: config.auth.adminUsername,
          password,
        });
      }
      await repo.createUser({ username: config.auth.adminUsername, passwordHash: await hashPassword(password) });
      logger.info('Admin user created', { username: config.auth.adminUsername });
    },

    async login(username, password) {
      const user = await repo.findUserByUsername(String(username || ''));
      // Verify against a dummy hash when the user is missing to keep timing uniform.
      const ok = await verifyPassword(password, user?.password_hash || 'scrypt$16384$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AAAA');
      if (!user || !ok) return null;
      await repo.touchLogin(user.id);
      const token = signJwt({ sub: user.id, pv: fingerprint(user.password_hash) }, config.auth.jwtSecret, { expiresInSec: ttlSec });
      return { token, user: publicUser(user) };
    },

    async changePassword(userId, current, next) {
      const user = await repo.findUserById(userId);
      if (!user || !(await verifyPassword(current, user.password_hash))) {
        const err = new Error('Current password is incorrect');
        err.status = 400;
        throw err;
      }
      if (typeof next !== 'string' || next.length < 10) {
        const err = new Error('New password must be at least 10 characters');
        err.status = 400;
        throw err;
      }
      const hash = await hashPassword(next);
      await repo.setPassword(userId, hash);
      userCache.clear();
      return signJwt({ sub: userId, pv: fingerprint(hash) }, config.auth.jwtSecret, { expiresInSec: ttlSec });
    },

    /** Resolve the user from a cookie header and/or Authorization header. */
    async authenticate({ cookie, authorization }) {
      const bearer = /^Bearer\s+(.+)$/i.exec(authorization || '')?.[1];
      const token = bearer || parseCookies(cookie)[config.auth.cookieName];
      const payload = verifyJwt(token, config.auth.jwtSecret);
      if (!payload) return null;
      const user = await loadUser(payload.sub);
      if (!user || fingerprint(user.password_hash) !== payload.pv) return null;
      return publicUser(user);
    },

    cookie(token) {
      const parts = [`${config.auth.cookieName}=${encodeURIComponent(token)}`, 'Path=/', 'HttpOnly', 'SameSite=Strict', `Max-Age=${ttlSec}`];
      if (config.auth.cookieSecure) parts.push('Secure');
      return parts.join('; ');
    },
    clearCookie() {
      return `${config.auth.cookieName}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0${config.auth.cookieSecure ? '; Secure' : ''}`;
    },
  };
}

const publicUser = (u) => ({ id: u.id, username: u.username, role: u.role });
