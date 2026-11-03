import { test } from 'node:test';
import assert from 'node:assert/strict';
import { hashPassword, verifyPassword } from '../src/auth/password.js';
import { parseCookies, signJwt, verifyJwt } from '../src/auth/jwt.js';

test('password hash round-trip', async () => {
  const h = await hashPassword('correct horse battery');
  assert.match(h, /^scrypt\$/);
  assert.equal(await verifyPassword('correct horse battery', h), true);
  assert.equal(await verifyPassword('wrong', h), false);
  assert.equal(await verifyPassword('x', 'garbage'), false);
});

test('short passwords are rejected', async () => {
  await assert.rejects(hashPassword('short'));
});

test('jwt sign/verify, tamper and expiry', () => {
  const t = signJwt({ sub: 1 }, 'secret', { expiresInSec: 60 });
  assert.equal(verifyJwt(t, 'secret').sub, 1);
  assert.equal(verifyJwt(t, 'other'), null);
  const [h, , s] = t.split('.');
  const forged = `${h}.${Buffer.from(JSON.stringify({ sub: 2, exp: 9e9 })).toString('base64url')}.${s}`;
  assert.equal(verifyJwt(forged, 'secret'), null);
  assert.equal(verifyJwt(signJwt({ sub: 1 }, 'secret', { expiresInSec: -1 }), 'secret'), null);
});

test('cookie parsing', () => {
  assert.deepEqual(parseCookies('a=1; wa_session=abc%20d; b'), { a: '1', wa_session: 'abc d' });
});
