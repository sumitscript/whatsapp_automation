// Small helpers to keep API keys out of plain text at rest (AES-256-GCM, key derived from a server secret).
// Pure Node built-ins: unit-tested without any dependency.
import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'node:crypto';

const keyFrom = (secret) => createHash('sha256').update(`llm-keys|${secret}`).digest();

/** Encrypt text → "v1.<iv>.<tag>.<data>" (base64url parts). */
export function encryptSecret(plain, secret) {
  const iv = randomBytes(12);
  const c = createCipheriv('aes-256-gcm', keyFrom(secret), iv);
  const data = Buffer.concat([c.update(String(plain), 'utf8'), c.final()]);
  return ['v1', iv.toString('base64url'), c.getAuthTag().toString('base64url'), data.toString('base64url')].join('.');
}

/** Decrypt the output of encryptSecret. Returns null when the secret changed or the data was tampered with. */
export function decryptSecret(token, secret) {
  try {
    const [v, iv, tag, data] = String(token).split('.');
    if (v !== 'v1') return null;
    const d = createDecipheriv('aes-256-gcm', keyFrom(secret), Buffer.from(iv, 'base64url'));
    d.setAuthTag(Buffer.from(tag, 'base64url'));
    return Buffer.concat([d.update(Buffer.from(data, 'base64url')), d.final()]).toString('utf8');
  } catch {
    return null;
  }
}

/** "gsk_abcd…wxyz" → "gsk_…wxyz": enough to recognise a key, never enough to use it. */
export function maskKey(key) {
  const k = String(key || '');
  if (k.length <= 8) return '…';
  const prefix = k.includes('_') ? `${k.split('_')[0]}_` : '';
  return `${prefix}…${k.slice(-4)}`;
}
