import test from 'node:test';
import assert from 'node:assert/strict';
import { isSystemMessageType } from '../src/core/msgtypes.js';

test('system notices are recognised, so they never become messages or phantom chats', () => {
  for (const t of ['e2e_notification', 'notification_template', 'notification', 'gp2', 'protocol', 'call_log', 'revoked', 'ciphertext', 'broadcast_notification', 'group_notification', 'security', 'peer_broadcast_notification', 'E2E_NOTIFICATION']) {
    assert.equal(isSystemMessageType(t), true, t);
  }
});

test('anything a person sends is kept, including types we cannot read (they get the "can\'t open that" reply)', () => {
  for (const t of ['chat', 'text', 'image', 'video', 'audio', 'ptt', 'sticker', 'document', 'location', 'vcard', 'multi_vcard', 'poll_creation', 'unknown', undefined, null, '']) {
    assert.equal(isSystemMessageType(t), false, String(t));
  }
});
