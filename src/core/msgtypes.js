// WhatsApp message types that are not something a person wrote: encryption/security notices, group events,
// call logs, template notices, revoked placeholders. They must not be stored as chat messages or create a
// contact (they made empty chats appear in the console). Pure and dependency-free.
const SYSTEM_TYPES = new Set([
  'e2e_notification',
  'notification_template',
  'notification',
  'gp2',
  'protocol',
  'call_log',
  'revoked',
  'ciphertext',
  'broadcast_notification',
  'group_notification',
  'security',
  'peer_broadcast_notification',
]);

export function isSystemMessageType(type) {
  return SYSTEM_TYPES.has(String(type || '').toLowerCase());
}
