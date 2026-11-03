// Chat commands. Start/stop commands are configurable (default /hi and /bye)
// and case-insensitive: /hi, /Hi, /HI, "/hi!" and "/hi tell me about you" all start.

const BUILTIN = { '/reset': 'reset', '/help': 'help' };

const list = (v) =>
  String(v || '')
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);

/**
 * @returns {{ type: 'start'|'stop'|'reset'|'help', command: string, rest: string } | null}
 */
export function parseCommand(text, settings) {
  const m = /^\s*(\/[\p{L}\p{N}_-]+)[!.?,]*(?:\s+([\s\S]*))?$/u.exec(String(text ?? ''));
  if (!m) return null;
  const command = m[1].toLowerCase();
  const rest = (m[2] || '').trim();
  if (list(settings.startCommands).includes(command)) return { type: 'start', command, rest };
  if (list(settings.stopCommands).includes(command)) return { type: 'stop', command, rest };
  if (BUILTIN[command]) return { type: BUILTIN[command], command, rest };
  return null;
}

export function helpText(settings) {
  const first = (v) => list(v)[0] || '';
  return [
    `*${settings.personaName || 'eSumit'} commands*`,
    `${first(settings.startCommands) || '/hi'} — start chatting with me`,
    `${first(settings.stopCommands) || '/bye'} — end the chat`,
    '/reset — fresh start (I forget this conversation)',
    '/help — this list',
  ].join('\n');
}
