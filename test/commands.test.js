import { test } from 'node:test';
import assert from 'node:assert/strict';
import { helpText, parseCommand } from '../src/brain/commands.js';
import { DEFAULT_SETTINGS, renderPersona, renderTemplate, sanitizeSettings } from '../src/core/settings-schema.js';

const s = DEFAULT_SETTINGS;

test('start/stop commands are case-insensitive and tolerate punctuation', () => {
  for (const t of ['/hi', '/Hi', '/HI', ' /hi ', '/hi!', '/hI.']) assert.equal(parseCommand(t, s)?.type, 'start', t);
  for (const t of ['/bye', '/Bye', '/BYE', '/bye!!']) assert.equal(parseCommand(t, s)?.type, 'stop', t);
});

test('text after the start command is kept', () => {
  assert.deepEqual(parseCommand('/hi tell me about your projects', s), { type: 'start', command: '/hi', rest: 'tell me about your projects' });
});

test('non-commands and unknown commands return null', () => {
  assert.equal(parseCommand('hi', s), null);
  assert.equal(parseCommand('hi /bye', s), null);
  assert.equal(parseCommand('/history', s), null);
  assert.equal(parseCommand('/hiya', s), null);
});

test('built-in /reset and /help, configurable command lists', () => {
  assert.equal(parseCommand('/RESET', s)?.type, 'reset');
  assert.equal(parseCommand('/help', s)?.type, 'help');
  const custom = { ...s, startCommands: '/hi, /hello, /start', stopCommands: '/bye, /stop' };
  assert.equal(parseCommand('/Hello', custom)?.type, 'start');
  assert.equal(parseCommand('/STOP', custom)?.type, 'stop');
  assert.match(helpText(custom), /\/hi — start/);
});

test('templates fill name, persona, owner and commands', () => {
  const t = renderTemplate('Hey {name}! {persona} = {owner}. Send {stop}, later {start}.', s, { name: 'Ravi Kumar' });
  assert.equal(t, 'Hey Ravi! eSumit = Sumit. Send /bye, later /hi.');
  assert.match(renderTemplate(s.welcomeMessage, s, {}), /^Hey there!/);
});

test('persona is first person and names the twin', () => {
  const p = renderPersona(s);
  assert.match(p, /You are eSumit — the virtual version of Sumit/);
  assert.doesNotMatch(p, /\{owner\}|\{persona\}/);
});

test('command settings are validated', () => {
  assert.deepEqual(sanitizeSettings({ startCommands: '/hi,/hello' }).value, { startCommands: '/hi, /hello' });
  assert.equal(sanitizeSettings({ startCommands: 'hi' }).errors.length, 1);
  assert.equal(sanitizeSettings({ replyMode: 'whitelist' }).value.replyMode, 'session'); // legacy value
});
