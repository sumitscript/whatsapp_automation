import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { patchFile, patchSendMessage } from '../scripts/patch-wwebjs.js';

// The shape of window.WWebJS.sendMessage in whatsapp-web.js 1.34.7 (from the upstream diff's context).
const SAMPLE = `exports.LoadUtils = () => {
    window.WWebJS.sendMessage = async (chat, content, options = {}) => {
        const message = {
            id: newMsgKey,
            ack: 0,
            body: content,
            from: meUser,
            to: chat.id,
            local: true,
            self: 'out',
            t: parseInt(new Date().getTime() / 1000),
            isNewMsg: true,
            type: 'chat',
            ...ephemeralFields,
            ...mediaOptions,
            ...(mediaOptions.toJSON ? mediaOptions.toJSON() : {}),
            ...quotedMsgOptions,
            ...locationOptions,
            ...vcardOptions,
            ...buttonOptions,
            ...listOptions,
            ...extraOptions,
        };

        // Bot's won't reply if canonicalUrl is set (linking)
        if (botOptions) {
            delete message.canonicalUrl;
        }
    };
};
`;

test('the media-send patch goes right after the outgoing message object, once', () => {
  const r = patchSendMessage(SAMPLE);
  assert.equal(r.status, 'patched');
  assert.match(r.src, /\.\.\.extraOptions,\n\s*\};\n\n\s*\/\/ wwebjs\/whatsapp-web\.js#201922[^\n]*\n\s*delete message\.__x_id;\n\n\s*\/\/ Bot's won't reply/);
  assert.equal((r.src.match(/delete message\.__x_id/g) || []).length, 1);
  // still valid JavaScript
  assert.doesNotThrow(() => new Function(r.src.replace('exports.LoadUtils', 'var x')));
});

test('running it twice, or on a release that already has the fix, changes nothing', () => {
  const once = patchSendMessage(SAMPLE).src;
  const again = patchSendMessage(once);
  assert.equal(again.status, 'already');
  assert.equal(again.src, once);
});

test('unknown code is left alone and reported', () => {
  assert.equal(patchSendMessage('exports.LoadUtils = () => {};').status, 'notfound');
  assert.equal(patchSendMessage(`${SAMPLE}\n${SAMPLE.replace('delete message.__x_id', '')}`).status, 'ambiguous');
});

test('patchFile edits the installed file in place and reports a missing install', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwebjs-'));
  try {
    const file = path.join(dir, 'Utils.js');
    assert.equal(patchFile(file), 'missing');
    fs.writeFileSync(file, SAMPLE);
    assert.equal(patchFile(file), 'patched');
    assert.match(fs.readFileSync(file, 'utf8'), /delete message\.__x_id;/);
    assert.equal(patchFile(file), 'already');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
