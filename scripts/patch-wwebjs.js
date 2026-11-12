// Build-time patch for whatsapp-web.js 1.34.7: sending ANY media (images, so our invitation cards) fails with
// "Data passed to getter must include an id property (it's how we memoize) but got undefined".
//
// Cause (wwebjs/whatsapp-web.js#201922): in window.WWebJS.sendMessage the media model's private `__x_id` field is spread
// into the outgoing message object and replaces the real message key. The fix (one line, merged on their main branch in
// #201923 but not in any release yet) is `delete message.__x_id;` right after that object is built.
//
// Run from the image build after `npm ci` (see Dockerfile). Plain Node built-ins; safe to run twice. If a later
// whatsapp-web.js release already contains the fix, or its code moved, this reports it and leaves the file alone.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { patchSendMessage } from '../src/core/wwebjs-patch.js';

export { patchSendMessage };

export function patchFile(file) {
  if (!fs.existsSync(file)) return 'missing';
  const { src, status } = patchSendMessage(fs.readFileSync(file, 'utf8'));
  if (status === 'patched') fs.writeFileSync(file, src);
  return status;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = process.argv[2] || process.cwd();
  const file = path.join(root, 'node_modules', 'whatsapp-web.js', 'src', 'util', 'Injected', 'Utils.js');
  const status = patchFile(file);
  const say = {
    patched: 'patched: media sends work again (wwebjs#201922)',
    already: 'already contains the fix, nothing to do',
    missing: 'whatsapp-web.js is not installed here, skipped',
    notfound: 'WARNING: could not find the spot to patch (whatsapp-web.js changed?). Sending images may fail: check test-image in the console',
    ambiguous: 'WARNING: found more than one place to patch, left the file alone. Sending images may fail',
  }[status];
  console.log(`[patch-wwebjs] ${say}`);
}
