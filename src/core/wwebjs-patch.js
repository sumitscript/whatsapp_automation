// The one-line fix for whatsapp-web.js 1.34.7's media-send bug (wwebjs/whatsapp-web.js#201922), as a pure text transform.
//
// Bug: sending ANY media fails with "Data passed to getter must include an id property (it's how we memoize) but got
// undefined": in window.WWebJS.sendMessage the media model's private `__x_id` is spread into the outgoing message
// object and replaces the real message key. Upstream fix (merged on their main branch, in no release yet): delete that
// field right after the object is built.
//
// Used twice: at image build time on the installed file (scripts/patch-wwebjs.js) and at run time on the function
// source inside the WhatsApp page (WhatsAppManager.ensureMediaPatch), so it works whichever the server actually has.
export const MARK = 'delete message.__x_id';
// The end of the `const message = { ..., ...extraOptions, };` literal inside sendMessage.
const OBJECT_END = /(\.\.\.extraOptions,\s*\n\s*\};)/g;

/** @returns {{ src: string, status: 'patched'|'already'|'notfound'|'ambiguous' }} */
export function patchSendMessage(src) {
  if (src.includes(MARK)) return { src, status: 'already' };
  const hits = src.match(OBJECT_END) || [];
  if (hits.length !== 1) return { src, status: hits.length ? 'ambiguous' : 'notfound' };
  const patched = src.replace(
    OBJECT_END,
    "$1\n\n        // wwebjs/whatsapp-web.js#201922: the media model's private __x_id collides with the outgoing Msg's id\n        delete message.__x_id;",
  );
  return { src: patched, status: 'patched' };
}
