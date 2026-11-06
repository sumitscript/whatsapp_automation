import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MessageChunker, toWhatsApp } from '../src/brain/format.js';

test('markdown converts to WhatsApp formatting', () => {
  assert.equal(toWhatsApp('## Title\n**bold** and ~~gone~~\n- item'), '*Title*\n*bold* and ~gone~\n• item');
  assert.equal(toWhatsApp('see [site](https://x.dev)'), 'see site (https://x.dev)');
});

test('code fences are left alone (language tag dropped)', () => {
  assert.equal(toWhatsApp('```js\nconst **a** = 1;\n```'), '```\nconst **a** = 1;\n```');
});

test('chunker splits at paragraph breaks', () => {
  const c = new MessageChunker({ minChars: 10 });
  const out = [...c.push('First paragraph is here.\n\nSecond one'), ...c.push(' continues.'), ...c.flush()];
  assert.deepEqual(out, ['First paragraph is here.', 'Second one continues.']);
});

test('chunker never splits inside a code block', () => {
  const c = new MessageChunker({ minChars: 5 });
  const text = 'Here you go:\n\n```\nline one\n\nline two\n```\n\nDone now.';
  const out = [...c.push(text), ...c.flush()];
  assert.equal(out.length, 2);
  assert.match(out[0], /line one\n\nline two/);
  assert.equal(out[1], 'Done now.');
});

test('chunker caps the number of bubbles', () => {
  const c = new MessageChunker({ minChars: 1, maxBubbles: 2 });
  const out = [...c.push('a1 a1\n\nb2 b2\n\nc3 c3\n\n'), ...c.flush()];
  assert.equal(out.length, 2);
  assert.equal(out[1], 'b2 b2\n\nc3 c3');
});

test('split=false yields a single bubble', () => {
  const c = new MessageChunker({ split: false });
  assert.deepEqual(c.push('one\n\ntwo'), []);
  assert.deepEqual(c.flush(), ['one\n\ntwo']);
});
