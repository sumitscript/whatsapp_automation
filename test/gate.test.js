import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { Semaphore, sleep } from '../src/core/util.js';
import { OllamaClient } from '../src/llm/ollama.js';

test('semaphore: still plain FIFO with no priorities, and bounded', async () => {
  const s = new Semaphore(1);
  const order = [];
  await s.acquire();
  const waits = [1, 2, 3].map((n) => s.acquire().then(() => { order.push(n); s.release(); }));
  s.release();
  await Promise.all(waits);
  assert.deepEqual(order, [1, 2, 3]);
  assert.equal(s.free, 1);
});

test('semaphore: cheaper live replies go before long ones, background and think-ahead go last', async () => {
  const s = new Semaphore(1);
  await s.acquire(); // the model is busy
  const order = [];
  const take = (name, priority) => s.acquire({ priority }).then(() => { order.push(name); s.release(); });
  const all = [take('think-ahead', 4), take('fold', 3), take('story', 2), take('hi-1', 1), take('hi-2', 1)];
  s.release();
  await Promise.all(all);
  assert.deepEqual(order, ['hi-1', 'hi-2', 'story', 'fold', 'think-ahead']);
});

test('semaphore: a long job that has waited long enough is not starved by newer short ones', async () => {
  const s = new Semaphore(1, { agingMs: 20 });
  await s.acquire();
  const order = [];
  const take = (name, priority) => s.acquire({ priority }).then(() => { order.push(name); s.release(); });
  const story = take('story', 2);
  await sleep(70); // the story has now aged past two priority steps
  const hi = take('hi', 1);
  s.release();
  await Promise.all([story, hi]);
  assert.deepEqual(order, ['story', 'hi']);
});

test('semaphore: a waiter whose signal aborts leaves the queue and rejects; a slot is never lost', async () => {
  const s = new Semaphore(1);
  await s.acquire();
  const ctl = new AbortController();
  const w = s.acquire({ priority: 4, signal: ctl.signal });
  assert.equal(s.waiting, 1);
  ctl.abort(new Error('superseded'));
  await assert.rejects(w, /superseded/);
  assert.equal(s.waiting, 0);
  s.release();
  assert.equal(s.free, 1);
  await assert.rejects(s.acquire({ signal: ctl.signal }), /superseded/, 'an already-aborted signal never waits');
  assert.equal(s.free, 1);
});

test('semaphore: run() releases on error', async () => {
  const s = new Semaphore(1);
  await assert.rejects(s.run(async () => { throw new Error('boom'); }), /boom/);
  assert.equal(s.free, 1);
});

// A streamed reply may take longer than the request timeout as long as data keeps arriving.
function ndjsonServer(chunks) {
  return new Promise((resolve) => {
    const server = http.createServer(async (req, res) => {
      res.writeHead(200, { 'content-type': 'application/x-ndjson' });
      for (const [delay, line] of chunks) {
        await sleep(delay);
        if (res.destroyed) return;
        res.write(`${JSON.stringify(line)}\n`);
      }
      res.end();
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${server.address().port}` }));
  });
}

test('ollama: the timeout is an idle timeout while streaming, so a slow long reply is not cut off', async () => {
  const lines = [...'abcdefg'].map((c) => [60, { message: { content: c } }]);
  lines.push([60, { done: true, done_reason: 'stop', eval_count: 7 }]);
  const { server, url } = await ndjsonServer(lines);
  try {
    const c = new OllamaClient({ url, timeoutMs: 200 }); // the whole reply takes ~480 ms > 200 ms
    let text = '';
    let done;
    for await (const ev of c.chat({ model: 'x', messages: [{ role: 'user', content: 'hi' }] })) {
      if (ev.type === 'token') text += ev.text;
      else done = ev;
    }
    assert.equal(text, 'abcdefg');
    assert.equal(done.stats.doneReason, 'stop');
  } finally {
    server.closeAllConnections?.();
    server.close();
  }
});

test('ollama: a stalled stream still times out', async () => {
  const { server, url } = await ndjsonServer([[10, { message: { content: 'a' } }], [2000, { done: true }]]);
  try {
    const c = new OllamaClient({ url, timeoutMs: 150 });
    await assert.rejects(async () => {
      for await (const ev of c.chat({ model: 'x', messages: [{ role: 'user', content: 'hi' }] })) void ev;
    }, /no data for|aborted|timed out/i);
  } finally {
    server.closeAllConnections?.();
    server.close();
  }
});

import { deleteKeysMatching } from '../src/core/util.js';

test('deleteKeysMatching: scans in batches, unlinks what matches, returns the count', async () => {
  const store = new Set(['p:ctx:a', 'p:ctx:b', 'p:ctx:c', 'p:keep:x']);
  const unlinked = [];
  const redis = {
    scanStream({ match }) {
      const re = new RegExp(`^${match.replace('*', '.*')}$`);
      const hits = [...store].filter((k) => re.test(k));
      return (async function* () { yield hits.slice(0, 2); yield []; yield hits.slice(2); })();
    },
    async unlink(...ks) { unlinked.push(...ks); ks.forEach((k) => store.delete(k)); },
  };
  assert.equal(await deleteKeysMatching(redis, 'p:ctx:*'), 3);
  assert.deepEqual(unlinked.sort(), ['p:ctx:a', 'p:ctx:b', 'p:ctx:c']);
  assert.deepEqual([...store], ['p:keep:x'], 'unrelated keys are untouched');
  assert.equal(await deleteKeysMatching(redis, 'p:none:*'), 0);
});
