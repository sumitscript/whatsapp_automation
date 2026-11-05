import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { OllamaClient } from '../src/llm/ollama.js';

// A fake Ollama: DELETE /api/delete removes a model, 404 when it isn't installed.
function fakeOllama(installed) {
  const calls = [];
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      if (req.method === 'DELETE' && req.url === '/api/delete') {
        const { model } = JSON.parse(body || '{}');
        calls.push(model);
        if (!installed.has(model)) return res.writeHead(404).end(JSON.stringify({ error: `model '${model}' not found` }));
        installed.delete(model);
        return res.writeHead(200).end();
      }
      res.writeHead(404).end();
    });
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, calls, url: `http://127.0.0.1:${server.address().port}` })));
}

test('deleteModel removes an installed model and reports false for a missing one', async () => {
  const installed = new Set(['openchat:latest', 'gemma3:4b']);
  const { server, calls, url } = await fakeOllama(installed);
  try {
    const llm = new OllamaClient({ url, timeoutMs: 2000 });
    assert.equal(await llm.deleteModel('openchat:latest'), true);
    assert.equal(installed.has('openchat:latest'), false);
    assert.equal(installed.has('gemma3:4b'), true);
    assert.equal(await llm.deleteModel('nope:1b'), false);
    assert.deepEqual(calls, ['openchat:latest', 'nope:1b']);
  } finally {
    server.close();
  }
});

test('deleteModel surfaces server errors other than 404', async () => {
  const server = http.createServer((_req, res) => res.writeHead(500).end('boom'));
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  try {
    const llm = new OllamaClient({ url: `http://127.0.0.1:${server.address().port}`, timeoutMs: 2000 });
    await assert.rejects(() => llm.deleteModel('x:1'), /500/);
  } finally {
    server.close();
  }
});
