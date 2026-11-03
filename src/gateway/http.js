// REST API + static console. Contract documented in README.md.
import fsp from 'node:fs/promises';
import path from 'node:path';
import express from 'express';
import { config } from '../core/config.js';
import { channels, keys, recentLogs, withinRate } from '../core/redis.js';
import { catalogWithInstalled } from '../llm/catalog.js';

const CHAT_ID = /^[\w.\-:@]{3,120}$/;
const KB_FILE = /^[\w\-]{1,80}\.(md|txt)$/;

const httpError = (status, message) => Object.assign(new Error(message), { status });

export function createApp(deps) {
  const { auth, repo, redis, wa, inbox, settings, kb, llm, brain, memory, hub, logger } = deps;
  const log = logger.child({ component: 'http' });
  const app = express();
  app.disable('x-powered-by');
  if (config.http.trustProxy) app.set('trust proxy', 1);

  // ---------- security headers + same-origin guard ----------
  app.use((req, res, next) => {
    res.set({
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'Referrer-Policy': 'same-origin',
      'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
      'Content-Security-Policy':
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    });
    if (config.isProd) res.set('Strict-Transport-Security', 'max-age=31536000');
    if (!['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      const origin = req.get('origin');
      if (origin) {
        let host;
        try {
          host = new URL(origin).host;
        } catch {
          host = null;
        }
        const allowed = config.http.publicOrigin ? origin === config.http.publicOrigin : host === req.get('host');
        if (!allowed) return res.status(403).json({ error: 'Cross-origin request blocked' });
      }
    }
    next();
  });
  app.use(express.json({ limit: '1mb' }));

  // ---------- health (unauthenticated, used by Docker/Caddy/CI) ----------
  app.get('/healthz', (_req, res) => res.json({ ok: true }));
  app.get('/readyz', async (_req, res) => {
    const checks = await Promise.allSettled([repo.countUsers(), redis.ping()]);
    const ok = checks.every((c) => c.status === 'fulfilled');
    res.status(ok ? 200 : 503).json({ ok, wa: wa.state });
  });

  // ---------- auth ----------
  const requireAuth = async (req, res, next) => {
    const user = await auth.authenticate({ cookie: req.get('cookie'), authorization: req.get('authorization') });
    if (!user) return res.status(401).json({ error: 'Not signed in' });
    req.user = user;
    next();
  };
  const audit = (req, action, details) => repo.audit(req.user?.username, action, details).catch(() => {});

  app.post('/api/auth/login', async (req, res) => {
    const { username, password } = req.body || {};
    if (!(await withinRate(redis, 'login', req.ip, 10))) throw httpError(429, 'Too many login attempts — wait a minute');
    const result = await auth.login(username, password);
    if (!result) {
      log.warn('Failed login', { username: String(username || '').slice(0, 50), ip: req.ip });
      throw httpError(401, 'Invalid username or password');
    }
    res.set('Set-Cookie', auth.cookie(result.token));
    log.info('Login', { username: result.user.username, ip: req.ip });
    res.json({ user: result.user });
  });
  app.post('/api/auth/logout', (_req, res) => {
    res.set('Set-Cookie', auth.clearCookie());
    res.json({ ok: true });
  });

  const api = express.Router();
  api.use(requireAuth);

  api.get('/auth/me', (req, res) => res.json({ user: req.user }));
  api.post('/auth/password', async (req, res) => {
    const token = await auth.changePassword(req.user.id, req.body?.currentPassword, req.body?.newPassword);
    res.set('Set-Cookie', auth.cookie(token));
    audit(req, 'password.change');
    res.json({ ok: true });
  });

  // ---------- WhatsApp connection ----------
  api.get('/wa/status', (_req, res) => res.json(wa.status()));
  for (const action of ['connect', 'disconnect', 'logout', 'restart']) {
    api.post(`/wa/${action}`, async (req, res) => {
      audit(req, `wa.${action}`);
      log.info(`WhatsApp ${action} requested`, { by: req.user.username });
      const status = await wa[action]();
      res.json({ ok: true, state: status.state });
    });
  }

  // ---------- chats + messages ----------
  const chatParam = (req) => {
    const id = req.params.chatId;
    if (!CHAT_ID.test(id)) throw httpError(400, 'Invalid chat id');
    return id;
  };
  api.get('/chats', async (req, res) => {
    const chats = await repo.listChats({ search: String(req.query.search || ''), limit: Number(req.query.limit) || 100 });
    res.json({ chats });
  });
  api.get('/chats/:chatId/messages', async (req, res) => {
    const before = req.query.before ? Number(req.query.before) : undefined;
    const limit = Math.min(Number(req.query.limit) || 50, 200);
    res.json(await repo.listMessages(chatParam(req), { before, limit }));
  });
  api.post('/chats/:chatId/send', async (req, res) => {
    const chatId = chatParam(req);
    const body = String(req.body?.body || '').trim();
    if (!body || body.length > 4096) throw httpError(400, 'Message must be 1–4096 characters');
    const message = await inbox.sendManual(chatId, body, req.user.username);
    audit(req, 'message.send', { chatId });
    res.json({ message });
  });
  api.patch('/chats/:chatId', async (req, res) => {
    const chatId = chatParam(req);
    const patch = pickContactPatch(req.body, ['aiPaused', 'isBlocked', 'name']);
    const contact = await repo.updateContact({ chatId }, patch);
    if (!contact) throw httpError(404, 'Chat not found');
    if (patch.aiPaused === false) await redis.del(keys.takeover(chatId)); // resuming also ends an owner takeover
    audit(req, 'chat.update', { chatId, patch });
    const chat = await repo.getChat(chatId);
    hub.emit('chat:update', { chat });
    res.json({ chat });
  });
  api.post('/chats/:chatId/session', async (req, res) => {
    const chatId = chatParam(req);
    const action = req.body?.action;
    if (!['start', 'end'].includes(action)) throw httpError(400, 'action must be "start" or "end"');
    const chat = await inbox.consoleSession(chatId, action, { notify: req.body?.notify === true });
    if (!chat) throw httpError(404, 'Chat not found');
    audit(req, `session.${action}`, { chatId, notify: req.body?.notify === true });
    res.json({ chat });
  });
  api.get('/sessions', async (req, res) => {
    const sessionsList = await repo.listSessions({ active: req.query.active === 'true', limit: Number(req.query.limit) || 50 });
    res.json({ sessions: sessionsList });
  });
  api.get('/chats/:chatId/memory', async (req, res) => {
    const chatId = chatParam(req);
    const [longTerm, shortTerm] = await Promise.all([repo.getMemory(chatId), memory.get(chatId)]);
    res.json({ memory: longTerm.memory, updatedAt: longTerm.updatedAt, shortTerm });
  });
  api.put('/chats/:chatId/memory', async (req, res) => {
    const chatId = chatParam(req);
    const text = req.body?.memory;
    if (typeof text !== 'string' || text.length > 4000) throw httpError(400, 'memory must be a string under 4000 characters');
    await repo.setMemory(chatId, text.trim() || null);
    audit(req, 'memory.update', { chatId });
    hub.emit('chat:update', { chat: await repo.getChat(chatId) });
    res.json({ ok: true });
  });
  api.delete('/chats/:chatId/memory', async (req, res) => {
    const chatId = chatParam(req);
    await repo.setMemory(chatId, null);
    audit(req, 'memory.forget', { chatId });
    hub.emit('chat:update', { chat: await repo.getChat(chatId) });
    res.json({ ok: true });
  });
  api.delete('/chats/:chatId/context', async (req, res) => {
    const chatId = chatParam(req);
    await memory.clear(chatId);
    await redis.del(keys.takeover(chatId));
    audit(req, 'chat.clear_context', { chatId });
    res.json({ ok: true });
  });

  // ---------- contacts / whitelist ----------
  api.get('/contacts', async (_req, res) => res.json({ contacts: await repo.listContacts() }));
  api.patch('/contacts/:id', async (req, res) => {
    const patch = pickContactPatch(req.body, ['name', 'notes', 'isBlocked', 'aiPaused']);
    const contact = await repo.updateContact({ id: Number(req.params.id) }, patch);
    if (!contact) throw httpError(404, 'Contact not found');
    audit(req, 'contact.update', { id: contact.id, patch });
    if (contact.chatId) hub.emit('chat:update', { chat: await repo.getChat(contact.chatId) });
    res.json({ contact });
  });
  api.delete('/contacts/:id', async (req, res) => {
    if (!(await repo.deleteContact(Number(req.params.id)))) throw httpError(404, 'Contact not found');
    audit(req, 'contact.delete', { id: Number(req.params.id) });
    res.json({ ok: true });
  });

  // ---------- settings + models ----------
  api.get('/settings', (_req, res) => res.json({ settings: settings.get() }));
  api.put('/settings', async (req, res) => {
    const next = await settings.update(req.body || {});
    audit(req, 'settings.update', { keys: Object.keys(req.body || {}) });
    log.info('Settings updated', { by: req.user.username, keys: Object.keys(req.body || {}) });
    res.json({ settings: next });
  });
  api.get('/models', async (_req, res) => {
    const installed = await llm.listModels().catch(() => []);
    res.json({ current: settings.get().model, installed, catalog: catalogWithInstalled(installed) });
  });
  const pulling = new Set();
  api.post('/models/pull', async (req, res) => {
    const name = String(req.body?.name || '');
    if (!/^[\w.\-/:]{1,100}$/.test(name)) throw httpError(400, 'Invalid model name');
    if (pulling.has(name)) return res.json({ ok: true, already: true });
    pulling.add(name);
    audit(req, 'model.pull', { name });
    let lastEmit = 0;
    llm
      .pull(name, (ev) => {
        const now = Date.now();
        if (ev.status === 'success' || now - lastEmit > 400) {
          lastEmit = now;
          hub.emit('model:pull', { name, status: ev.status, completed: ev.completed ?? null, total: ev.total ?? null });
        }
      })
      .then(() => log.info('Model pulled', { name }))
      .catch((err) => {
        log.error('Model pull failed', { name, err });
        hub.emit('model:pull', { name, status: 'error', error: err.message });
      })
      .finally(() => pulling.delete(name));
    res.json({ ok: true });
  });

  // ---------- knowledge base ----------
  api.get('/kb', async (_req, res) => {
    const stats = kb.stats();
    const files = await Promise.all(
      stats.files.map(async (f) => {
        const st = await fsp.stat(path.join(kb.dir, f.name)).catch(() => null);
        return { ...f, bytes: st?.size ?? 0, updatedAt: st?.mtime ?? null };
      }),
    );
    res.json({ version: stats.version, sectionCount: stats.sectionCount, files });
  });
  const kbPath = (name) => {
    if (!KB_FILE.test(name)) throw httpError(400, 'File name must look like 11_topic.md');
    return path.join(kb.dir, name);
  };
  api.get('/kb/:name', async (req, res) => {
    const content = await fsp.readFile(kbPath(req.params.name), 'utf8').catch(() => {
      throw httpError(404, 'File not found');
    });
    res.json({ name: req.params.name, content });
  });
  api.put('/kb/:name', async (req, res) => {
    const file = kbPath(req.params.name);
    const content = req.body?.content;
    if (typeof content !== 'string' || content.length > 500_000) throw httpError(400, 'Content must be a string under 500 KB');
    const tmp = `${file}.tmp-${process.pid}`;
    await fsp.writeFile(tmp, content, 'utf8');
    await fsp.rename(tmp, file);
    await kb.load();
    await redis.publish(channels.kb, kb.version);
    audit(req, 'kb.save', { name: req.params.name, bytes: content.length });
    log.info('Knowledge base file saved', { name: req.params.name, by: req.user.username, version: kb.version });
    res.json({ ok: true, version: kb.version });
  });
  api.post('/kb/search', (req, res) => {
    const query = String(req.body?.query || '').slice(0, 500);
    const r = kb.search(query);
    res.json({
      tookMs: r.tookMs,
      topScore: r.topScore,
      expandedTerms: r.expandedTerms,
      results: r.sections.map((s) => ({ path: s.path, file: s.file, score: s.score, snippet: s.text.slice(0, 280) })),
    });
  });

  // ---------- playground (test the bot without WhatsApp) ----------
  api.post('/playground', async (req, res) => {
    const message = String(req.body?.message || '').trim().slice(0, 2000);
    if (!message) throw httpError(400, 'Message required');
    const history = (Array.isArray(req.body?.history) ? req.body.history : [])
      .filter((t) => t && ['user', 'assistant'].includes(t.role) && typeof t.content === 'string')
      .slice(-20)
      .map((t) => ({ role: t.role, content: t.content.slice(0, 4000) }));
    const bubbles = [];
    let meta = {};
    for await (const ev of brain.respond({ text: message, settings: settings.get(), history })) {
      if (ev.type === 'chunk') bubbles.push(ev.text);
      else if (ev.type === 'done') meta = ev.meta;
    }
    res.json({ reply: bubbles.join('\n\n'), bubbles, ...meta });
  });

  // ---------- logs + stats ----------
  api.get('/logs', async (req, res) => {
    const logs = await recentLogs(redis, {
      limit: Math.min(Number(req.query.limit) || 500, config.logs.bufferSize),
      level: req.query.level ? String(req.query.level) : undefined,
      service: req.query.service ? String(req.query.service) : undefined,
    });
    res.json({ logs });
  });
  api.get('/stats', async (_req, res) => {
    const [db, cacheHits] = await Promise.all([repo.stats(), redis.get(keys.cacheHits(new Date().toISOString().slice(0, 10)))]);
    res.json({ ...db, waState: wa.state, model: settings.get().model, cacheHits: Number(cacheHits) || 0, kbSections: kb.stats().sectionCount });
  });

  app.use('/api', api);
  app.use('/api', (_req, res) => res.status(404).json({ error: 'Not found' }));

  // ---------- static console ----------
  app.use(
    express.static(path.join(config.root, 'public'), {
      index: 'index.html',
      maxAge: config.isProd ? '5m' : 0,
      setHeaders: (res, file) => {
        if (file.endsWith('.html')) res.set('Cache-Control', 'no-cache');
      },
    }),
  );

  // ---------- errors ----------
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, _next) => {
    if (err.code === '23505') return res.status(409).json({ error: 'A contact with that phone or chat already exists' });
    if (err.type === 'entity.parse.failed') return res.status(400).json({ error: 'Invalid JSON body' });
    const status = err.status || err.statusCode || 500;
    if (status >= 500) log.error('Request failed', { method: req.method, path: req.path, err });
    res.status(status).json({ error: status >= 500 ? 'Internal server error' : err.message });
  });

  return app;
}

function str(v, max) {
  return typeof v === 'string' && v.trim() ? v.trim().slice(0, max) : null;
}

function pickContactPatch(body, allowed) {
  const patch = {};
  for (const k of allowed) {
    const v = body?.[k];
    if (v === undefined) continue;
    if (['aiPaused', 'isBlocked'].includes(k)) {
      if (typeof v !== 'boolean') throw httpError(400, `${k} must be true or false`);
      patch[k] = v;
    } else {
      patch[k] = str(v, k === 'notes' ? 500 : 80);
    }
  }
  return patch;
}
