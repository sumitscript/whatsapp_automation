// Data access. Rows are mapped to the camelCase shapes the UI consumes.

const toSession = (r) =>
  r && {
    id: r.id,
    chatId: r.chat_id,
    active: !r.ended_at,
    startedAt: r.started_at,
    lastActivityAt: r.last_activity_at,
    endedAt: r.ended_at ?? null,
    endedReason: r.ended_reason ?? null,
    messageCount: r.message_count,
  };

const toContact = (r) =>
  r && {
    id: r.id,
    chatId: r.chat_id,
    phone: r.phone,
    name: r.name,
    isGroup: r.is_group,
    isBlocked: r.is_blocked,
    aiPaused: r.ai_paused,
    notes: r.notes,
    hasMemory: !!r.memory,
    memoryUpdatedAt: r.memory_updated_at ?? null,
    lastMessageAt: r.last_message_at,
    createdAt: r.created_at,
    session: r.s_id
      ? { id: r.s_id, active: true, startedAt: r.s_started_at, lastActivityAt: r.s_last_activity_at, messageCount: r.s_message_count }
      : null,
  };

const toMessage = (r) =>
  r && {
    id: r.id,
    waMessageId: r.wa_message_id,
    chatId: r.chat_id,
    direction: r.direction,
    body: r.body,
    type: r.type,
    isAi: r.is_ai,
    sentBy: r.sent_by,
    model: r.model,
    intent: r.intent,
    language: r.language,
    latencyMs: r.latency_ms,
    cached: r.cached,
    sources: r.sources,
    createdAt: r.created_at,
  };

const toAgenda = (r) =>
  r && {
    id: Number(r.id),
    chatId: r.chat_id,
    contactName: r.contact_name ?? null,
    phone: r.contact_phone ?? null,
    kind: r.kind,
    title: r.title,
    details: r.details,
    dueAt: r.due_at,
    allDay: r.all_day,
    dueText: r.due_text,
    location: r.location,
    status: r.status,
    source: r.source,
    sourceMessage: r.source_message,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  };

const AGENDA_SELECT = `SELECT a.*, c.name AS contact_name, c.phone AS contact_phone
  FROM agenda_items a LEFT JOIN contacts c ON c.chat_id = a.chat_id`;
const AGENDA_FIELDS = {
  kind: 'kind', title: 'title', details: 'details', dueAt: 'due_at', allDay: 'all_day', dueText: 'due_text',
  location: 'location', status: 'status', chatId: 'chat_id',
};

const toChat = (r) => ({
  ...toContact(r),
  lastMessage: r.last_body ?? null,
  lastDirection: r.last_direction ?? null,
  lastMessageAt: r.last_at ?? r.last_message_at,
  messageCount: r.message_count ?? 0,
});

// Contact columns + its open session (if any).
const CONTACT_COLS =
  'c.*, s.id AS s_id, s.started_at AS s_started_at, s.last_activity_at AS s_last_activity_at, s.message_count AS s_message_count';
const CONTACT_FROM = 'FROM contacts c LEFT JOIN sessions s ON s.chat_id = c.chat_id AND s.ended_at IS NULL';
const CONTACT_SELECT = `${CONTACT_COLS} ${CONTACT_FROM}`;

const CHAT_SELECT = `
  SELECT ${CONTACT_COLS}, m.body AS last_body, m.direction AS last_direction, m.created_at AS last_at,
         (SELECT count(*)::int FROM messages x WHERE x.chat_id = c.chat_id) AS message_count
    ${CONTACT_FROM}
    LEFT JOIN LATERAL (SELECT body, direction, created_at FROM messages WHERE chat_id = c.chat_id ORDER BY id DESC LIMIT 1) m ON true`;

const CONTACT_FIELDS = {
  name: 'name',
  notes: 'notes',
  isBlocked: 'is_blocked',
  aiPaused: 'ai_paused',
};

export function createRepo(pool) {
  const q = (text, params) => pool.query(text, params);

  const repo = {
    // ---------- users ----------
    async countUsers() {
      return (await q('SELECT count(*)::int AS n FROM users')).rows[0].n;
    },
    async findUserByUsername(username) {
      return (await q('SELECT * FROM users WHERE lower(username) = lower($1)', [username])).rows[0] || null;
    },
    async findUserById(id) {
      return (await q('SELECT * FROM users WHERE id = $1', [id])).rows[0] || null;
    },
    async createUser({ username, passwordHash, role = 'admin' }) {
      return (await q('INSERT INTO users (username, password_hash, role) VALUES ($1, $2, $3) RETURNING *', [username, passwordHash, role])).rows[0];
    },
    async setPassword(id, passwordHash) {
      await q('UPDATE users SET password_hash = $2 WHERE id = $1', [id, passwordHash]);
    },
    async touchLogin(id) {
      await q('UPDATE users SET last_login_at = now() WHERE id = $1', [id]);
    },

    // ---------- contacts ----------
    /** Find or create the contact row for an inbound chat. */
    async upsertContactForChat({ chatId, phone, name, isGroup = false }) {
      const { rows } = await q(
        `INSERT INTO contacts (chat_id, phone, name, is_group, last_message_at)
         VALUES ($1, $2, $3, $4, now())
         ON CONFLICT (chat_id) DO UPDATE SET
           name = COALESCE(contacts.name, EXCLUDED.name),
           phone = COALESCE(contacts.phone, EXCLUDED.phone),
           last_message_at = now(), updated_at = now()
         RETURNING id`,
        [chatId, phone, name, isGroup],
      ).catch(async (err) => {
        // Same phone already linked to another chat id (e.g. @c.us vs @lid): store without phone.
        if (err.code !== '23505') throw err;
        return q(
          `INSERT INTO contacts (chat_id, name, is_group, last_message_at) VALUES ($1, $2, $3, now())
           ON CONFLICT (chat_id) DO UPDATE SET last_message_at = now() RETURNING id`,
          [chatId, name, isGroup],
        );
      });
      return repo.getContactById(rows[0].id);
    },
    async getContactById(id) {
      return toContact((await q(`SELECT ${CONTACT_SELECT} WHERE c.id = $1`, [id])).rows[0]);
    },
    async getContactByChatId(chatId) {
      return toContact((await q(`SELECT ${CONTACT_SELECT} WHERE c.chat_id = $1`, [chatId])).rows[0]);
    },
    async listContacts() {
      return (await q(`SELECT ${CONTACT_SELECT} ORDER BY (s.id IS NOT NULL) DESC, c.last_message_at DESC NULLS LAST, c.id DESC`)).rows.map(toContact);
    },
    async updateContact(where, patch) {
      const sets = [];
      const params = [];
      for (const [k, col] of Object.entries(CONTACT_FIELDS)) {
        if (patch[k] === undefined) continue;
        params.push(patch[k]);
        sets.push(`${col} = $${params.length}`);
      }
      const col = where.id ? 'id' : 'chat_id';
      const key = where.id ?? where.chatId;
      if (sets.length) {
        params.push(key);
        await q(`UPDATE contacts SET ${sets.join(', ')}, updated_at = now() WHERE ${col} = $${params.length}`, params);
      }
      return where.id ? repo.getContactById(where.id) : repo.getContactByChatId(where.chatId);
    },
    async deleteContact(id) {
      return (await q('DELETE FROM contacts WHERE id = $1 RETURNING id', [id])).rows[0] ? true : false;
    },

    // ---------- long-term memory ----------
    async getMemory(chatId) {
      const r = (await q('SELECT memory, memory_updated_at, name FROM contacts WHERE chat_id = $1', [chatId])).rows[0];
      return r ? { memory: r.memory, updatedAt: r.memory_updated_at, name: r.name } : { memory: null, updatedAt: null, name: null };
    },
    async setMemory(chatId, memory) {
      await q('UPDATE contacts SET memory = $2, memory_updated_at = now(), updated_at = now() WHERE chat_id = $1', [chatId, memory || null]);
    },

    // ---------- sessions ----------
    async getOpenSession(chatId) {
      return toSession((await q('SELECT * FROM sessions WHERE chat_id = $1 AND ended_at IS NULL', [chatId])).rows[0]) || null;
    },
    async startSession(chatId) {
      await q(`INSERT INTO sessions (chat_id) VALUES ($1) ON CONFLICT (chat_id) WHERE ended_at IS NULL DO NOTHING`, [chatId]);
      return repo.getOpenSession(chatId);
    },
    async touchSession(id) {
      await q('UPDATE sessions SET last_activity_at = now(), message_count = message_count + 1 WHERE id = $1', [id]);
    },
    async endSession(chatId, reason) {
      const { rows } = await q(
        `UPDATE sessions SET ended_at = now(), ended_reason = $2 WHERE chat_id = $1 AND ended_at IS NULL RETURNING *`,
        [chatId, reason],
      );
      return toSession(rows[0]) || null;
    },
    async idleSessionChats(minutes) {
      const { rows } = await q(
        `SELECT chat_id FROM sessions WHERE ended_at IS NULL AND last_activity_at < now() - make_interval(mins => $1)`,
        [minutes],
      );
      return rows.map((r) => r.chat_id);
    },
    async listSessions({ active = false, limit = 50 } = {}) {
      const { rows } = await q(
        `SELECT s.*, c.name, c.phone FROM sessions s LEFT JOIN contacts c ON c.chat_id = s.chat_id
          WHERE ($1::boolean = false OR s.ended_at IS NULL)
          ORDER BY s.started_at DESC LIMIT $2`,
        [active, Math.min(limit, 500)],
      );
      return rows.map((r) => ({ ...toSession(r), name: r.name, phone: r.phone }));
    },

    // ---------- messages ----------
    async insertMessage(m) {
      const { rows } = await q(
        `INSERT INTO messages (wa_message_id, chat_id, direction, body, type, is_ai, sent_by, model, intent, language, latency_ms, cached, sources)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
         ON CONFLICT (wa_message_id) DO NOTHING RETURNING *`,
        [
          m.waMessageId ?? null,
          m.chatId,
          m.direction,
          m.body ?? '',
          m.type ?? 'chat',
          m.isAi ?? false,
          m.sentBy ?? null,
          m.model ?? null,
          m.intent ?? null,
          m.language ?? null,
          m.latencyMs ?? null,
          m.cached ?? false,
          m.sources ? JSON.stringify(m.sources) : null,
        ],
      );
      if (rows[0]) await q('UPDATE contacts SET last_message_at = now() WHERE chat_id = $1', [m.chatId]);
      return toMessage(rows[0]) || null; // null = duplicate
    },
    async listMessages(chatId, { before, limit = 50 } = {}) {
      const params = [chatId, Math.min(limit, 200) + 1];
      let where = 'chat_id = $1';
      if (before) {
        params.push(before);
        where += ' AND id < $3';
      }
      const { rows } = await q(`SELECT * FROM messages WHERE ${where} ORDER BY id DESC LIMIT $2`, params);
      const hasMore = rows.length > limit;
      return { messages: rows.slice(0, limit).reverse().map(toMessage), hasMore };
    },
    async listChats({ search = '', limit = 100 } = {}) {
      const { rows } = await q(
        `${CHAT_SELECT}
          WHERE c.chat_id IS NOT NULL
            AND ($1 = '' OR c.name ILIKE '%' || $1 || '%' OR c.phone ILIKE '%' || $1 || '%' OR c.chat_id ILIKE '%' || $1 || '%')
          ORDER BY COALESCE(m.created_at, c.last_message_at) DESC NULLS LAST
          LIMIT $2`,
        [search.trim(), Math.min(limit, 500)],
      );
      return rows.map(toChat);
    },
    async getChat(chatId) {
      const { rows } = await q(`${CHAT_SELECT} WHERE c.chat_id = $1`, [chatId]);
      return rows[0] ? toChat(rows[0]) : null;
    },

    // ---------- settings ----------
    async getSettings() {
      const { rows } = await q('SELECT key, value FROM settings');
      return Object.fromEntries(rows.map((r) => [r.key, r.value]));
    },
    async saveSettings(patch) {
      const client = await pool.connect();
      try {
        await client.query('BEGIN');
        for (const [k, v] of Object.entries(patch)) {
          await client.query(
            `INSERT INTO settings (key, value) VALUES ($1, $2::jsonb)
             ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()`,
            [k, JSON.stringify(v)],
          );
        }
        await client.query('COMMIT');
      } catch (err) {
        await client.query('ROLLBACK').catch(() => {});
        throw err;
      } finally {
        client.release();
      }
    },

    // ---------- agenda ----------
    /** @param {{ view?: 'review'|'open'|'done'|'all', kind?: string, chatId?: string, limit?: number }} f */
    async listAgenda({ view = 'all', kind, chatId, limit = 300 } = {}) {
      const where = [];
      const params = [];
      if (view === 'review') where.push("a.kind = 'appointment' AND a.status = 'pending'");
      else if (view === 'open') where.push("a.status IN ('pending', 'confirmed')");
      else if (view === 'done') where.push("a.status IN ('done', 'declined')");
      if (kind) where.push(`a.kind = $${params.push(kind)}`);
      if (chatId) where.push(`a.chat_id = $${params.push(chatId)}`);
      const order = view === 'done' ? 'a.updated_at DESC' : 'a.due_at ASC NULLS LAST, a.created_at DESC';
      params.push(Math.min(Number(limit) || 300, 1000));
      const { rows } = await q(`${AGENDA_SELECT} ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY ${order} LIMIT $${params.length}`, params);
      return rows.map(toAgenda);
    },
    async getAgendaItem(id) {
      return toAgenda((await q(`${AGENDA_SELECT} WHERE a.id = $1`, [id])).rows[0]) || null;
    },
    async createAgendaItem(it) {
      const { rows } = await q(
        `INSERT INTO agenda_items (chat_id, kind, title, details, due_at, all_day, due_text, location, status, source, source_message)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING id`,
        [it.chatId ?? null, it.kind, it.title, it.details ?? null, it.dueAt ?? null, !!it.allDay, it.dueText ?? null, it.location ?? null, it.status || 'pending', it.source || 'ai', it.sourceMessage ?? null],
      );
      return repo.getAgendaItem(rows[0].id);
    },
    async updateAgendaItem(id, patch) {
      const sets = [];
      const params = [];
      for (const [k, col] of Object.entries(AGENDA_FIELDS)) {
        if (patch[k] === undefined) continue;
        sets.push(`${col} = $${params.push(patch[k])}`);
      }
      if (!sets.length) return repo.getAgendaItem(id);
      params.push(id);
      const { rowCount } = await q(`UPDATE agenda_items SET ${sets.join(', ')}, updated_at = now() WHERE id = $${params.length}`, params);
      return rowCount ? repo.getAgendaItem(id) : null;
    },
    async deleteAgendaItem(id) {
      return (await q('DELETE FROM agenda_items WHERE id = $1', [id])).rowCount > 0;
    },

    // ---------- audit + stats ----------
    async audit(username, action, details = null) {
      await q('INSERT INTO audit_log (username, action, details) VALUES ($1, $2, $3)', [username, action, details ? JSON.stringify(details) : null]);
    },
    async stats() {
      const [today, latency, active, contacts, sessions, hourly] = await Promise.all([
        q(`SELECT count(*)::int AS total,
                  count(*) FILTER (WHERE direction = 'in')::int AS inbound,
                  count(*) FILTER (WHERE is_ai)::int AS ai
             FROM messages WHERE created_at >= date_trunc('day', now())`),
        q(`SELECT avg(latency_ms)::int AS avg,
                  percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::int AS p95
             FROM messages WHERE is_ai AND latency_ms IS NOT NULL AND created_at >= now() - interval '24 hours'`),
        q(`SELECT count(DISTINCT chat_id)::int AS n FROM messages WHERE created_at >= now() - interval '24 hours'`),
        q(`SELECT count(*)::int AS total FROM contacts`),
        q(`SELECT count(*) FILTER (WHERE ended_at IS NULL)::int AS active,
                  count(*) FILTER (WHERE started_at >= date_trunc('day', now()))::int AS today
             FROM sessions`),
        q(`SELECT date_trunc('hour', created_at) AS hour,
                  count(*) FILTER (WHERE direction = 'in')::int AS inbound,
                  count(*) FILTER (WHERE direction = 'out')::int AS outbound
             FROM messages WHERE created_at >= now() - interval '24 hours'
            GROUP BY 1 ORDER BY 1`),
      ]);
      return {
        messagesToday: today.rows[0].total,
        inboundToday: today.rows[0].inbound,
        aiRepliesToday: today.rows[0].ai,
        avgLatencyMs: latency.rows[0].avg ?? 0,
        p95LatencyMs: latency.rows[0].p95 ?? 0,
        activeChats24h: active.rows[0].n,
        contacts: contacts.rows[0].total,
        activeSessions: sessions.rows[0].active,
        sessionsToday: sessions.rows[0].today,
        hourly: hourly.rows.map((r) => ({ hour: r.hour, inbound: r.inbound, outbound: r.outbound })),
      };
    },
  };
  return repo;
}
