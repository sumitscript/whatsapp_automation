// Contacts: session state, memory indicator, block/pause toggles, notes.
import { api } from '../api.js';
import { icons } from '../icons.js';
import { $, confirmDialog, debounce, displayPhone, formatTime, hueFor, html, initials, raw, relativeTime, toast } from '../util.js';

const MODE_TEXT = {
  session: 'eSumit replies only while a person has an active AI session.',
  all: 'eSumit replies to everyone except blocked contacts.',
  off: 'AI replies are turned off for everyone.',
};

const firstCmd = (list, fallback) => String(list || fallback).split(',').map((x) => x.trim()).filter(Boolean)[0] || fallback;

export async function mount(root, { hub }) {
  const s = { contacts: [], filter: 'all', q: '', settings: null };

  root.innerHTML = html`
    <div class="page">
      <header class="page-header">
        <div>
          <h1>Contacts</h1>
          <p class="muted" id="session-help">Loading…</p>
          <p class="muted small" id="mode-line"></p>
        </div>
      </header>

      <section class="card">
        <div class="toolbar">
          <div class="chip-group" role="radiogroup" aria-label="Filter contacts">
            <button class="chip active" data-filter="all" role="radio" aria-checked="true">All</button>
            <button class="chip" data-filter="session" role="radio" aria-checked="false">Active session</button>
            <button class="chip" data-filter="blocked" role="radio" aria-checked="false">Blocked</button>
            <button class="chip" data-filter="paused" role="radio" aria-checked="false">AI paused</button>
          </div>
          <div class="search-bar compact">
            ${raw(icons.search)}
            <input type="search" id="contact-q" placeholder="Search" aria-label="Search contacts" />
          </div>
        </div>
        <div id="contact-table" class="contact-table"></div>
      </section>
    </div>`;

  const tableEl = $('#contact-table', root);

  function filtered() {
    const q = s.q.toLowerCase();
    return s.contacts.filter((c) => {
      if (s.filter === 'session' && !c.session?.active) return false;
      if (s.filter === 'blocked' && !c.isBlocked) return false;
      if (s.filter === 'paused' && !c.aiPaused) return false;
      if (!q) return true;
      return [c.name, c.phone, c.notes, c.chatId].some((v) => String(v || '').toLowerCase().includes(q));
    });
  }

  const toggle = (c, key, label) =>
    html`<label class="switch-cell" title="${label}">
      <input type="checkbox" class="switch" data-key="${key}" ${raw(c[key] ? 'checked' : '')} aria-label="${label} for ${c.name || c.phone || 'contact'}" />
    </label>`;

  const sessionCell = (c) =>
    c.session?.active
      ? html`<span class="badge badge-ok badge-dot" title="${c.session.messageCount ?? 0} messages">AI on · ${formatTime(c.session.startedAt)}</span>`
      : '<span class="muted small">—</span>';

  function render() {
    const rows = filtered();
    if (!rows.length) {
      tableEl.innerHTML = '<div class="list-empty">No contacts here yet. People appear once they message you.</div>';
      return;
    }
    tableEl.innerHTML = html`
      <div class="ct-row ct-head" role="row">
        <span>Contact</span><span>Notes</span><span>Session</span><span>AI paused</span><span>Blocked</span><span>Last message</span><span></span>
      </div>
      ${rows.map((c) =>
        raw(html`
          <div class="ct-row" role="row" data-id="${c.id}">
            <span class="ct-contact">
              <span class="avatar avatar-sm" style="--h:${hueFor(c.chatId || c.phone)}">${initials(c.name || c.phone)}</span>
              <span>
                <span class="ct-name">${c.name || displayPhone(c.phone) || c.chatId}${raw(c.hasMemory ? ' <span class="badge" title="Has long-term memory">🧠</span>' : '')}</span>
                <span class="ct-phone muted">${displayPhone(c.phone) || c.chatId || ''}</span>
              </span>
            </span>
            <span class="ct-notes"><input class="inline-input" data-key="notes" value="${c.notes || ''}" placeholder="Add note" aria-label="Notes" /></span>
            <span data-label="Session">${raw(sessionCell(c))}</span>
            <span data-label="AI paused">${raw(toggle(c, 'aiPaused', 'AI paused'))}</span>
            <span data-label="Blocked">${raw(toggle(c, 'isBlocked', 'Blocked'))}</span>
            <span class="muted ct-last">${c.lastMessageAt ? relativeTime(c.lastMessageAt) : '—'}</span>
            <span class="ct-actions">
              ${raw(c.chatId ? html`<a class="icon-btn" href="#/chats/${encodeURIComponent(c.chatId)}" aria-label="Open chat" title="Open chat">${raw(icons.chats)}</a>` : '')}
              <button class="icon-btn danger" data-act="delete" aria-label="Delete contact" title="Delete">${raw(icons.trash)}</button>
            </span>
          </div>`)
      )}`;
  }

  function renderHelp() {
    const st = s.settings || {};
    const start = firstCmd(st.startCommands, '/hi');
    const stop = firstCmd(st.stopCommands, '/bye');
    $('#session-help', root).innerHTML = html`People start <strong>${st.personaName || 'eSumit'}</strong> by sending <code class="wa-inline">${start}</code> and stop with <code class="wa-inline">${stop}</code>.`;
    $('#mode-line', root).innerHTML = st.replyMode
      ? html`Reply mode: <strong>${st.replyMode}</strong> — ${MODE_TEXT[st.replyMode] || ''} <a href="#/settings">Change</a>`
      : '<a href="#/settings">Open settings</a>';
  }

  async function load() {
    try {
      const [{ contacts }, settingsRes] = await Promise.all([api.contacts(), api.settings().catch(() => null)]);
      s.contacts = contacts || [];
      s.settings = settingsRes?.settings || null;
      renderHelp();
      render();
    } catch (e) {
      tableEl.innerHTML = html`<div class="list-empty">Could not load contacts: ${e.message}</div>`;
    }
  }

  function replace(contact) {
    const i = s.contacts.findIndex((c) => String(c.id) === String(contact.id));
    if (i >= 0) s.contacts[i] = { ...s.contacts[i], ...contact };
  }

  tableEl.addEventListener('change', async (e) => {
    const row = e.target.closest('.ct-row[data-id]');
    const key = e.target.dataset.key;
    if (!row || !key) return;
    const isBox = e.target.type === 'checkbox';
    const value = isBox ? e.target.checked : e.target.value.trim();
    try {
      const { contact } = await api.patchContact(row.dataset.id, { [key]: value });
      if (contact) replace(contact);
      if (isBox && s.filter !== 'all') render();
      else toast('Saved', 'success', 1200);
    } catch (ex) {
      if (isBox) e.target.checked = !value;
      toast(ex.message, 'error');
    }
  });

  tableEl.addEventListener('click', async (e) => {
    const btn = e.target.closest('button[data-act="delete"]');
    if (!btn) return;
    const id = btn.closest('.ct-row').dataset.id;
    const c = s.contacts.find((x) => String(x.id) === id);
    const ok = await confirmDialog({
      title: 'Delete contact?',
      message: `Remove ${c?.name || displayPhone(c?.phone) || 'this contact'}. Message history is kept.`,
      confirmText: 'Delete',
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteContact(id);
      s.contacts = s.contacts.filter((x) => String(x.id) !== id);
      render();
      toast('Contact deleted', 'success');
    } catch (ex) {
      toast(ex.message, 'error');
    }
  });

  root.querySelector('.chip-group').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    s.filter = chip.dataset.filter;
    root.querySelectorAll('.chip-group .chip').forEach((c) => {
      c.classList.toggle('active', c === chip);
      c.setAttribute('aria-checked', String(c === chip));
    });
    render();
  });

  $('#contact-q', root).addEventListener(
    'input',
    debounce((e) => {
      s.q = e.target.value.trim();
      render();
    }, 150)
  );

  const offs = [
    hub.on('chat:update', ({ chat } = {}) => {
      if (!chat) return;
      const c = s.contacts.find((x) => (chat.chatId && x.chatId === chat.chatId) || (chat.phone && x.phone === chat.phone));
      if (!c) return;
      for (const k of ['isBlocked', 'aiPaused', 'name', 'session', 'hasMemory', 'memoryUpdatedAt', 'lastMessageAt']) {
        if (chat[k] !== undefined) c[k] = chat[k];
      }
      render();
    }),
  ];

  await load();
  return { destroy: () => offs.forEach((o) => o()) };
}
