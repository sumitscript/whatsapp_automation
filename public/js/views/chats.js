// Chats: WhatsApp-like list + conversation with live updates and manual send.
import { api } from '../api.js';
import { icons } from '../icons.js';
import {
  $, confirmDialog, dayLabel, debounce, displayPhone, formatDateTime, formatMs, formatTime,
  formatWhatsApp, hueFor, html, initials, isSameDay, listStamp, raw, toast,
} from '../util.js';

const PAGE = 50;

export async function mount(root, { params, hub, navigate }) {
  const s = {
    chats: [],
    search: '',
    activeId: null,
    messages: [],
    hasMore: false,
    loadingOlder: false,
    typing: new Set(),
    destroyed: false,
    loadToken: 0,
  };

  root.innerHTML = html`
    <div class="split chats-layout">
      <aside class="pane list-pane" aria-label="Chat list">
        <header class="pane-header">
          <h2>Chats</h2>
          <button class="icon-btn" id="chats-refresh" aria-label="Refresh chats" title="Refresh">${raw(icons.refresh)}</button>
        </header>
        <div class="search-bar">
          ${raw(icons.search)}
          <input type="search" id="chat-search" placeholder="Search name, number or message" aria-label="Search chats" />
        </div>
        <ul class="chat-list" id="chat-list" role="list"></ul>
      </aside>
      <section class="pane conversation-pane" id="conversation" aria-live="off"></section>
    </div>`;

  const layout = $('.chats-layout', root);
  const listEl = $('#chat-list', root);
  const convEl = $('#conversation', root);

  // ---------- chat list ----------
  const chatName = (c) => c?.name || displayPhone(c?.phone) || c?.chatId || 'Unknown';

  function chatItem(c) {
    const typing = s.typing.has(c.chatId);
    const preview = typing
      ? '<span class="typing-text">typing…</span>'
      : html`${raw(c.lastDirection === 'out' ? '<span class="tick">✓✓</span> ' : '')}${c.lastMessage || ''}`;
    return html`
      <li>
        <a class="chat-item ${raw(c.chatId === s.activeId ? 'active' : '')}" href="#/chats/${encodeURIComponent(c.chatId)}" data-id="${c.chatId}">
          <span class="avatar" style="--h:${hueFor(c.chatId)}">${initials(chatName(c))}</span>
          <span class="chat-item-body">
            <span class="chat-item-top">
              <span class="chat-item-name">${chatName(c)}</span>
              <span class="chat-item-time">${listStamp(c.lastMessageAt)}</span>
            </span>
            <span class="chat-item-bottom">
              <span class="chat-item-preview">${raw(preview)}</span>
              <span class="chat-badges">
                ${raw(c.isBlocked ? '<span class="badge badge-bad" title="Blocked">Blocked</span>' : '')}
                ${raw(c.aiPaused ? '<span class="badge badge-warn" title="AI paused">Paused</span>' : '')}
                ${raw(c.session?.active ? '<span class="badge badge-ok badge-dot" title="AI session active">AI on</span>' : '')}
                ${raw(c.hasMemory ? '<span class="badge" title="Has long-term memory">🧠</span>' : '')}
              </span>
            </span>
          </span>
        </a>
      </li>`;
  }

  function renderList() {
    if (!s.chats.length) {
      listEl.innerHTML = html`<li class="list-empty">${s.search ? 'No chats match your search.' : 'No conversations yet. Messages appear here as they arrive.'}</li>`;
      return;
    }
    listEl.innerHTML = s.chats.map(chatItem).join('');
  }

  function updateListItem(chat) {
    const a = listEl.querySelector(`.chat-item[data-id="${CSS.escape(chat.chatId)}"]`);
    if (a) a.parentElement.outerHTML = chatItem(chat);
  }

  async function loadChats() {
    try {
      const { chats } = await api.chats(s.search, 100);
      if (s.destroyed) return;
      s.chats = chats || [];
      renderList();
      if (s.activeId) renderHeader();
    } catch (e) {
      listEl.innerHTML = html`<li class="list-empty">Could not load chats: ${e.message}</li>`;
    }
  }

  function matchesSearch(c) {
    if (!s.search) return true;
    const q = s.search.toLowerCase();
    return [c.name, c.phone, c.lastMessage, c.chatId].some((v) => String(v || '').toLowerCase().includes(q));
  }

  function upsertChat(chat, { moveTop = false } = {}) {
    if (!chat?.chatId) return;
    const idx = s.chats.findIndex((c) => c.chatId === chat.chatId);
    const merged = idx >= 0 ? { ...s.chats[idx], ...chat } : chat;
    if (idx >= 0 && !moveTop) {
      s.chats[idx] = merged;
      updateListItem(merged);
    } else {
      if (idx >= 0) s.chats.splice(idx, 1);
      if (!matchesSearch(merged)) return;
      s.chats.unshift(merged);
      renderList();
    }
    if (chat.chatId === s.activeId) renderHeader();
  }

  $('#chat-search', root).addEventListener(
    'input',
    debounce((e) => {
      s.search = e.target.value.trim();
      loadChats();
    }, 250)
  );
  $('#chats-refresh', root).addEventListener('click', loadChats);

  // ---------- conversation ----------
  const activeChat = () => s.chats.find((c) => c.chatId === s.activeId) || { chatId: s.activeId };

  function renderEmpty() {
    layout.classList.remove('show-detail');
    convEl.innerHTML = html`
      <div class="conversation-empty">
        <div class="empty-illustration">${raw(icons.chats)}</div>
        <h2>eSumit Console</h2>
        <p class="muted">Pick a chat to read the conversation, pause the AI, or reply yourself.</p>
      </div>`;
  }

  function renderConversationShell() {
    layout.classList.add('show-detail');
    convEl.innerHTML = html`
      <header class="pane-header conv-header" id="conv-header"></header>
      <div class="messages wallpaper" id="messages" tabindex="0" aria-label="Messages">
        <div class="load-older" id="load-older" hidden><div class="spinner spinner-sm"></div></div>
        <div id="message-list"></div>
      </div>
      <form class="composer" id="composer">
        <textarea id="composer-input" rows="1" placeholder="Type a message (sent as you, not the AI)" aria-label="Message"></textarea>
        <button class="send-btn" type="submit" aria-label="Send message" title="Send">${raw(icons.send)}</button>
      </form>`;

    const box = $('#messages', convEl);
    box.addEventListener('scroll', () => {
      if (box.scrollTop < 80) loadOlder();
    });

    const input = $('#composer-input', convEl);
    const form = $('#composer', convEl);
    const autosize = () => {
      input.style.height = 'auto';
      input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
    };
    input.addEventListener('input', autosize);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        form.requestSubmit();
      }
    });
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const body = input.value.trim();
      if (!body || !s.activeId) return;
      const chatId = s.activeId;
      input.value = '';
      autosize();
      try {
        const { message } = await api.send(chatId, body);
        if (message && chatId === s.activeId) addMessage(message);
      } catch (ex) {
        input.value = body;
        autosize();
        toast(`Send failed: ${ex.message}`, 'error');
      }
    });
    renderHeader();
  }

  function renderHeader() {
    const header = $('#conv-header', convEl);
    if (!header) return;
    const c = activeChat();
    const typing = s.typing.has(c.chatId);
    const sub = typing
      ? '<span class="typing-text">typing…</span>'
      : html`${displayPhone(c.phone) || c.chatId}${c.messageCount != null ? ` · ${c.messageCount} messages` : ''}`;
    header.innerHTML = html`
      <button class="icon-btn back-btn" id="conv-back" aria-label="Back to chats">${raw(icons.back)}</button>
      <span class="avatar" style="--h:${hueFor(c.chatId)}">${initials(chatName(c))}</span>
      <div class="conv-title">
        <div class="conv-name">${chatName(c)}</div>
        <div class="conv-sub">${raw(sub)}</div>
      </div>
      <div class="conv-actions">
        <button class="icon-btn ${raw(c.aiPaused ? 'is-on warn' : '')}" data-act="pause" aria-pressed="${String(!!c.aiPaused)}"
          aria-label="${c.aiPaused ? 'Resume AI replies' : 'Pause AI replies (human takeover)'}"
          title="${c.aiPaused ? 'AI paused — click to resume' : 'Pause AI for this chat'}">${raw(c.aiPaused ? icons.play : icons.pause)}</button>
        ${raw(
          c.session?.active
            ? html`<span class="session-pill" title="Started ${formatDateTime(c.session.startedAt)}"><span class="pill-dot"></span>AI session · since ${formatTime(c.session.startedAt)}</span>
                <button class="btn btn-ghost btn-sm" data-act="session-end" aria-label="End AI session">End session</button>`
            : '<button class="btn btn-primary btn-sm" data-act="session-start" aria-label="Start AI session">Start AI session</button>'
        )}
        <button class="icon-btn ${raw(c.hasMemory ? 'is-on ok' : '')}" data-act="memory" aria-label="Open memory" title="Memory">${raw(icons.brain)}</button>
        <button class="icon-btn ${raw(c.isBlocked ? 'is-on bad' : '')}" data-act="block" aria-pressed="${String(!!c.isBlocked)}"
          aria-label="${c.isBlocked ? 'Unblock' : 'Block (never reply)'}"
          title="${c.isBlocked ? 'Blocked — click to unblock' : 'Block: AI never replies'}">${raw(icons.block)}</button>
        <button class="icon-btn" data-act="clear" aria-label="Clear AI short-term memory" title="Clear AI memory for this chat">${raw(icons.eraser)}</button>
      </div>`;
    $('#conv-back', header).addEventListener('click', () => navigate('#/chats'));
    header.querySelector('.conv-actions').addEventListener('click', onHeaderAction);
  }

  async function onHeaderAction(e) {
    const btn = e.target.closest('button[data-act]');
    if (!btn) return;
    const c = activeChat();
    const act = btn.dataset.act;
    try {
      if (act === 'memory') {
        openMemory(c.chatId);
        return;
      }
      if (act === 'session-start' || act === 'session-end') {
        const starting = act === 'session-start';
        const choice = await sessionDialog(starting, chatName(c));
        if (!choice) return;
        btn.disabled = true;
        const { chat } = await api.session(c.chatId, starting ? 'start' : 'end', choice.notify);
        upsertChat(chat || { ...c, session: starting ? { active: true, startedAt: new Date().toISOString() } : null });
        toast(starting ? 'AI session started' : 'AI session ended', 'success');
        return;
      }
      if (act === 'clear') {
        const ok = await confirmDialog({
          title: 'Clear AI memory?',
          message: 'The assistant forgets the recent conversation context for this chat. Stored messages are kept.',
          confirmText: 'Clear memory',
        });
        if (!ok) return;
        await api.clearContext(c.chatId);
        toast('AI memory cleared for this chat.', 'success');
        return;
      }
      const patch = act === 'pause' ? { aiPaused: !c.aiPaused } : { isBlocked: !c.isBlocked };
      btn.disabled = true;
      const { chat } = await api.patchChat(c.chatId, patch);
      upsertChat(chat || { ...c, ...patch });
    } catch (ex) {
      toast(ex.message, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  /** Ask whether to start/end a session and whether to notify the contact. */
  function sessionDialog(starting, name) {
    return new Promise((resolve) => {
      const dlg = document.createElement('dialog');
      dlg.className = 'dialog';
      dlg.innerHTML = html`
        <form method="dialog">
          <h3>${starting ? 'Start AI session?' : 'End AI session?'}</h3>
          <p>${starting
            ? `eSumit will start replying to ${name} until the session ends.`
            : `eSumit stops replying to ${name}. Memory is kept.`}</p>
          <label class="switch-row"><input type="checkbox" class="switch" name="notify" ${raw(starting ? 'checked' : '')} />
            <span>${starting ? 'Send the welcome message' : 'Send the goodbye message'}</span></label>
          <div class="dialog-actions">
            <button value="cancel" class="btn btn-ghost">Cancel</button>
            <button value="ok" class="btn ${raw(starting ? 'btn-primary' : 'btn-danger')}">${starting ? 'Start session' : 'End session'}</button>
          </div>
        </form>`;
      document.body.appendChild(dlg);
      dlg.addEventListener('close', () => {
        const ok = dlg.returnValue === 'ok';
        resolve(ok ? { notify: dlg.querySelector('[name="notify"]').checked } : null);
        dlg.remove();
      });
      dlg.showModal();
    });
  }

  async function openMemory(chatId) {
    convEl.querySelector('.drawer')?.remove();
    const drawer = document.createElement('aside');
    drawer.className = 'drawer';
    drawer.setAttribute('aria-label', 'Contact memory');
    drawer.innerHTML = html`
      <header class="drawer-head">
        <h2>Memory</h2>
        <button class="icon-btn" data-mem="close" aria-label="Close memory panel">${raw(icons.x)}</button>
      </header>
      <div class="drawer-body"><div class="view-loading"><div class="spinner"></div></div></div>`;
    convEl.appendChild(drawer);
    const close = () => drawer.remove();
    drawer.querySelector('[data-mem="close"]').addEventListener('click', close);
    drawer.addEventListener('keydown', (e) => e.key === 'Escape' && close());

    let data;
    try {
      data = await api.memory(chatId);
    } catch (e) {
      drawer.querySelector('.drawer-body').innerHTML = html`<p class="form-error">Could not load memory: ${e.message}</p>`;
      return;
    }
    if (!drawer.isConnected || chatId !== s.activeId) return;
    const turns = data.shortTerm || [];
    drawer.querySelector('.drawer-body').innerHTML = html`
      <section class="stack">
        <div>
          <h3 class="drawer-sub">Long-term memory</h3>
          <p class="muted small">What eSumit remembers about this person across sessions${data.updatedAt ? ` · updated ${formatDateTime(data.updatedAt)}` : ''}.</p>
        </div>
        <textarea class="code-area small-area" rows="9" id="mem-text" aria-label="Long-term memory" placeholder="Nothing remembered yet.">${data.memory || ''}</textarea>
        <div class="form-actions">
          <button class="btn btn-danger btn-sm" data-mem="forget">${raw(icons.trash)} Forget</button>
          <button class="btn btn-primary btn-sm" data-mem="save">${raw(icons.save)} Save</button>
        </div>
      </section>
      <section class="stack">
        <div class="card-title-row">
          <h3 class="drawer-sub">Short-term context (${turns.length})</h3>
          <button class="btn btn-ghost btn-sm" data-mem="clear">${raw(icons.eraser)} Clear</button>
        </div>
        <ol class="turns">${turns.length
          ? turns.map((t) => raw(html`<li class="turn turn-${t.role}"><span class="turn-role">${t.role}</span><span class="turn-text">${t.content}</span></li>`))
          : raw('<li class="muted small">No recent turns in context.</li>')}</ol>
      </section>`;

    drawer.querySelector('.drawer-body').addEventListener('click', async (e) => {
      const b = e.target.closest('[data-mem]');
      if (!b) return;
      const act = b.dataset.mem;
      try {
        if (act === 'save') {
          await api.saveMemory(chatId, drawer.querySelector('#mem-text').value);
          toast('Memory saved', 'success');
        } else if (act === 'forget') {
          const ok = await confirmDialog({
            title: 'Forget this person?',
            message: 'Deletes the long-term memory eSumit keeps for this contact. Message history is kept.',
            confirmText: 'Forget',
            danger: true,
          });
          if (!ok) return;
          await api.forgetMemory(chatId);
          drawer.querySelector('#mem-text').value = '';
          toast('Long-term memory deleted', 'success');
        } else if (act === 'clear') {
          await api.clearContext(chatId);
          drawer.querySelector('.turns').innerHTML = '<li class="muted small">No recent turns in context.</li>';
          toast('Short-term context cleared', 'success');
        }
      } catch (ex) {
        toast(ex.message, 'error');
      }
    });
    drawer.querySelector('#mem-text').focus();
  }

  function bubble(m) {
    const out = m.direction === 'out';
    let meta = '';
    if (out) {
      if (m.isAi) {
        const parts = ['AI'];
        if (m.cached) parts.push('cached');
        else {
          if (m.model) parts.push(m.model);
          if (m.latencyMs != null) parts.push(formatMs(m.latencyMs));
        }
        meta = html`<span class="meta-tag ai" title="${m.intent ? `Intent: ${m.intent}` : 'AI reply'}">${parts.join(' · ')}</span>`;
      } else {
        meta = '<span class="meta-tag manual">Manual</span>';
      }
    }
    const isText = !m.type || m.type === 'chat' || m.type === 'text';
    const body = m.body
      ? formatWhatsApp(m.body)
      : html`<span class="muted">[${m.type || 'message'}]</span>`;
    return html`
      <div class="msg-row ${raw(out ? 'out' : 'in')}" data-id="${m.id}">
        <div class="bubble ${raw(out ? 'bubble-out' : 'bubble-in')}">
          ${raw(!isText && m.body ? html`<div class="msg-type">${m.type}</div>` : '')}
          <div class="msg-body">${raw(body)}</div>
          <div class="msg-meta">${raw(meta)}<time datetime="${m.createdAt}" title="${formatDateTime(m.createdAt)}">${formatTime(m.createdAt)}</time>${raw(out ? '<span class="tick">✓✓</span>' : '')}</div>
        </div>
      </div>`;
  }

  const separator = (d) => html`<div class="day-sep"><span>${dayLabel(d)}</span></div>`;

  function renderMessages() {
    const list = $('#message-list', convEl);
    if (!list) return;
    if (!s.messages.length) {
      list.innerHTML = '<div class="day-sep"><span>No messages yet</span></div>';
      return;
    }
    let out = '';
    let prev = null;
    for (const m of s.messages) {
      if (!prev || !isSameDay(prev.createdAt, m.createdAt)) out += separator(m.createdAt);
      out += bubble(m);
      prev = m;
    }
    list.innerHTML = out;
  }

  function scrollToBottom(force = false) {
    const box = $('#messages', convEl);
    if (!box) return;
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 160;
    if (force || nearBottom) box.scrollTop = box.scrollHeight;
  }

  function addMessage(m) {
    if (s.messages.some((x) => x.id === m.id)) return;
    const box = $('#messages', convEl);
    const wasNearBottom = box && box.scrollHeight - box.scrollTop - box.clientHeight < 160;
    const prev = s.messages[s.messages.length - 1];
    s.messages.push(m);
    const list = $('#message-list', convEl);
    if (!list) return;
    if (!prev) {
      renderMessages();
    } else {
      const frag = (!isSameDay(prev.createdAt, m.createdAt) ? separator(m.createdAt) : '') + bubble(m);
      list.insertAdjacentHTML('beforeend', frag);
      list.lastElementChild?.classList.add('is-new'); // live message: pop-in animation
    }
    if (wasNearBottom || m.direction === 'out') scrollToBottom(true);
  }

  async function openChat(chatId) {
    const token = ++s.loadToken;
    s.activeId = chatId;
    s.messages = [];
    s.hasMore = false;
    listEl.querySelectorAll('.chat-item').forEach((a) => a.classList.toggle('active', a.dataset.id === chatId));
    renderConversationShell();
    $('#message-list', convEl).innerHTML = '<div class="view-loading"><div class="spinner"></div></div>';
    try {
      const { messages, hasMore } = await api.messages(chatId, undefined, PAGE);
      if (token !== s.loadToken || s.destroyed) return;
      s.messages = messages || [];
      s.hasMore = !!hasMore;
      renderMessages();
      scrollToBottom(true);
      if (!s.chats.some((c) => c.chatId === chatId)) loadChats();
    } catch (e) {
      if (token !== s.loadToken) return;
      $('#message-list', convEl).innerHTML = html`<div class="day-sep"><span>Could not load messages: ${e.message}</span></div>`;
    }
  }

  async function loadOlder() {
    if (!s.hasMore || s.loadingOlder || !s.messages.length) return;
    s.loadingOlder = true;
    const token = s.loadToken;
    const box = $('#messages', convEl);
    const spinner = $('#load-older', convEl);
    spinner.hidden = false;
    try {
      const before = Math.min(...s.messages.map((m) => Number(m.id)).filter((n) => !Number.isNaN(n)));
      const { messages, hasMore } = await api.messages(s.activeId, before, PAGE);
      if (token !== s.loadToken) return;
      const fromBottom = box.scrollHeight - box.scrollTop;
      const known = new Set(s.messages.map((m) => m.id));
      s.messages = [...(messages || []).filter((m) => !known.has(m.id)), ...s.messages];
      s.hasMore = !!hasMore;
      renderMessages();
      box.scrollTop = box.scrollHeight - fromBottom;
    } catch (e) {
      toast(`Could not load older messages: ${e.message}`, 'error');
    } finally {
      s.loadingOlder = false;
      if (spinner) spinner.hidden = true;
    }
  }

  // ---------- live events ----------
  const offs = [
    hub.on('message:new', ({ message, chat } = {}) => {
      if (!message) return;
      const chatId = message.chatId || chat?.chatId;
      if (chatId === s.activeId) addMessage(message);
      const base = s.chats.find((c) => c.chatId === chatId) || {};
      upsertChat(
        {
          ...base,
          ...(chat || {}),
          chatId,
          lastMessage: message.body || `[${message.type || 'message'}]`,
          lastMessageAt: message.createdAt,
          lastDirection: message.direction,
          messageCount: chat?.messageCount ?? (base.messageCount != null ? base.messageCount + 1 : undefined),
        },
        { moveTop: true }
      );
    }),
    hub.on('chat:update', ({ chat } = {}) => chat && upsertChat(chat)),
    hub.on('typing', ({ chatId, on } = {}) => {
      if (!chatId) return;
      if (on) s.typing.add(chatId);
      else s.typing.delete(chatId);
      const c = s.chats.find((x) => x.chatId === chatId);
      if (c) updateListItem(c);
      if (chatId === s.activeId) renderHeader();
    }),
    hub.on('socket:state', (st) => {
      if (st === 'connected') {
        loadChats();
        if (s.activeId) openChat(s.activeId);
      }
    }),
  ];

  function update(p) {
    const id = p?.[0] || null;
    if (id && id === s.activeId) return;
    if (id) openChat(id);
    else {
      s.activeId = null;
      s.loadToken++;
      listEl.querySelectorAll('.chat-item.active').forEach((a) => a.classList.remove('active'));
      renderEmpty();
    }
  }

  await loadChats();
  update(params);

  return {
    update,
    destroy() {
      s.destroyed = true;
      offs.forEach((off) => off());
    },
  };
}
