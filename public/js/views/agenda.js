// Agenda: appointments and to-dos other people ask of the owner (extracted by the AI) plus manual items.
import { api } from '../api.js';
import { icons } from '../icons.js';
import {
  $, confirmDialog, debounce, formatNumber, formatTime, html, raw, storageGet, storageSet, toast,
} from '../util.js';

const NOTIFY_KEY = 'wa-agenda-notify';

const FILTERS = [
  { id: 'review', label: 'Needs review', params: { status: 'review' } },
  { id: 'upcoming', label: 'Upcoming', params: { status: 'open', kind: 'appointment' } },
  { id: 'todos', label: 'To-dos', params: { status: 'open', kind: 'todo' } },
  { id: 'done', label: 'Done', params: { status: 'done' } },
  { id: 'all', label: 'All', params: { status: 'all' } },
];

const pad = (n) => String(n).padStart(2, '0');

function toDate(v) {
  if (!v) return null;
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate());
const dayKey = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const fmtDate = (d) => d.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' });

/** ISO -> value for <input type="datetime-local"> / <input type="date"> in browser-local time. */
function isoToInput(iso, allDay) {
  const d = toDate(iso);
  if (!d) return '';
  const date = dayKey(d);
  return allDay ? date : `${date}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Input value (browser-local) -> ISO string, or null when empty/invalid. */
function inputToIso(value, allDay) {
  if (!value) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}))?$/.exec(value);
  if (!m) return null;
  const d = allDay
    ? new Date(+m[1], +m[2] - 1, +m[3])
    : new Date(+m[1], +m[2] - 1, +m[3], +(m[4] || 0), +(m[5] || 0));
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

function isOverdue(item, now) {
  if (!['pending', 'confirmed'].includes(item.status)) return false;
  const d = toDate(item.dueAt);
  if (!d) return false;
  return item.allDay ? d < startOfDay(now) : d < now;
}

function groupOf(item, now) {
  const d = toDate(item.dueAt);
  if (!d) return { key: 'none', label: 'No date', order: 9e15 };
  if (isOverdue(item, now)) return { key: 'overdue', label: 'Overdue', order: -1 };
  const today = startOfDay(now);
  const diff = Math.round((startOfDay(d) - today) / 86400000);
  const label = diff === 0 ? 'Today' : diff === 1 ? 'Tomorrow' : fmtDate(d);
  return { key: dayKey(d), label, order: startOfDay(d).getTime() };
}

function whenText(item) {
  const d = toDate(item.dueAt);
  if (!d) return '';
  const day = fmtDate(d);
  return item.allDay ? `${day} · all day` : `${day} · ${formatTime(d)}`;
}

const STATUS_LABEL = { pending: 'Needs review', confirmed: 'Confirmed', declined: 'Declined', done: 'Done' };
const STATUS_TONE = { pending: 'badge-warn', confirmed: 'badge-ok', declined: 'badge-bad', done: '' };

function statusBadge(item) {
  let label = STATUS_LABEL[item.status] || item.status;
  if (item.kind === 'todo') {
    if (item.status === 'pending') label = 'Open';
    if (item.status === 'declined') label = 'Dismissed';
  }
  const tone = item.kind === 'todo' && item.status === 'pending' ? '' : STATUS_TONE[item.status] || '';
  return html`<span class="badge ${raw(tone)}">${label}</span>`;
}

export async function mount(root, { hub }) {
  const s = {
    items: [],
    summary: null,
    filter: 'review',
    picked: false, // user chose a tab explicitly
    loading: true,
    error: '',
    busy: new Set(),
    token: 0,
  };

  root.innerHTML = html`
    <div class="page agenda-page">
      <header class="page-header">
        <div><h1>Agenda</h1><p class="muted" id="ag-sub">Appointments and to-dos people ask of you, collected from your chats.</p></div>
        <div class="agenda-head-actions">
          <button class="icon-btn" id="ag-refresh" type="button" aria-label="Refresh agenda" title="Refresh">${raw(icons.refresh)}</button>
          <button class="btn btn-primary" id="ag-add" type="button" aria-label="Add agenda item">${raw(icons.plus)}<span>Add item</span></button>
        </div>
      </header>
      <div class="stat-grid" id="ag-tiles"></div>
      <section class="card">
        <div class="toolbar agenda-toolbar">
          <div class="chip-group" id="ag-filters" role="radiogroup" aria-label="Filter agenda"></div>
          <label class="switch-row agenda-notify small">
            <input type="checkbox" class="switch" id="ag-notify" aria-label="Also message the contact when I confirm or decline" />
            <span>Also message the contact when I confirm or decline</span>
          </label>
        </div>
        <div id="ag-list" aria-live="polite"></div>
      </section>
    </div>`;

  const listEl = $('#ag-list', root);
  const tilesEl = $('#ag-tiles', root);
  const filtersEl = $('#ag-filters', root);
  const notifyEl = $('#ag-notify', root);
  notifyEl.checked = storageGet(NOTIFY_KEY, '0') === '1';
  notifyEl.addEventListener('change', () => storageSet(NOTIFY_KEY, notifyEl.checked ? '1' : '0'));

  // ---------- rendering ----------
  function renderTiles() {
    const m = s.summary;
    if (!m) {
      tilesEl.innerHTML = '';
      return;
    }
    const tile = (label, value, tone = '') =>
      html`<div class="stat-tile ${raw(tone)}"><div class="stat-label">${label}</div><div class="stat-value">${formatNumber(value)}</div></div>`;
    tilesEl.innerHTML = [
      tile('Needs review', m.review, m.review > 0 ? 'agenda-tile-warn' : ''),
      tile('Today', m.today),
      tile('Next 7 days', m.next7),
      tile('Open to-dos', m.openTodos),
      tile('Overdue', m.overdue, m.overdue > 0 ? 'agenda-tile-bad' : ''),
    ].join('');
  }

  function renderFilters() {
    const m = s.summary;
    const count = { review: m?.review, todos: m?.openTodos, all: m?.total };
    filtersEl.innerHTML = FILTERS.map((f) => {
      const on = f.id === s.filter;
      const n = count[f.id];
      return html`<button type="button" class="chip ${raw(on ? 'active' : '')}" role="radio" aria-checked="${on}" data-filter="${f.id}">${f.label}${raw(
        n != null ? html` <span class="agenda-count">${formatNumber(n)}</span>` : ''
      )}</button>`;
    }).join('');
  }

  function actionBtn(act, id, label, cls = 'btn-ghost', icon = '') {
    return html`<button type="button" class="btn btn-sm ${raw(cls)}" data-act="${act}" data-id="${id}" aria-label="${label}">${raw(icon)}<span>${label}</span></button>`;
  }

  function actionsFor(item) {
    const a = [];
    const id = item.id;
    if (item.kind === 'appointment' && item.status === 'pending') {
      a.push(actionBtn('confirm', id, 'Confirm', 'btn-primary', icons.check));
      a.push(actionBtn('decline', id, 'Decline', 'btn-danger', icons.x));
    } else if (item.status === 'confirmed' || (item.kind === 'todo' && item.status === 'pending')) {
      a.push(actionBtn('done', id, 'Mark done', 'btn-primary', icons.check));
    }
    if (item.status === 'done' || item.status === 'declined') a.push(actionBtn('reopen', id, 'Reopen', 'btn-ghost', icons.refresh));
    a.push(actionBtn('edit', id, 'Edit'));
    a.push(actionBtn('delete', id, 'Delete', 'btn-ghost agenda-del', icons.trash));
    return a.join('');
  }

  function itemCard(item, now) {
    const when = whenText(item);
    const overdue = isOverdue(item, now);
    const name = item.contactName || item.phone;
    const who = item.chatId && name
      ? html`<a class="agenda-contact" href="#/chats/${encodeURIComponent(item.chatId)}">${name}</a>`
      : name
        ? html`<span class="agenda-contact">${name}</span>`
        : '';
    const meta = [
      when ? html`<span class="badge ${raw(overdue ? 'badge-bad' : 'badge-ok')}">${when}</span>` : '',
      item.dueText ? html`<span class="muted small">they said: “${item.dueText}”</span>` : '',
      item.location ? html`<span class="agenda-loc small">${item.location}</span>` : '',
    ].filter(Boolean);
    return html`
      <article class="agenda-item ${raw(html`agenda-${item.kind} agenda-st-${item.status}`)}" data-id="${item.id}">
        <div class="agenda-ico" aria-hidden="true">${raw(item.kind === 'todo' ? icons.checklist : icons.agenda)}</div>
        <div class="agenda-body">
          <div class="agenda-title-row">
            <h3 class="agenda-title">${item.title}</h3>
            ${raw(statusBadge(item))}
            <span class="badge agenda-src">${item.source === 'ai' ? 'AI' : 'Manual'}</span>
          </div>
          ${raw(meta.length ? `<div class="agenda-meta">${meta.join('')}</div>` : '')}
          ${raw(who ? html`<div class="agenda-who small">${raw(icons.contacts)}${raw(who)}</div>` : '')}
          ${raw(item.details ? html`<p class="agenda-details">${item.details}</p>` : '')}
          ${raw(item.sourceMessage ? html`<blockquote class="agenda-quote muted small">${item.sourceMessage}</blockquote>` : '')}
          <div class="agenda-actions">${raw(actionsFor(item))}</div>
        </div>
      </article>`;
  }

  function render() {
    renderTiles();
    renderFilters();
    if (s.loading && !s.items.length) {
      listEl.innerHTML = '<div class="view-loading"><div class="spinner" aria-label="Loading"></div></div>';
      return;
    }
    if (s.error && !s.items.length) {
      listEl.innerHTML = html`<div class="empty-state"><h3>Could not load the agenda</h3><p class="muted">${s.error}</p>
        <button class="btn btn-ghost" type="button" data-act="reload">Try again</button></div>`;
      return;
    }
    if (!s.items.length) {
      const msg = {
        review: 'Nothing needs your decision. New proposals from chats will show up here.',
        upcoming: 'No upcoming appointments.',
        todos: 'No open to-dos.',
        done: 'Nothing completed or declined yet.',
        all: 'The agenda is empty. Add an item, or wait for someone to ask you something in chat.',
      }[s.filter];
      listEl.innerHTML = html`<div class="empty-state"><h3>All clear</h3><p class="muted">${msg}</p></div>`;
      return;
    }
    const now = new Date();
    let groups;
    if (s.filter === 'done') {
      groups = [{ key: 'done', label: '', items: s.items }];
    } else {
      const map = new Map();
      for (const item of s.items) {
        const g = groupOf(item, now);
        if (!map.has(g.key)) map.set(g.key, { ...g, items: [] });
        map.get(g.key).items.push(item);
      }
      groups = [...map.values()].sort((a, b) => a.order - b.order);
    }
    listEl.innerHTML = groups.map((g) => html`
      <section class="agenda-group ${raw(g.key === 'overdue' ? 'agenda-group-overdue' : '')}">
        ${raw(g.label ? html`<h2 class="agenda-day">${g.label}<span class="agenda-count">${g.items.length}</span></h2>` : '')}
        <div class="agenda-items">${raw(g.items.map((i) => itemCard(i, now)).join(''))}</div>
      </section>`).join('');
  }

  // ---------- data ----------
  async function load({ quiet = false } = {}) {
    const token = ++s.token;
    if (!quiet) {
      s.loading = true;
      if (!s.items.length) render();
    }
    try {
      let f = FILTERS.find((x) => x.id === s.filter);
      let data = await api.agenda({ ...f.params, limit: 300 });
      if (token !== s.token) return;
      // Default tab: review when something awaits a decision, else upcoming.
      if (!s.picked && s.filter === 'review' && !(data.summary?.review > 0)) {
        s.filter = 'upcoming';
        f = FILTERS.find((x) => x.id === 'upcoming');
        data = await api.agenda({ ...f.params, limit: 300 });
        if (token !== s.token) return;
      }
      s.items = Array.isArray(data.items) ? data.items : [];
      s.summary = data.summary || null;
      s.error = '';
      if (data.timezone) $('#ag-sub', root).title = `Server timezone: ${data.timezone}. Times are shown in your browser's timezone.`;
    } catch (e) {
      if (token !== s.token) return;
      s.error = e.message;
      if (quiet || s.items.length) toast(`Could not refresh agenda: ${e.message}`, 'error');
    } finally {
      if (token === s.token) {
        s.loading = false;
        render();
      }
    }
  }

  async function mutate(id, fn, okMsg) {
    if (s.busy.has(id)) return;
    s.busy.add(id);
    listEl.querySelectorAll(`[data-id="${CSS.escape(String(id))}"] button`).forEach((b) => (b.disabled = true));
    try {
      await fn();
      if (okMsg) toast(okMsg, 'success');
    } catch (e) {
      toast(e.message || 'Action failed', 'error');
    } finally {
      s.busy.delete(id);
      await load({ quiet: true });
    }
  }

  const notifyNow = () => notifyEl.checked;

  // ---------- add / edit dialog ----------
  function openEditor(item) {
    const editing = !!item;
    const allDay = !!item?.allDay;
    const dlg = document.createElement('dialog');
    dlg.className = 'dialog agenda-dialog';
    dlg.innerHTML = html`
      <form id="ag-form" novalidate>
        <h3>${editing ? 'Edit item' : 'Add item'}</h3>
        <label class="field"><span>Title</span><input name="title" type="text" maxlength="200" required autocomplete="off" aria-label="Title" /></label>
        <div class="form-grid">
          <label class="field"><span>Kind</span>
            <select name="kind" aria-label="Kind"><option value="appointment">Appointment</option><option value="todo">To-do</option></select>
          </label>
          <label class="field"><span>Date and time</span><input name="due" type="${allDay ? 'date' : 'datetime-local'}" aria-label="Date and time" /></label>
        </div>
        <label class="switch-row"><input type="checkbox" class="switch" name="allDay" aria-label="All day (no time)" /><span>All day (no time)</span></label>
        <label class="field"><span>Location</span><input name="location" type="text" maxlength="200" autocomplete="off" aria-label="Location" /></label>
        <label class="field"><span>Details</span><textarea name="details" rows="3" maxlength="2000" aria-label="Details"></textarea></label>
        <p class="form-error" id="ag-err" role="alert" hidden></p>
        <div class="dialog-actions">
          <button type="button" class="btn btn-ghost" data-close>Cancel</button>
          <button type="submit" class="btn btn-primary">${editing ? 'Save' : 'Add'}</button>
        </div>
      </form>`;
    document.body.appendChild(dlg);
    const form = $('#ag-form', dlg);
    const f = form.elements;
    // Values are set via properties (never interpolated into markup).
    f.title.value = item?.title || '';
    f.kind.value = item?.kind || 'appointment';
    f.allDay.checked = allDay;
    f.due.value = isoToInput(item?.dueAt, allDay);
    f.location.value = item?.location || '';
    f.details.value = item?.details || '';
    const err = $('#ag-err', dlg);

    f.allDay.addEventListener('change', () => {
      const prev = f.due.value;
      const nowAllDay = f.allDay.checked;
      f.due.type = nowAllDay ? 'date' : 'datetime-local';
      if (prev) f.due.value = nowAllDay ? prev.slice(0, 10) : `${prev.slice(0, 10)}T09:00`;
    });
    dlg.addEventListener('close', () => dlg.remove());
    $('[data-close]', dlg).addEventListener('click', () => dlg.close());

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const title = f.title.value.trim();
      if (!title) {
        err.textContent = 'A title is required.';
        err.hidden = false;
        f.title.focus();
        return;
      }
      err.hidden = true;
      const isAllDay = f.allDay.checked;
      const body = {
        kind: f.kind.value,
        title,
        details: f.details.value.trim() || null,
        dueAt: inputToIso(f.due.value, isAllDay),
        allDay: isAllDay && !!f.due.value,
        location: f.location.value.trim() || null,
      };
      const submit = $('button[type="submit"]', dlg);
      submit.disabled = true;
      try {
        if (editing) await api.agendaUpdate(item.id, body);
        else await api.agendaCreate(body);
        toast(editing ? 'Item updated' : 'Item added', 'success');
        dlg.close();
        load({ quiet: true });
      } catch (ex) {
        err.textContent = ex.message || 'Could not save';
        err.hidden = false;
        submit.disabled = false;
      }
    });
    dlg.showModal();
    f.title.focus();
  }

  // ---------- events ----------
  const find = (id) => s.items.find((i) => String(i.id) === String(id));

  listEl.addEventListener('click', async (e) => {
    const btn = e.target.closest('button[data-act]');
    if (!btn) return;
    const act = btn.dataset.act;
    if (act === 'reload') return load();
    const item = find(btn.dataset.id);
    if (!item) return;
    if (act === 'edit') return openEditor(item);
    if (act === 'confirm') {
      return mutate(item.id, () => api.agendaUpdate(item.id, { status: 'confirmed', notify: notifyNow() }), 'Confirmed');
    }
    if (act === 'decline') {
      return mutate(item.id, () => api.agendaUpdate(item.id, { status: 'declined', notify: notifyNow() }), 'Declined');
    }
    if (act === 'done') return mutate(item.id, () => api.agendaUpdate(item.id, { status: 'done' }), 'Marked done');
    if (act === 'reopen') return mutate(item.id, () => api.agendaUpdate(item.id, { status: 'pending' }), 'Reopened');
    if (act === 'delete') {
      const ok = await confirmDialog({
        title: 'Delete this item?',
        message: `“${item.title}” will be removed from the agenda.`,
        confirmText: 'Delete',
        danger: true,
      });
      if (ok) mutate(item.id, () => api.agendaDelete(item.id), 'Deleted');
    }
  });

  filtersEl.addEventListener('click', (e) => {
    const b = e.target.closest('[data-filter]');
    if (!b || b.dataset.filter === s.filter) return;
    s.filter = b.dataset.filter;
    s.picked = true;
    s.items = [];
    load();
  });

  $('#ag-refresh', root).addEventListener('click', () => load());
  $('#ag-add', root).addEventListener('click', () => openEditor(null));

  const reload = debounce(() => load({ quiet: true }), 400);
  const off = hub.on('agenda:update', reload);

  await load();
  return {
    destroy() {
      off();
      s.token++;
    },
  };
}
