// Live log tail with filters.
import { api } from '../api.js';
import { subscribeLogs } from '../socket.js';
import { icons } from '../icons.js';
import { $, debounce, escapeHtml, html, raw } from '../util.js';

const MAX_LINES = 2000;
const LEVELS = ['debug', 'info', 'warn', 'error'];
const RESERVED = new Set(['ts', 'level', 'service', 'msg', 'time', 'message']);

export async function mount(root, { hub }) {
  const s = {
    levels: new Set(['info', 'warn', 'error']),
    service: 'all',
    text: '',
    paused: false,
    autoscroll: true,
    buffer: [], // entries received while paused
    count: 0,
  };

  root.innerHTML = html`
    <div class="page page-full logs-page">
      <header class="page-header">
        <div><h1>Logs</h1><p class="muted" id="log-status">Connecting to live tail…</p></div>
      </header>
      <div class="toolbar logs-toolbar">
        <div class="chip-group" aria-label="Log levels">
          ${LEVELS.map((l) =>
            raw(html`<button class="chip lvl-${l} ${raw(s.levels.has(l) ? 'active' : '')}" data-level="${l}" aria-pressed="${String(s.levels.has(l))}">${l}</button>`)
          )}
        </div>
        <select id="log-service" class="select" aria-label="Service">
          <option value="all">All services</option>
          <option value="gateway">gateway</option>
          <option value="worker">worker</option>
        </select>
        <div class="search-bar compact">
          ${raw(icons.search)}
          <input type="search" id="log-text" placeholder="Filter text" aria-label="Filter log text" />
        </div>
        <label class="switch-row"><input type="checkbox" class="switch" id="log-autoscroll" checked /> <span>Autoscroll</span></label>
        <button class="btn btn-ghost btn-sm" id="log-pause">${raw(icons.pause)} Pause</button>
        <button class="btn btn-ghost btn-sm" id="log-clear">${raw(icons.trash)} Clear</button>
      </div>
      <div class="log-view" id="log-view" role="log" aria-live="off"></div>
    </div>`;

  const view = $('#log-view', root);
  const status = $('#log-status', root);

  function matches(e) {
    const lvl = String(e.level || 'info').toLowerCase();
    if (!s.levels.has(lvl === 'fatal' ? 'error' : lvl)) return false;
    if (s.service !== 'all' && e.service !== s.service) return false;
    if (s.text) {
      const hay = `${e.msg || ''} ${JSON.stringify(e)}`.toLowerCase();
      if (!hay.includes(s.text)) return false;
    }
    return true;
  }

  function metaOf(e) {
    const meta = {};
    for (const [k, v] of Object.entries(e)) if (!RESERVED.has(k)) meta[k] = v;
    return Object.keys(meta).length ? meta : null;
  }

  function line(e) {
    const lvl = String(e.level || 'info').toLowerCase();
    const ts = e.ts ? new Date(e.ts) : null;
    const t = ts && !Number.isNaN(ts.getTime())
      ? ts.toLocaleTimeString([], { hour12: false }) + '.' + String(ts.getMilliseconds()).padStart(3, '0')
      : '';
    const meta = metaOf(e);
    const head = `<span class="log-ts">${escapeHtml(t)}</span><span class="log-lvl lvl-${escapeHtml(lvl)}">${escapeHtml(lvl.toUpperCase())}</span><span class="log-svc">${escapeHtml(e.service || '')}</span><span class="log-msg">${escapeHtml(e.msg ?? e.message ?? '')}</span>`;
    if (!meta) return `<div class="log-line">${head}</div>`;
    return `<details class="log-line"><summary>${head}<span class="log-meta-hint">${escapeHtml(
      Object.keys(meta).slice(0, 4).join(', ')
    )}</span></summary><pre class="log-meta">${escapeHtml(JSON.stringify(meta, null, 2))}</pre></details>`;
  }

  function trim() {
    while (view.childElementCount > MAX_LINES) view.firstElementChild.remove();
  }

  function append(entries) {
    const visible = entries.filter(matches);
    if (!visible.length) return;
    view.insertAdjacentHTML('beforeend', visible.map(line).join(''));
    s.count += visible.length;
    trim();
    if (s.autoscroll) view.scrollTop = view.scrollHeight;
  }

  // Keep a bounded copy of everything so filter changes can re-render.
  let all = [];
  function remember(entries) {
    all = all.concat(entries);
    if (all.length > MAX_LINES * 2) all = all.slice(-MAX_LINES * 2);
  }

  function rerender() {
    view.innerHTML = '';
    append(all.slice(-MAX_LINES * 2));
  }

  // Initial backlog.
  try {
    const { logs } = await api.logs(500);
    remember(logs || []);
    rerender();
    status.textContent = 'Live — new entries stream in as they happen.';
  } catch (e) {
    status.textContent = `Could not load backlog: ${e.message}`;
  }

  const unsubscribe = subscribeLogs();

  let pending = [];
  let raf = 0;
  const flush = () => {
    raf = 0;
    const batch = pending;
    pending = [];
    append(batch);
  };

  const offs = [
    hub.on('log', (entry) => {
      if (!entry) return;
      remember([entry]);
      if (s.paused) {
        s.buffer.push(entry);
        $('#log-pause', root).innerHTML = `${icons.play} Resume (${s.buffer.length})`;
        return;
      }
      pending.push(entry);
      if (!raf) raf = requestAnimationFrame(flush);
    }),
    hub.on('socket:state', (st) => {
      status.textContent = st === 'connected' ? 'Live — new entries stream in as they happen.' : `Live tail ${st}.`;
    }),
  ];

  root.querySelector('.chip-group').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const l = chip.dataset.level;
    if (s.levels.has(l)) s.levels.delete(l);
    else s.levels.add(l);
    chip.classList.toggle('active', s.levels.has(l));
    chip.setAttribute('aria-pressed', String(s.levels.has(l)));
    rerender();
  });
  $('#log-service', root).addEventListener('change', (e) => {
    s.service = e.target.value;
    rerender();
  });
  $('#log-text', root).addEventListener(
    'input',
    debounce((e) => {
      s.text = e.target.value.trim().toLowerCase();
      rerender();
    }, 200)
  );
  $('#log-autoscroll', root).addEventListener('change', (e) => {
    s.autoscroll = e.target.checked;
    if (s.autoscroll) view.scrollTop = view.scrollHeight;
  });
  $('#log-pause', root).addEventListener('click', (e) => {
    s.paused = !s.paused;
    const btn = e.currentTarget;
    if (s.paused) {
      btn.innerHTML = `${icons.play} Resume`;
    } else {
      btn.innerHTML = `${icons.pause} Pause`;
      append(s.buffer);
      s.buffer = [];
    }
  });
  $('#log-clear', root).addEventListener('click', () => {
    view.innerHTML = '';
    all = [];
    s.buffer = [];
  });

  return {
    destroy() {
      unsubscribe();
      offs.forEach((o) => o());
      if (raf) cancelAnimationFrame(raf);
    },
  };
}
