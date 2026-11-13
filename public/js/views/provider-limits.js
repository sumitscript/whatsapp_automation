// AI accounts (Groq, Cloudflare Workers AI): one compact row per account, a usage bar that opens the limit details on hover
// or tap, and the Cloudflare account wiring. The Settings view owns the surrounding card.
import { api } from '../api.js';
import { $, confirmDialog, formatNumber, html, raw, toast } from '../util.js';

const pad = (n) => String(n).padStart(2, '0');

/** 7_380_000 → "2h 03m", 45_000 → "45s" */
export function formatLeft(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 90) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${pad(m % 60)}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

/** Clock time in the viewer's zone; adds the weekday when it is not today. */
export function formatAt(ts) {
  const d = new Date(ts);
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return d.toDateString() === new Date().toDateString() ? time : `${d.toLocaleDateString([], { weekday: 'short' })} ${time}`;
}

const compact = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(Math.round(n)));
const tone = (pct, full) => (full || pct >= 85 ? 'bad' : pct >= 60 ? 'warn' : 'ok');

// ---------------------------------------------------------------- limit panel (hover / tap on the usage bar)

function bar({ label, used, limit, resetAt, now }) {
  const pct = limit ? Math.min(100, Math.max(0, (used / limit) * 100)) : 0;
  const full = !!limit && used >= limit;
  const reset = resetAt && resetAt > now ? `resets in ${formatLeft(resetAt - now)} (${formatAt(resetAt)})` : '';
  return html`<div class="lim-row">
    <div class="lim-head"><span>${label}</span><span>${limit ? `${compact(used)} / ${compact(limit)}${full ? ' · exhausted' : ''}` : 'not known yet'}</span></div>
    <div class="meter" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(pct)}"><span class="meter-${tone(pct, full)}" style="width:${pct.toFixed(1)}%"></span></div>
    ${raw(reset ? `<div class="lim-reset muted small">${reset}</div>` : '')}
  </div>`;
}

function modelSection(m, now) {
  const name = String(m.id).split('/').pop();
  const lim = m.limits || {};
  const bars = [];
  if (lim.neurons) bars.push(bar({ label: 'Free neurons today (estimate)', used: lim.neurons.used, limit: lim.neurons.limit, resetAt: lim.neurons.resetAt, now }));
  if (lim.requests?.limit != null) bars.push(bar({ label: 'Requests today', used: lim.requests.limit - (lim.requests.remaining ?? lim.requests.limit), limit: lim.requests.limit, resetAt: lim.requests.resetAt, now }));
  if (lim.tokens?.limit != null) bars.push(bar({ label: 'Tokens per minute', used: lim.tokens.limit - (lim.tokens.remaining ?? lim.tokens.limit), limit: lim.tokens.limit, resetAt: lim.tokens.resetAt, now }));
  let note = '';
  if (m.gone) note = html`<div class="lim-note lim-bad">Not available on this account</div>`;
  else if (m.state === 'cooling' && m.restUntil) {
    const what = m.hit?.what ? ` ${m.hit.what}` : '';
    note = html`<div class="lim-note lim-bad">Limit reached${what}. Back in <strong>${formatLeft(m.restUntil - now)}</strong> (${formatAt(m.restUntil)})</div>`;
  } else if (m.state === 'disabled') note = html`<div class="lim-note lim-bad">Key rejected</div>`;
  if (!bars.length && !note) bars.push(html`<p class="muted small">Figures appear after the first request</p>`);
  const state = m.state === 'ready' ? html`<span class="badge badge-ok">ready</span>` : html`<span class="badge badge-bad">${m.gone ? 'unavailable' : 'resting'}</span>`;
  return html`<section class="lim-model"><h4>${name} ${raw(state)}</h4>${raw(note)}${raw(bars.join(''))}</section>`;
}

/** The panel's contents for one account. */
export function limitsPanel(key, kind = 'groq') {
  const now = Date.now();
  const u = key.usage?.today;
  const usage = u ? html`<p class="muted small lim-usage">Today: ${formatNumber(u.calls)} calls${u.limited ? ` · ${u.limited} rate-limited` : ''}${kind === 'cloudflare' ? ` · ~${formatNumber(Math.round(u.neurons || 0))} neurons` : ` · ${compact(u.tokens)} tokens`}</p>` : '';
  return html`<h3>${key.label}</h3>${raw((key.models || []).map((m) => modelSection(m, now)).join(''))}${raw(usage)}`;
}

// ---------------------------------------------------------------- account rows

/** Share of the daily allowance used (0..1) or null when nothing is known yet. Resting because a daily limit was hit counts as full. */
export function usedFraction(key) {
  let worst = null;
  for (const m of key.models || []) {
    if (m.state === 'cooling' && m.hit?.what && /day|allowance/i.test(m.hit.what)) return 1;
    const l = m.limits || {};
    const fr = [];
    if (l.neurons?.limit) fr.push(l.neurons.used / l.neurons.limit);
    if (l.requests?.limit) fr.push((l.requests.limit - (l.requests.remaining ?? l.requests.limit)) / l.requests.limit);
    for (const f of fr) worst = Math.max(worst ?? 0, f);
  }
  return worst == null ? null : Math.min(1, Math.max(0, worst));
}

const stateOf = (k) => {
  const models = k.models || [];
  if (k.state === 'disabled') return { dot: 'bad', badge: 'Rejected', tone: 'bad' };
  if (models.length && models.every((m) => m.gone)) return { dot: 'bad', badge: 'No access', tone: 'bad' };
  if (k.state === 'cooling') return { dot: 'warn', badge: `Resting ${formatLeft(k.cooldownMs || 0)}`, tone: 'warn' };
  return { dot: 'ok', badge: '', tone: 'ok' };
};

/**
 * One account as a single line: status dot, name, key hint, usage bar (hover for details), actions menu.
 * @param {object} k account from the status endpoint
 * @param {{ kind: 'groq'|'cloudflare', consoleIndex: number, consoleCount: number }} o
 */
export function accountRow(k, { kind, consoleIndex, consoleCount }) {
  const st = stateOf(k);
  const frac = usedFraction(k);
  const pct = frac == null ? 0 : frac * 100;
  const own = k.source === 'console';
  const same = kind === 'groq' && k.sameOrgAs ? html`<span class="badge badge-warn" title="Same Groq organization as ${k.sameOrgAs}: shared limits, no extra capacity">shared limits</span>` : '';
  const menu = html`<details class="menu"><summary aria-label="Account actions" title="Actions">⋯</summary><div class="menu-pop">
      ${raw(own ? html`<button type="button" data-act="up" data-id="${k.id}" ${consoleIndex <= 0 ? 'disabled' : ''}>Move up</button><button type="button" data-act="down" data-id="${k.id}" ${consoleIndex >= consoleCount - 1 ? 'disabled' : ''}>Move down</button>` : '')}
      <button type="button" data-act="test" data-id="${k.id}">Test key</button>
      ${raw(own ? html`<button type="button" class="menu-danger" data-act="remove" data-id="${k.id}" data-label="${k.label}">Remove</button>` : '')}
    </div></details>`;
  return html`<div class="acct">
    <span class="dot dot-${st.dot}" aria-hidden="true"></span>
    <div class="acct-name"><strong>${k.label}</strong><span class="muted small">${k.hint}${own ? '' : ' · env'}</span></div>
    ${raw(st.badge ? html`<span class="badge badge-${st.tone}">${st.badge}</span>` : '')}${raw(same)}
    <button type="button" class="usage-btn" data-kind="${kind}" data-id="${k.id}" aria-label="Usage and limits for ${k.label}">
      <span class="meter mini"><span class="meter-${tone(pct, frac === 1)}" style="width:${pct.toFixed(1)}%"></span></span>
      <span class="usage-pct">${frac == null ? '—' : `${Math.round(pct)}%`}</span>
    </button>
    ${raw(menu)}
  </div>`;
}

/** Re-render an account list unless one of its menus is open (a refresh would slam it shut). */
export function renderAccounts(container, keys, kind, empty) {
  if (container.querySelector('details[open]')) return;
  const own = keys.filter((k) => k.source === 'console');
  container.innerHTML = keys.length
    ? keys.map((k) => accountRow(k, { kind, consoleIndex: own.findIndex((x) => x.id === k.id), consoleCount: own.length })).join('')
    : `<p class="empty muted small">${empty}</p>`;
}

/**
 * Show the details while a usage bar is hovered or focused (tap toggles on touch screens). `find(kind, id)` returns the
 * account. One panel for the page, placed beside the bar, so scrolling containers never clip it.
 */
export function installLimitPanel(root, find) {
  const pop = document.createElement('div');
  pop.className = 'limit-pop';
  pop.hidden = true;
  root.appendChild(pop);
  let current = null;
  const show = (btn) => {
    const key = find(btn.dataset.kind, btn.dataset.id);
    if (!key) return;
    current = btn;
    pop.innerHTML = limitsPanel(key, btn.dataset.kind);
    pop.hidden = false;
    const r = btn.getBoundingClientRect();
    const w = Math.min(340, window.innerWidth - 16);
    pop.style.width = `${w}px`;
    pop.style.left = `${Math.min(Math.max(8, r.right - w), window.innerWidth - w - 8)}px`;
    pop.style.top = `${Math.min(r.bottom + 6, Math.max(8, window.innerHeight - pop.offsetHeight - 8))}px`;
  };
  const hide = () => {
    pop.hidden = true;
    current = null;
  };
  root.addEventListener('mouseover', (e) => {
    const b = e.target.closest?.('.usage-btn');
    if (b && b !== current) show(b);
  });
  root.addEventListener('mouseout', (e) => {
    if (e.target.closest?.('.usage-btn') && !e.relatedTarget?.closest?.('.limit-pop')) hide();
  });
  root.addEventListener('focusin', (e) => e.target.closest?.('.usage-btn') && show(e.target.closest('.usage-btn')));
  root.addEventListener('focusout', (e) => e.target.closest?.('.usage-btn') && hide());
  root.addEventListener('click', (e) => {
    const b = e.target.closest?.('.usage-btn');
    if (!b) return;
    if (current === b && !pop.hidden) hide();
    else show(b);
  });
  window.addEventListener('scroll', hide, { passive: true, capture: true });
  return { refresh: () => current && !pop.hidden && show(current), destroy: () => pop.remove() };
}

// ---------------------------------------------------------------- Cloudflare accounts

/**
 * Wire the Cloudflare part of the AI provider card (its markup lives in the Settings view).
 * `modelInput` is the Cloudflare model field in the advanced section.
 */
export function mountCloudflare(root, { modelInput, onData } = {}) {
  const st = { data: null, timer: null };
  const list = $('#cf-keys', root);
  const stripId = (id) => String(id).replace(/^cf-db:/, '');

  function render() {
    const d = st.data;
    if (!d) return;
    if (modelInput && document.activeElement !== modelInput) modelInput.value = d.model || '';
    renderAccounts(list, d.keys, 'cloudflare', 'No Cloudflare account yet.');
    const af = $('#cf-add', root);
    if (!d.keys.length && !af.dataset.touched) af.hidden = false; // nothing added yet: show the form right away
    onData?.(d);
  }
  async function load() {
    try {
      st.data = await api.cfStatus();
      render();
    } catch (e) {
      list.innerHTML = `<p class="empty muted small">Could not load Cloudflare: ${e.message}</p>`;
    }
  }

  $('#cf-add', root).addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      const r = await api.cfAddKey(f.label.value.trim(), f.accountId.value.trim(), f.token.value.trim());
      toast(r.note || 'Account added', 'success');
      f.reset();
      f.hidden = true;
      await load();
    } catch (ex) {
      toast(ex.message || 'Could not add the account', 'error');
    }
  });
  list.addEventListener('click', async (e) => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    b.closest('details')?.removeAttribute('open');
    const id = b.dataset.id;
    try {
      if (b.dataset.act === 'test') {
        const r = await api.cfTest(id);
        toast(r.ok ? r.note || `Works (${r.ms} ms)` : r.error || 'The account did not work', r.ok ? 'success' : 'error');
      } else if (b.dataset.act === 'up' || b.dataset.act === 'down') {
        await api.cfMoveKey(stripId(id), b.dataset.act);
        await load();
      } else if (b.dataset.act === 'remove') {
        const ok = await confirmDialog({ title: 'Remove this account?', message: `“${b.dataset.label}” will be deleted from the server.`, confirmText: 'Remove', danger: true });
        if (!ok) return;
        await api.cfDeleteKey(stripId(id));
        toast('Account removed', 'success');
        await load();
      }
    } catch (ex) {
      toast(ex.message, 'error');
    }
  });

  load();
  st.timer = setInterval(load, 5000);
  return { find: (id) => st.data?.keys.find((k) => k.id === id), data: () => st.data, reload: load, stop: () => clearInterval(st.timer) };
}
