// Limit bars for the hosted AI providers (Groq, Cloudflare Workers AI): the "i" button next to every account opens a small
// panel with, per model, how much of each limit is used, whether it is exhausted, how long until it resets and at what time.
// Also the Cloudflare card (its keys, model and live status).
import { api } from '../api.js';
import { $, confirmDialog, formatNumber, html, raw, toast } from '../util.js';

const pad = (n) => String(n).padStart(2, '0');

/** 7_380_000 → "2h 3m", 45_000 → "45s" */
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

/** One limit as a bar: label, used/limit, and when it resets. `limit` null → just text. */
function bar({ label, used, limit, resetAt, now }) {
  const pct = limit ? Math.min(100, Math.max(0, (used / limit) * 100)) : 0;
  const tone = limit && used >= limit ? 'bad' : pct >= 85 ? 'bad' : pct >= 60 ? 'warn' : 'ok';
  const full = limit && used >= limit;
  const reset = resetAt && resetAt > now ? `resets in ${formatLeft(resetAt - now)} (at ${formatAt(resetAt)})` : 'no reset time known yet';
  return html`<div class="lim-row">
    <div class="lim-head"><span>${label}</span><span>${limit ? `${compact(used)} of ${compact(limit)} used${full ? ' · exhausted' : ` · ${Math.round(100 - pct)}% left`}` : 'limit not known yet'}</span></div>
    <div class="meter" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(pct)}"><span class="meter-${tone}" style="width:${pct.toFixed(1)}%"></span></div>
    <div class="lim-reset muted small">${reset}</div>
  </div>`;
}

function modelSection(m, kind, now) {
  const name = String(m.id).split('/').pop();
  const lim = m.limits || {};
  const bars = [];
  if (lim.neurons) bars.push(bar({ label: 'Free neurons today (estimate)', used: lim.neurons.used, limit: lim.neurons.limit, resetAt: lim.neurons.resetAt, now }));
  if (lim.requests?.limit != null) bars.push(bar({ label: 'Requests today', used: lim.requests.limit - (lim.requests.remaining ?? lim.requests.limit), limit: lim.requests.limit, resetAt: lim.requests.resetAt, now }));
  if (lim.tokens?.limit != null) bars.push(bar({ label: 'Tokens this minute', used: lim.tokens.limit - (lim.tokens.remaining ?? lim.tokens.limit), limit: lim.tokens.limit, resetAt: lim.tokens.resetAt, now }));
  let status = '';
  if (m.gone) {
    status = html`<div class="lim-note lim-bad">This account cannot use this model (Groq says it does not exist for it).</div>`;
  } else if (m.state === 'cooling' && m.restUntil) {
    const what = m.hit?.what ? ` ${m.hit.what}${m.hit.limit ? ` (limit ${formatNumber(m.hit.limit)}${m.hit.used != null ? `, used ${formatNumber(m.hit.used)}` : ''})` : ''}` : '';
    status = html`<div class="lim-note lim-bad">Limit reached:${what || ' rate limit'}. Available again in <strong>${formatLeft(m.restUntil - now)}</strong> (at ${formatAt(m.restUntil)}).</div>`;
  } else if (m.state === 'disabled') {
    status = html`<div class="lim-note lim-bad">The provider rejected this key.</div>`;
  }
  if (!bars.length && !status) bars.push(html`<p class="muted small">No limit figures yet: they appear after this model answers its first request.</p>`);
  const state = m.state === 'ready' ? html`<span class="badge badge-ok">ready</span>` : html`<span class="badge badge-bad">${m.gone ? 'not available' : 'resting'}</span>`;
  return html`<section class="lim-model"><h4>${name} ${raw(state)}</h4>${raw(status)}${raw(bars.join(''))}</section>`;
}

/** The panel's contents for one account. */
export function limitsPanel(key, kind = 'groq') {
  const now = Date.now();
  const u = key.usage?.today;
  const sections = (key.models || []).map((m) => modelSection(m, kind, now)).join('');
  const usage = u ? html`<p class="muted small lim-usage">Today: ${formatNumber(u.calls)} calls${u.limited ? ` · ${u.limited} rate-limited` : ''}${kind === 'cloudflare' ? ` · ~${formatNumber(Math.round(u.neurons || 0))} neurons` : ` · ${compact(u.tokens)} tokens`}</p>` : '';
  const note = kind === 'cloudflare' ? html`<p class="muted small">Cloudflare sends no usage headers: the bar is our estimate from token counts. The free allowance is shared by all models and resets at 00:00 UTC.</p>` : html`<p class="muted small">Groq reports limits per model: requests per day and tokens per minute.</p>`;
  return html`<h3>${key.label}</h3>${raw(sections)}${raw(usage)}${raw(note)}`;
}

/** The "i" button. `data-kind` + `data-id` find the account again when it is hovered. */
export const infoButton = (kind, id) => html`<button type="button" class="info-btn" data-kind="${kind}" data-id="${id}" aria-label="Usage limits and reset time" title="Usage limits and reset time">i</button>`;

/**
 * Show the panel while an "i" button is hovered or focused (and toggle it on tap). `find(kind, id)` returns the account.
 * One panel for the whole page, positioned next to the button, so scrolling tables never clip it.
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
    const w = Math.min(360, window.innerWidth - 16);
    pop.style.width = `${w}px`;
    const left = Math.min(Math.max(8, r.right - w), window.innerWidth - w - 8);
    pop.style.left = `${left}px`;
    const below = r.bottom + 6;
    pop.style.top = `${Math.min(below, Math.max(8, window.innerHeight - pop.offsetHeight - 8))}px`;
  };
  const hide = () => {
    pop.hidden = true;
    current = null;
  };
  root.addEventListener('mouseover', (e) => {
    const b = e.target.closest?.('.info-btn');
    if (b && b !== current) show(b);
  });
  root.addEventListener('mouseout', (e) => {
    if (e.target.closest?.('.info-btn') && !e.relatedTarget?.closest?.('.limit-pop')) hide();
  });
  root.addEventListener('focusin', (e) => e.target.closest?.('.info-btn') && show(e.target.closest('.info-btn')));
  root.addEventListener('focusout', (e) => e.target.closest?.('.info-btn') && hide());
  root.addEventListener('click', (e) => {
    const b = e.target.closest?.('.info-btn'); // touch screens have no hover: tap toggles
    if (!b) return;
    if (current === b && !pop.hidden) hide();
    else show(b);
  });
  window.addEventListener('scroll', hide, { passive: true, capture: true });
  return { refresh: () => current && !pop.hidden && show(current), destroy: () => pop.remove() };
}

// ---------------------------------------------------------------- Cloudflare card

// Rough replies per day on the 10,000 free neurons, for a ~1,500-token prompt and a ~150-token reply (Cloudflare's rates).
const CF_MODELS = [
  ['@cf/google/gemma-4-26b-a4b-it', 'about 560 replies a day · strong, handles Hindi well (recommended)'],
  ['@cf/meta/llama-3.1-8b-instruct-fp8', 'about 400 replies a day · solid all-rounder'],
  ['@cf/meta/llama-3.2-3b-instruct', 'about 870 replies a day · lighter, a bit weaker'],
  ['@cf/ibm-granite/granite-4.0-h-micro', 'about 3,500 replies a day · very light, weak on nuance'],
];

const CF_CARD = html`
<section class="card" id="cf-card">
  <h2 class="card-title">Cloudflare Workers AI <span class="muted small">(second backup after Groq)</span></h2>
  <p class="muted small" id="cf-summary">Loading…</p>
  <form id="cf-form" class="stack">
    <label class="field"><span>Model <span class="muted small">(free tier: 10,000 neurons a day for the whole account, shared by all models, resets 00:00 UTC)</span></span>
      <input name="cloudflareModel" list="cf-models" placeholder="@cf/google/gemma-4-26b-a4b-it" autocomplete="off" />
      <datalist id="cf-models">${raw(CF_MODELS.map(([id, label]) => `<option value="${id}">${label}</option>`).join(''))}</datalist></label>
    <div class="form-actions"><button class="btn btn-primary" type="submit">Save model</button></div>
  </form>
  <h3 class="section-title">Cloudflare accounts</h3>
  <div id="cf-keys" class="table-scroll"></div>
  <form id="cf-add" class="inline-form">
    <input name="label" placeholder="Label (optional)" aria-label="Label" maxlength="60" />
    <input name="accountId" placeholder="Account ID (32 characters)" autocomplete="off" aria-label="Cloudflare account id" required />
    <input name="token" type="password" placeholder="API token (Workers AI)" autocomplete="off" aria-label="Cloudflare API token" required />
    <button class="btn btn-ghost" type="submit">Add account</button>
  </form>
  <p class="muted small">Create the token in Cloudflare → My Profile → API Tokens → "Workers AI" template. Order: Groq first, then these accounts, then the local model. Each Cloudflare account has its own 10,000 free neurons a day.</p>
</section>`;

/** Insert the Cloudflare card after the Groq card and keep it live. Returns { find, stop }. */
export function mountCloudflare(root, { onChange } = {}) {
  $('#groq-card', root).insertAdjacentHTML('afterend', CF_CARD);
  const st = { data: null, timer: null };
  const form = $('#cf-form', root);
  const stripId = (id) => String(id).replace(/^cf-db:/, '');

  function render() {
    const d = st.data;
    if (!d) return;
    if (!form.contains(document.activeElement)) form.elements.cloudflareModel.value = d.model || '';
    const ready = d.keys.filter((k) => k.state === 'ready').length;
    $('#cf-summary', root).textContent = d.keys.length
      ? `${ready} of ${d.keys.length} account${d.keys.length === 1 ? '' : 's'} ready · model ${d.model}. Used after Groq runs out (Auto and Cloud-only modes), before the local model.`
      : 'No Cloudflare account added yet. Add one below to get a free second backup after Groq.';
    const consoleKeys = d.keys.filter((k) => k.source === 'console');
    const rows = d.keys.map((k, i) => {
      const m = (k.models || [])[0];
      const n = m?.limits?.neurons;
      const pct = n ? Math.min(100, (n.used / n.limit) * 100) : 0;
      const tone = !n ? 'ok' : n.remaining <= 0 ? 'bad' : pct >= 85 ? 'bad' : pct >= 60 ? 'warn' : 'ok';
      const chip = k.state === 'disabled' ? html`<span class="badge badge-bad">Rejected by Cloudflare</span>` : k.state === 'ready' ? html`<span class="badge badge-ok">ready</span>` : html`<span class="badge badge-warn">allowance used up</span>`;
      const ci = consoleKeys.findIndex((x) => x.id === k.id);
      const move = k.source === 'console' ? html`<button class="btn btn-ghost btn-sm" data-act="up" data-id="${k.id}" ${ci <= 0 ? 'disabled' : ''} aria-label="Move up">↑</button><button class="btn btn-ghost btn-sm" data-act="down" data-id="${k.id}" ${ci < 0 || ci >= consoleKeys.length - 1 ? 'disabled' : ''} aria-label="Move down">↓</button> ` : '';
      const remove = k.source === 'console' ? html` <button class="btn btn-ghost btn-sm agenda-del" data-act="remove" data-id="${k.id}" data-label="${k.label}">Remove</button>` : '';
      const meter = n ? html`<div class="meter mini" title="${Math.round(n.used)} of ${n.limit} neurons"><span class="meter-${tone}" style="width:${pct.toFixed(1)}%"></span></div><div class="muted small">~${formatNumber(Math.round(n.used))} of ${formatNumber(n.limit)} neurons today</div>` : html`<span class="muted small">No usage yet</span>`;
      return html`<tr><td><strong>Account ${k.priority ?? i + 1}: ${k.label}</strong> ${raw(infoButton('cloudflare', k.id))}<div class="muted small">${k.hint} · ${k.source === 'env' ? 'server env' : 'console'}</div></td><td>${raw(chip)}${raw(k.lastError && k.state !== 'ready' ? html`<div class="muted small">${k.lastError}</div>` : '')}</td><td>${raw(meter)}</td><td class="actions">${raw(move)}<button class="btn btn-ghost btn-sm" data-act="test" data-id="${k.id}">Test</button>${raw(remove)}</td></tr>`;
    });
    $('#cf-keys', root).innerHTML = rows.length
      ? html`<table class="table"><thead><tr><th>Account (priority order)</th><th>Status</th><th>Free allowance today</th><th></th></tr></thead><tbody>${raw(rows.join(''))}</tbody></table>`
      : '';
  }

  async function load() {
    try {
      st.data = await api.cfStatus();
      render();
    } catch (e) {
      $('#cf-summary', root).textContent = `Could not load the Cloudflare status: ${e.message}`;
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      await api.saveSettings({ cloudflareModel: form.elements.cloudflareModel.value.trim() });
      toast('Cloudflare model saved', 'success');
      await load();
    } catch (ex) {
      toast(ex.message || 'Could not save', 'error');
    }
  });
  $('#cf-add', root).addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      const r = await api.cfAddKey(f.label.value.trim(), f.accountId.value.trim(), f.token.value.trim());
      toast(r.note || 'Cloudflare account added. It is used after Groq runs out', 'success');
      f.reset();
      await load();
      onChange?.();
    } catch (ex) {
      toast(ex.message || 'Could not add the account', 'error');
    }
  });
  $('#cf-keys', root).addEventListener('click', async (e) => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    const id = b.dataset.id;
    try {
      if (b.dataset.act === 'test') {
        b.disabled = true;
        const r = await api.cfTest(id);
        toast(r.ok ? r.note || `Account works (${r.ms} ms)` : r.error || 'The account did not work', r.ok ? 'success' : 'error');
        b.disabled = false;
      } else if (b.dataset.act === 'up' || b.dataset.act === 'down') {
        await api.cfMoveKey(stripId(id), b.dataset.act);
        await load();
      } else if (b.dataset.act === 'remove') {
        const ok = await confirmDialog({ title: 'Remove this account?', message: `“${b.dataset.label}” will be deleted from the server. Other accounts keep working.`, confirmText: 'Remove', danger: true });
        if (!ok) return;
        await api.cfDeleteKey(stripId(id));
        toast('Account removed', 'success');
        await load();
      }
    } catch (ex) {
      b.disabled = false;
      toast(ex.message, 'error');
    }
  });

  load();
  st.timer = setInterval(load, 5000);
  return { find: (id) => st.data?.keys.find((k) => k.id === id), stop: () => clearInterval(st.timer) };
}
