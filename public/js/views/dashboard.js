// Dashboard: KPI tiles + hourly inbound/outbound chart (inline SVG).
import { api } from '../api.js';
import { icons } from '../icons.js';
import {
  $, displayPhone, escapeHtml, formatBytes, formatMs, formatNumber, formatTime, hueFor, html, initials, raw, relativeTime,
} from '../util.js';

const REFRESH_MS = 30000;

function tile(label, value, sub = '') {
  return html`<div class="stat-tile"><div class="stat-label">${label}</div><div class="stat-value">${value}</div>${raw(
    sub ? html`<div class="stat-sub muted">${sub}</div>` : ''
  )}</div>`;
}

function linkTile(href, label, value, sub = '') {
  return html`<a class="stat-tile agenda-tile-link" href="${href}"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-sub muted">${sub}</div></a>`;
}

function chart(hourly) {
  const data = (hourly || []).slice(-24);
  if (!data.length) return '<p class="muted">No traffic in the last 24 hours.</p>';
  const W = 720;
  const H = 220;
  const pad = { l: 34, r: 8, t: 12, b: 26 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const max = Math.max(1, ...data.map((d) => Math.max(d.inbound || 0, d.outbound || 0)));
  const nice = Math.ceil(max / 4) * 4 || 4;
  const slot = iw / data.length;
  const bw = Math.max(2, Math.min(14, slot / 2 - 2));
  const y = (v) => pad.t + ih - (v / nice) * ih;

  let grid = '';
  for (let i = 0; i <= 4; i++) {
    const v = (nice / 4) * i;
    const yy = y(v).toFixed(1);
    grid += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${yy}" y2="${yy}" class="grid"/><text x="${pad.l - 6}" y="${yy}" class="axis" text-anchor="end" dominant-baseline="middle">${v}</text>`;
  }
  let bars = '';
  data.forEach((d, i) => {
    const cx = pad.l + slot * i + slot / 2;
    const hour = new Date(d.hour);
    const label = Number.isNaN(hour.getTime()) ? '' : hour.toLocaleTimeString([], { hour: '2-digit', hour12: false });
    const tip = `${label}:00 — in ${d.inbound || 0}, out ${d.outbound || 0}`;
    const inH = (d.inbound || 0) / nice * ih;
    const outH = (d.outbound || 0) / nice * ih;
    bars += `<g><title>${escapeHtml(tip)}</title>
      <rect x="${(cx - bw - 1).toFixed(1)}" y="${(pad.t + ih - inH).toFixed(1)}" width="${bw.toFixed(1)}" height="${inH.toFixed(1)}" rx="2" class="bar-in"/>
      <rect x="${(cx + 1).toFixed(1)}" y="${(pad.t + ih - outH).toFixed(1)}" width="${bw.toFixed(1)}" height="${outH.toFixed(1)}" rx="2" class="bar-out"/></g>`;
    if (i % Math.ceil(data.length / 8) === 0) bars += `<text x="${cx.toFixed(1)}" y="${H - 8}" class="axis" text-anchor="middle">${escapeHtml(label)}</text>`;
  });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Messages per hour, last 24 hours">${grid}${bars}</svg>
    <div class="legend"><span><i class="sw sw-in"></i>Inbound</span><span><i class="sw sw-out"></i>Outbound</span></div>`;
}

export async function mount(root, { hub }) {
  root.innerHTML = html`
    <div class="page">
      <header class="page-header">
        <div><h1>Dashboard</h1><p class="muted" id="dash-sub">Loading…</p></div>
        <button class="icon-btn" id="dash-refresh" aria-label="Refresh" title="Refresh">${raw(icons.refresh)}</button>
      </header>
      <section class="card"><h2 class="card-title">LLM health</h2><div id="dash-llm"><p class="muted">Checking…</p></div></section>
      <div class="stat-grid focus-cards" id="dash-tiles"></div>
      <section class="card"><h2 class="card-title">Messages per hour (24 h)</h2><div id="dash-chart"></div></section>
      <section class="card"><h2 class="card-title">Active sessions</h2><div id="dash-sessions"></div></section>
    </div>`;

  async function loadSessions() {
    const box = $('#dash-sessions', root);
    try {
      const { sessions } = await api.sessions(true, 50);
      box.innerHTML = (sessions || []).length
        ? html`<ul class="session-list" role="list">${sessions.map((x) =>
            raw(html`<li><a class="session-item" href="#/chats/${encodeURIComponent(x.chatId)}">
              <span class="avatar avatar-sm" style="--h:${hueFor(x.chatId)}">${initials(x.name || x.phone)}</span>
              <span class="session-main"><span>${x.name || displayPhone(x.phone) || x.chatId}</span>
                <span class="muted small">${displayPhone(x.phone)}</span></span>
              <span class="muted small">since ${formatTime(x.startedAt)} · ${relativeTime(x.startedAt)}</span>
              <span class="badge">${formatNumber(x.messageCount)} msgs</span>
            </a></li>`)
          )}</ul>`
        : '<p class="muted">No one is chatting with eSumit right now.</p>';
    } catch (e) {
      box.innerHTML = html`<p class="muted">Could not load sessions: ${e.message}</p>`;
    }
  }

  async function loadLlm() {
    const box = $('#dash-llm', root);
    try {
      const l = await api.llmStatus();
      const tone = { ok: 'badge-ok', degraded: 'badge-warn', down: 'badge-bad' }[l.state] || 'badge-warn';
      const label = { ok: 'Healthy', degraded: 'Degraded', down: 'Down' }[l.state] || l.state;
      const w = l.worker;
      const r = w?.lastReply;
      const row = (k, v) => html`<div class="llm-row"><span class="muted">${k}</span><span>${v}</span></div>`;
      const rows = [
        row('Model', l.model),
        row('Ollama', l.ollama.up ? `up · v${l.ollama.version || '?'}` : 'unreachable'),
        row('Installed', l.installed == null ? '—' : l.installed ? 'yes' : 'no'),
        row('In memory', l.resident ? formatBytes(l.resident.sizeBytes) : 'not loaded'),
        row('Worker', w ? `up · ${w.inFlight} running, ${w.lanes} chats queued (max ${w.concurrency})` : 'no heartbeat'),
        row('Last reply', r ? `${formatMs(r.latencyMs)}${r.tokensPerSec ? ` · ${r.tokensPerSec} tok/s` : ''}${r.cached ? ' · cached' : ''} · ${relativeTime(r.at)}` : '—'),
        row('Last error', w?.lastError ? `${w.lastError.message} · ${relativeTime(w.lastError.at)}` : 'none'),
      ];
      box.innerHTML = html`
        <p><span class="badge badge-dot ${tone}">${label}</span> <span class="muted">${l.message}</span></p>
        <div class="llm-grid">${raw(rows.join(''))}</div>`;
    } catch (e) {
      box.innerHTML = html`<p class="muted">Could not load LLM status: ${e.message}</p>`;
    }
  }

  async function load() {
    loadSessions();
    loadLlm();
    try {
      const st = await api.stats();
      $('#dash-sub', root).innerHTML = html`WhatsApp <strong>${st.waState || '—'}</strong> · model <strong>${st.model || '—'}</strong>`;
      $('#dash-tiles', root).innerHTML = [
        tile('Messages today', formatNumber(st.messagesToday), `${formatNumber(st.inboundToday)} inbound`),
        tile('AI replies today', formatNumber(st.aiRepliesToday), `${formatNumber(st.cacheHits)} from cache`),
        tile('Avg latency', formatMs(st.avgLatencyMs), `p95 ${formatMs(st.p95LatencyMs)}`),
        tile('Active chats (24 h)', formatNumber(st.activeChats24h)),
        tile('Active sessions', formatNumber(st.activeSessions), `${formatNumber(st.sessionsToday)} started today`),
        ...(st.agenda
          ? [linkTile('#/agenda', 'Agenda', formatNumber(st.agenda.review), `needs review · ${formatNumber(st.agenda.today)} today · ${formatNumber(st.agenda.overdue)} overdue`)]
          : []),
        tile('Contacts', formatNumber(st.contacts)),
        tile('Knowledge sections', formatNumber(st.kbSections)),
      ].join('');
      $('#dash-chart', root).innerHTML = chart(st.hourly);
    } catch (e) {
      $('#dash-sub', root).textContent = `Could not load stats: ${e.message}`;
    }
  }

  $('#dash-refresh', root).addEventListener('click', load);
  const timer = setInterval(() => document.visibilityState === 'visible' && load(), REFRESH_MS);
  const offStatus = hub.on('wa:status', (s) => {
    const strong = $('#dash-sub strong', root);
    if (strong && s?.state) strong.textContent = s.state;
  });
  let sessTimer = 0;
  const offChat = hub.on('chat:update', () => {
    clearTimeout(sessTimer);
    sessTimer = setTimeout(loadSessions, 500);
  });
  let llmTimer = 0;
  const offLlm = hub.on('llm:status', () => {
    clearTimeout(llmTimer);
    llmTimer = setTimeout(loadLlm, 400);
  });
  const off = () => {
    offLlm();
    clearTimeout(llmTimer);
    offStatus();
    offChat();
    clearTimeout(sessTimer);
  };
  await load();
  return {
    destroy() {
      clearInterval(timer);
      off();
    },
  };
}
