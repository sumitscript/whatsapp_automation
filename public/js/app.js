// App bootstrap: auth gate, shell (nav rail), hash router, shared state.
import { api, onUnauthorized } from './api.js';
import { connectSocket, disconnectSocket } from './socket.js';
import { icons } from './icons.js';
import { $, Hub, html, raw, storageGet, storageSet, toast } from './util.js';

const hub = new Hub();

export const state = {
  user: null,
  wa: { state: 'DISCONNECTED', me: null, qr: null, since: null, lastError: null },
  socket: 'disconnected',
};

const NAV = [
  { route: 'chats', label: 'Chats', icon: icons.chats },
  { route: 'contacts', label: 'Contacts', icon: icons.contacts },
  { route: 'agenda', label: 'Agenda', icon: icons.agenda },
  { route: 'dashboard', label: 'Dashboard', icon: icons.dashboard },
  { route: 'logs', label: 'Logs', icon: icons.logs },
  { route: 'kb', label: 'Knowledge base', icon: icons.kb },
  { route: 'settings', label: 'Settings & models', icon: icons.settings },
  { route: 'connect', label: 'WhatsApp connection', icon: icons.connect },
];

const VIEWS = {
  chats: () => import('./views/chats.js'),
  contacts: () => import('./views/contacts.js'),
  agenda: () => import('./views/agenda.js'),
  dashboard: () => import('./views/dashboard.js'),
  logs: () => import('./views/logs.js'),
  kb: () => import('./views/knowledge.js'),
  settings: () => import('./views/settings.js'),
  connect: () => import('./views/connection.js'),
};

// ---------- theme ----------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $('#theme-toggle');
  if (btn) {
    btn.innerHTML = theme === 'dark' ? icons.sun : icons.moon;
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    btn.title = btn.getAttribute('aria-label');
  }
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = theme === 'dark' ? '#111b21' : '#008069';
}
function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  storageSet('wa-admin-theme', next);
  applyTheme(next);
}

// ---------- router ----------
export function navigate(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  const [name, ...rest] = h.split('/');
  return { name: name || 'chats', params: rest.map((p) => decodeURIComponent(p)) };
}

let current = { name: null, cleanup: null, token: 0 };

async function route() {
  if (!state.user) return;
  const { name, params } = parseHash();
  const viewName = VIEWS[name] ? name : 'chats';
  const token = ++current.token;

  // Chats view handles its own sub-route changes without remounting.
  if (current.name === viewName && current.instance?.update) {
    current.instance.update(params);
    return;
  }
  teardownView();
  const root = $('#view');
  root.innerHTML = '<div class="view-loading"><div class="spinner" aria-label="Loading"></div></div>';
  document.querySelectorAll('.rail-item').forEach((a) => {
    const active = a.dataset.route === viewName;
    a.classList.toggle('active', active);
    if (active) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  try {
    const mod = await VIEWS[viewName]();
    if (token !== current.token) return;
    root.innerHTML = '';
    root.dataset.view = viewName;
    const instance = (await mod.mount(root, { params, hub, state, navigate })) || {};
    if (token !== current.token) {
      instance.destroy?.();
      return;
    }
    current = { ...current, name: viewName, instance };
    const nav = NAV.find((n) => n.route === viewName);
    document.title = `${nav ? nav.label : 'Admin'} · eSumit Console`;
  } catch (e) {
    console.error(e);
    root.innerHTML = html`<div class="empty-state"><h3>Could not load this page</h3><p>${e.message}</p></div>`;
  }
}

function teardownView() {
  try {
    current.instance?.destroy?.();
  } catch (e) {
    console.error(e);
  }
  current = { ...current, name: null, instance: null };
}

// ---------- shell ----------
function renderShell() {
  document.body.innerHTML = html`
    <div class="app">
      <nav class="rail" aria-label="Main">
        <div class="rail-brand" title="eSumit Console">${raw(icons.logo)}</div>
        <div class="rail-items">
          ${NAV.map((n) =>
            raw(html`<a class="rail-item" href="#/${n.route}" data-route="${n.route}" aria-label="${n.label}" title="${n.label}">
              ${raw(n.icon)}${raw(n.route === 'connect' ? '<span class="status-dot" id="wa-dot"></span>' : '')}
              <span class="rail-label">${n.label.split(' ')[0]}</span>
            </a>`)
          )}
        </div>
        <div class="rail-bottom">
          <button class="rail-item" id="theme-toggle" type="button"></button>
          <button class="rail-item" id="logout-btn" type="button" aria-label="Sign out" title="Sign out">${raw(icons.logout)}</button>
        </div>
      </nav>
      <div class="main">
        <div class="qr-banner" id="qr-banner" hidden>
          <span>${raw(icons.connect)} WhatsApp is waiting for you to scan the QR code.</span>
          <a href="#/connect" class="btn btn-sm btn-primary">Open QR</a>
        </div>
        <main id="view" class="view" tabindex="-1"></main>
      </div>
    </div>`;
  applyTheme(document.documentElement.dataset.theme || 'dark');
  $('#theme-toggle').addEventListener('click', toggleTheme);
  $('#logout-btn').addEventListener('click', async () => {
    try {
      await api.logout();
    } catch {
      /* ignore */
    }
    showLogin();
  });
  renderWaIndicators();
}

function waTone(s) {
  if (s === 'READY') return 'ok';
  if (s === 'QR' || s === 'INITIALIZING' || s === 'AUTHENTICATED') return 'warn';
  return 'bad';
}

function renderWaIndicators() {
  const dot = $('#wa-dot');
  if (dot) {
    dot.dataset.tone = waTone(state.wa.state);
    dot.title = `WhatsApp: ${state.wa.state}`;
  }
  const banner = $('#qr-banner');
  if (banner) banner.hidden = !(state.wa.state === 'QR' && parseHash().name !== 'connect');
}

hub.on('wa:status', (s) => {
  state.wa = { ...state.wa, ...s };
  renderWaIndicators();
});
hub.on('socket:state', (s) => {
  state.socket = s;
});
window.addEventListener('hashchange', () => {
  route();
  renderWaIndicators();
});

// ---------- auth ----------
async function showLogin() {
  teardownView();
  disconnectSocket();
  state.user = null;
  const { mount } = await import('./views/login.js');
  document.body.innerHTML = '<div id="login-root"></div>';
  mount($('#login-root'), {
    onLogin: (user) => startApp(user),
  });
}

async function startApp(user) {
  state.user = user;
  renderShell();
  connectSocket(hub, { onUnauthorized: () => showLogin() });
  api
    .waStatus()
    .then((s) => hub.emit('wa:status', s))
    .catch(() => {});
  route();
  // Warm the other views while the browser is idle so switching tabs is instant.
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 1200));
  idle(() => Object.values(VIEWS).forEach((load) => load().catch(() => {})));
}

onUnauthorized(() => {
  if (state.user) toast('Session expired — please sign in again.', 'warn');
  showLogin();
});

async function boot() {
  const saved = storageGet('wa-admin-theme');
  const prefersLight = window.matchMedia?.('(prefers-color-scheme: light)').matches;
  applyTheme(saved || (prefersLight ? 'light' : 'dark'));
  try {
    const { user } = await api.me();
    if (user) return startApp(user);
  } catch {
    /* not signed in */
  }
  showLogin();
}

boot();
