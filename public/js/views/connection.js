// WhatsApp connection control: QR, status, connect/disconnect/logout/restart.
import { api } from '../api.js';
import { icons } from '../icons.js';
import { evervault, evervaultMarkup } from '../fx.js';
import { $, confirmDialog, displayPhone, formatDateTime, html, raw, relativeTime, toast } from '../util.js';

const STATE_TEXT = {
  DISCONNECTED: { title: 'Disconnected', tone: 'bad', text: 'The WhatsApp client is not running. Connect to start it.' },
  INITIALIZING: { title: 'Starting…', tone: 'warn', text: 'Launching WhatsApp Web on the server. This takes a few seconds.' },
  QR: { title: 'Scan the QR code', tone: 'warn', text: 'Link this server as a device on your phone.' },
  AUTHENTICATED: { title: 'Authenticated', tone: 'warn', text: 'Phone linked. Syncing chats…' },
  READY: { title: 'Connected', tone: 'ok', text: 'The assistant is online and handling messages.' },
};

export async function mount(root, { hub, state }) {
  root.innerHTML = '<div class="page connect-page"><div id="conn"></div></div>';
  const el = $('#conn', root);
  let busy = null;
  let stopFx = () => {};

  function render(s) {
    const info = STATE_TEXT[s.state] || STATE_TEXT.DISCONNECTED;
    const st = s.state;
    const can = {
      connect: st === 'DISCONNECTED',
      disconnect: st !== 'DISCONNECTED',
      logout: st === 'READY' || st === 'AUTHENTICATED',
      restart: st !== 'INITIALIZING',
    };
    el.innerHTML = html`
      <section class="card conn-card">
        <div class="conn-status">
          <span class="status-pill tone-${info.tone}"><span class="pill-dot"></span>${info.title}</span>
          <span class="muted small">${s.since ? `since ${relativeTime(s.since)}` : ''}</span>
        </div>
        <div class="conn-body">
          <div class="conn-text">
            <h1>${info.title}</h1>
            <p class="muted">${info.text}</p>
            ${raw(
              st === 'QR'
                ? `<ol class="steps">
                    <li>Open <strong>WhatsApp</strong> on your phone</li>
                    <li>Tap <strong>Menu ⋮</strong> or <strong>Settings</strong> → <strong>Linked devices</strong></li>
                    <li>Tap <strong>Link a device</strong> and point your phone at this screen</li>
                  </ol>
                  <p class="muted small">The code refreshes automatically about every 20 seconds.</p>`
                : ''
            )}
            ${raw(
              st === 'READY' && s.me
                ? html`<div class="me-card">
                    <span class="avatar" style="--h:150">${(s.me.pushname || '?').slice(0, 1).toUpperCase()}</span>
                    <div><div class="conv-name">${s.me.pushname || 'WhatsApp account'}</div>
                    <div class="muted small">${displayPhone(s.me.phone) || s.me.wid || ''}</div></div>
                  </div>`
                : ''
            )}
            ${raw(s.lastError ? html`<p class="form-error">Last error: ${s.lastError}</p>` : '')}
            <div class="conn-actions">
              <button class="btn btn-primary" data-act="connect" ${raw(can.connect && !busy ? '' : 'disabled')}>${raw(icons.play)} Connect</button>
              <button class="btn btn-ghost" data-act="disconnect" ${raw(can.disconnect && !busy ? '' : 'disabled')}>${raw(icons.pause)} Disconnect</button>
              <button class="btn btn-ghost" data-act="restart" ${raw(can.restart && !busy ? '' : 'disabled')}>${raw(icons.refresh)} Restart</button>
              <button class="btn btn-danger" data-act="logout" ${raw(can.logout && !busy ? '' : 'disabled')}>${raw(icons.logout)} Log out</button>
            </div>
            <p class="muted small">Disconnect keeps the saved session so you can reconnect without scanning. Log out unlinks the device and deletes the session; you can also delete the chats, messages and AI memory stored on this server.</p>
          </div>
          <div class="conn-visual">
            ${raw(
              st === 'QR' && s.qr
                ? html`<div class="qr-ev">${raw(evervaultMarkup(''))}<div class="qr-frame"><img src="${s.qr}" alt="WhatsApp pairing QR code" width="264" height="264" /></div></div>`
                : st === 'INITIALIZING' || st === 'AUTHENTICATED' || (st === 'QR' && !s.qr)
                  ? '<div class="qr-frame placeholder"><div class="spinner"></div></div>'
                  : html`<div class="qr-frame placeholder tone-${info.tone}">${raw(st === 'READY' ? icons.check : icons.phone)}</div>`
            )}
          </div>
        </div>
        ${raw(s.since ? html`<p class="muted small conn-foot">State changed ${formatDateTime(s.since)}</p>` : '')}
      </section>`;
    // Evervault glow around the QR (pointer effect only; the code itself stays on white).
    stopFx();
    const ev = el.querySelector('.qr-ev .ev');
    if (ev) stopFx = evervault(ev, { length: 1200 });
  }

  el.addEventListener('click', async (e) => {
    const btn = e.target.closest('button[data-act]');
    if (!btn || btn.disabled) return;
    const act = btn.dataset.act;
    let body = {};
    if (act === 'logout') {
      const { ok, checked } = await confirmDialog({
        title: 'Log out of WhatsApp?',
        message: 'This unlinks the server from your phone and wipes the saved WhatsApp session. You will need to scan a new QR code. Chats, messages and the AI’s memory of people are stored on this server and stay unless you also delete them below.',
        confirmText: 'Log out',
        danger: true,
        option: { label: 'Also delete all chats, messages, agenda and AI memory from this server (cannot be undone)', checked: true },
      });
      if (!ok) return;
      body = { wipeData: checked };
    }
    busy = act;
    render(state.wa);
    try {
      const r = await api.waAction(act, body);
      if (r?.state) hub.emit('wa:status', { ...state.wa, state: r.state });
    } catch (ex) {
      toast(`${act} failed: ${ex.message}`, 'error');
    } finally {
      busy = null;
      render(state.wa);
    }
  });

  const off = hub.on('wa:status', () => render(state.wa));
  render(state.wa);
  try {
    hub.emit('wa:status', await api.waStatus());
  } catch (e) {
    toast(`Could not load status: ${e.message}`, 'error');
  }
  return {
    destroy() {
      off();
      stopFx();
    },
  };
}
