// Small, dependency-free helpers shared by every view.

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** Escape any value for safe insertion into HTML. Always use for user/message content. */
export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ESC[c]);
}

/** Tagged template that escapes interpolations unless wrapped with raw(). */
export function html(strings, ...values) {
  let out = '';
  strings.forEach((s, i) => {
    out += s;
    if (i < values.length) {
      const v = values[i];
      if (v && v.__raw) out += v.html;
      else if (Array.isArray(v)) out += v.map((x) => (x && x.__raw ? x.html : escapeHtml(x))).join('');
      else out += escapeHtml(v);
    }
  });
  return out;
}

/** Mark a trusted HTML string so html`` won't escape it. */
export function raw(str) {
  return { __raw: true, html: String(str ?? '') };
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function debounce(fn, ms = 250) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function toDate(d) {
  if (!d) return null;
  const date = d instanceof Date ? d : new Date(d);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatTime(d) {
  const date = toDate(d);
  if (!date) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function formatDateTime(d) {
  const date = toDate(d);
  if (!date) return '';
  return date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' });
}

export function isSameDay(a, b) {
  const x = toDate(a);
  const y = toDate(b);
  return !!x && !!y && x.toDateString() === y.toDateString();
}

/** "Today", "Yesterday", weekday within a week, else a date. */
export function dayLabel(d) {
  const date = toDate(d);
  if (!date) return '';
  const now = new Date();
  if (isSameDay(date, now)) return 'Today';
  const y = new Date(now);
  y.setDate(now.getDate() - 1);
  if (isSameDay(date, y)) return 'Yesterday';
  const diffDays = (now - date) / 86400000;
  if (diffDays < 7) return date.toLocaleDateString([], { weekday: 'long' });
  return date.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' });
}

/** Chat-list style stamp: time today, "Yesterday", or short date. */
export function listStamp(d) {
  const date = toDate(d);
  if (!date) return '';
  if (isSameDay(date, new Date())) return formatTime(date);
  const label = dayLabel(date);
  if (label === 'Yesterday') return label;
  return date.toLocaleDateString([], { day: '2-digit', month: '2-digit', year: '2-digit' });
}

export function relativeTime(d) {
  const date = toDate(d);
  if (!date) return '—';
  const s = Math.round((Date.now() - date.getTime()) / 1000);
  if (s < 45) return 'just now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const days = Math.round(h / 24);
  if (days < 30) return `${days} d ago`;
  return date.toLocaleDateString();
}

export function formatMs(ms) {
  if (ms == null || Number.isNaN(Number(ms))) return '—';
  const n = Number(ms);
  return n < 1000 ? `${Math.round(n)} ms` : `${(n / 1000).toFixed(n < 10000 ? 1 : 0)} s`;
}

export function formatBytes(n) {
  const b = Number(n) || 0;
  if (b < 1024) return `${b} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = b / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function formatNumber(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString();
}

export function initials(name) {
  const s = String(name || '').trim();
  if (!s) return '?';
  if (/^\+?\d/.test(s)) return '#';
  const parts = s.split(/\s+/).filter(Boolean);
  return ((parts[0]?.[0] || '') + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase();
}

/** Stable hue per id so avatars keep their color. */
export function hueFor(id) {
  let h = 0;
  const s = String(id || '');
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

export function displayPhone(phone) {
  const d = String(phone || '').replace(/\D/g, '');
  return d ? `+${d}` : '';
}

/**
 * Render WhatsApp-style formatting safely: escape first, then apply
 * ```code```, `inline`, *bold*, _italic_, ~strike~ and links.
 */
export function formatWhatsApp(text) {
  const blocks = [];
  let s = escapeHtml(text ?? '');
  s = s.replace(/```([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre class="wa-code"><code>${code.replace(/^\n/, '')}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  s = s.replace(/`([^`\n]+)`/g, (_, c) => {
    blocks.push(`<code class="wa-inline">${c}</code>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  const edge = '(^|[\\s([{>.,!?;:"\'-])';
  s = s.replace(new RegExp(`${edge}\\*(\\S(?:[^*\\n]*?\\S)?)\\*(?=$|[\\s)\\]}.,!?;:<"'-])`, 'g'), '$1<strong>$2</strong>');
  s = s.replace(new RegExp(`${edge}_(\\S(?:[^_\\n]*?\\S)?)_(?=$|[\\s)\\]}.,!?;:<"'-])`, 'g'), '$1<em>$2</em>');
  s = s.replace(new RegExp(`${edge}~(\\S(?:[^~\\n]*?\\S)?)~(?=$|[\\s)\\]}.,!?;:<"'-])`, 'g'), '$1<del>$2</del>');
  s = s.replace(/\bhttps?:\/\/[^\s<]+[^\s<.,;:!?)\]'"]/g, (url) => {
    // url is already escaped; &amp; etc. are fine inside href.
    return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`;
  });
  s = s.replace(/\n/g, '<br>');
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
  return s;
}

// ---------- toast ----------
let toastHost;
export function toast(message, type = 'info', ms = 3200) {
  if (!toastHost) {
    toastHost = document.createElement('div');
    toastHost.className = 'toast-host';
    toastHost.setAttribute('role', 'status');
    toastHost.setAttribute('aria-live', 'polite');
    document.body.appendChild(toastHost);
  }
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = String(message);
  toastHost.appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 250);
  }, ms);
}

/** Promise-based confirm dialog using <dialog>. */
/**
 * Modal confirm. Resolves true/false. With `option: { label, checked }` it shows a checkbox and resolves
 * `{ ok, checked }` instead (the checkbox state is read when the dialog closes).
 */
export function confirmDialog({ title, message, confirmText = 'Confirm', danger = false, option = null }) {
  return new Promise((resolve) => {
    const dlg = document.createElement('dialog');
    dlg.className = 'dialog';
    dlg.innerHTML = html`
      <form method="dialog">
        <h3>${title}</h3>
        <p>${message}</p>
        ${raw(option ? html`<label class="switch-row"><input type="checkbox" class="switch" id="dlg-option" ${raw(option.checked ? 'checked' : '')} aria-label="${option.label}" /><span>${option.label}</span></label>` : '')}
        <div class="dialog-actions">
          <button value="cancel" class="btn btn-ghost">Cancel</button>
          <button value="ok" class="btn ${raw(danger ? 'btn-danger' : 'btn-primary')}">${confirmText}</button>
        </div>
      </form>`;
    document.body.appendChild(dlg);
    dlg.addEventListener('close', () => {
      const ok = dlg.returnValue === 'ok';
      resolve(option ? { ok, checked: !!dlg.querySelector('#dlg-option')?.checked } : ok);
      dlg.remove();
    });
    dlg.showModal();
  });
}

export function storageGet(key, fallback = null) {
  try {
    const v = localStorage.getItem(key);
    return v == null ? fallback : v;
  } catch {
    return fallback;
  }
}

export function storageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: ignore */
  }
}

/** Simple event hub so views can subscribe to socket events and clean up. */
export class Hub {
  constructor() {
    this.map = new Map();
  }
  on(evt, fn) {
    if (!this.map.has(evt)) this.map.set(evt, new Set());
    this.map.get(evt).add(fn);
    return () => this.map.get(evt)?.delete(fn);
  }
  emit(evt, payload) {
    this.map.get(evt)?.forEach((fn) => {
      try {
        fn(payload);
      } catch (e) {
        console.error(e);
      }
    });
  }
}
