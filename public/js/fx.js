// Motion effects without libraries: vanilla ports of Aceternity's EvervaultCard,
// ContainerScroll and MacbookScroll (FocusCards is pure CSS: `.focus-cards`).
// Everything is driven by CSS custom properties written at most once per frame;
// with prefers-reduced-motion the effects render in their final, static state.
import { icons } from './icons.js';
import { html, raw } from './util.js';

export const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Piecewise-linear map, like motion's useTransform(progress, input, output). */
export function mapRange(p, input, output) {
  if (p <= input[0]) return output[0];
  for (let i = 1; i < input.length; i++) {
    if (p <= input[i]) {
      const t = (p - input[i - 1]) / (input[i] - input[i - 1] || 1);
      return output[i - 1] + t * (output[i] - output[i - 1]);
    }
  }
  return output.at(-1);
}

// ---------------- scroll progress (shared, rAF-batched) ----------------

/**
 * Track a target's scroll progress inside `scroller` and call `onProgress(p)` (0..1) at most once per frame.
 * mode 'enter': 0 when the target's top meets the viewport bottom → 1 when its bottom meets the viewport bottom
 *               (motion's default offset ["start end", "end end"]).
 * mode 'exit':  0 when the target's top meets the viewport top → 1 when its bottom meets the viewport top
 *               (offset ["start start", "end start"]).
 * Only targets near the viewport are measured (IntersectionObserver). Returns a cleanup function.
 */
export function trackScroll(target, scroller, onProgress, { mode = 'enter', reduced = 1 } = {}) {
  if (reducedMotion()) {
    onProgress(reduced);
    return () => {};
  }
  let visible = true;
  let frame = 0;
  const measure = () => {
    frame = 0;
    if (!visible) return;
    const r = target.getBoundingClientRect();
    const vh = scroller === window ? window.innerHeight : scroller.clientHeight;
    const top = scroller === window ? r.top : r.top - scroller.getBoundingClientRect().top;
    const p = mode === 'exit' ? -top / r.height : (vh - top) / r.height;
    onProgress(clamp01(p));
  };
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(measure);
  };
  const io = new IntersectionObserver(
    ([e]) => {
      visible = e.isIntersecting;
      if (visible) schedule();
    },
    { root: scroller === window ? null : scroller, rootMargin: '200px 0px' },
  );
  io.observe(target);
  scroller.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule, { passive: true });
  schedule();
  return () => {
    io.disconnect();
    cancelAnimationFrame(frame);
    scroller.removeEventListener('scroll', schedule);
    window.removeEventListener('resize', schedule);
  };
}

const isMobile = () => window.innerWidth <= 768;
const setVars = (el, vars) => {
  for (const k in vars) el.style.setProperty(k, vars[k]);
};

// ---------------- Evervault card ----------------

const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
export function randomString(length) {
  let out = '';
  for (let i = 0; i < length; i++) out += CHARS[(Math.random() * CHARS.length) | 0];
  return out;
}

/** Markup for an Evervault card; call `evervault(el)` on the returned element to animate it. */
export function evervaultMarkup(text, { className = '' } = {}) {
  return html`<div class="ev ${raw(className)}">
    <div class="ev-inner">
      <div class="ev-pattern" aria-hidden="true">
        <div class="ev-gradient"></div>
        <div class="ev-chars"><p></p></div>
      </div>
      <div class="ev-label"><span class="ev-blur"></span><span class="ev-text">${text}</span></div>
    </div>
  </div>`;
}

/** Mouse-following reveal of a random character field. Regenerates the string at most once per frame. */
export function evervault(card, { length = 1500 } = {}) {
  const inner = card.querySelector('.ev-inner') || card;
  const chars = card.querySelector('.ev-chars p');
  if (chars) chars.textContent = randomString(length);
  if (reducedMotion()) return () => {};
  let frame = 0;
  let x = 0;
  let y = 0;
  const paint = () => {
    frame = 0;
    inner.style.setProperty('--mx', `${x}px`);
    inner.style.setProperty('--my', `${y}px`);
    if (chars) chars.textContent = randomString(length);
  };
  const onMove = (e) => {
    const r = inner.getBoundingClientRect();
    x = e.clientX - r.left;
    y = e.clientY - r.top;
    if (!frame) frame = requestAnimationFrame(paint);
  };
  inner.addEventListener('pointermove', onMove, { passive: true });
  return () => {
    cancelAnimationFrame(frame);
    inner.removeEventListener('pointermove', onMove);
  };
}

/** The "+" corner marks used around Evervault cards. */
export const plusIcon = (cls) =>
  `<svg class="${cls}" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m6-6H6"/></svg>`;

// ---------------- Container scroll ----------------

export function containerScrollMarkup(titleHtml, bodyHtml) {
  return `<div class="cs"><div class="cs-stage">
    <div class="cs-header">${titleHtml}</div>
    <div class="cs-card"><div class="cs-screen">${bodyHtml}</div></div>
  </div></div>`;
}

/** Card tilts from 20° to flat and the title rises as the section scrolls in. */
export function containerScroll(section, scroller) {
  return trackScroll(
    section,
    scroller,
    (p) => {
      const [s0, s1] = isMobile() ? [0.7, 0.9] : [1.05, 1];
      setVars(section, {
        '--cs-rotate': `${mapRange(p, [0, 1], [20, 0])}deg`,
        '--cs-scale': mapRange(p, [0, 1], [s0, s1]),
        '--cs-translate': `${mapRange(p, [0, 1], [0, -100])}px`,
      });
    },
    { mode: 'enter', reduced: 1 },
  );
}

// ---------------- Macbook scroll ----------------

const KEY_ROWS = [
  [['esc', 'wide'], 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', ['●', 'power']],
  [['~ `'], ['! 1'], ['@ 2'], ['# 3'], ['$ 4'], ['% 5'], ['^ 6'], ['& 7'], ['* 8'], ['( 9'], [') 0'], ['— _'], ['+ ='], ['delete', 'wide end']],
  [['tab', 'wide'], 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', ['{ ['], ['} ]'], ['| \\']],
  [['caps lock', 'caps'], 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', [': ;'], ['" \''], ['return', 'ret end']],
  [['shift', 'shift'], 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ['< ,'], ['> .'], ['? /'], ['shift', 'shift end']],
  [['fn'], ['control'], ['option'], ['command', 'cmd'], ['', 'space'], ['command', 'cmd'], ['option'], ['◂ ▾ ▸', 'arrows']],
];

function keyboard() {
  return KEY_ROWS.map(
    (row) =>
      `<div class="kb-row">${row
        .map((k) => {
          const [label, cls = ''] = Array.isArray(k) ? k : [k];
          const twoLine = !cls && /^\S \S$/.test(label); // shifted symbol above, base below
          const parts = twoLine ? label.split(' ').map((p) => `<span>${p}</span>`).join('') : `<span>${label}</span>`;
          return `<div class="kb-key ${cls}"><div class="kb-cap">${parts}</div></div>`;
        })
        .join('')}</div>`,
  ).join('');
}

export function macbookMarkup({ titleHtml, screenHtml, badgeHtml = '' }) {
  return `<div class="mb">
    <h2 class="mb-title">${titleHtml}</h2>
    <div class="mb-lid">
      <div class="mb-lid-back"><div class="mb-lid-logo">${icons.bot}</div></div>
      <div class="mb-screen"><div class="mb-screen-bg"></div><div class="mb-screen-content">${screenHtml}</div></div>
    </div>
    <div class="mb-base">
      <div class="mb-hinge"><div></div></div>
      <div class="mb-deck">
        <div class="mb-speaker"></div>
        <div class="mb-keys">${keyboard()}</div>
        <div class="mb-speaker"></div>
      </div>
      <div class="mb-trackpad"></div>
      <div class="mb-notch"></div>
      ${badgeHtml ? `<div class="mb-badge">${badgeHtml}</div>` : ''}
    </div>
  </div>`;
}

/** Lid opens, the screen scales up and slides down as the section scrolls past. */
export function macbookScroll(section, scroller) {
  const screen = section.querySelector('.mb-screen');
  const title = section.querySelector('.mb-title');
  if (reducedMotion()) {
    // Static: lid open and flat, title visible, no scale-up.
    setVars(screen, { '--mb-sx': 1, '--mb-sy': 1, '--mb-rotate': '0deg', '--mb-translate': '0px' });
    return () => {};
  }
  return trackScroll(
    section,
    scroller,
    (p) => {
      const m = isMobile();
      setVars(screen, {
        '--mb-sx': mapRange(p, [0, 0.3], [1.2, m ? 1 : 1.5]),
        '--mb-sy': mapRange(p, [0, 0.3], [0.6, m ? 1 : 1.5]),
        '--mb-rotate': `${mapRange(p, [0.1, 0.12, 0.3], [-28, -28, 0])}deg`,
        '--mb-translate': `${mapRange(p, [0, 1], [0, 1500])}px`,
      });
      setVars(title, { '--mb-tt': `${mapRange(p, [0, 0.3], [0, 100])}px`, '--mb-to': mapRange(p, [0, 0.2], [1, 0]) });
    },
    { mode: 'exit' },
  );
}
