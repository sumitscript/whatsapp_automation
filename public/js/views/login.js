// Sign-in + scroll showcase. First screen: Evervault brand card and the login form.
// Below: ContainerScroll (how /hi works), MacbookScroll (the console) and FocusCards (features).
import { api } from '../api.js';
import { icons } from '../icons.js';
import {
  containerScroll, containerScrollMarkup, evervault, evervaultMarkup, macbookMarkup, macbookScroll, plusIcon,
} from '../fx.js';
import { $, formatWhatsApp, html, raw } from '../util.js';

// Illustrative conversation explaining the commands (static copy, not data).
const DEMO_CHAT = [
  ['in', '/hi'],
  ['out', "Hey! 👋 *eSumit* here — Sumit's AI twin. Ask me anything: work, projects, code, or just chat. Send /bye whenever you're done 🙂"],
  ['in', 'bhai closures samjha do 2 line mein'],
  ['out', 'Simple hai 😄 closure = function jo apne bahar ke variables *yaad* rakhta hai:'],
  ['out', '```\nconst counter = () => { let n = 0; return () => ++n; };\n```'],
  ['in', 'haha perfect. /bye'],
  ['out', 'Bye! 👋 Send /hi anytime to talk to eSumit again.'],
];

const FEATURES = [
  { title: 'Starts with /hi', text: 'Replies only inside a session. /bye, /reset and /help work in any case.', icon: icons.chats, hue: 160 },
  { title: 'Vectorless RAG', text: 'BM25F over a markdown knowledge base. About 1 ms per lookup.', icon: icons.kb, hue: 200 },
  { title: 'Remembers people', text: 'Recent turns in Redis, long-term notes in Postgres, folded while idle.', icon: icons.contacts, hue: 260 },
  { title: 'Speaks Indian languages', text: 'Hindi, Hinglish, Marathi, Tamil, Bengali and more, in the right script.', icon: icons.bot, hue: 30 },
  { title: 'Local LLM', text: 'Ollama on your own server. No cloud API, no per-message cost.', icon: icons.settings, hue: 330 },
  { title: 'Live control', text: 'QR pairing, sessions, memory, models and log tail from the browser.', icon: icons.logs, hue: 100 },
];

function demoChat() {
  return `<div class="demo-chat wallpaper">${DEMO_CHAT.map(
    ([dir, text]) =>
      `<div class="msg-row ${dir}"><div class="bubble bubble-${dir}"><div class="msg-body">${formatWhatsApp(text)}</div></div></div>`,
  ).join('')}</div>`;
}

// A drawing of the console layout for the laptop screen (shapes only, no data).
function consoleSketch() {
  const items = Array.from({ length: 7 }, (_, i) => `<div class="sk-item"><i style="--h:${(i * 53) % 360}"></i><span><b></b><em></em></span></div>`).join('');
  return `<div class="sk"><div class="sk-rail">${'<i></i>'.repeat(6)}</div>
    <div class="sk-list"><div class="sk-search"></div>${items}</div>
    <div class="sk-conv wallpaper"><div class="sk-head"></div>
      <div class="sk-b in" style="width:46%"></div><div class="sk-b out" style="width:58%"></div>
      <div class="sk-b out" style="width:34%"></div><div class="sk-b in" style="width:40%"></div>
      <div class="sk-b out" style="width:62%"></div><div class="sk-compose"></div></div></div>`;
}

export function mount(root, { onLogin }) {
  root.innerHTML = html`
    <div class="lp">
      <header class="lp-top">
        <span class="lp-brand">${raw(icons.bot)} eSumit Console</span>
        <a class="btn btn-ghost btn-sm" href="#features" id="lp-features">Features</a>
      </header>

      <section class="lp-hero">
        <div class="lp-ev-wrap">
          ${raw(plusIcon('ev-plus tl'))}${raw(plusIcon('ev-plus bl'))}${raw(plusIcon('ev-plus tr'))}${raw(plusIcon('ev-plus br'))}
          ${raw(evervaultMarkup('eSumit'))}
          <h2 class="lp-ev-caption">My AI twin on WhatsApp. It chats as me, in first person, from my own server.</h2>
          <p class="lp-pill">Local LLM · Indian languages · /hi to start</p>
        </div>

        <div class="login-card" role="main">
          <div class="login-logo">${raw(icons.bot)}</div>
          <h1>Sign in</h1>
          <p class="muted">Manage chats, sessions, memory and models.</p>
          <form id="login-form" novalidate>
            <label class="field">
              <span>Username</span>
              <input name="username" autocomplete="username" required autofocus />
            </label>
            <label class="field">
              <span>Password</span>
              <input name="password" type="password" autocomplete="current-password" required />
            </label>
            <p class="form-error" id="login-error" role="alert" hidden></p>
            <button class="btn btn-primary btn-block" type="submit" id="login-submit">Sign in</button>
          </form>
        </div>
      </section>

      <a class="lp-cue" href="#how" aria-label="Scroll to see how it works">${raw('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>')}</a>

      <section id="how" class="lp-section">
        ${raw(containerScrollMarkup(
          '<h2 class="cs-title">Talk to my twin<br><span class="cs-title-big">/hi</span></h2><p class="muted">One command opens a session, /bye closes it. Outside a session it stays silent.</p>',
          demoChat(),
        ))}
      </section>

      <section class="lp-section lp-mb">
        ${raw(macbookMarkup({
          titleHtml: 'Every chat, session and memory<br>in one console.',
          screenHtml: consoleSketch(),
          badgeHtml: `<span class="mb-badge-icon">${icons.bot}</span>`,
        }))}
      </section>

      <section id="features" class="lp-section lp-features">
        <h2 class="lp-h2">What's inside</h2>
        <div class="focus-cards">
          ${FEATURES.map((f) =>
            raw(html`<article class="focus-card" style="--h:${f.hue}">
              <div class="focus-art" aria-hidden="true">${raw(f.icon)}</div>
              <div class="focus-overlay"><h3>${f.title}</h3><p>${f.text}</p></div>
            </article>`),
          )}
        </div>
      </section>

      <footer class="lp-foot muted small">eSumit · self-hosted · <a href="#top" id="lp-totop">Back to sign in</a></footer>
    </div>`;

  // Effects (each returns a cleanup; the page is replaced wholesale on login).
  const scroller = root;
  const cleanups = [
    evervault($('.ev', root)),
    containerScroll($('.cs', root), scroller),
    macbookScroll($('.mb', root), scroller),
  ];

  const toTop = (e) => {
    e.preventDefault();
    root.scrollTo({ top: 0, behavior: 'smooth' });
    $('input[name=username]', root).focus({ preventScroll: true });
  };
  $('#lp-totop', root).addEventListener('click', toTop);
  for (const a of root.querySelectorAll('a[href^="#"]:not(#lp-totop)')) {
    a.addEventListener('click', (e) => {
      const target = root.querySelector(a.getAttribute('href'));
      if (!target) return;
      e.preventDefault(); // keep the hash router out of in-page jumps
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  const form = $('#login-form', root);
  const err = $('#login-error', root);
  const submit = $('#login-submit', root);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = form.username.value.trim();
    const password = form.password.value;
    if (!username || !password) {
      err.textContent = 'Enter your username and password.';
      err.hidden = false;
      return;
    }
    err.hidden = true;
    submit.disabled = true;
    submit.textContent = 'Signing in…';
    try {
      const { user } = await api.login(username, password);
      cleanups.forEach((fn) => fn());
      onLogin(user);
    } catch (ex) {
      err.textContent = ex.status === 429 ? 'Too many attempts. Wait a minute and try again.' : ex.message;
      err.hidden = false;
      form.password.select();
    } finally {
      submit.disabled = false;
      submit.textContent = 'Sign in';
    }
  });
}
