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
  { tag: 'Sessions', title: 'Starts with /hi', text: 'Replies only inside a session. /bye, /reset and /help work in any case.' },
  { tag: 'Retrieval', title: 'Vectorless RAG', text: 'BM25F over a markdown knowledge base. About 1 ms per lookup, no embeddings to maintain.' },
  { tag: 'Memory', title: 'Remembers people', text: 'Recent turns in Redis, long-term notes in Postgres, folded while the server is idle.' },
  { tag: 'Languages', title: 'Speaks Indian languages', text: 'Hindi, Hinglish, Marathi, Tamil, Bengali and more, replied in the right script.' },
  { tag: 'Runtime', title: 'Local LLM', text: 'Ollama on your own server. No cloud API and no per-message cost.' },
  { tag: 'Console', title: 'Live control', text: 'QR pairing, sessions, memory, models and a log tail, all from the browser.' },
];

function demoChat() {
  return `<div class="demo-chat wallpaper">${DEMO_CHAT.map(
    ([dir, text]) =>
      `<div class="msg-row ${dir}"><div class="bubble bubble-${dir}"><div class="msg-body">${formatWhatsApp(text)}</div></div></div>`,
  ).join('')}</div>`;
}

// What the laptop screen shows: a faithful miniature of the console (static copy, not data).
const MOCK_CHATS = [
  ['AM', 'Aarav Mehta', '10:42', 'bhai closures samjha do 2 line mein', 160, true],
  ['PS', 'Priya Shah', '10:31', 'Thanks! That helped a lot', 330, false],
  ['RK', 'Rohan Kulkarni', '09:58', 'Can we meet tomorrow at 6?', 40, false],
  ['NI', 'Neha Iyer', 'Yesterday', 'Voice note', 280, false],
  ['KP', 'Karan Patel', 'Yesterday', 'Send me the repo link', 210, false],
  ['SD', 'Sana Desai', 'Tuesday', 'Great, talk soon', 100, false],
];
function consoleMock() {
  const rail = [icons.logo, icons.chats, icons.contacts, icons.kb, icons.logs, icons.settings]
    .map((s, i) => `<i class="${i === 1 ? 'on' : ''}">${s}</i>`).join('');
  const rows = MOCK_CHATS.map(([ini, name, time, prev, h, sel]) =>
    `<div class="cm-row${sel ? ' sel' : ''}"><i class="cm-av" style="--h:${h}">${ini}</i><div class="cm-meta"><div class="cm-top"><b>${name}</b><span>${time}</span></div><div class="cm-prev">${prev}</div></div></div>`).join('');
  return `<div class="cm"><div class="cm-rail">${rail}</div>
    <div class="cm-list"><div class="cm-search">Search chats</div>${rows}</div>
    <div class="cm-conv wallpaper">
      <div class="cm-head"><i class="cm-av" style="--h:160">AM</i><div><b>Aarav Mehta</b><em>typing…</em></div></div>
      <div class="cm-msgs">
        <div class="cm-b in">/hi<i>10:40</i></div>
        <div class="cm-b out">Hey! 👋 <b>eSumit</b> here, Sumit's AI twin. Ask me anything.<i>10:40 ✓✓</i></div>
        <div class="cm-b in">bhai closures samjha do 2 line mein<i>10:42</i></div>
        <div class="cm-b out">Simple hai: closure = function jo apne bahar ke variables <b>yaad</b> rakhta hai.<i>10:42 ✓✓</i></div>
      </div>
      <div class="cm-compose">Type a message</div>
    </div></div>`;
}

export function mount(root, { onLogin }) {
  root.innerHTML = html`
    <div class="lp">
      <header class="lp-top">
        <span class="lp-brand">${raw(icons.logo)} eSumit</span>
        <a class="btn btn-ghost btn-sm" href="#features" id="lp-features">Features</a>
      </header>

      <section class="lp-hero">
        <div class="lp-ev-wrap">
          ${raw(plusIcon('ev-plus tl'))}${raw(plusIcon('ev-plus bl'))}${raw(plusIcon('ev-plus tr'))}${raw(plusIcon('ev-plus br'))}
          ${raw(evervaultMarkup(raw(`<span class="ev-mark">${icons.logo}</span>eSumit`)))}
          <h2 class="lp-ev-caption">My AI twin on WhatsApp. It chats as me, in first person, from my own server.</h2>
          <ul class="lp-points">
            <li>Local LLM</li>
            <li>Indian languages</li>
            <li><code>/hi</code> to start</li>
          </ul>
        </div>

        <div class="login-card" role="main">
          <div class="login-logo">${raw(icons.logo)}</div>
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
          screenHtml: consoleMock(),
          
        }))}
      </section>

      <section id="features" class="lp-section lp-features">
        <h2 class="lp-h2">What's inside</h2>
        <ol class="spec">
          ${FEATURES.map((f, i) =>
            raw(html`<li class="spec-row">
              <span class="spec-n">${String(i + 1).padStart(2, '0')}</span>
              <h3>${f.title}</h3>
              <p>${f.text}</p>
              <span class="spec-tag">${f.tag}</span>
            </li>`),
          )}
        </ol>
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
