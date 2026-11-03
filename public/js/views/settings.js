// Settings, model management, playground and password change.
import { api } from '../api.js';
import { icons } from '../icons.js';
import { $, escapeHtml, formatBytes, formatMs, formatWhatsApp, html, raw, toast } from '../util.js';

const TIER_LABEL = { fast: 'Fast', balanced: 'Balanced', quality: 'Quality' };

export async function mount(root, { hub }) {
  const s = { settings: null, models: null, pulls: new Map(), history: [] };

  root.innerHTML = html`
    <div class="page settings-page">
      <header class="page-header"><div><h1>Settings &amp; models</h1><p class="muted">Changes apply to the next message — no restart needed.</p></div></header>

      <div class="settings-grid">
        <section class="card">
          <h2 class="card-title">Assistant behaviour</h2>
          <form id="settings-form" class="stack">
            <fieldset class="field">
              <legend>Who gets AI replies</legend>
              <div class="radio-cards">
                <label class="radio-card"><input type="radio" name="replyMode" value="session" /><span><strong>Session</strong><small>Only after the person sends a start command</small></span></label>
                <label class="radio-card"><input type="radio" name="replyMode" value="all" /><span><strong>Everyone</strong><small>All chats except blocked/paused</small></span></label>
                <label class="radio-card"><input type="radio" name="replyMode" value="off" /><span><strong>Off</strong><small>Log messages, never reply</small></span></label>
              </div>
            </fieldset>
            <label class="field"><span>Owner name</span><input name="ownerName" placeholder="Name the assistant represents" /></label>
            <label class="field"><span>Persona / system prompt</span>
              <textarea name="persona" rows="8" class="code-area small-area" placeholder="How the assistant should talk and behave"></textarea>
            </label>
            <div class="form-grid">
              <label class="field"><span>Temperature <output id="temp-out"></output></span>
                <input type="range" name="temperature" min="0" max="1.5" step="0.05" />
              </label>
              <label class="field"><span>Max reply tokens</span><input type="number" name="maxTokens" min="32" max="4096" step="16" /></label>
              <label class="field"><span>Debounce (ms)</span><input type="number" name="debounceMs" min="0" max="10000" step="100" />
                <small class="muted">Wait for follow-up messages and answer them together.</small></label>
            </div>
            <div class="toggles">
              <label class="switch-row"><input type="checkbox" class="switch" name="humanize" /><span>Human-like pacing<small class="muted"> — typing indicator &amp; split long replies</small></span></label>
              <label class="switch-row"><input type="checkbox" class="switch" name="answerCache" /><span>Answer cache<small class="muted"> — instant replies to repeated factual questions</small></span></label>
              <label class="switch-row"><input type="checkbox" class="switch" name="replyInGroups" /><span>Reply in groups</span></label>
            </div>

            <h3 class="section-title">eSumit persona &amp; sessions</h3>
            <div class="form-grid">
              <label class="field"><span>Persona name</span><input name="personaName" placeholder="eSumit" /></label>
              <label class="field"><span>Start commands</span><input name="startCommands" placeholder="/hi" />
                <small class="muted">Comma-separated, e.g. /hi, /start</small></label>
              <label class="field"><span>Stop commands</span><input name="stopCommands" placeholder="/bye" />
                <small class="muted">Comma-separated, e.g. /bye, /stop</small></label>
              <label class="field"><span>Auto-end after idle (minutes)</span><input type="number" name="sessionIdleMinutes" min="0" max="1440" />
                <small class="muted">0 = never end automatically</small></label>
              <label class="field"><span>Pause AI after I reply from my phone (seconds)</span><input type="number" name="takeoverSeconds" min="0" max="86400" />
                <small class="muted">0 = never pause. Default 30 s.</small></label>
            </div>
            <label class="field"><span>Welcome message</span>
              <textarea name="welcomeMessage" rows="3" placeholder="Hi {name}! I'm {persona}, {owner}'s AI twin. Send {stop} to end."></textarea>
            </label>
            <label class="field"><span>Goodbye message</span>
              <textarea name="goodbyeMessage" rows="3" placeholder="Bye {name}! Send {start} any time to chat with {persona} again."></textarea>
            </label>
            <p class="muted small placeholder-help">Placeholders: <code class="wa-inline">{name}</code> contact name ·
              <code class="wa-inline">{persona}</code> persona name · <code class="wa-inline">{owner}</code> owner name ·
              <code class="wa-inline">{start}</code> first start command · <code class="wa-inline">{stop}</code> first stop command</p>

            <h3 class="section-title">Memory &amp; context</h3>
            <div class="form-grid">
              <label class="field"><span>Max turns kept verbatim</span><input type="number" name="historyTurns" min="0" max="50" /></label>
              <label class="field"><span>History token budget</span><input type="number" name="historyTokens" min="200" max="6000" step="50" />
                <small class="muted">Smaller = faster replies on CPU</small></label>
            </div>
            <label class="switch-row"><input type="checkbox" class="switch" name="memoryEnabled" /><span>Long-term memory<small class="muted"> — remember facts about each contact across sessions</small></span></label>
            <div class="form-actions"><button class="btn btn-primary" type="submit">${raw(icons.save)} Save settings</button></div>
          </form>
        </section>

        <section class="card">
          <div class="card-title-row">
            <h2 class="card-title">AI model</h2>
            <button class="icon-btn" id="models-refresh" aria-label="Refresh models" title="Refresh">${raw(icons.refresh)}</button>
          </div>
          <p class="muted small" id="model-current">Loading…</p>
          <div id="model-catalog" class="model-grid focus-cards"></div>
          <details class="installed">
            <summary>Installed on server</summary>
            <div id="model-installed"></div>
          </details>
          <form id="pull-form" class="inline-form">
            <input name="name" placeholder="Pull any Ollama model, e.g. qwen3:4b" aria-label="Model name to pull" />
            <button class="btn btn-ghost" type="submit">${raw(icons.download)} Pull</button>
          </form>
          <div id="pull-progress"></div>
        </section>

        <section class="card playground">
          <div class="card-title-row">
            <h2 class="card-title">Playground</h2>
            <button class="btn btn-ghost btn-sm" id="pg-reset">${raw(icons.eraser)} Reset</button>
          </div>
          <p class="muted small">Talk to the bot exactly as a WhatsApp contact would — nothing is sent to WhatsApp.</p>
          <div class="pg-messages wallpaper" id="pg-messages"></div>
          <form class="composer" id="pg-form">
            <textarea id="pg-input" rows="1" placeholder="Try: “Hi! Sumit ke projects batao” or tell it a joke" aria-label="Playground message"></textarea>
            <button class="send-btn" type="submit" aria-label="Send">${raw(icons.send)}</button>
          </form>
        </section>

        <section class="card">
          <h2 class="card-title">Change password</h2>
          <form id="pw-form" class="stack">
            <label class="field"><span>Current password</span><input type="password" name="currentPassword" autocomplete="current-password" required /></label>
            <label class="field"><span>New password</span><input type="password" name="newPassword" autocomplete="new-password" minlength="8" required /></label>
            <label class="field"><span>Confirm new password</span><input type="password" name="confirm" autocomplete="new-password" minlength="8" required /></label>
            <div class="form-actions"><button class="btn btn-primary" type="submit">Update password</button></div>
          </form>
        </section>
      </div>
    </div>`;

  const form = $('#settings-form', root);

  // ---------- settings ----------
  function fillSettings(st) {
    s.settings = st;
    const set = (name, v) => {
      const el = form.elements[name];
      if (!el || v === undefined) return;
      if (el instanceof RadioNodeList) {
        Array.from(el).forEach((r) => (r.checked = r.value === v));
      } else if (el.type === 'checkbox') el.checked = !!v;
      else el.value = v ?? '';
    };
    [
      'replyMode', 'ownerName', 'persona', 'temperature', 'maxTokens', 'historyTurns', 'historyTokens', 'debounceMs',
      'humanize', 'answerCache', 'replyInGroups', 'memoryEnabled', 'personaName', 'startCommands', 'stopCommands',
      'welcomeMessage', 'goodbyeMessage', 'sessionIdleMinutes', 'takeoverSeconds',
    ].forEach((k) => set(k, st[k]));
    $('#temp-out', root).textContent = Number(form.temperature.value).toFixed(2);
  }

  form.temperature.addEventListener('input', () => {
    $('#temp-out', root).textContent = Number(form.temperature.value).toFixed(2);
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const num = (n) => (form[n].value === '' ? undefined : Number(form[n].value));
    const partial = {
      replyMode: form.querySelector('input[name="replyMode"]:checked')?.value,
      ownerName: form.ownerName.value.trim(),
      persona: form.persona.value,
      temperature: num('temperature'),
      maxTokens: num('maxTokens'),
      historyTurns: num('historyTurns'),
      historyTokens: num('historyTokens'),
      debounceMs: num('debounceMs'),
      sessionIdleMinutes: num('sessionIdleMinutes'),
      takeoverSeconds: num('takeoverSeconds'),
      humanize: form.humanize.checked,
      answerCache: form.answerCache.checked,
      replyInGroups: form.replyInGroups.checked,
      memoryEnabled: form.memoryEnabled.checked,
      personaName: form.personaName.value.trim(),
      startCommands: form.startCommands.value.trim(),
      stopCommands: form.stopCommands.value.trim(),
      welcomeMessage: form.welcomeMessage.value,
      goodbyeMessage: form.goodbyeMessage.value,
    };
    try {
      const { settings } = await api.saveSettings(partial);
      fillSettings(settings || { ...s.settings, ...partial });
      toast('Settings saved', 'success');
    } catch (ex) {
      toast(ex.message, 'error');
    }
  });

  // ---------- models ----------
  function pullBar(name) {
    const p = s.pulls.get(name);
    if (!p) return '';
    if (p.error) return html`<div class="pull-row error"><span>${name}: ${p.error}</span></div>`;
    const pct = p.total ? Math.min(100, (p.completed / p.total) * 100) : null;
    return html`<div class="pull-row">
      <div class="pull-label"><span>${name}</span><span class="muted small">${p.status || ''}${pct != null ? ` · ${pct.toFixed(0)}% of ${formatBytes(p.total)}` : ''}</span></div>
      <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" ${raw(pct != null ? `aria-valuenow="${pct.toFixed(0)}"` : '')}>
        <span class="${raw(pct == null ? 'indeterminate' : '')}" style="width:${pct != null ? pct.toFixed(1) : 30}%"></span>
      </div>
    </div>`;
  }

  function renderPulls() {
    $('#pull-progress', root).innerHTML = Array.from(s.pulls.keys()).map(pullBar).join('');
  }

  function renderModels() {
    const m = s.models;
    if (!m) return;
    $('#model-current', root).innerHTML = html`Current model: <strong>${m.current || '—'}</strong>`;
    const installedNames = new Set((m.installed || []).map((x) => x.name));
    const catalog = m.catalog || [];
    $('#model-catalog', root).innerHTML = catalog.length
      ? catalog
          .map((c) => {
            const installed = c.installed ?? installedNames.has(c.name);
            const isCurrent = c.name === m.current;
            const pulling = s.pulls.has(c.name) && !s.pulls.get(c.name).done && !s.pulls.get(c.name).error;
            return html`<article class="model-card ${raw(isCurrent ? 'current' : '')}">
              <div class="model-head">
                <span class="model-name">${c.name}</span>
                <span class="badge tier-${c.tier || 'balanced'}">${TIER_LABEL[c.tier] || c.tier || ''}</span>
              </div>
              <div class="muted small">${[c.params, c.languages].filter(Boolean).join(' · ')}</div>
              <p class="model-note">${c.note || ''}</p>
              <div class="model-actions">
                ${raw(
                  isCurrent
                    ? '<span class="badge badge-ok">In use</span>'
                    : installed
                      ? html`<button class="btn btn-primary btn-sm" data-use="${c.name}">Use</button>`
                      : html`<button class="btn btn-ghost btn-sm" data-pull="${c.name}" ${raw(pulling ? 'disabled' : '')}>${raw(icons.download)} ${pulling ? 'Pulling…' : 'Pull'}</button>`
                )}
                ${raw(installed && !isCurrent ? '<span class="muted small">installed</span>' : '')}
              </div>
            </article>`;
          })
          .join('')
      : '<p class="muted">No catalog available.</p>';

    const inst = m.installed || [];
    $('#model-installed', root).innerHTML = inst.length
      ? html`<table class="simple-table"><thead><tr><th>Name</th><th>Params</th><th>Quant</th><th>Size</th><th></th></tr></thead><tbody>
          ${inst.map((x) =>
            raw(html`<tr><td>${x.name}</td><td>${x.parameterSize || '—'}</td><td>${x.quantization || '—'}</td><td>${formatBytes(x.size)}</td>
              <td>${raw(x.name === m.current ? '<span class="badge badge-ok">In use</span>' : html`<button class="btn btn-ghost btn-sm" data-use="${x.name}">Use</button>`)}</td></tr>`)
          )}</tbody></table>`
      : '<p class="muted small">No models installed yet.</p>';
  }

  async function loadModels() {
    try {
      s.models = await api.models();
      renderModels();
    } catch (e) {
      $('#model-current', root).textContent = `Could not load models: ${e.message}`;
    }
  }

  async function useModel(name) {
    try {
      const { settings } = await api.saveSettings({ model: name });
      if (settings) fillSettings(settings);
      s.models.current = name;
      renderModels();
      toast(`Now using ${name}`, 'success');
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function pull(name) {
    if (!name) return;
    s.pulls.set(name, { status: 'starting' });
    renderPulls();
    renderModels();
    try {
      await api.pullModel(name);
    } catch (e) {
      s.pulls.set(name, { error: e.message });
      renderPulls();
      renderModels();
    }
  }

  root.addEventListener('click', (e) => {
    const use = e.target.closest('[data-use]');
    if (use) return useModel(use.dataset.use);
    const p = e.target.closest('[data-pull]');
    if (p) return pull(p.dataset.pull);
  });
  $('#models-refresh', root).addEventListener('click', loadModels);
  $('#pull-form', root).addEventListener('submit', (e) => {
    e.preventDefault();
    const name = e.target.name.value.trim();
    if (name) {
      pull(name);
      e.target.reset();
    }
  });

  const offs = [
    hub.on('model:pull', (p) => {
      if (!p?.name) return;
      const done = p.status === 'success';
      s.pulls.set(p.name, { ...p, done });
      renderPulls();
      if (done || p.error) {
        if (done) toast(`${p.name} is ready`, 'success');
        setTimeout(() => {
          s.pulls.delete(p.name);
          renderPulls();
        }, 4000);
        loadModels();
      }
    }),
  ];

  // ---------- playground ----------
  const pgBox = $('#pg-messages', root);
  const pgInput = $('#pg-input', root);

  function pgRender() {
    if (!s.history.length) {
      pgBox.innerHTML = '<div class="day-sep"><span>Start a test conversation</span></div>';
      return;
    }
    pgBox.innerHTML = s.history
      .map((m) => {
        const out = m.role === 'assistant';
        const meta = m.meta
          ? html`<span class="meta-tag ai">${[m.meta.intent, m.meta.language, m.meta.cached ? 'cached' : formatMs(m.meta.latencyMs), m.meta.model]
              .filter(Boolean)
              .join(' · ')}</span>`
          : '';
        const sources = m.meta?.sources?.length
          ? `<div class="pg-sources">${m.meta.sources
              .map((x) => `<span class="badge" title="score ${escapeHtml(Number(x.score).toFixed(3))}">${escapeHtml(x.path)}</span>`)
              .join(' ')}</div>`
          : '';
        return `<div class="msg-row ${out ? 'out' : 'in'}"><div class="bubble ${out ? 'bubble-out' : 'bubble-in'} ${m.pending ? 'pending' : ''}">
          <div class="msg-body">${m.pending ? '<span class="typing-dots"><i></i><i></i><i></i></span>' : formatWhatsApp(m.content)}</div>
          ${sources}<div class="msg-meta">${meta}</div></div></div>`;
      })
      .join('');
    pgBox.scrollTop = pgBox.scrollHeight;
  }

  // Playground "out" = assistant side (like the bot's own phone), "in" = tester.
  $('#pg-form', root).addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = pgInput.value.trim();
    if (!message) return;
    pgInput.value = '';
    const history = s.history.filter((m) => !m.pending).map(({ role, content }) => ({ role, content }));
    s.history.push({ role: 'user', content: message });
    const pending = { role: 'assistant', content: '', pending: true };
    s.history.push(pending);
    pgRender();
    try {
      const r = await api.playground(message, history);
      Object.assign(pending, { content: r.reply || '', pending: false, meta: r });
    } catch (ex) {
      Object.assign(pending, { content: `⚠️ ${ex.message}`, pending: false });
    }
    pgRender();
  });
  pgInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $('#pg-form', root).requestSubmit();
    }
  });
  $('#pg-reset', root).addEventListener('click', () => {
    s.history = [];
    pgRender();
  });
  pgRender();

  // ---------- password ----------
  $('#pw-form', root).addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    if (f.newPassword.value !== f.confirm.value) {
      toast('New passwords do not match', 'warn');
      return;
    }
    try {
      await api.changePassword(f.currentPassword.value, f.newPassword.value);
      f.reset();
      toast('Password updated', 'success');
    } catch (ex) {
      toast(ex.message, 'error');
    }
  });

  try {
    const { settings } = await api.settings();
    fillSettings(settings || {});
  } catch (e) {
    toast(`Could not load settings: ${e.message}`, 'error');
  }
  loadModels();

  return { destroy: () => offs.forEach((o) => o()) };
}
