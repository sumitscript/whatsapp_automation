// Settings, model management, playground and password change.
import { api } from '../api.js';
import { icons } from '../icons.js';
import { installLimitPanel, mountCloudflare, renderAccounts } from './provider-limits.js';
import { $, confirmDialog, escapeHtml, formatBytes, formatMs, formatNumber, formatWhatsApp, html, raw, toast } from '../util.js';

const TIER_LABEL = { fast: 'Fast', balanced: 'Balanced', quality: 'Quality' };

export async function mount(root, { hub }) {
  const s = { settings: null, models: null, pulls: new Map(), history: [] };

  root.innerHTML = html`
    <div class="page settings-page">
      <header class="page-header"><div><h1>Settings</h1><p class="muted">Changes apply to the next message.</p></div></header>

      <div class="settings-grid">
        <section class="card">
          <h2 class="card-title">Assistant behaviour</h2>
          <form id="settings-form" class="stack">
            <fieldset class="field">
              <legend>Who gets AI replies</legend>
              <div class="radio-cards">
                <label class="radio-card"><input type="radio" name="replyMode" value="session" /><span><strong>Session</strong><small>After a start command</small></span></label>
                <label class="radio-card"><input type="radio" name="replyMode" value="all" /><span><strong>Everyone</strong><small>All chats</small></span></label>
                <label class="radio-card"><input type="radio" name="replyMode" value="off" /><span><strong>Off</strong><small>Never reply</small></span></label>
              </div>
            </fieldset>
            <label class="field"><span>Owner name</span><input name="ownerName" placeholder="Name the assistant represents" /></label>
            <label class="field"><span>Persona prompt</span>
              <textarea name="persona" rows="8" class="code-area small-area" placeholder="Empty = built-in persona"></textarea>
            </label>
            <div class="form-grid">
              <label class="field"><span>Temperature <output id="temp-out"></output></span>
                <input type="range" name="temperature" min="0" max="1.5" step="0.05" />
              </label>
              <label class="field"><span>Max reply tokens</span><input type="number" name="maxTokens" min="32" max="4096" step="16" /></label>
              <label class="field"><span>Wait for follow-ups (ms)</span><input type="number" name="debounceMs" min="0" max="10000" step="100" /></label>
            </div>
            <div class="toggles">
              <label class="switch-row"><input type="checkbox" class="switch" name="humanize" /><span>Human-like pacing</span></label>
              <label class="switch-row"><input type="checkbox" class="switch" name="answerCache" /><span>Answer cache</span></label>
              <label class="switch-row"><input type="checkbox" class="switch" name="prefillAhead" /><span>Think ahead</span></label>
              <label class="switch-row"><input type="checkbox" class="switch" name="replyInGroups" /><span>Reply in groups</span></label>
            </div>

            <h3 class="section-title">eSumit persona &amp; sessions</h3>
            <div class="form-grid">
              <label class="field"><span>Persona name</span><input name="personaName" placeholder="eSumit" /></label>
              <label class="field"><span>Start commands</span><input name="startCommands" placeholder="/hi, /start" /></label>
              <label class="field"><span>Stop commands</span><input name="stopCommands" placeholder="/bye, /stop" /></label>
              <label class="field"><span>End session after idle (min, 0 = never)</span><input type="number" name="sessionIdleMinutes" min="0" max="1440" /></label>
              <label class="field"><span>Pause AI after I reply (sec, 0 = never)</span><input type="number" name="takeoverSeconds" min="0" max="86400" /></label>
            </div>
            <label class="field"><span>Welcome message</span>
              <textarea name="welcomeMessage" rows="3" placeholder="Hi {name}! I'm {persona}, {owner}'s AI twin. Send {stop} to end."></textarea>
            </label>
            <label class="field"><span>Goodbye message</span>
              <textarea name="goodbyeMessage" rows="3" placeholder="Bye {name}! Send {start} any time to chat with {persona} again."></textarea>
            </label>
            <label class="switch-row"><input type="checkbox" class="switch" name="notifyOnTimeout" /><span>Tell people when their session times out</span></label>
            <label class="field"><span>Timeout message</span>
              <textarea name="timeoutMessage" rows="3" placeholder="Hey {name}, it's been quiet… send {start} to chat again."></textarea>
            </label>
            <label class="field"><span>Reply to voice notes, photos &amp; stickers</span>
              <textarea name="unsupportedMessage" rows="3" placeholder="Hey {name}! I can only read text, so I can't open {kind}…"></textarea>
            </label>
            <p class="muted small placeholder-help">Placeholders: <code class="wa-inline">{name}</code> <code class="wa-inline">{persona}</code> <code class="wa-inline">{owner}</code> <code class="wa-inline">{start}</code> <code class="wa-inline">{stop}</code> <code class="wa-inline">{kind}</code></p>

            <h3 class="section-title">Agenda &amp; alerts</h3>
            <label class="switch-row"><input type="checkbox" class="switch" name="agendaEnabled" /><span>Collect appointments &amp; to-dos</span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="notifyOwnerOnAgenda" /><span>WhatsApp me about new agenda items</span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="meetingAck" /><span>Instant reply to meeting requests</span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="agendaConfirm" /><span>Confirm to the contact when I accept</span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="agendaCard" /><span>Attach an invitation card</span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="alertOwner" /><span>WhatsApp me if the AI goes down</span></label>
            <label class="field"><span>Timezone</span><input name="timezone" placeholder="Asia/Kolkata" /></label>

            <h3 class="section-title">Availability</h3>
            <label class="switch-row"><input type="checkbox" class="switch" name="enforceAvailability" /><span>Check availability before confirming</span></label>
            <div class="form-grid">
              <label class="field"><span>Work starts</span><input type="time" name="workStart" /></label>
              <label class="field"><span>Work ends</span><input type="time" name="workEnd" /></label>
              <label class="field"><span>Meeting length (min)</span><input type="number" name="meetingMinutes" min="5" max="480" step="5" /></label>
              <label class="field"><span>Gap between meetings (min)</span><input type="number" name="meetingBufferMinutes" min="0" max="240" step="5" /></label>
            </div>
            <fieldset class="field"><legend>Meeting days</legend>
              <div class="chip-group"><label class="chip"><input type="checkbox" name="wd" value="mon" /> Mon</label><label class="chip"><input type="checkbox" name="wd" value="tue" /> Tue</label><label class="chip"><input type="checkbox" name="wd" value="wed" /> Wed</label><label class="chip"><input type="checkbox" name="wd" value="thu" /> Thu</label><label class="chip"><input type="checkbox" name="wd" value="fri" /> Fri</label><label class="chip"><input type="checkbox" name="wd" value="sat" /> Sat</label><label class="chip"><input type="checkbox" name="wd" value="sun" /> Sun</label></div>
            </fieldset>

            <h3 class="section-title">Memory &amp; context</h3>
            <div class="form-grid">
              <label class="field"><span>Messages kept in context</span><input type="number" name="historyTurns" min="0" max="50" /></label>
              <label class="field"><span>Context token budget</span><input type="number" name="historyTokens" min="200" max="6000" step="50" /></label>
            </div>
            <label class="switch-row"><input type="checkbox" class="switch" name="memoryEnabled" /><span>Long-term memory</span></label>
            <div class="form-actions"><button class="btn btn-primary" type="submit">${raw(icons.save)} Save settings</button></div>
          </form>
        </section>

        <div class="settings-col">
        <section class="card" id="groq-card">
          <h2 class="card-title">AI provider</h2>
          <form id="gq-form" class="stack">
            <div class="seg" role="radiogroup" aria-label="AI provider mode">
              <label><input type="radio" name="llmProvider" value="auto" /><b>Auto</b><small>Cloud, local as backup</small></label>
              <label><input type="radio" name="llmProvider" value="ollama" /><b>Local</b><small>This server only</small></label>
              <label><input type="radio" name="llmProvider" value="groq" /><b>Cloud only</b><small>Never local</small></label>
            </div>
            <p class="muted small" id="gq-summary">Loading…</p>
          </form>

          <div class="prov-head"><h3>Groq</h3><button type="button" class="btn btn-ghost btn-sm" data-add="gq-add">${raw(icons.plus)} Add</button></div>
          <div id="gq-keys" class="acct-list"></div>
          <form id="gq-add" class="add-form" hidden>
            <input name="label" placeholder="Name (optional)" aria-label="Name" maxlength="60" />
            <input name="key" type="password" placeholder="Groq API key (gsk_…)" autocomplete="off" aria-label="Groq API key" required />
            <button class="btn btn-primary btn-sm" type="submit">Add</button>
          </form>

          <div class="prov-head"><h3>Cloudflare</h3><button type="button" class="btn btn-ghost btn-sm" data-add="cf-add">${raw(icons.plus)} Add</button></div>
          <div id="cf-keys" class="acct-list"></div>
          <form id="cf-add" class="add-form" hidden>
            <input name="label" placeholder="Name (optional)" aria-label="Name" maxlength="60" />
            <input name="accountId" placeholder="Account ID" autocomplete="off" aria-label="Cloudflare account id" required />
            <input name="token" type="password" placeholder="API token" autocomplete="off" aria-label="Cloudflare API token" required />
            <button class="btn btn-primary btn-sm" type="submit">Add</button>
          </form>

          <details class="adv">
            <summary>Advanced</summary>
            <div class="stack">
              <label class="field"><span>Groq models, best first</span>
                <input form="gq-form" name="groqModels" list="gq-models" placeholder="openai/gpt-oss-120b, openai/gpt-oss-20b" autocomplete="off" /><datalist id="gq-models"></datalist></label>
              <label class="field"><span>Cloudflare model</span>
                <input form="gq-form" name="cloudflareModel" list="cf-models" placeholder="@cf/google/gemma-4-26b-a4b-it" autocomplete="off" />
                <datalist id="cf-models"><option value="@cf/google/gemma-4-26b-a4b-it"></option><option value="@cf/meta/llama-3.1-8b-instruct-fp8"></option><option value="@cf/meta/llama-3.2-3b-instruct"></option></datalist></label>
              <div class="field"><span>Account order</span>
                <div class="seg" role="radiogroup" aria-label="Account order">
                  <label><input form="gq-form" type="radio" name="groqStrategy" value="priority" /><b>Priority</b><small>Account 1 first</small></label>
                  <label><input form="gq-form" type="radio" name="groqStrategy" value="spread" /><b>Spread</b><small>Take turns</small></label>
                </div>
              </div>
              <label class="field"><span>Break message <span class="muted small">({name} {persona} {owner})</span></span>
                <textarea form="gq-form" name="groqBusyMessage" rows="3" maxlength="1000"></textarea></label>
              <div class="form-actions"><button class="btn btn-primary" type="submit" form="gq-form">${raw(icons.save)} Save</button></div>
            </div>
          </details>
        </section>

        <section class="card">
          <div class="card-title-row">
            <h2 class="card-title">AI model</h2>
            <button class="icon-btn" id="models-refresh" aria-label="Refresh models" title="Refresh">${raw(icons.refresh)}</button>
          </div>
          <p class="muted small" id="model-current">Loading…</p>
          <div id="model-installed" class="table-scroll"></div>
          <form id="pull-form" class="inline-form">
            <input name="name" list="model-suggestions" placeholder="Install another model, e.g. qwen3:4b" aria-label="Model name to pull" />
            <datalist id="model-suggestions"></datalist>
            <button class="btn btn-ghost" type="submit">${raw(icons.download)} Pull</button>
          </form>
          <div id="pull-progress"></div>
        </section>

        <section class="card playground">
          <div class="card-title-row">
            <h2 class="card-title">Playground</h2>
            <button class="btn btn-ghost btn-sm" id="pg-reset">${raw(icons.eraser)} Reset</button>
          </div>
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
      'humanize', 'answerCache', 'prefillAhead', 'replyInGroups', 'memoryEnabled', 'personaName', 'startCommands', 'stopCommands',
      'welcomeMessage', 'goodbyeMessage', 'sessionIdleMinutes', 'takeoverSeconds',
      'notifyOnTimeout', 'timeoutMessage', 'unsupportedMessage', 'agendaEnabled', 'notifyOwnerOnAgenda', 'meetingAck', 'agendaConfirm', 'agendaCard', 'alertOwner', 'timezone', 'workStart', 'workEnd', 'meetingMinutes', 'meetingBufferMinutes', 'enforceAvailability',
    ].forEach((k) => set(k, st[k]));
    const wd = String(st.workDays || '').split(',');
    form.querySelectorAll('input[name="wd"]').forEach((cb) => (cb.checked = wd.includes(cb.value)));
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
      prefillAhead: form.prefillAhead.checked,
      replyInGroups: form.replyInGroups.checked,
      memoryEnabled: form.memoryEnabled.checked,
      personaName: form.personaName.value.trim(),
      startCommands: form.startCommands.value.trim(),
      stopCommands: form.stopCommands.value.trim(),
      welcomeMessage: form.welcomeMessage.value,
      goodbyeMessage: form.goodbyeMessage.value,
      notifyOnTimeout: form.notifyOnTimeout.checked,
      timeoutMessage: form.timeoutMessage.value,
      unsupportedMessage: form.unsupportedMessage.value,
      agendaEnabled: form.agendaEnabled.checked,
      notifyOwnerOnAgenda: form.notifyOwnerOnAgenda.checked,
      meetingAck: form.meetingAck.checked,
      agendaConfirm: form.agendaConfirm.checked,
      agendaCard: form.agendaCard.checked,
      alertOwner: form.alertOwner.checked,
      timezone: form.timezone.value.trim(),
      workStart: form.workStart.value,
      workEnd: form.workEnd.value,
      workDays: [...form.querySelectorAll('input[name="wd"]:checked')].map((c) => c.value).join(',') || undefined,
      meetingMinutes: num('meetingMinutes'),
      meetingBufferMinutes: num('meetingBufferMinutes'),
      enforceAvailability: form.enforceAvailability.checked,
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
    const inst = m.installed || [];
    // Suggestions for the pull box: catalog entries that aren't installed yet.
    const have = new Set(inst.map((x) => x.name));
    $('#model-suggestions', root).innerHTML = (m.catalog || [])
      .filter((c) => !have.has(c.name))
      .map((c) => html`<option value="${c.name}"></option>`)
      .join('');
    $('#model-installed', root).innerHTML = inst.length
      ? html`<table class="simple-table models-table"><thead><tr><th>Name</th><th>Params</th><th>Quant</th><th>Size</th><th></th></tr></thead><tbody>
          ${inst.map((x) => {
            const isCurrent = x.name === m.current;
            return raw(html`<tr class="${raw(isCurrent ? 'current' : '')}">
              <td class="mono">${x.name}</td><td>${x.parameterSize || '—'}</td><td>${x.quantization || '—'}</td><td>${formatBytes(x.size)}</td>
              <td class="row-actions">
                ${raw(isCurrent ? '<span class="badge badge-ok">In use</span>' : html`<button class="btn btn-primary btn-sm" data-use="${x.name}">Use</button>`)}
                <button class="btn btn-ghost btn-sm" data-del="${x.name}" ${raw(isCurrent ? 'disabled title="Switch to another model before deleting this one"' : 'title="Delete from the server"')}>${raw(icons.trash)} Delete</button>
              </td></tr>`);
          })}</tbody></table>`
      : '<p class="muted small">No models installed yet. Pull one below.</p>';
  }

  async function deleteModel(name) {
    const ok = await confirmDialog({
      title: 'Delete model?',
      message: `${name} will be removed from the server and its disk space freed. You can pull it again later.`,
      confirmText: 'Delete',
      danger: true,
    });
    if (!ok) return;
    try {
      const r = await api.deleteModel(name);
      if (s.models) s.models.installed = r.installed || s.models.installed.filter((x) => x.name !== name);
      renderModels();
      toast(`${name} deleted`, 'success');
    } catch (e) {
      toast(e.message, 'error');
    }
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
    const del = e.target.closest('[data-del]');
    if (del && !del.disabled) return deleteModel(del.dataset.del);
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

  // ---------- AI provider: mode, accounts, advanced ----------
  const gq = { data: null, timer: null, modelsLoaded: false, busyMessage: null };
  const gqForm = $('#gq-form', root);
  const editing = () => Array.from(gqForm.elements).includes(document.activeElement);

  // one short status line from both providers' account lists
  function renderSummary() {
    const g = gq.data;
    if (!g) return;
    const cfData = cf.data();
    const keys = [...g.keys, ...(cfData?.keys || [])];
    const ready = keys.filter((k) => k.state === 'ready').length;
    let line;
    if (g.provider === 'ollama') line = 'Local model only.';
    else if (!keys.length) line = g.provider === 'groq' ? 'No account yet: people get the break message.' : 'No cloud account yet: the local model answers.';
    else if (g.provider === 'groq') line = ready ? `${ready} of ${keys.length} accounts ready.` : 'No account can answer: people get the break message.';
    else line = ready ? `${ready} of ${keys.length} accounts ready.` : 'All accounts are resting: the local model answers.';
    const lp = g.lastFallback;
    const recent = lp && Date.now() - lp.at < 10 * 60_000 && g.provider !== 'ollama' && !ready;
    $('#gq-summary', root).textContent = recent ? `${line} Last problem: ${lp.reason}` : line;
  }

  function renderGroq() {
    const d = gq.data;
    if (!d) return;
    if (!editing()) {
      Array.from(gqForm.elements.llmProvider).forEach((r) => (r.checked = r.value === d.provider));
      gqForm.elements.groqModels.value = (d.models?.length ? d.models : [d.model]).filter(Boolean).join(', ');
      Array.from(gqForm.elements.groqStrategy).forEach((r) => (r.checked = r.value === (d.strategy || 'priority')));
      if (gq.busyMessage != null) gqForm.elements.groqBusyMessage.value = gq.busyMessage;
    }
    renderAccounts($('#gq-keys', root), d.keys, 'groq', d.unreadable ? `${d.unreadable} saved key(s) cannot be read: add them again.` : 'No Groq key yet.');
    renderSummary();
  }

  async function loadGroq() {
    try {
      if (gq.busyMessage == null) gq.busyMessage = (await api.settings()).settings?.groqBusyMessage ?? '';
      gq.data = await api.groqStatus();
      renderGroq();
      if (!gq.modelsLoaded && gq.data.keys.length) {
        gq.modelsLoaded = true;
        api
          .groqModels()
          .then(({ models }) => {
            $('#gq-models', root).innerHTML = models.map((m) => html`<option value="${m}"></option>`).join('');
          })
          .catch(() => {});
      }
    } catch (e) {
      $('#gq-summary', root).textContent = `Could not load the status: ${e.message}`;
    }
  }

  gqForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const el = gqForm.elements;
    const body = { llmProvider: el.llmProvider.value, groqModels: el.groqModels.value.trim() || 'openai/gpt-oss-120b', groqStrategy: el.groqStrategy.value || 'priority', groqBusyMessage: el.groqBusyMessage.value.trim() };
    if (!body.groqBusyMessage) delete body.groqBusyMessage;
    if (el.cloudflareModel.value.trim()) body.cloudflareModel = el.cloudflareModel.value.trim();
    try {
      await api.saveSettings(body);
      gq.busyMessage = body.groqBusyMessage ?? gq.busyMessage;
      toast('Saved', 'success');
      await loadGroq();
    } catch (ex) {
      toast(ex.message || 'Could not save', 'error');
    }
  });
  // the mode is a single choice: apply it as soon as it is picked
  Array.from(gqForm.elements.llmProvider).forEach((r) => r.addEventListener('change', () => gqForm.requestSubmit()));

  // "+ Add" opens the small form under its list
  root.addEventListener('click', (e) => {
    const add = e.target.closest('[data-add]');
    if (!add) return;
    const f = $(`#${add.dataset.add}`, root);
    f.hidden = !f.hidden;
    if (!f.hidden) f.querySelector('input')?.focus();
  });
  $('#gq-add', root).addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      await api.groqAddKey(f.label.value.trim(), f.key.value.trim());
      toast('Account added', 'success');
      f.reset();
      f.hidden = true;
      await loadGroq();
    } catch (ex) {
      toast(ex.message || 'Could not add the key', 'error');
    }
  });
  $('#gq-keys', root).addEventListener('click', async (e) => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    b.closest('details')?.removeAttribute('open');
    const id = b.dataset.id;
    try {
      if (b.dataset.act === 'test') {
        const r = await api.groqTest(id);
        toast(r.ok ? `Works (${r.ms} ms)` : r.error || 'The key did not work', r.ok ? 'success' : 'error');
      } else if (b.dataset.act === 'up' || b.dataset.act === 'down') {
        await api.groqMoveKey(String(id).replace(/^db:/, ''), b.dataset.act);
        await loadGroq();
      } else if (b.dataset.act === 'remove') {
        const ok = await confirmDialog({ title: 'Remove this account?', message: `“${b.dataset.label}” will be deleted from the server.`, confirmText: 'Remove', danger: true });
        if (!ok) return;
        await api.groqDeleteKey(String(id).replace(/^db:/, ''));
        toast('Account removed', 'success');
        await loadGroq();
      }
    } catch (ex) {
      toast(ex.message, 'error');
    }
  });
  const cf = mountCloudflare(root, { modelInput: gqForm.elements.cloudflareModel, onData: renderSummary });
  const limitPanel = installLimitPanel(root, (kind, id) => (kind === 'cloudflare' ? cf.find(id) : gq.data?.keys.find((k) => k.id === id)));
  loadGroq();
  gq.timer = setInterval(loadGroq, 5000); // live account status

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

  return {
    destroy: () => {
      clearInterval(gq.timer);
      cf.stop();
      limitPanel.destroy();
      offs.forEach((o) => o());
    },
  };
}
