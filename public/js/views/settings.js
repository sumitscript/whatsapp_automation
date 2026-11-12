// Settings, model management, playground and password change.
import { api } from '../api.js';
import { icons } from '../icons.js';
import { infoButton, installLimitPanel, mountCloudflare } from './provider-limits.js';
import { $, confirmDialog, escapeHtml, formatBytes, formatMs, formatNumber, formatWhatsApp, html, raw, toast } from '../util.js';

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
            <label class="field"><span>Persona / system prompt <span class="muted small">(leave empty to use the built-in eSumit persona, which improves with updates)</span></span>
              <textarea name="persona" rows="8" class="code-area small-area" placeholder="Empty = built-in eSumit persona (recommended)"></textarea>
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
              <label class="switch-row"><input type="checkbox" class="switch" name="prefillAhead" /><span>Think ahead<small class="muted"> — pre-read the prompt while they type, so replies start sooner</small></span></label>
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
            <label class="switch-row"><input type="checkbox" class="switch" name="notifyOnTimeout" /><span>Tell people when their session times out<small class="muted"> — sent once, with how to start again</small></span></label>
            <label class="field"><span>Timeout message</span>
              <textarea name="timeoutMessage" rows="3" placeholder="Hey {name}, it's been quiet… send {start} to chat again."></textarea>
            </label>
            <label class="field"><span>Reply to voice notes, photos &amp; stickers</span>
              <textarea name="unsupportedMessage" rows="3" placeholder="Hey {name}! I can only read text, so I can't open {kind}…"></textarea>
            </label>
            <p class="muted small placeholder-help">Placeholders: <code class="wa-inline">{name}</code> contact name ·
              <code class="wa-inline">{persona}</code> persona name · <code class="wa-inline">{owner}</code> owner name ·
              <code class="wa-inline">{start}</code> first start command · <code class="wa-inline">{stop}</code> first stop command ·
              <code class="wa-inline">{kind}</code> what they sent (voice notes, photos…, only in the media reply)</p>

            <h3 class="section-title">Agenda &amp; alerts</h3>
            <label class="switch-row"><input type="checkbox" class="switch" name="agendaEnabled" /><span>Collect appointments &amp; to-dos from chats<small class="muted"> — "let's meet tomorrow at 5" shows up in the Agenda</small></span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="notifyOwnerOnAgenda" /><span>WhatsApp me about new agenda items<small class="muted"> — sent to your own number</small></span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="meetingAck" /><span>Instant reply to meeting requests<small class="muted"> — in their language (English, Hinglish, Hindi, Marathi, Gujarati); says it was noted and passed on, never agrees for you</small></span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="agendaConfirm" /><span>Tell the contact when I accept an appointment<small class="muted"> — the Agenda's Confirm button sends a confirmation message</small></span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="agendaCard" /><span>Attach an invitation card to the confirmation<small class="muted"> — an image generated fresh for each appointment</small></span></label>
            <label class="switch-row"><input type="checkbox" class="switch" name="alertOwner" /><span>WhatsApp me if the AI goes down<small class="muted"> — and when it recovers</small></span></label>
            <label class="field"><span>Timezone for "tomorrow at 5pm"</span><input name="timezone" placeholder="Asia/Kolkata" />
              <small class="muted">IANA name, e.g. Asia/Kolkata, Europe/London, America/New_York</small></label>

            <h3 class="section-title">Availability</h3>
            <p class="muted small">Meetings are only offered and confirmed inside these hours (in the timezone above) and never on top of another confirmed meeting. A request for a taken time gets "not available" plus your free slots.</p>
            <label class="switch-row"><input type="checkbox" class="switch" name="enforceAvailability" /><span>Check availability<small class="muted"> — tell people a taken or out-of-hours time is unavailable, and ask you before double-booking</small></span></label>
            <div class="form-grid">
              <label class="field"><span>Working hours start</span><input type="time" name="workStart" /></label>
              <label class="field"><span>Working hours end</span><input type="time" name="workEnd" /></label>
              <label class="field"><span>Meeting length (minutes)</span><input type="number" name="meetingMinutes" min="5" max="480" step="5" /></label>
              <label class="field"><span>Gap around meetings (minutes)</span><input type="number" name="meetingBufferMinutes" min="0" max="240" step="5" /></label>
            </div>
            <fieldset class="field"><legend>Days you take meetings</legend>
              <div class="chip-group"><label class="chip"><input type="checkbox" name="wd" value="mon" /> Mon</label><label class="chip"><input type="checkbox" name="wd" value="tue" /> Tue</label><label class="chip"><input type="checkbox" name="wd" value="wed" /> Wed</label><label class="chip"><input type="checkbox" name="wd" value="thu" /> Thu</label><label class="chip"><input type="checkbox" name="wd" value="fri" /> Fri</label><label class="chip"><input type="checkbox" name="wd" value="sat" /> Sat</label><label class="chip"><input type="checkbox" name="wd" value="sun" /> Sun</label></div>
            </fieldset>

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

        <div class="settings-col">
        <section class="card" id="groq-card">
          <h2 class="card-title">AI provider</h2>
          <p class="muted small" id="gq-summary">Loading…</p>
          <form id="gq-form" class="stack">
            <div class="radio-cards">
              <label class="radio-card"><input type="radio" name="llmProvider" value="auto" /><span><strong>Auto <span class="muted small">(recommended)</span></strong><small>Groq first, then Cloudflare, while a key can answer; the local model takes over when every account is used up, the providers are down, or no key is added</small></span></label>
              <label class="radio-card"><input type="radio" name="llmProvider" value="ollama" /><span><strong>Local only</strong><small>Always the model on this server's CPU. Private, but slower. Groq is never used</small></span></label>
              <label class="radio-card"><input type="radio" name="llmProvider" value="groq" /><span><strong>Cloud only</strong><small>Groq, then Cloudflare, never the local model. When no account can answer, people get the short-break message below</small></span></label>
            </div>
            <label class="field"><span>Models, best first <span class="muted small">(tried in this order inside every account; Groq's limits are per model)</span></span>
              <input name="groqModels" list="gq-models" placeholder="openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b" autocomplete="off" /><datalist id="gq-models"></datalist>
              <small class="muted">Up to 8, separated by commas. A model an account cannot use is skipped for that account.</small></label>
            <fieldset class="field"><legend>Which account first?</legend>
              <div class="radio-cards">
                <label class="radio-card"><input type="radio" name="groqStrategy" value="priority" /><span><strong>Priority</strong><small>Account 1 with all its models first, then account 2, then 3… (top of the list below is first)</small></span></label>
                <label class="radio-card"><input type="radio" name="groqStrategy" value="spread" /><span><strong>Spread</strong><small>Accounts take turns, so every account's quota is used evenly</small></span></label>
              </div>
            </fieldset>
            <label class="field"><span>Short-break message <span class="muted small">(sent in Groq-only mode when Groq cannot answer)</span></span>
              <textarea name="groqBusyMessage" rows="3" maxlength="1000"></textarea>
              <small class="muted">Placeholders: <code class="wa-inline">{name}</code> <code class="wa-inline">{persona}</code> <code class="wa-inline">{owner}</code>. Sent at most once every 5 minutes per chat.</small></label>
            <div class="form-actions"><button class="btn btn-primary" type="submit">${raw(icons.save)} Save provider</button></div>
          </form>
          <h3 class="section-title">Groq API keys</h3>
          <div id="gq-keys" class="table-scroll"></div>
          <form id="gq-add" class="inline-form">
            <input name="label" placeholder="Label (optional)" aria-label="Key label" maxlength="60" />
            <input name="key" type="password" placeholder="gsk_…" autocomplete="off" aria-label="Groq API key" required />
            <button class="btn btn-ghost" type="submit">${raw(icons.plus)} Add key</button>
          </form>
          <p class="muted small">Keys are stored encrypted on the server and never shown again. When one key hits its rate limit the next one takes over instantly; when all are used up, the local model answers. Keys set in the server's <code class="wa-inline">GROQ_API_KEYS</code> are listed here read-only.</p>
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

  // ---------- Groq provider + keys ----------
  const gq = { data: null, timer: null, modelsLoaded: false, busyMessage: null };
  const gqForm = $('#gq-form', root);
  const cooldownText = (ms) => (ms >= 90_000 ? `${Math.ceil(ms / 60_000)} min` : `${Math.ceil(ms / 1000)} s`);

  function renderGroq() {
    const d = gq.data;
    if (!d) return;
    if (!gqForm.contains(document.activeElement)) {
      Array.from(gqForm.elements.llmProvider).forEach((r) => (r.checked = r.value === d.provider));
      gqForm.elements.groqModels.value = (d.models?.length ? d.models : [d.model]).filter(Boolean).join(', ');
      Array.from(gqForm.elements.groqStrategy).forEach((r) => (r.checked = r.value === (d.strategy || 'priority')));
      if (gq.busyMessage != null) gqForm.elements.groqBusyMessage.value = gq.busyMessage;
    }
    const ready = d.keys.filter((k) => k.state === 'ready').length;
    const c = d.counters;
    const served = c ? html` <span class="muted">Served: ${c.groq} Groq · ${c.local} local${c.fallbacks ? ` · ${c.fallbacks} fell back` : ''}.</span>` : '';
    const keysText = d.keys.length ? `${ready} of ${d.keys.length} key${d.keys.length === 1 ? '' : 's'} ready.` : 'No Groq key added yet.';
    $('#gq-summary', root).innerHTML =
      d.provider === 'ollama'
        ? html`Mode: <strong>Local only</strong>. Groq is not used${raw(d.keys.length ? html` (you have ${d.keys.length} key${d.keys.length === 1 ? '' : 's'} added: switch to <strong>Auto</strong> to use them)` : '')}.${raw(served)}`
        : d.provider === 'groq'
          ? html`Mode: <strong>Groq only</strong> (${d.model}). ${keysText} ${ready ? '' : 'No key can answer right now, so people get the short-break message.'}${raw(served)}`
          : html`Mode: <strong>Auto</strong>. ${d.keys.length ? `Groq (${d.model}) is answering${ready ? '' : ', but every key is resting, so the local model is answering for now'}. ${keysText}` : 'No Groq key yet, so the local model is answering. Add a key below and Groq starts right away.'}${raw(served)}`;
    // The last time Groq could not answer (wrong model name, every key resting, Groq down…), so it is never a mystery.
    const lp = d.lastFallback;
    const problem = lp && Date.now() - lp.at < 30 * 60_000
      ? html` <div class="muted small">Last Groq problem (${new Date(lp.at).toLocaleTimeString()}): ${lp.reason}${/does not exist|access|404/i.test(lp.reason) ? ' Choose a model your key can use from the list in the model box.' : ''}</div>`
      : '';
    $('#gq-summary', root).insertAdjacentHTML('beforeend', problem);
    const compact = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));
    const modelChip = (m) => {
      const name = String(m.id).split('/').pop();
      if (m.state === 'ready') return html`<span class="badge badge-ok" title="${m.id}">${name}</span>`;
      if (m.gone) return html`<span class="badge badge-bad" title="${m.id}: ${m.lastError || ''}">${name} · not available</span>`;
      if (m.state === 'cooling') return html`<span class="badge badge-warn" title="${m.id}: ${m.lastError || ''}">${name} · ${cooldownText(m.cooldownMs)}</span>`;
      return html`<span class="badge badge-bad" title="${m.id}">${name} · off</span>`;
    };
    const consoleKeys = d.keys.filter((k) => k.source === 'console');
    const rows = d.keys.map((k, i) => {
      // Counted in Redis by both the worker and the gateway, so it matches console.groq.com and survives restarts.
      const u = k.usage;
      const perModel = u && u.today.models ? Object.entries(u.today.models).map(([m, v]) => `${m.split('/').pop()}: ${formatNumber(v.calls)} calls, ${compact(v.tokens)}`).join(' · ') : '';
      const used = u
        ? html`Today: ${formatNumber(u.today.calls)} calls · ${compact(u.today.tokens)} tokens${u.today.limited ? ` · ${u.today.limited} rate-limited` : ''}${perModel ? html`<div>${perModel}</div>` : ''}<div>30 days: ${formatNumber(u.d30.calls)} calls · ${compact(u.d30.tokens)} tokens</div>`
        : html`<span>No usage recorded yet</span>`;
      const ci = consoleKeys.findIndex((x) => x.id === k.id);
      const move = k.source === 'console' ? html`<button class="btn btn-ghost btn-sm" data-act="up" data-id="${k.id}" ${ci <= 0 ? 'disabled' : ''} aria-label="Move up" title="Higher priority">↑</button><button class="btn btn-ghost btn-sm" data-act="down" data-id="${k.id}" ${ci < 0 || ci >= consoleKeys.length - 1 ? 'disabled' : ''} aria-label="Move down" title="Lower priority">↓</button> ` : '';
      const remove = k.source === 'console' ? html` <button class="btn btn-ghost btn-sm agenda-del" data-act="remove" data-id="${k.id}" data-label="${k.label}">Remove</button>` : '';
      const state = k.state === 'disabled' ? html`<span class="badge badge-bad">Rejected by Groq</span>` : '';
      const org = k.sameOrgAs ? html`<div class="small" style="color: var(--warn)">Same Groq organization as ${k.sameOrgAs}: they share limits, so this adds no extra capacity</div>` : k.org ? html`<div class="muted small">Organization ${k.org}</div>` : '';
      const why = k.lastError && k.state !== 'ready' ? html`<div class="muted small">${k.lastError}</div>` : '';
      return html`<tr><td><strong>Account ${k.priority ?? i + 1}: ${k.label}</strong> ${raw(infoButton('groq', k.id))}<div class="muted small">${k.hint} · ${k.source === 'env' ? 'server env' : 'console'}</div>${raw(org)}</td><td>${raw(state)}<div class="chip-group">${raw((k.models || []).map(modelChip).join(''))}</div>${raw(why)}</td><td class="muted small">${raw(used)}</td><td class="actions">${raw(move)}<button class="btn btn-ghost btn-sm" data-act="test" data-id="${k.id}">Test</button>${raw(remove)}</td></tr>`;
    });
    $('#gq-keys', root).innerHTML = rows.length
      ? html`<table class="table"><thead><tr><th>Account (priority order)</th><th>Models</th><th>Usage</th><th></th></tr></thead><tbody>${raw(rows.join(''))}</tbody></table>`
      : html`<p class="muted small">No keys yet. Add your Groq API key below (you can add several, one per Groq account).</p>`;
    if (d.unreadable) $('#gq-keys', root).insertAdjacentHTML('beforeend', html`<p class="muted small">${d.unreadable} saved key(s) cannot be read (the server secret changed). Add them again.</p>`);
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
      $('#gq-summary', root).textContent = `Could not load the Groq status: ${e.message}`;
    }
  }
  gqForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const body = { llmProvider: gqForm.elements.llmProvider.value, groqModels: gqForm.elements.groqModels.value.trim() || 'openai/gpt-oss-120b', groqStrategy: gqForm.elements.groqStrategy.value || 'priority', groqBusyMessage: gqForm.elements.groqBusyMessage.value.trim() };
    if (!body.groqBusyMessage) delete body.groqBusyMessage;
    try {
      await api.saveSettings(body);
      gq.busyMessage = body.groqBusyMessage ?? gq.busyMessage;
      toast(body.llmProvider === 'groq' && !gq.data?.keys.length ? 'Saved. Add a Groq key: until then people get the short-break message' : 'AI provider saved', 'success');
      await loadGroq();
    } catch (ex) {
      toast(ex.message || 'Could not save', 'error');
    }
  });
  $('#gq-add', root).addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      const r = await api.groqAddKey(f.label.value.trim(), f.key.value.trim());
      toast(gq.data?.provider === 'ollama' ? `Key added (${r.models} models). The mode is Local only: choose Auto to use Groq` : `Key added (${r.models} models). Groq is active`, 'success');
      f.reset();
      await loadGroq();
    } catch (ex) {
      toast(ex.message || 'Could not add the key', 'error');
    }
  });
  $('#gq-keys', root).addEventListener('click', async (e) => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    const id = b.dataset.id;
    if (b.dataset.act === 'test') {
      b.disabled = true;
      try {
        const r = await api.groqTest(id);
        toast(r.ok ? `Key works (${r.ms} ms, ${r.models} models)` : r.error || 'The key did not work', r.ok ? 'success' : 'error');
      } catch (ex) {
        toast(ex.message, 'error');
      } finally {
        b.disabled = false;
      }
    } else if (b.dataset.act === 'up' || b.dataset.act === 'down') {
      try {
        await api.groqMoveKey(String(id).replace(/^db:/, ''), b.dataset.act);
        await loadGroq();
      } catch (ex) {
        toast(ex.message, 'error');
      }
    } else if (b.dataset.act === 'remove') {
      const ok = await confirmDialog({ title: 'Remove this key?', message: `“${b.dataset.label}” will be deleted from the server. Other keys keep working.`, confirmText: 'Remove', danger: true });
      if (!ok) return;
      try {
        await api.groqDeleteKey(String(id).replace(/^db:/, ''));
        toast('Key removed', 'success');
        await loadGroq();
      } catch (ex) {
        toast(ex.message, 'error');
      }
    }
  });
  loadGroq();
  gq.timer = setInterval(loadGroq, 5000); // live key status: who is cooling, who is ready
  // Cloudflare Workers AI card + the usage-limit panels behind every account's "i" button
  const cf = mountCloudflare(root, {});
  const limitPanel = installLimitPanel(root, (kind, id) => (kind === 'cloudflare' ? cf.find(id) : gq.data?.keys.find((k) => k.id === id)));

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
