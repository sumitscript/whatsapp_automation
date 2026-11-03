// Knowledge base: browse/edit files and test vectorless retrieval.
import { api } from '../api.js';
import { icons } from '../icons.js';
import { $, confirmDialog, escapeHtml, formatBytes, html, raw, relativeTime, toast } from '../util.js';

export async function mount(root) {
  const s = { files: [], active: null, original: '', version: null };

  root.innerHTML = html`
    <div class="page page-full kb-page">
      <header class="page-header">
        <div>
          <h1>Knowledge base</h1>
          <p class="muted" id="kb-summary">Loading…</p>
        </div>
      </header>
      <div class="kb-grid">
        <aside class="card kb-files">
          <h2 class="card-title">Files</h2>
          <ul id="kb-file-list" class="kb-file-list" role="list"></ul>
        </aside>
        <section class="card kb-editor">
          <div class="kb-editor-head">
            <h2 class="card-title" id="kb-file-title">Select a file</h2>
            <span class="muted small" id="kb-dirty" hidden>Unsaved changes</span>
            <button class="btn btn-primary btn-sm" id="kb-save" disabled>${raw(icons.save)} Save <kbd>Ctrl S</kbd></button>
          </div>
          <textarea id="kb-text" class="code-area" spellcheck="false" disabled aria-label="File content"
            placeholder="Choose a file on the left to edit it. Saving re-indexes instantly — no embeddings to rebuild."></textarea>
        </section>
        <section class="card kb-search">
          <h2 class="card-title">Retrieval tester</h2>
          <form id="kb-search-form" class="inline-form">
            <input id="kb-query" placeholder="Ask something, e.g. “Sumit ne kaunse projects banaye?”" aria-label="Test query" />
            <button class="btn btn-primary" type="submit">${raw(icons.search)} Search</button>
          </form>
          <div id="kb-results" class="kb-results"><p class="muted small">See which sections the assistant would read for a question.</p></div>
        </section>
      </div>
    </div>`;

  const listEl = $('#kb-file-list', root);
  const text = $('#kb-text', root);
  const saveBtn = $('#kb-save', root);
  const dirtyEl = $('#kb-dirty', root);

  const isDirty = () => s.active && text.value !== s.original;

  function renderFiles() {
    if (!s.files.length) {
      listEl.innerHTML = '<li class="list-empty">No knowledge files found.</li>';
      return;
    }
    listEl.innerHTML = s.files
      .map((f) =>
        html`<li><button class="kb-file ${raw(f.name === s.active ? 'active' : '')}" data-name="${f.name}">
          ${raw(icons.file)}
          <span class="kb-file-body">
            <span class="kb-file-title">${f.title || f.name}</span>
            <span class="muted small">${f.name} · ${f.sections ?? 0} sections · ${formatBytes(f.bytes)}${f.updatedAt ? ` · ${relativeTime(f.updatedAt)}` : ''}</span>
          </span>
        </button></li>`
      )
      .join('');
  }

  async function loadIndex() {
    try {
      const data = await api.kb();
      s.files = data.files || [];
      s.version = data.version;
      $('#kb-summary', root).textContent = `${s.files.length} files · ${data.sectionCount ?? 0} sections · index version ${data.version ?? '—'}`;
      renderFiles();
    } catch (e) {
      $('#kb-summary', root).textContent = `Could not load: ${e.message}`;
    }
  }

  async function openFile(name) {
    if (isDirty()) {
      const ok = await confirmDialog({ title: 'Discard changes?', message: `You have unsaved edits in ${s.active}.`, confirmText: 'Discard', danger: true });
      if (!ok) return;
    }
    try {
      const { content } = await api.kbFile(name);
      s.active = name;
      s.original = content ?? '';
      text.value = s.original;
      text.disabled = false;
      saveBtn.disabled = true;
      dirtyEl.hidden = true;
      $('#kb-file-title', root).textContent = name;
      renderFiles();
      text.focus();
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function save() {
    if (!s.active || !isDirty()) return;
    saveBtn.disabled = true;
    try {
      const { version } = await api.kbSave(s.active, text.value);
      s.original = text.value;
      dirtyEl.hidden = true;
      toast(`Saved and re-indexed${version ? ` (v${version})` : ''}`, 'success');
      loadIndex();
    } catch (e) {
      saveBtn.disabled = false;
      toast(`Save failed: ${e.message}`, 'error');
    }
  }

  listEl.addEventListener('click', (e) => {
    const b = e.target.closest('.kb-file');
    if (b) openFile(b.dataset.name);
  });
  text.addEventListener('input', () => {
    const d = isDirty();
    saveBtn.disabled = !d;
    dirtyEl.hidden = !d;
  });
  text.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      save();
    }
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault();
      text.setRangeText('  ', text.selectionStart, text.selectionEnd, 'end');
      text.dispatchEvent(new Event('input'));
    }
  });
  saveBtn.addEventListener('click', save);

  const beforeUnload = (e) => {
    if (isDirty()) {
      e.preventDefault();
      e.returnValue = '';
    }
  };
  window.addEventListener('beforeunload', beforeUnload);

  $('#kb-search-form', root).addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = $('#kb-query', root).value.trim();
    const out = $('#kb-results', root);
    if (!q) return;
    out.innerHTML = '<div class="spinner spinner-sm"></div>';
    try {
      const r = await api.kbSearch(q);
      const terms = (r.expandedTerms || []).map((t) => `<span class="badge">${escapeHtml(t)}</span>`).join(' ');
      const top = Math.max(...(r.results || []).map((x) => Number(x.score) || 0), 1);
      out.innerHTML = html`
        <p class="muted small">${(r.results || []).length} results in ${r.tookMs != null ? `${Number(r.tookMs).toFixed(2)} ms` : '—'}</p>
        ${raw(terms ? `<div class="term-list"><span class="muted small">Expanded terms:</span> ${terms}</div>` : '')}
        ${raw(
          (r.results || []).length
            ? (r.results || [])
                .map((x) =>
                  html`<article class="kb-result">
                    <div class="kb-result-head">
                      <span class="kb-path">${x.path}</span>
                      <span class="score">${Number(x.score).toFixed(3)}</span>
                    </div>
                    <div class="score-bar"><span style="width:${Math.max(4, (Number(x.score) / top) * 100).toFixed(0)}%"></span></div>
                    <p class="kb-snippet">${x.snippet || ''}</p>
                    <span class="muted small">${x.file || ''}</span>
                  </article>`
                )
                .join('')
            : '<p class="muted">No section matched. The assistant would answer from general knowledge or say it doesn’t know.</p>'
        )}`;
    } catch (ex) {
      out.innerHTML = html`<p class="form-error">${ex.message}</p>`;
    }
  });

  await loadIndex();
  return { destroy: () => window.removeEventListener('beforeunload', beforeUnload) };
}
