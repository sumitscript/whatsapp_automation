// Classic (non-module) script: apply the saved theme before first paint.
// Kept as a file rather than inline so the strict CSP (script-src 'self') allows it.
try {
  var t = localStorage.getItem('wa-admin-theme');
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
} catch (e) {}
