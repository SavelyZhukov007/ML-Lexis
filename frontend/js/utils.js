const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);
const tx = (id, v) => { const e = $(id); if (e) e.textContent = (v === null || v === undefined) ? '—' : v; };
const show = id => $(id)?.classList.remove('hidden');
const hide = id => $(id)?.classList.add('hidden');
const toggle = (id, cond) => cond ? show(id) : hide(id);

function fmt(n, d = 4) { if (n == null) return '—'; return typeof n === 'number' ? n.toFixed(d) : String(n); }
function fmtK(n) {
  if (!n && n !== 0) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}
function fmtLR(lr) { if (!lr) return '—'; return lr < 0.001 ? lr.toExponential(2) : lr.toFixed(6); }
function nowTime() { return new Date().toTimeString().slice(0, 8); }
function escapeHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function fmtTs(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit' });
}

// API helpers
async function apiGet(url) {
  const r = await fetch(url); return r.json();
}
async function apiPost(url, body = {}) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return r.json();
}
async function apiDel(url) {
  const r = await fetch(url, { method: 'DELETE' }); return r.json();
}

// Log
const MAX_LOG = 200;
function addLog(boxId, msg, type = 'info') {
  const el = $(boxId); if (!el) return;
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-ts">${nowTime()}</span><span class="log-msg ${type}">${escapeHtml(msg)}</span>`;
  el.appendChild(line);
  while (el.children.length > MAX_LOG) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
}

// Modal
function openModal(id) { $(id)?.classList.remove('hidden'); }
function closeModal(id) { $(id)?.classList.add('hidden'); }
function closeOverlay(e, id) { if (e.target.id === id) closeModal(id); }

// Slider bind
function bindSlider(sid, vid, fmt) {
  const s = $(sid), v = $(vid); if (!s || !v) return;
  const upd = () => { v.textContent = fmt ? fmt(s.value) : s.value; };
  s.addEventListener('input', upd); upd();
}

// Nav active
function initNav() {
  const path = location.pathname.replace(/\.html$/, '').replace(/\/$/, '') || '/';
  $$('.nav-link').forEach(a => {
    const href = a.getAttribute('href')?.replace(/\.html$/, '').replace(/\/$/, '') || '';
    a.classList.toggle('active', path === href || (path === '/' && href === '/'));
  });
}

// SSE
window._sseHandlers = [];
function onSSE(fn) { window._sseHandlers.push(fn); }

function connectSSE() {
  const es = new EventSource('/api/stream');
  es.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      window._lastData = d;
      window._sseHandlers.forEach(fn => fn(d));
    } catch (_) { }
  };
  es.onerror = () => setTimeout(connectSSE, 3000);
}

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  connectSSE();
});
