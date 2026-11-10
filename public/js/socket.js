// Socket.IO connection that re-broadcasts server events on an app-wide hub.
// The Socket.IO client (global `io`) is loaded lazily after sign-in, so the login page stays light.

const EVENTS = ['wa:status', 'message:new', 'chat:update', 'typing', 'log', 'model:pull', 'agenda:update', 'llm:status', 'chats:wiped'];

let socket = null;

let loading = null;
let generation = 0; // bumped on disconnect so a slow script load can't revive a closed socket
function loadClient() {
  if (typeof window.io === 'function') return Promise.resolve();
  loading ??= new Promise((resolve) => {
    const s = document.createElement('script');
    s.src = '/socket.io/socket.io.js';
    s.onload = s.onerror = () => resolve();
    document.head.appendChild(s);
  });
  return loading;
}

export async function connectSocket(hub, { onUnauthorized } = {}) {
  if (socket) return socket;
  const gen = generation;
  await loadClient();
  if (socket || gen !== generation) return socket;
  if (typeof window.io !== 'function') {
    console.warn('Socket.IO client not loaded; live updates disabled.');
    hub.emit('socket:state', 'unavailable');
    return null;
  }
  socket = window.io({ path: '/socket.io', withCredentials: true, transports: ['websocket', 'polling'] });

  socket.on('connect', () => {
    hub.emit('socket:state', 'connected');
    // Re-establish log subscription after reconnects.
    if (logSubscribers > 0) socket.emit('logs:subscribe');
  });
  socket.on('disconnect', () => hub.emit('socket:state', 'disconnected'));
  socket.on('connect_error', (err) => {
    hub.emit('socket:state', 'error');
    if (String(err?.message || '').toLowerCase().includes('unauthorized')) {
      disconnectSocket();
      onUnauthorized?.();
    }
  });
  EVENTS.forEach((evt) => socket.on(evt, (payload) => hub.emit(evt, payload)));
  return socket;
}

export function disconnectSocket() {
  generation++;
  if (socket) {
    socket.removeAllListeners();
    socket.disconnect();
    socket = null;
  }
  logSubscribers = 0;
}

let logSubscribers = 0;
/** Ref-counted log stream subscription; returns an unsubscribe function. */
export function subscribeLogs() {
  logSubscribers++;
  if (logSubscribers === 1) socket?.emit('logs:subscribe');
  let done = false;
  return () => {
    if (done) return;
    done = true;
    logSubscribers = Math.max(0, logSubscribers - 1);
    if (logSubscribers === 0) socket?.emit('logs:unsubscribe');
  };
}
