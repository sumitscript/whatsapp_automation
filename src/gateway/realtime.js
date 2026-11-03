// Socket.IO: pushes QR/connection status, new messages, typing, model pull
// progress and the live log tail to authenticated console sessions.
import { Server } from 'socket.io';
import { channels } from '../core/redis.js';

export async function createRealtime(httpServer, { auth, wa, subscriber, logger }) {
  const log = logger.child({ component: 'realtime' });
  const io = new Server(httpServer, {
    path: '/socket.io',
    serveClient: true,
    pingInterval: 20_000,
    pingTimeout: 20_000,
    maxHttpBufferSize: 64 * 1024,
  });

  io.use(async (socket, next) => {
    try {
      const user = await auth.authenticate({ cookie: socket.handshake.headers.cookie, authorization: socket.handshake.auth?.token && `Bearer ${socket.handshake.auth.token}` });
      if (!user) return next(new Error('unauthorized'));
      socket.data.user = user;
      next();
    } catch {
      next(new Error('unauthorized'));
    }
  });

  io.on('connection', (socket) => {
    socket.join('admins');
    socket.emit('wa:status', wa.status());
    socket.on('logs:subscribe', () => socket.join('logs'));
    socket.on('logs:unsubscribe', () => socket.leave('logs'));
    log.debug('Console connected', { user: socket.data.user.username });
  });

  // Live log tail: every service publishes its log lines to Redis.
  // (No logging inside this handler — it would feed back into itself.)
  await subscriber.subscribe(channels.logs);
  subscriber.on('message', (ch, line) => {
    if (ch !== channels.logs) return;
    if (!io.sockets.adapter.rooms.get('logs')?.size) return;
    try {
      io.to('logs').emit('log', JSON.parse(line));
    } catch {
      /* ignore malformed */
    }
  });

  return {
    io,
    emit: (event, data) => io.to('admins').emit(event, data),
    close: () => new Promise((r) => io.close(() => r())),
  };
}
