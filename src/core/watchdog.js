// Decides when a connection that still says "ready" has really died. WhatsApp Web inside the headless browser can go stale
// after hours of idleness (the tab is throttled or its socket is dropped) without emitting any "disconnected" event, so the
// console kept showing a live session that received nothing until someone pressed Disconnect and Connect.
// Pure bookkeeping (no I/O), unit-tested: feed it the result of each health probe.
export class Watchdog {
  /** @param {{ failLimit?: number }} o  consecutive failed probes before the connection counts as dead */
  constructor({ failLimit = 3 } = {}) {
    this.failLimit = failLimit;
    this.fails = 0;
  }

  /** Record one probe. Returns true when the connection should be restarted (and starts counting again). */
  record(ok) {
    if (ok) {
      this.fails = 0;
      return false;
    }
    this.fails++;
    if (this.fails < this.failLimit) return false;
    this.fails = 0;
    return true;
  }

  reset() {
    this.fails = 0;
  }
}

/** Reject after `ms` so a frozen browser cannot hang a health probe forever. */
export function withTimeout(promise, ms, what = 'operation') {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out after ${Math.round(ms / 1000)}s`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
