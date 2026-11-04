// Watches the LLM (Ollama + worker heartbeat). Pushes every state change to the
// console and, when the AI has been down for a while, messages the owner on
// their own WhatsApp number — once when it breaks and once when it recovers.
import { DownAlerter } from '../llm/status.js';

const CHECK_MS = 30_000;
const STARTUP_GRACE_MS = 5 * 60_000; // first boot pulls/loads the model: not an outage

const minutes = (ms) => (ms < 90_000 ? `${Math.round(ms / 1000)} s` : `${Math.round(ms / 60_000)} min`);

export class LlmMonitor {
  /** @param {{ probe: () => Promise<{state:string,message:string}>, settings, inbox, hub, logger }} deps */
  constructor({ probe, settings, inbox, hub, logger, checkMs = CHECK_MS, graceMs = STARTUP_GRACE_MS }) {
    Object.assign(this, { probe, settings, inbox, hub, checkMs });
    this.log = logger.child({ component: 'llm-monitor' });
    this.alerter = new DownAlerter({ afterMs: 90_000 });
    this.bootedAt = Date.now();
    this.graceMs = graceMs;
    this.last = null;
    this.timer = null;
  }

  start() {
    this.timer = setInterval(() => this.tick().catch((err) => this.log.warn('LLM check failed', { err })), this.checkMs);
    this.timer.unref();
    setTimeout(() => this.tick().catch(() => {}), 10_000).unref();
  }

  stop() {
    clearInterval(this.timer);
  }

  async tick() {
    const status = await this.probe();
    if (status.state !== this.last) {
      this.log[status.state === 'down' ? 'error' : 'info']('LLM status changed', { from: this.last, to: status.state, message: status.message });
      this.last = status.state;
      this.hub.emit('llm:status', { state: status.state, message: status.message });
    }
    const inGrace = Date.now() - this.bootedAt < this.graceMs;
    const ev = this.alerter.update(inGrace && status.state === 'down' ? 'degraded' : status.state, status.message);
    if (!ev || !this.settings.get().alertOwner) return;
    const owner = String(this.settings.get().ownerName || 'there').split(/\s+/)[0];
    const text =
      ev.type === 'down'
        ? `⚠️ ${owner}, the AI brain is down — people messaging me get no replies.\n${status.message}\nCheck Dashboard → LLM health in the console.`
        : `✅ ${owner}, the AI brain is back (it was down for about ${minutes(ev.downForMs)}).`;
    const sent = await this.inbox.notifyOwner(text);
    this.log.warn(ev.type === 'down' ? 'LLM down alert' : 'LLM recovery notice', { sent });
  }
}
