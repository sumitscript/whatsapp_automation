// One reply per burst of messages, not one per message.
//
// People chat in bursts ("Sumit I want to talk to you" / "Stop this" / "Haa"). Each burst after the
// debounce becomes its own job, and a CPU model takes 10–40 s per reply, so by the time job 1 is
// answered the person has sent two more messages. The old worker then answered every one of them in
// turn: stale, repeated replies that ignored the newest message.
//
// Coalescer fixes that per chat:
//   • queued jobs: only the newest one runs; it answers everything the person said since the last reply.
//   • a job that is generating but has not sent a bubble yet is aborted and re-run with the new text.
//   • once a bubble is out it is never recalled: the new message simply gets its own reply afterwards.
// Pure (no I/O) so it is unit-tested without Redis or an LLM.

export class Coalescer {
  #pending = new Map(); // chatId → { texts: string[], latest: jobId }
  #active = new Map(); // chatId → { jobId, texts: string[], ctl: AbortController, emitted: boolean }

  /** A job arrived for the chat. Returns true when it aborted a generation that had not spoken yet. */
  add(chatId, jobId, text) {
    const p = this.#pending.get(chatId) || { texts: [], latest: jobId };
    p.texts.push(text);
    p.latest = jobId;
    this.#pending.set(chatId, p);
    const a = this.#active.get(chatId);
    if (a && !a.emitted && !a.ctl.signal.aborted) {
      a.ctl.abort(new Error('superseded'));
      return true;
    }
    return false;
  }

  /**
   * The job is about to run. Returns null when a newer job exists (this one is folded into it),
   * otherwise { text, signal } with every message not yet answered.
   */
  claim(chatId, jobId) {
    const p = this.#pending.get(chatId);
    if (!p || p.latest !== jobId) return null;
    this.#pending.delete(chatId);
    const entry = { jobId, texts: p.texts, ctl: new AbortController(), emitted: false };
    this.#active.set(chatId, entry);
    return { text: p.texts.join('\n'), signal: entry.ctl.signal };
  }

  /** The first bubble went out: from here the reply is committed. */
  markEmitted(chatId) {
    const a = this.#active.get(chatId);
    if (a) a.emitted = true;
  }

  /** The job ended. If it was aborted, its messages go back in front of the newer ones. */
  release(chatId, jobId) {
    const a = this.#active.get(chatId);
    if (!a || a.jobId !== jobId) return false;
    this.#active.delete(chatId);
    if (!a.ctl.signal.aborted) return false;
    const p = this.#pending.get(chatId);
    if (p) p.texts.unshift(...a.texts);
    else this.#pending.set(chatId, { texts: [...a.texts], latest: jobId });
    return true;
  }

  isSuperseded(signal) {
    return !!signal?.aborted && signal.reason?.message === 'superseded';
  }

  get size() {
    return this.#pending.size + this.#active.size;
  }
}
