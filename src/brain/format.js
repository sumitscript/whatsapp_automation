// Turns streamed LLM markdown into natural WhatsApp messages.
//  - converts Markdown to WhatsApp formatting (*bold*, _italic_, ~strike~, ```code```)
//  - splits long answers into several bubbles at paragraph boundaries, the way
//    a person texts, so the first bubble goes out while the rest is generated
//  - never splits inside a code fence

/** Markdown → WhatsApp formatting (code fences are left untouched). */
export function toWhatsApp(text) {
  const parts = String(text ?? '').split(/(```[\s\S]*?```)/g);
  return parts
    .map((part) => {
      if (part.startsWith('```')) return part.replace(/^```[a-zA-Z0-9_+-]*\n/, '```\n');
      return part
        .replace(/^\s{0,3}#{1,6}\s+(.+?)\s*#*$/gm, '*$1*') // headings → bold line
        .replace(/\*\*(.+?)\*\*/g, '*$1*') // **bold** → *bold*
        .replace(/__(.+?)__/g, '_$1_')
        .replace(/~~(.+?)~~/g, '~$1~')
        .replace(/^\s*[-*+]\s+/gm, '• ') // bullets
        .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '$1 ($2)') // [text](url)
        .replace(/^(\s*)(assistant|ai|bot)\s*:\s*/i, '$1');
    })
    .join('')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export class MessageChunker {
  /**
   * @param {{ minChars?: number, maxChars?: number, split?: boolean }} opts
   *   split=false keeps the whole reply as one bubble; after maxBubbles-1 splits
   *   the rest of the answer stays together in the final bubble.
   */
  constructor({ minChars = 60, maxChars = 700, maxBubbles = 4, split = true } = {}) {
    this.minChars = minChars;
    this.maxBubbles = maxBubbles;
    this.maxChars = maxChars;
    this.split = split;
    this.buf = '';
    this.sent = 0;
  }

  /** Feed streamed text; returns zero or more finished bubbles. */
  push(delta) {
    this.buf += delta;
    if (!this.split) return [];
    const out = [];
    while (this.sent < this.maxBubbles - 1) {
      const cut = this.#findCut();
      if (cut < 0) break;
      const piece = toWhatsApp(this.buf.slice(0, cut));
      this.buf = this.buf.slice(cut).replace(/^\s+/, '');
      if (piece) {
        out.push(piece);
        this.sent++;
      }
    }
    return out;
  }

  /** Remaining text at end of stream. */
  flush() {
    const piece = toWhatsApp(this.buf);
    this.buf = '';
    if (piece) this.sent++;
    return piece ? [piece] : [];
  }

  #findCut() {
    const b = this.buf;
    // Outside a code fence only: count fences before each candidate cut.
    const fenceOpenAt = (i) => (b.slice(0, i).match(/```/g) || []).length % 2 === 1;

    let idx = b.indexOf('\n\n');
    while (idx >= 0) {
      if (idx >= this.minChars && !fenceOpenAt(idx)) {
        // Keep a lead-in line ending with ':' attached to what follows (lists, code).
        const before = b.slice(0, idx).trimEnd();
        if (!before.endsWith(':')) return idx + 2;
      }
      idx = b.indexOf('\n\n', idx + 2);
    }
    if (b.length > this.maxChars && !fenceOpenAt(b.length)) {
      const window = b.slice(0, this.maxChars);
      const sentence = Math.max(window.lastIndexOf('. '), window.lastIndexOf('? '), window.lastIndexOf('! '), window.lastIndexOf('। '));
      if (sentence > this.minChars) return sentence + 2;
    }
    return -1;
  }
}
