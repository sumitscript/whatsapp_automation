// The reply pipeline. Latency on CPU is dominated by LLM tokens, so everything
// before generation is O(ms): regex intent + language, BM25F retrieval, Redis
// short-term memory, Postgres long-term memory, and a memoized answer cache
// that skips the LLM entirely for repeated factual questions.
//
// Prompt layout (keeps Ollama's KV-cache prefix stable across turns):
//   [system: persona]  [history …]  [user: facts + memory + who + message + language hint]
import { INTENT_PROFILES, isFollowUp, resolveIntent } from './intent.js';
import { replyLanguage } from './lang.js';
import { MessageChunker } from './format.js';
import { estimateTokens, normalize } from '../rag/text.js';
import { ownerAliasList, renderPersona } from '../core/settings-schema.js';
import { clamp, sha1 } from '../core/util.js';

const FALLBACK = "Hmm, I couldn't put a good answer together for that — could you say it another way?";
const SAFETY_TOKENS = 256;
const REFERS_TO_PAST = /\b(remember|recall|last time|earlier|before|previously|already told|i told you|yaad|pichli baar|pichhli baar|pehle|kal|bataya tha|याद|पिछली)\b/i;
const WANTS_ANSWER =
  /\b(not sure|no idea|don'?t know|dont know|idk|tell me the (correct |right )?answer|give me the (correct |right )?answer|(correct|right) answer|just tell me|solution batao|answer batao|pata nahi|nahi pata|nhi pata|bata do|samjha do)\b/i;

/** "yes", "ok sure", "haan bhai", "no", "go on" — a reply to my last message rather than a new topic. */
export function isShortReply(text) {
  const s = String(text ?? '').trim();
  return s.length > 0 && s.length <= 24 && s.split(/\s+/).length <= 3 && !/\?$/.test(s);
}

/**
 * The model ran out of tokens mid-sentence: cut back to the last sentence end, or
 * close an open code fence. Leaves complete text alone.
 */
export function trimDangling(text) {
  const s = String(text ?? '').trimEnd();
  if (!s) return s;
  if ((s.match(/```/g) || []).length % 2 === 1) return `${s}\n\`\`\``;
  if (/[.!?।…)\]"'*_~`]$|\p{Extended_Pictographic}$/u.test(s)) return s;
  let cut = -1;
  for (const m of s.matchAll(/[.!?।…](?=\s)|\p{Extended_Pictographic}(?=\s)/gu)) cut = m.index + m[0].length;
  return cut >= s.length * 0.4 ? s.slice(0, cut) : `${s.replace(/\s+\S*$/, '')}…`;
}

/**
 * Pick the most recent turns that fit both the message cap and the token
 * budget, merging consecutive same-role messages (owner replies from the phone
 * can create assistant→assistant runs, which some chat templates dislike).
 */
export function selectHistory(turns, { maxMessages, maxTokens }) {
  const picked = [];
  let used = 0;
  for (let i = turns.length - 1; i >= 0 && picked.length < maxMessages; i--) {
    const t = turns[i];
    const cost = estimateTokens(t.content) + 4;
    if (used + cost > maxTokens) break;
    used += cost;
    picked.unshift({ role: t.role, content: t.content });
  }
  while (picked.length && picked[0].role !== 'user') picked.shift(); // start on a user turn
  const merged = [];
  for (const t of picked) {
    const last = merged.at(-1);
    if (last && last.role === t.role) last.content += `\n${t.content}`;
    else merged.push({ ...t });
  }
  return { history: merged, tokens: used, dropped: turns.length - picked.length };
}

export function createBrain({ kb, llm, memory, cache, logger }) {
  const log = logger?.child({ component: 'brain' });

  function buildUserTurn({ text, lang, retrieval, intent, contactName, longTerm, owner, replyTo, refersToPast, hadNotes, askedQuestionLast, wantsAnswer }) {
    const parts = [];
    const hasFacts = !!retrieval?.sections.length;
    if (hasFacts) parts.push(`[Facts about me]\n${retrieval.context}`);
    else if (intent === 'about_owner') parts.push(`[Facts about me]\n(none match — don't guess; say the real ${owner} can answer that one)`);
    if (longTerm) {
      parts.push(`[What I remember about ${contactName || 'them'}]\n(from our earlier chats — use it when it helps, e.g. they continue a topic or mention last time; don't recite it)\n${longTerm}`);
    }
    if (contactName) {
      // Testing from a phone named like the owner made the model think it was talking to itself.
      const sameAsOwner = contactName.split(/\s+/)[0].toLowerCase() === owner.toLowerCase();
      parts.push(`[Chatting with] ${contactName} (their WhatsApp name${sameAsOwner ? ` — a different person from me` : ''}; if they told me another name, use that)`);
    }
    // "yes" / "ok" / "haan" only make sense next to what I just asked; small models
    // otherwise answer an older topic from the history or the notes.
    if (replyTo) parts.push(`[My last message]\n${replyTo.length > 280 ? `${replyTo.slice(0, 280)}…` : replyTo}`);

    const rules = [];
    // Without facts a small model fills the gap with an invented job or project.
    if (!hasFacts && intent !== 'about_owner') rules.push(`No facts about me came with this message: don't mention my job, company, projects, studies or personal life.`);
    if (refersToPast) {
      rules.push(hadNotes
        ? `They're referring to an earlier chat: answer from what I remember about them.`
        : `They're referring to an earlier chat, but I have no notes about it: say so honestly and ask for a one-line reminder.`);
    }
    if (wantsAnswer) rules.push('They want the answer: give it directly and correctly first; if their idea was wrong, correct it kindly.');
    if (askedQuestionLast) rules.push("My last message already asked a question: don't end this one with a question.");
    rules.push(replyTo
      ? 'Continue exactly what my last message offered or asked. Do not switch to another topic.'
      : 'Answer the message in the flow of our conversation so far. Use the facts/notes only if they directly help; otherwise ignore them and never bring them up.');

    const hint = lang.code === 'und' ? '' : `\n\n(Reply in ${lang.name}${lang.reason === 'preference' ? ' — they asked for it' : ''}.)`;
    const message = replyTo ? `Message (their reply to my last message): ${text}` : `Message: ${text}`;
    const head = parts.length ? `${parts.join('\n\n')}\n\n---\n` : '';
    return `${head}${message}${hint}\n(${rules.join(' ')})`;
  }

  /**
   * Generate a reply. Yields {type:'chunk', text, index, meta} for every
   * WhatsApp bubble as soon as it is ready, then {type:'done', reply, meta}.
   * @param {{ chatId?: string, text: string, settings: object, history?: {role:string, content:string}[],
   *           contactName?: string, longTermMemory?: string, signal?: AbortSignal }} p
   */
  async function* respond({ chatId, text, settings, history, contactName, longTermMemory, signal }) {
    const started = performance.now();
    const owner = String(settings.ownerName || 'the owner').split(/\s+/)[0];
    const turns = history ?? (chatId ? await memory.get(chatId) : []);
    const theirMessages = turns.filter((t) => t.role === 'user').map((t) => t.content);
    // "I prefer Hinglish" said once (or kept in the notes) outranks one English message.
    const lang = replyLanguage(text, theirMessages, settings.memoryEnabled ? longTermMemory : null);
    // "one more" / "another" / "why?" inherit the topic of the previous turn instead of reading as small talk.
    const { intent, inherited } = resolveIntent(text, turns, { ownerAliases: ownerAliasList(settings) });
    const profile = INTENT_PROFILES[intent];
    const followUp = isFollowUp(text);
    const longTermAvailable = settings.memoryEnabled ? longTermMemory || null : null;
    // Long-term notes bridge sessions. Mid-conversation the history already carries the
    // context, and the notes' old topics pull small models off-topic, so they're skipped
    // unless the chat is fresh or the message refers to the past.
    const refersToPast = REFERS_TO_PAST.test(text);
    const longTerm = longTermAvailable && (turns.length < 6 || refersToPast) ? longTermAvailable : null;
    const lastTurn = turns.at(-1);
    const replyTo = isShortReply(text) && lastTurn?.role === 'assistant' ? lastTurn.content : null;
    // Stop the "…? …? …?" loop: every reply ending in a question feels like an interrogation.
    const askedQuestionLast = lastTurn?.role === 'assistant' && /\?\s*[\p{Extended_Pictographic}\s]*$/u.test(lastTurn.content);
    const wantsAnswer = WANTS_ANSWER.test(text);

    // --- Retrieval (vectorless, ~1ms) ---
    let retrieval = null;
    // Tiny general messages ("ok", "2+2", "one more") never deserve knowledge-base facts.
    const tinyGeneral = intent === 'general' && text.trim().split(/\s+/).length <= 3;
    if (profile.useKb && !tinyGeneral) {
      const lastUser = [...turns].reverse().find((t) => t.role === 'user')?.content;
      const query = followUp && lastUser ? `${lastUser} ${text}` : text;
      retrieval = kb.search(query);
      const confident = retrieval.topScore >= kb.minScore * (profile.useKb === 'auto' ? 1.6 : 1);
      if (!confident) retrieval = profile.useKb === 'auto' ? null : { ...retrieval, sections: [], context: '' };
    }

    const system = `${renderPersona(settings)}\n\nToday's date: ${new Date().toDateString()}.`;
    const persona = sha1(system).slice(0, 10);
    const sources = retrieval?.sections.map((s) => ({ path: s.path, score: s.score })) ?? [];
    const baseMeta = { model: settings.model, intent, inheritedIntent: inherited, language: lang.code, sources, retrievalMs: retrieval?.tookMs ?? 0 };

    // --- Memoized answer cache (standalone factual questions only) ---
    const cacheKey =
      settings.answerCache && intent === 'about_owner' && !followUp && retrieval?.sections.length
        ? `ans:${sha1(`${settings.model}|${kb.version}|${persona}|${lang.code}|${normalize(text).replace(/[^\p{L}\p{N}]+/gu, ' ').trim()}`)}`
        : null;
    if (cacheKey) {
      const hit = await cache.get(cacheKey).catch(() => null);
      if (hit?.bubbles?.length) {
        const meta = { ...baseMeta, cached: true };
        for (const [index, b] of hit.bubbles.entries()) yield { type: 'chunk', text: b, index, meta };
        const reply = hit.bubbles.join('\n\n');
        if (chatId) await memory.append(chatId, [{ role: 'user', content: text }, { role: 'assistant', content: reply }]);
        const latencyMs = Math.round(performance.now() - started);
        yield { type: 'done', reply, meta: { ...meta, latencyMs, firstChunkMs: latencyMs } };
        return;
      }
    }

    // --- Context window: system + notes + reply budget first, history gets the rest ---
    const userTurn = buildUserTurn({
      text, lang, retrieval, intent, contactName, longTerm, owner, replyTo,
      refersToPast, hadNotes: !!longTermAvailable, askedQuestionLast, wantsAnswer,
    });
    const maxTokens = clamp(Math.round(settings.maxTokens * profile.tokenScale), 48, 2048);
    const numCtx = llm.numCtx || 4096;
    const historyBudget = Math.max(0, Math.min(settings.historyTokens ?? 1500, numCtx - estimateTokens(system) - estimateTokens(userTurn) - maxTokens - SAFETY_TOKENS));
    const { history: picked, tokens: historyTokens } = selectHistory(turns, { maxMessages: settings.historyTurns, maxTokens: historyBudget });

    const messages = [{ role: 'system', content: system }, ...picked, { role: 'user', content: userTurn }];
    const temperature = profile.temperature ?? settings.temperature;
    const liveMeta = { ...baseMeta, cached: false };
    const chunker = new MessageChunker({ split: settings.humanize });
    const bubbles = [];
    let firstChunkMs = null;
    let stats = null;

    const emit = (piece) => {
      firstChunkMs ??= Math.round(performance.now() - started);
      bubbles.push(piece);
      return { type: 'chunk', text: piece, index: bubbles.length - 1, meta: liveMeta };
    };

    for await (const ev of llm.chat({ model: settings.model, messages, temperature, maxTokens, signal })) {
      if (ev.type === 'done') {
        stats = ev.stats;
        break;
      }
      for (const piece of chunker.push(ev.text)) yield emit(piece);
    }
    // Hit the token limit mid-sentence? Don't send "…and apne fav tech project ke baareme bhi ba".
    for (const piece of chunker.flush()) {
      const clean = stats?.doneReason === 'length' ? trimDangling(piece) : piece;
      if (clean) yield emit(clean);
    }
    if (!bubbles.length) yield emit(FALLBACK);

    const reply = bubbles.join('\n\n');
    if (chatId) await memory.append(chatId, [{ role: 'user', content: text }, { role: 'assistant', content: reply }]);
    if (cacheKey && reply !== FALLBACK && stats?.doneReason !== 'length') {
      await cache.set(cacheKey, { bubbles }).catch((err) => log?.warn('answer cache write failed', { err }));
    }

    const latencyMs = Math.round(performance.now() - started);
    const tps = stats?.evalMs ? Number(((stats.evalTokens / stats.evalMs) * 1000).toFixed(1)) : null;
    log?.info('Reply generated', {
      chatId,
      intent,
      lang: lang.code,
      langReason: lang.reason,
      latencyMs,
      firstChunkMs,
      tps,
      promptTokens: stats?.promptTokens,
      historyMsgs: picked.length,
      historyTokens,
      memory: !!longTermAvailable, // notes exist for this contact
      memoryUsed: !!longTerm, // notes were put in this prompt
      replyTo: !!replyTo,
      truncated: stats?.doneReason === 'length',
      sources: sources.length,
    });
    yield { type: 'done', reply, meta: { ...liveMeta, latencyMs, firstChunkMs, stats, tokensPerSec: tps, historyMessages: picked.length } };
  }

  /**
   * Fold conversation turns into the contact's long-term memory notes.
   * Runs off the hot path (worker idle queue).
   */
  async function summarize({ previous, turns, contactName, settings }) {
    const owner = String(settings.ownerName || 'the owner').split(/\s+/)[0];
    const who = contactName || 'this contact';
    const lines = [];
    let budget = 2500;
    for (let i = turns.length - 1; i >= 0; i--) {
      const line = `${turns[i].role === 'user' ? 'Them' : 'Me'}: ${turns[i].content}`;
      budget -= estimateTokens(line);
      if (budget < 0) break;
      lines.unshift(line);
    }
    if (!lines.length) return previous || null;
    const messages = [
      {
        role: 'system',
        content: `You keep ${settings.personaName || 'eSumit'}'s private memory notes about a WhatsApp contact (${who}). ${settings.personaName || 'eSumit'} is ${owner}'s AI twin.
Write concise English bullet points, at most 120 words in total, in past tense ("Asked about…", "Likes…"). Start with "- Name: …" using the name they introduced themselves with (their WhatsApp name is only "${who}" and may belong to someone else). Keep: who they are, how they know ${owner} if stated, interests and preferences, topics they discussed, anything they asked to pass on to the real ${owner}, preferred language/tone (write it as "Prefers Hinglish" etc. if they said so). These are background facts, not a to-do list: never write things like "wants to continue" or "still needs". Drop greetings, jokes and small talk. Never store passwords, OTPs, card or bank details.
Return ONLY the updated notes.`,
      },
      { role: 'user', content: `Existing notes:\n${previous || '(none)'}\n\nNew conversation:\n${lines.join('\n')}` },
    ];
    let out = '';
    for await (const ev of llm.chat({ model: settings.model, messages, temperature: 0.2, maxTokens: 220 })) {
      if (ev.type === 'token') out += ev.text;
    }
    out = out.trim();
    return out || previous || null;
  }

  return { respond, summarize };
}

/** In-process stores (tests, playground without Redis). */
export function createMemoryStores() {
  const ctx = new Map();
  const answers = new Map();
  return {
    memory: {
      async get(chatId) {
        return [...(ctx.get(chatId) || [])];
      },
      async append(chatId, msgs) {
        ctx.set(chatId, [...(ctx.get(chatId) || []), ...msgs].slice(-80));
      },
      async trimOldest(chatId, n) {
        ctx.set(chatId, (ctx.get(chatId) || []).slice(n));
      },
      async clear(chatId) {
        ctx.delete(chatId);
      },
    },
    cache: {
      async get(k) {
        return answers.get(k) ?? null;
      },
      async set(k, v) {
        answers.set(k, v);
      },
    },
  };
}
