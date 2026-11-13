// The reply pipeline. Latency on CPU is dominated by LLM tokens, so everything
// before generation is O(ms): regex intent + language, BM25F retrieval, Redis
// short-term memory, Postgres long-term memory, and a memoized answer cache
// that skips the LLM entirely for repeated factual questions.
//
// Prompt layout (keeps Ollama's KV-cache prefix stable across turns):
//   [system: persona]  [history …]  [user: facts + memory + who + message + language hint]
import { INTENT_PROFILES, isFollowUp, resolveIntent } from './intent.js';
import { replyLanguage, statedLanguagePreference } from './lang.js';
import { MessageChunker } from './format.js';
import { isRepeat, recentBubbles } from './repeat.js';
import { asksAboutPlans, buildExtractionMessages, guessLocation, guessLocationFromContext, heuristicItem, localParts, isMeetingRequest, isPassOnRequest, meetingKind, meetingMode, meetingTitle, meetingTopic, mentionsMeeting, parseExtraction, parseWhen, resolveItems, wantsToMeet, zonedToUtc } from './agenda.js';
import { meetingAck, passOnAck, slotTakenAck } from './invite.js';
import { slotsText } from './slots.js';
import { estimateTokens, normalize } from '../rag/text.js';
import { ownerAliasList, renderPersona } from '../core/settings-schema.js';
import { clamp, sha1 } from '../core/util.js';

const FALLBACK = "Hmm, I couldn't put a good answer together for that — could you say it another way?";
const SAFETY_TOKENS = 256;
const BACKGROUND_PRIORITY = 3; // LLM gate priority for memory folds and agenda extraction (live replies are 1–2, think-ahead 4)
const TYPICAL_TURN_TOKENS = 150; // size of the final user turn (facts + rules + message) assumed when pre-reading
const REFERS_TO_PAST = /\b(remember|recall|last time|earlier|before|previously|already told|i told you|yaad|pichli baar|pichhli baar|pehle|kal|bataya tha|याद|पिछली)\b/i;
const WANTS_ANSWER =
  /\b(not sure|no idea|don'?t know|dont know|idk|tell me the (correct |right )?answer|give me the (correct |right )?answer|(correct|right) answer|just tell me|solution batao|answer batao|pata nahi|nahi pata|nhi pata|bata do|samjha do)\b/i;

const COMPACT_ASSISTANT = 600;
const COMPACT_USER = 1200;
/**
 * Shorten an old turn for the prompt: keep its start and end (where offers and questions
 * live). Deterministic, so the history prefix stays identical turn to turn and Ollama's
 * prompt cache keeps working.
 */
export function compactTurn(t) {
  const max = t.role === 'assistant' ? COMPACT_ASSISTANT : COMPACT_USER;
  const s = String(t.content ?? '');
  if (s.length <= max) return s;
  if (/```/.test(s) && t.role === 'user') return s.slice(0, max); // their code: the start matters
  const head = s.slice(0, Math.round(max * 0.6)).replace(/\s+\S*$/, '');
  const tail = s.slice(-Math.round(max * 0.3)).replace(/^\S*\s+/, '');
  return `${head} […] ${tail}`;
}

/** "minimum 200 words", "300 shabd mein" → 200 / 300 (capped at 800), else null. */
export function requestedWords(text) {
  const m = String(text ?? '').match(/\b(\d{2,4})\s*(?:\+\s*)?(?:words?|shabd|शब्द)\b/i);
  return m ? Math.min(Number(m[1]), 800) : null;
}

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
    // The newest turn stays whole (a "yes" may refer to its last line); older long replies are
    // shortened, which keeps prompt evaluation fast on CPU and stops old stories anchoring new ones.
    const content = i === turns.length - 1 ? t.content : compactTurn(t);
    const cost = estimateTokens(content) + 4;
    if (used + cost > maxTokens) break;
    used += cost;
    picked.unshift({ role: t.role, content });
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

export function createBrain({ kb, llm, memory, cache, logger, availability = null }) {
  const log = logger?.child({ component: 'brain' });

  // These blocks sit inside the *user* message, where "me/my" naturally reads as the
  // user. Every label therefore says explicitly whose information it is — with
  // "[Facts about me]" a 4B model credited the owner's projects to the contact and
  // retold the contact's interview prep as its own life.
  function buildUserTurn({ text, lang, retrieval, intent, contactName, longTerm, owner, replyTo, refersToPast, hadNotes, askedQuestionLast, wantsAnswer, wordTarget, wantsAnother }) {
    const parts = [];
    const them = contactName || 'them';
    const hasFacts = !!retrieval?.sections.length;
    const factsLabel = `[My facts — about ${owner}, the one replying (me). Not about ${them}]`;
    if (hasFacts) parts.push(`${factsLabel}\n${retrieval.context}`);
    else if (intent === 'about_owner') parts.push(`${factsLabel}\n(none match — don't guess; say the real ${owner} can answer that one)`);
    if (longTerm) {
      parts.push(`[Notes about ${them} — the person I'm chatting with. Their life, not mine]\n(from our earlier chats — use it when it helps, e.g. they continue a topic or mention last time; don't recite it)\n${longTerm}`);
    }
    if (contactName) {
      // Testing from a phone named like the owner made the model think it was talking to itself.
      const sameAsOwner = contactName.split(/\s+/)[0].toLowerCase() === owner.toLowerCase();
      parts.push(`[I'm chatting with] ${contactName} (their WhatsApp name${sameAsOwner ? ` — a different person from me` : ''}; if they told me another name, use that)`);
    }
    // "yes" / "ok" / "haan" only make sense next to what I just asked; small models
    // otherwise answer an older topic from the history or the notes.
    if (replyTo) parts.push(`[What I said last]\n${replyTo.length > 280 ? `${replyTo.slice(0, 280)}…` : replyTo}`);

    const rules = [];
    if (intent === 'creative') {
      // Fiction must not borrow my real life or theirs, and must not rerun earlier stories.
      rules.push(`This is creative writing: invent original characters, places and plot — it is not about me or ${them} unless they asked for that. Pick a fresh theme and setting; don't reuse topics from earlier stories in this chat.`);
    } else if (!hasFacts && intent !== 'about_owner') {
      // Without facts a small model fills the gap with an invented job or project.
      rules.push(`No facts about me came with this message: don't mention my job, company, projects, studies or personal life.`);
    }
    if (hasFacts && longTerm) rules.push(`"My facts" are about me; the notes and their messages are about ${them} — never mix the two up.`);
    if (refersToPast) {
      rules.push(hadNotes
        ? `They're referring to an earlier chat: answer from the notes about ${them}.`
        : `They're referring to an earlier chat, but I have no notes about it: say so honestly and ask for a one-line reminder.`);
    }
    if (wantsAnswer) rules.push('They want the answer: give it directly and correctly first; if their idea was wrong, correct it kindly.');
    if (askedQuestionLast) rules.push("What I said last already asked a question: don't end this one with a question.");
    if (isMeetingRequest(text)) rules.push(`This asks ${owner} for a meeting, call or favour. Don't agree, refuse or promise anything for me: say I'll pass it on and the real ${owner} will confirm.`);
    if (asksAboutPlans(text)) rules.push(`They ask what I'm doing or if I'm free. I don't know my real plans or schedule: never invent an activity. Say I'm not sure and that I can ask the real ${owner}.`);
    if (wantsAnother) rules.push(`They want another one: give a NEW ${intent === 'joke' ? 'joke' : 'one'}, different from everything I already said in this chat. Never repeat or reword an earlier one.`);
    if (intent === 'math') rules.push('Give the result in plain text (like "2 + 2 = 4"), not in a code block.');
    if (wordTarget) rules.push(`They asked for about ${wordTarget} words: write at least that much.`);
    rules.push(replyTo
      ? 'Continue exactly what I said last offered or asked. Do not switch to another topic.'
      : 'Answer the message in the flow of our conversation so far. Use the facts/notes only if they directly help; otherwise ignore them and never bring them up.');

    // Small models copy the language of the earlier replies in the chat; say it outright, and that the old replies don't count.
    if (lang.code === 'en') rules.push('Write the whole reply in plain English with no Hindi or Hinglish words (no "bhai", "yaar", "haan", "nahi"), even if my earlier replies in this chat used another language.');
    else if (lang.reason === 'preference') rules.push(`Write the whole reply in ${lang.name}, even if my earlier replies in this chat used another language.`);
    const hint = lang.code === 'und' ? '' : `\n\n(Reply in ${lang.name}${lang.reason === 'preference' ? ' — they asked for it' : ''}.)`;
    const message = replyTo ? `Their message (a reply to what I said last): ${text}` : `Their message: ${text}`;
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
    // A "Prefers Hinglish" note must not fight the language they just asked for.
    const stripPref = (n) => n.split('\n').filter((l) => !/\bprefers?\s+(?:hinglish|hindi|english|marathi|gujarati|tamil|telugu|bengali|kannada|malayalam|punjabi|urdu)\b/i.test(l)).join('\n').trim();
    const longTermAvailable = settings.memoryEnabled && longTermMemory ? (lang.reason === 'preference' ? stripPref(longTermMemory) || null : longTermMemory) : null;
    // Long-term notes bridge sessions. Mid-conversation the history already carries the
    // context, and the notes' old topics pull small models off-topic, so they're skipped
    // unless the chat is fresh or the message refers to the past.
    const refersToPast = REFERS_TO_PAST.test(text);
    // Creative writing never gets the notes: the contact's life ended up as "my" story.
    const longTerm = longTermAvailable && intent !== 'creative' && (turns.length < 6 || refersToPast) ? longTermAvailable : null;
    const wordTarget = requestedWords(text);
    const lastTurn = turns.at(-1);
    // "one more" after a joke/story asks for a NEW one. Anchoring it to my last message made the
    // model repeat that message, so these follow-ups skip the "what I said last" anchor.
    const wantsAnother = inherited && (intent === 'joke' || intent === 'creative');
    const replyTo = !wantsAnother && isShortReply(text) && lastTurn?.role === 'assistant' ? lastTurn.content : null;
    // Stop the "…? …? …?" loop: every reply ending in a question feels like an interrogation.
    const askedQuestionLast = lastTurn?.role === 'assistant' && /\?\s*[\p{Extended_Pictographic}\s]*$/u.test(lastTurn.content);
    const wantsAnswer = WANTS_ANSWER.test(text);

    // --- Meeting requests: an instant template in their language, never an agreement ---
    // "let's meet tomorrow at 6", "kal subah 6 baje station pe milte hain", "कल शाम 5 बजे मीटिंग", or a
    // follow-up that only carries the time ("tomorrow at six near the office" after "I want to meet").
    // A small model gets these wrong (it says "sure!" or invents a plan); a template cannot. The worker still
    // files the request in the Agenda for the owner to confirm.
    // Naming the owner ("I want to meet Sumit at 8") classifies as about_owner and "Hello?" as a greeting, so every
    // intent qualifies except the ones that are clearly something else (code, maths, jokes, stories).
    const meetingish = isMeetingRequest(text) || wantsToMeet(text);
    if (settings.meetingAck !== false && !['creative', 'math', 'joke'].includes(intent) && (intent !== 'coding' || (meetingish && !text.includes('```'))) && !asksAboutPlans(text) && text.length <= 400) {
      const tz = settings.timezone || 'Asia/Kolkata';
      let hit = parseWhen(text, { timeZone: tz });
      const before = theirMessages.slice(-3).join('\n');
      // We just offered free times or asked for one: a bare "12 pm" or "ok" belongs to that conversation, not to the model
      // (which answered "12 pm is booked" on its own). A time that matches exactly one offered slot picks that slot.
      const mtgKey = chatId ? `mtg:${chatId}` : null;
      const offered = mtgKey ? await cache.get(mtgKey).catch(() => null) : null;
      const followingOffer = !!offered?.at && Date.now() - offered.at < 30 * 60_000;
      // "tomorrow on Google Meet" asked for the time; the answer "9 PM" belongs to tomorrow, not to today.
      if (followingOffer && hit && !hit.hasDay && !hit.allDay && offered.askedDate) {
        const day = localParts(new Date(offered.askedDate), tz);
        const t = localParts(hit.dueAt, tz);
        hit = { ...hit, dueAt: zonedToUtc(day.y, day.m, day.d, t.h, t.mi, tz), hasDay: true };
      }
      if (followingOffer && hit && !hit.hasDay && offered.slots?.length) {
        const want = localParts(hit.dueAt, tz);
        const same = offered.slots.map((x) => new Date(x)).filter((d) => {
          const p = localParts(d, tz);
          return p.h === want.h && p.mi === want.mi;
        });
        if (same.length === 1) hit = { ...hit, dueAt: same[0], hasDay: true, allDay: false };
      }
      let ack = null;
      let ackIntent = 'meeting';
      let offeredNow = [];
      // "I want to meet you" / "mujhe aapse milna hai" (no time yet) is answered with the "which day, time and place?" template too.
      if (isMeetingRequest(text) || (hit && before && mentionsMeeting(before)) || wantsToMeet(text) || (followingOffer && hit)) {
        ack = meetingAck({
          code: lang.code,
          name: contactName,
          owner: settings.ownerName,
          kind: meetingKind(`${before}\n${text}`),
          dueAt: hit?.dueAt,
          allDay: hit?.allDay,
          place: guessLocation(text) || guessLocationFromContext(theirMessages.slice(-3), { timeZone: tz }),
          topic: meetingTopic(text) || meetingTopic(before),
          mode: meetingMode(`${before}\n${text}`),
          timeZone: tz,
        });
        // The asked time is taken, blocked, outside working hours or past: say it is not available and offer free slots.
        // (A time that only overlaps another request that is still waiting is accepted: the owner decides and is told.)
        if (availability && hit && !hit.allDay) {
          const av = await availability.assess(hit.dueAt).catch((err) => {
            log?.warn('Availability check failed; treating the time as free', { err });
            return null;
          });
          if (av && !av.ok) {
            const free = await availability.freeSlots({ count: 4, perDay: 2 }).catch(() => []);
            offeredNow = free;
            ack = slotTakenAck({ code: lang.code, name: contactName, owner: settings.ownerName, dueAt: hit.dueAt, allDay: false, timeZone: tz, slots: slotsText(free, tz, ['hi', 'mr', 'gu'].includes(lang.code) ? lang.code : 'en') });
            ackIntent = 'meeting_unavailable';
          }
        }
      } else if (isPassOnRequest(text, { ownerNames: ownerAliasList(settings) })) {
        // "remind him", "tell Sumit …", "unhe bol dena": noted and passed on, no promise made
        ack = passOnAck({ code: lang.code, name: contactName, owner: settings.ownerName });
        ackIntent = 'pass_on';
      }
      if (ack) {
        const meta = { model: 'template', intent: ackIntent, inheritedIntent: false, language: lang.code, sources: [], retrievalMs: 0, cached: false };
        // Remember that we are waiting for an answer (a time, or a choice among the offered slots); a complete request clears it.
        if (mtgKey && ackIntent !== 'pass_on') {
          const waiting = ackIntent === 'meeting_unavailable' || !hit || hit.allDay;
          await cache.set(mtgKey, waiting ? { at: Date.now(), slots: offeredNow.map((d) => d.toISOString()), askedDate: hit?.allDay && hit.dueAt ? hit.dueAt.toISOString() : offered?.askedDate ?? null } : { at: 0 }).catch(() => {});
        }
        // A complete, free request travels with the reply so the worker files it at once, without the model and without a queue.
        if (ackIntent === 'meeting' && hit && !hit.allDay) {
          meta.meeting = { dueAt: hit.dueAt.toISOString(), dueText: hit.matched || null, place: guessLocation(text) || guessLocationFromContext(theirMessages.slice(-3), { timeZone: tz }), kind: meetingKind(`${before}\n${text}`), topic: meetingTopic(text) || meetingTopic(before), mode: meetingMode(`${before}\n${text}`) };
        }
        yield { type: 'chunk', text: ack, index: 0, meta };
        if (chatId) await memory.append(chatId, [{ role: 'user', content: text }, { role: 'assistant', content: ack }]);
        const latencyMs = Math.round(performance.now() - started);
        yield { type: 'done', reply: ack, meta: { ...meta, latencyMs, firstChunkMs: latencyMs } };
        return;
      }
    }

    // --- Retrieval (vectorless, ~1ms) ---
    let retrieval = null;
    // Tiny general messages ("ok", "2+2", "one more") and meeting requests never deserve knowledge-base facts.
    const tinyGeneral = intent === 'general' && (text.trim().split(/\s+/).length <= 3 || isMeetingRequest(text));
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
      refersToPast, hadNotes: !!longTermAvailable, askedQuestionLast, wantsAnswer, wordTarget, wantsAnother,
    });
    // "minimum 200 words" needs ~1.6 tokens per word; otherwise the reply budget follows the intent.
    const maxTokens = clamp(Math.max(Math.round(settings.maxTokens * profile.tokenScale), wordTarget ? Math.ceil(wordTarget * 1.6) + 40 : 0), 48, 1400);
    const numCtx = llm.numCtx || 4096;
    const historyBudget = Math.max(0, Math.min(settings.historyTokens ?? 1500, numCtx - estimateTokens(system) - estimateTokens(userTurn) - maxTokens - SAFETY_TOKENS));
    const { history: picked, tokens: historyTokens } = selectHistory(turns, { maxMessages: settings.historyTurns, maxTokens: historyBudget });

    const messages = [{ role: 'system', content: system }, ...picked, { role: 'user', content: userTurn }];
    const temperature = profile.temperature ?? settings.temperature;
    const liveMeta = { ...baseMeta, cached: false };
    let chunker = new MessageChunker({ split: settings.humanize });
    const bubbles = [];
    let firstChunkMs = null;
    let stats = null;

    const emit = (piece) => {
      firstChunkMs ??= Math.round(performance.now() - started);
      bubbles.push(piece);
      return { type: 'chunk', text: piece, index: bubbles.length - 1, meta: liveMeta };
    };

    // A small model sometimes starts with what it already said. Catch that on the first bubble (nothing
    // is sent yet) and retry once, warmer and told to say something new; later bubbles stream as before.
    const earlier = recentBubbles(turns);
    for (let attempt = 0; attempt < 2; attempt++) {
      const own = new AbortController();
      const sig = signal ? AbortSignal.any([signal, own.signal]) : own.signal;
      const retry = attempt === 1;
      const req = retry ? [...messages.slice(0, -1), { role: 'user', content: `${userTurn}\n(I already said almost exactly that. Say something new and different, shorter.)` }] : messages;
      let repeated = false;
      stats = null;
      chunker = new MessageChunker({ split: settings.humanize });
      try {
        for await (const ev of llm.chat({ model: settings.model, messages: req, temperature: retry ? Math.min(1.1, temperature + 0.25) : temperature, maxTokens, signal: sig, provider: settings.llmProvider, groqModel: settings.groqModel, groqModels: settings.groqModels })) {
          if (ev.type === 'meta') {
            // The router says which provider/model is answering (Groq key pool or the local model); show it, not the setting.
            liveMeta.model = ev.model || liveMeta.model;
            liveMeta.provider = ev.provider;
            continue;
          }
          if (ev.type === 'done') {
            stats = ev.stats;
            break;
          }
          for (const piece of chunker.push(ev.text)) {
            if (!retry && !bubbles.length && isRepeat(piece, earlier)) {
              repeated = true;
              own.abort();
              break;
            }
            yield emit(piece);
          }
          if (repeated) break;
        }
        // Hit the token limit mid-sentence? Don't send "…and apne fav tech project ke baareme bhi ba".
        if (!repeated) {
          for (const piece of chunker.flush()) {
            const clean = stats?.doneReason === 'length' ? trimDangling(piece) : piece;
            if (!clean) continue;
            if (!retry && !bubbles.length && isRepeat(clean, earlier)) {
              repeated = true;
              break;
            }
            yield emit(clean);
          }
        }
      } catch (err) {
        if (!repeated) throw err; // our own abort is expected; a caller's abort or a real failure is not
      }
      if (!repeated) break;
      log?.info('Repeated reply caught, regenerating', { chatId });
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
  async function summarize({ previous, turns, contactName, settings, signal }) {
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
    for await (const ev of llm.chat({ model: settings.model, messages, temperature: 0.2, maxTokens: 220, priority: BACKGROUND_PRIORITY, signal })) {
      if (ev.type === 'token') out += ev.text;
    }
    out = out.trim() || previous || '';
    // The summary model sometimes drops "I prefer Hinglish"; the next session then answers in
    // English. Keep a stated preference in the notes deterministically.
    const stated = statedLanguagePreference(turns.filter((t) => t.role === 'user').map((t) => t.content));
    if (stated) {
      const label = stated.name.split(' ')[0];
      if (!new RegExp(`prefers?\\s+${label}`, 'i').test(out)) out = `${out.replace(/\n?.*\bprefers?\s+(hinglish|hindi|english|marathi|gujarati|tamil|telugu|bengali|kannada|malayalam|punjabi|urdu)\b.*/gi, '')}\n- Prefers ${label}`.trim();
    }
    return out || null;
  }

  /**
   * Read the conversation for commitments (meetings, to-dos). Runs off the hot path.
   * The model decides what the item is; dates are read deterministically.
   * @returns {Promise<object[]>} items with { kind, title, details, location, dueAt, allDay, dueText }
   */
  async function extractAgenda({ turns, text, contactName, settings, now = new Date(), signal }) {
    const timeZone = settings.timezone || 'Asia/Kolkata';
    let items = null;
    try {
      let out = '';
      const messages = buildExtractionMessages({ ownerName: settings.ownerName, contactName, turns, now, timeZone });
      for await (const ev of llm.chat({ model: settings.model, messages, temperature: 0.1, maxTokens: 220, priority: BACKGROUND_PRIORITY, signal })) {
        if (ev.type === 'token') out += ev.text;
      }
      items = parseExtraction(out);
    } catch (err) {
      if (signal?.aborted) throw err; // pre-empted by a live message: the caller re-queues, no half-made item
      log?.warn('Agenda extraction failed', { err });
    }
    const userTexts = turns.filter((t) => t.role === 'user').map((t) => t.content);
    const resolved = items ? resolveItems(items, { text, userTexts, now, timeZone, contactName }) : [];
    // A small model often leaves `location` empty even when the message names a place.
    // The place named in the CURRENT message wins; an earlier one is only a fallback (joining them let the older place win).
    const place = guessLocation(text) || guessLocationFromContext(userTexts.slice(-4, -1), { now, timeZone });
    const withPlace = (list) => list.map((it) => (it.location || !place ? it : { ...it, location: place }));
    // A small model sometimes files "Confirm meeting with Aditya" as a to-do: a meeting with a date is an appointment.
    const ctx = [...userTexts.slice(-3), text].join('\n');
    const asMeeting = (list) =>
      list.map((it) => {
        if (!(it.kind === 'todo' && it.dueAt && mentionsMeeting(it.title))) return it;
        const topic = meetingTopic(text) || meetingTopic(ctx);
        return { ...it, kind: 'appointment', topic, mode: meetingMode(ctx), title: meetingTitle({ kind: meetingKind(ctx), topic, contactName }) };
      });
    if (resolved.length) return withPlace(asMeeting(resolved));
    // The model failed or found nothing: a plain "let's meet tomorrow at 5" is still worth keeping.
    const h = heuristicItem({ text, contextText: userTexts.slice(-4, -1).join('\n'), contactName, now, timeZone });
    return h && (items === null || h.kind === 'appointment') ? withPlace([h]) : [];
  }

  /**
   * Think-ahead (predictive prefill). About 85% of every prompt (persona + recent history) is known before the
   * person writes their next message, and on a CPU reading it is most of the wait (~150 tokens/s). So after a
   * reply, while they read and type, the worker has the model read that known part; the real message then only
   * pays for its own few tokens. The prefix is built exactly like respond() builds it (same system text, same
   * deterministic selectHistory), with a placeholder where the new message will go.
   * @param {{ chatId?: string, history?: {role:string, content:string}[], settings: object, signal?: AbortSignal }} p
   */
  async function warm({ chatId, history, settings, signal }) {
    const turns = history ?? (chatId ? await memory.get(chatId) : []);
    const system = `${renderPersona(settings)}\n\nToday's date: ${new Date().toDateString()}.`;
    const numCtx = llm.numCtx || 4096;
    const maxTokens = clamp(settings.maxTokens, 48, 1400);
    const budget = Math.max(0, Math.min(settings.historyTokens ?? 1500, numCtx - estimateTokens(system) - TYPICAL_TURN_TOKENS - maxTokens - SAFETY_TOKENS));
    const { history: picked } = selectHistory(turns, { maxMessages: settings.historyTurns, maxTokens: budget });
    const messages = [{ role: 'system', content: system }, ...picked, { role: 'user', content: '…' }];
    return llm.prefill({ model: settings.model, messages, signal, provider: settings.llmProvider });
  }

  return { respond, summarize, extractAgenda, warm };
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
