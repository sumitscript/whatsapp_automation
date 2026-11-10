# Speed architecture

How replies are made fast on a CPU-only server, what was measured, and what is not built yet.

## Where the time goes (measured on the live 8 vCPU server)

A reply has two costs. **Reading** the prompt (persona + history + the new message, about 1,000 to 1,400 tokens) and **writing** the answer (about 40 tokens).

| Model | Reads prompt | Writes | Typical reply |
|---|---|---|---|
| llama3.2:3b | ~530 tok/s | ~22 tok/s | 3.5 s |
| openchat (7B) | ~320 tok/s | ~13 tok/s | 7.7 s |
| gemma3:4b | ~140 tok/s | ~19 tok/s | 9.8 s |

Reading is 60 to 80% of the wait. Two replies generated at the same moment each ran ~3x slower (the cores are shared), so parallel generation does not add capacity on a CPU.

## The design: four ideas

```
 person types ─▶ [ inbox ] ─▶ wa.inbound ─▶ [ worker ]  ──▶ reply bubbles ─▶ person reads
                                                │                              │
   1 serialise, don't share the CPU ────────────┤ LLM_CONCURRENCY=1            │
   2 think ahead ───────────────────────────────┴─ after the reply: pre-read ◀───┘ (while they read/type)
   3 background work waits for a quiet gap          persona + history; cancelled the moment a message arrives
   4 answer cache for repeated factual questions
```

1. **Serialise generations (`LLM_CONCURRENCY=1`).** One reply uses every core. A queue gives the first person 8 s instead of 18 s and costs the others nothing.
2. **Think-ahead (predictive prefill).** About 85% of the next prompt (persona + history) is known before the person writes. After each reply the worker has the model read that part (`brain.warm` → `OllamaClient.prefill`: the same messages the real reply will send, a placeholder where the message goes, one token generated). It runs at lowest priority and is cancelled the instant a real message arrives. A test pins that the pre-read messages are exactly the prefix of the real request.
3. **Quiet-gap background work.** Meeting extraction and memory summaries start only after `WORKER_BG_QUIET_MS` (8 s) without messages, so they never split the CPU with the next live reply.
4. **Answer cache** (existing). Standalone factual questions about the owner are answered in milliseconds.

Setting: `prefillAhead` (Settings → Think ahead). The worker also pre-reads the persona alone after a restart or model change.

## Scheduling (what waits for what)

- **The gate wraps model calls only.** Template replies (meeting requests), answer-cache hits and commands never touch the model, so they are answered at once instead of queueing behind a 30-second story.
- **Priorities.** Waiting model calls are served 1 = normal reply, 2 = long reply (story, code: token budget over 450), 3 = background (memory folds, agenda extraction), 4 = think-ahead. A waiter gains one step per 20 s of waiting, so long replies are delayed by short ones but never starved.
- **Live work pre-empts the rest.** A new message aborts think-ahead and any running background job (which is re-queued); once background work finishes, the persona prefix it overwrote in Ollama's cache is read back.
- **One reply per burst** (`core/coalesce.js`): queued jobs of a chat fold into the newest; a generation that has not spoken yet is aborted and re-run with the new text. Debounce is therefore short (400 ms).
- **Idle timeout.** `LLM_TIMEOUT_MS` is "no data for N seconds" for streamed replies (restarted on every chunk), not a cap on the whole reply, so long stories are no longer cut off. A model that is not ready within `WORKER_MODEL_WAIT_MS` (20 s) fails the job with a notice instead of parking every chat.

## Measured effect of think-ahead

Prompt-reading time of the real reply, neutral messages (no knowledge-base lookups), 3 repetitions each:

| Model | Cold | Persona pre-read | Think-ahead |
|---|---|---|---|
| gemma3:4b | 6.3 s | 6.4 s | 6.4 s (no change) |
| llama3.2:3b | 2.0 s | 1.8 s | 1.5 s (−25%) |
| openchat | 4.8 s | 4.5 s | 3.3 s (−31%) |

**gemma3 cannot benefit.** Its sliding-window attention cannot rewind the cache to a partly matching prompt, so any prompt that changes after the persona is read from scratch. `OllamaClient.reusesPrefix()` returns false for `gemma3*` and the worker skips think-ahead for it (no wasted CPU). Llama/Qwen-style models do reuse the prefix.

The bigger lever is therefore the model: llama3.2:3b reads the same prompt about 3x faster than gemma3:4b before any pre-reading.

`POST /api/playground/warm` runs the pre-read for a given history and returns its timings; the experiment is reproducible through it.

## Not built yet (designs)

- **Cache-affinity sharding.** To scale past one server, route a chat to the same worker every time (hash of the chat id over per-worker Redis streams) so its history stays warm in that worker's KV cache. The Redis bus already supports several workers; only the stream-per-shard routing is missing.
- **Fast lane.** Greetings, thanks and emoji replies could use a tiny model with a very short prompt, with `OLLAMA_MAX_LOADED_MODELS=2` so both models stay resident. This changes the voice for those messages, so it needs the owner's approval and a quality test.
- **GPU worker.** A GPU box can serve many chats at once. Point one worker's `OLLAMA_URL` at it; the rest of the architecture is unchanged. This is the only way to answer 100 simultaneous messages without a queue.
