// Cloudflare Workers AI through its OpenAI-compatible endpoint, as one more provider after Groq.
//
// Free tier: 10,000 "neurons" per day per Cloudflare account, shared by every model, reset at 00:00 UTC. A neuron is a unit of
// GPU work, charged per token with a rate that depends on the model, so the number of replies a day depends on the model.
// A key here is "ACCOUNT_ID:API_TOKEN" (the account id selects the URL, the token authorises it).
import { GroqClient, msToUtcMidnight } from './groq.js';

export const CF_DAILY_NEURONS = 10_000;
export const CF_DEFAULT_MODEL = '@cf/google/gemma-4-26b-a4b-it';

// Neurons per million input / output tokens (Cloudflare's pricing page, Nov 2026).
export const CF_NEURON_RATES = {
  '@cf/google/gemma-4-26b-a4b-it': [9091, 27273],
  '@cf/meta/llama-3.1-8b-instruct-fp8': [13778, 26128],
  '@cf/meta/llama-3.2-3b-instruct': [4625, 30475],
  '@cf/ibm-granite/granite-4.0-h-micro': [1542, 10158],
  '@cf/openai/gpt-oss-20b': [18182, 27273],
  '@cf/qwen/qwen3-30b-a3b-fp8': [4625, 30475],
};
const FALLBACK_RATE = [13778, 26128]; // unknown model: assume a dear one, so the estimate errs on the safe side

export function estimateNeurons(model, promptTokens = 0, completionTokens = 0) {
  const [i, o] = CF_NEURON_RATES[model] || FALLBACK_RATE;
  return Math.round(((promptTokens * i + completionTokens * o) / 1e6) * 100) / 100;
}

/** "<32 hex account id>:<token>" → { accountId, token } or null. */
export function parseCloudflareKey(secret) {
  const m = /^([0-9a-f]{32}):([\w-]{20,200})$/i.exec(String(secret || '').trim());
  return m ? { accountId: m[1].toLowerCase(), token: m[2] } : null;
}

export const isCloudflareKeyFormat = (secret) => !!parseCloudflareKey(secret);

const base = (secret, baseUrl) => {
  const k = parseCloudflareKey(secret);
  if (!k) throw new Error('Cloudflare key must look like ACCOUNT_ID:API_TOKEN');
  return `${String(baseUrl).replace(/\/+$/, '')}/accounts/${k.accountId}/ai/v1`;
};

/** Cloudflare's "you have used up your daily free allocation" (code 3036, HTTP 429): rest until 00:00 UTC. */
export const allowanceUsedUp = (status, detail) => (/daily free allocation|3036/i.test(String(detail)) && (status === 429 || status === 403) ? msToUtcMidnight() : null);

export function createCloudflareClient({ pool, baseUrl = 'https://api.cloudflare.com/client/v4', timeoutMs = 30_000, logger, fetchImpl } = {}) {
  return new GroqClient({
    pool,
    provider: 'cloudflare',
    baseUrl,
    timeoutMs,
    logger,
    fetchImpl,
    reasoning: false,
    urlFor: (secret) => `${base(secret, baseUrl)}/chat/completions`,
    tokenFor: (secret) => parseCloudflareKey(secret)?.token ?? secret,
    exhausted: allowanceUsedUp,
    neuronsFor: estimateNeurons,
  });
}

/** Check a Cloudflare account id + token by listing one model: costs no neurons and does not depend on which model is chosen. { ok, status, ms, error?, note? } */
export async function probeCloudflare(secret, { baseUrl = 'https://api.cloudflare.com/client/v4', fetchImpl, timeoutMs = 15_000 } = {}) {
  const f = fetchImpl || ((...a) => fetch(...a));
  const k = parseCloudflareKey(secret);
  if (!k) return { ok: false, status: 0, ms: 0, error: 'The key must look like ACCOUNT_ID:API_TOKEN (the 32-character account id, a colon, then the API token)' };
  const t0 = performance.now();
  try {
    const res = await f(`${String(baseUrl).replace(/\/+$/, '')}/accounts/${k.accountId}/ai/models/search?per_page=1`, {
      headers: { authorization: `Bearer ${k.token}` },
      signal: AbortSignal.timeout(timeoutMs),
    });
    const ms = Math.round(performance.now() - t0);
    if (res.ok) return { ok: true, status: 200, ms };
    const text = await res.text().catch(() => '');
    if (allowanceUsedUp(res.status, text)) return { ok: true, status: res.status, ms, note: "The key works, but today's free allowance is already used up" };
    const why = res.status === 401 || res.status === 403 ? 'Cloudflare rejected this token or account id (check the token has the Workers AI permission)' : res.status === 404 ? 'Cloudflare did not find that account id' : `Cloudflare answered HTTP ${res.status}`;
    return { ok: false, status: res.status, ms, error: why };
  } catch (err) {
    return { ok: false, status: 0, ms: Math.round(performance.now() - t0), error: `Could not reach Cloudflare: ${err.message}` };
  }
}
