// Curated Ollama models for CPU-only hosting on an 8 vCPU / 32 GB KVM, ranked
// for conversational quality in English + Indian languages (incl. Hinglish).
// Speed figures are rough Q4 CPU estimates; run `npm run bench` on the box
// for real numbers.

export const MODEL_CATALOG = Object.freeze([
  {
    name: 'gemma3:4b',
    tier: 'balanced',
    params: '4B',
    ramGb: 4,
    languages: '140+ languages; strong Hindi, Bengali, Marathi, Tamil, Telugu, Gujarati and Hinglish for its size',
    note: 'Default. Best multilingual quality per CPU cycle, natural tone, good at jokes and casual chat. ≈10–15 tok/s on 8 vCPU.',
    recommended: true,
  },
  {
    name: 'qwen3:4b',
    tier: 'balanced',
    params: '4B',
    ramGb: 3.5,
    languages: '119 languages; good Hindi, decent other Indic languages',
    note: 'Strongest coding/reasoning at 4B. Thinking is disabled automatically for speed. ≈10–15 tok/s.',
  },
  {
    name: 'llama3.2:3b',
    tier: 'fast',
    params: '3B',
    ramGb: 2.5,
    languages: 'Officially 8 languages incl. Hindi; weaker on other Indic scripts',
    note: 'Fastest model that still holds a good conversation. ≈15–22 tok/s.',
  },
  {
    name: 'qwen2.5:3b',
    tier: 'fast',
    params: '3B',
    ramGb: 2.3,
    languages: '29+ languages; okay Hindi',
    note: 'Fast and good at short factual answers and simple code. ≈15–22 tok/s.',
  },
  {
    name: 'gemma3:12b',
    tier: 'quality',
    params: '12B',
    ramGb: 9,
    languages: '140+ languages; best Indic quality in this list',
    note: 'Noticeably smarter and more fluent, but ≈4–6 tok/s on CPU — replies take several seconds.',
  },
  {
    name: 'qwen3:8b',
    tier: 'quality',
    params: '8B',
    ramGb: 6,
    languages: '119 languages',
    note: 'Best coding + reasoning that is still usable on CPU. ≈6–9 tok/s.',
  },
  {
    name: 'aya-expanse:8b',
    tier: 'quality',
    params: '8B',
    ramGb: 6,
    languages: '23 languages incl. Hindi; tuned specifically for multilingual chat',
    note: 'Very natural Hindi chat; weaker at code. ≈6–9 tok/s.',
  },
]);

export function catalogWithInstalled(installed = []) {
  const names = new Set(installed.map((m) => m.name));
  return MODEL_CATALOG.map((m) => ({ ...m, installed: names.has(m.name) }));
}
