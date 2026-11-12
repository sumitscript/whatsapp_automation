// Thin fetch wrapper around the gateway REST API.

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

let unauthorizedHandler = () => {};
export function onUnauthorized(fn) {
  unauthorizedHandler = fn;
}

async function request(method, path, body, { silent401 = false } = {}) {
  const opts = {
    method,
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new ApiError('Network error — is the server reachable?', 0, null);
  }
  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text.slice(0, 200) };
    }
  }
  if (!res.ok) {
    if (res.status === 401 && !silent401) unauthorizedHandler();
    throw new ApiError(data?.error || `${res.status} ${res.statusText}`, res.status, data);
  }
  return data ?? {};
}

const enc = encodeURIComponent;
function qs(params) {
  const p = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') p.set(k, v);
  });
  const s = p.toString();
  return s ? `?${s}` : '';
}

export const api = {
  // auth
  login: (username, password) => request('POST', '/api/auth/login', { username, password }, { silent401: true }),
  logout: () => request('POST', '/api/auth/logout', {}, { silent401: true }),
  me: () => request('GET', '/api/auth/me', undefined, { silent401: true }),
  changePassword: (currentPassword, newPassword) =>
    request('POST', '/api/auth/password', { currentPassword, newPassword }),

  // whatsapp
  waStatus: () => request('GET', '/api/wa/status'),
  waAction: (action, body = {}) => request('POST', `/api/wa/${action}`, body),

  // chats
  chats: (search = '', limit = 100) => request('GET', `/api/chats${qs({ search, limit })}`),
  messages: (chatId, before, limit = 50) =>
    request('GET', `/api/chats/${enc(chatId)}/messages${qs({ before, limit })}`),
  send: (chatId, body) => request('POST', `/api/chats/${enc(chatId)}/send`, { body }),
  patchChat: (chatId, patch) => request('PATCH', `/api/chats/${enc(chatId)}`, patch),
  clearContext: (chatId) => request('DELETE', `/api/chats/${enc(chatId)}/context`),
  session: (chatId, action, notify = false) =>
    request('POST', `/api/chats/${enc(chatId)}/session`, { action, notify }),
  memory: (chatId) => request('GET', `/api/chats/${enc(chatId)}/memory`),
  saveMemory: (chatId, memory) => request('PUT', `/api/chats/${enc(chatId)}/memory`, { memory }),
  forgetMemory: (chatId) => request('DELETE', `/api/chats/${enc(chatId)}/memory`),
  sessions: (active = true, limit = 50) => request('GET', `/api/sessions${qs({ active, limit })}`),

  // contacts
  contacts: () => request('GET', '/api/contacts'),
  patchContact: (id, patch) => request('PATCH', `/api/contacts/${enc(id)}`, patch),
  deleteContact: (id) => request('DELETE', `/api/contacts/${enc(id)}`),

  // settings & models
  settings: () => request('GET', '/api/settings'),
  saveSettings: (partial) => request('PUT', '/api/settings', partial),
  models: () => request('GET', '/api/models'),
  groqStatus: () => request('GET', '/api/llm/groq'),
  groqModels: () => request('GET', '/api/llm/groq/models'),
  groqAddKey: (label, key) => request('POST', '/api/llm/groq/keys', { label, key }),
  groqDeleteKey: (id) => request('DELETE', `/api/llm/groq/keys/${enc(id)}`),
  groqMoveKey: (id, dir) => request('POST', `/api/llm/groq/keys/${enc(id)}/move`, { dir }),
  groqTest: (id) => request('POST', '/api/llm/groq/test', { id }),
  llmStatus: () => request('GET', '/api/llm/status'),
  pullModel: (name) => request('POST', '/api/models/pull', { name }),
  deleteModel: (name) => request('DELETE', `/api/models?name=${encodeURIComponent(name)}`),
  playground: (message, history) => request('POST', '/api/playground', { message, history }),

  // knowledge base
  kb: () => request('GET', '/api/kb'),
  kbFile: (name) => request('GET', `/api/kb/${enc(name)}`),
  kbSave: (name, content) => request('PUT', `/api/kb/${enc(name)}`, { content }),
  kbSearch: (query) => request('POST', '/api/kb/search', { query }),

  // agenda
  agenda: (params) => request('GET', `/api/agenda${qs(params)}`),
  agendaCreate: (body) => request('POST', '/api/agenda', body),
  agendaUpdate: (id, patch) => request('PATCH', `/api/agenda/${enc(id)}`, patch),
  agendaDelete: (id) => request('DELETE', `/api/agenda/${enc(id)}`),
  availability: () => request('GET', '/api/availability'),
  addBlock: (body) => request('POST', '/api/availability/blocks', body),
  deleteBlock: (id) => request('DELETE', `/api/availability/blocks/${enc(id)}`),
  agendaSendCard: (id) => request('POST', `/api/agenda/${enc(id)}/send-card`),

  // ops
  logs: (limit = 500, level) => request('GET', `/api/logs${qs({ limit, level })}`),
  stats: () => request('GET', '/api/stats'),
};
