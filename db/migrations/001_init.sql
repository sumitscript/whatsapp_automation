-- Users of the web console
CREATE TABLE IF NOT EXISTS users (
  id            SERIAL PRIMARY KEY,
  username      TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL DEFAULT 'admin',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_login_at TIMESTAMPTZ
);

-- One row per WhatsApp chat and/or whitelisted phone number.
-- A number can be whitelisted before it ever messages (chat_id NULL); the
-- first inbound message links the chat to the row by phone.
CREATE TABLE IF NOT EXISTS contacts (
  id              SERIAL PRIMARY KEY,
  chat_id         TEXT UNIQUE,
  phone           TEXT UNIQUE,
  name            TEXT,
  is_group        BOOLEAN NOT NULL DEFAULT false,
  is_whitelisted  BOOLEAN NOT NULL DEFAULT false,
  is_blocked      BOOLEAN NOT NULL DEFAULT false,
  ai_paused       BOOLEAN NOT NULL DEFAULT false,
  notes           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_message_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS contacts_last_message_at ON contacts (last_message_at DESC NULLS LAST);

-- Permanent conversation log (inbound, AI replies, manual replies)
CREATE TABLE IF NOT EXISTS messages (
  id            BIGSERIAL PRIMARY KEY,
  wa_message_id TEXT UNIQUE,
  chat_id       TEXT NOT NULL,
  direction     TEXT NOT NULL CHECK (direction IN ('in', 'out')),
  body          TEXT NOT NULL DEFAULT '',
  type          TEXT NOT NULL DEFAULT 'chat',
  is_ai         BOOLEAN NOT NULL DEFAULT false,
  sent_by       TEXT,             -- ai | console:<user> | phone
  model         TEXT,
  intent        TEXT,
  language      TEXT,
  latency_ms    INTEGER,
  cached        BOOLEAN NOT NULL DEFAULT false,
  sources       JSONB,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_chat_id_id ON messages (chat_id, id DESC);
CREATE INDEX IF NOT EXISTS messages_created_at ON messages (created_at);

CREATE TABLE IF NOT EXISTS settings (
  key        TEXT PRIMARY KEY,
  value      JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
  id         BIGSERIAL PRIMARY KEY,
  username   TEXT,
  action     TEXT NOT NULL,
  details    JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_log_created_at ON audit_log (created_at DESC);
