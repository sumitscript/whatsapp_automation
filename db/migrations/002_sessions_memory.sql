-- v2.1: number whitelisting is replaced by /hi … /bye AI sessions, and each
-- contact gets a long-term memory (rolling summary of past conversations).

CREATE TABLE IF NOT EXISTS sessions (
  id               BIGSERIAL PRIMARY KEY,
  chat_id          TEXT NOT NULL,
  started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_activity_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at         TIMESTAMPTZ,
  ended_reason     TEXT,          -- command | timeout | console
  message_count    INTEGER NOT NULL DEFAULT 0
);
-- At most one open session per chat.
CREATE UNIQUE INDEX IF NOT EXISTS sessions_one_open_per_chat ON sessions (chat_id) WHERE ended_at IS NULL;
CREATE INDEX IF NOT EXISTS sessions_started_at ON sessions (started_at DESC);

ALTER TABLE contacts ADD COLUMN IF NOT EXISTS memory TEXT;
ALTER TABLE contacts ADD COLUMN IF NOT EXISTS memory_updated_at TIMESTAMPTZ;
ALTER TABLE contacts DROP COLUMN IF EXISTS is_whitelisted;

UPDATE settings SET value = '"session"'::jsonb WHERE key = 'replyMode' AND value = '"whitelist"'::jsonb;
