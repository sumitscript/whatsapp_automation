-- v2.2: Agenda — appointments and to-dos that people ask of the owner in chats
-- (extracted by the worker) plus items added by hand in the console.

CREATE TABLE IF NOT EXISTS agenda_items (
  id             BIGSERIAL PRIMARY KEY,
  chat_id        TEXT,                              -- NULL for manual items without a contact
  kind           TEXT NOT NULL DEFAULT 'todo',      -- appointment | todo
  title          TEXT NOT NULL,
  details        TEXT,
  due_at         TIMESTAMPTZ,
  all_day        BOOLEAN NOT NULL DEFAULT false,    -- true: a date without a time of day
  due_text       TEXT,                              -- the words they used: "tomorrow at 5pm"
  location       TEXT,
  status         TEXT NOT NULL DEFAULT 'pending',   -- pending | confirmed | declined | done
  source         TEXT NOT NULL DEFAULT 'ai',        -- ai | manual
  source_message TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agenda_status_due ON agenda_items (status, due_at);
CREATE INDEX IF NOT EXISTS agenda_chat ON agenda_items (chat_id);
