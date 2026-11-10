-- v2.3: API keys for hosted LLM providers (Groq), added from the console.
-- The key itself is stored encrypted (AES-256-GCM, derived from JWT_SECRET); `hint` is the masked form shown in the UI.

CREATE TABLE IF NOT EXISTS llm_keys (
  id          BIGSERIAL PRIMARY KEY,
  provider    TEXT NOT NULL DEFAULT 'groq',
  label       TEXT NOT NULL,
  key_enc     TEXT NOT NULL,
  hint        TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
