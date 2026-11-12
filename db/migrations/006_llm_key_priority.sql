-- v2.5: the order of the Groq accounts is the priority order (account 1 is used up before account 2).
ALTER TABLE llm_keys ADD COLUMN IF NOT EXISTS position INTEGER;
UPDATE llm_keys SET position = id WHERE position IS NULL;
