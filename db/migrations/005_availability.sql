-- v2.4: Availability. Free slots are derived live (working hours minus confirmed meetings minus the blocks below),
-- so they can never drift out of sync with the agenda. `time_blocks` holds the periods the owner marked as busy
-- (leave, lunch, travel…) that are not meetings.

CREATE TABLE IF NOT EXISTS time_blocks (
  id          BIGSERIAL PRIMARY KEY,
  starts_at   TIMESTAMPTZ NOT NULL,
  ends_at     TIMESTAMPTZ NOT NULL,
  label       TEXT NOT NULL DEFAULT 'Busy',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (ends_at > starts_at)
);
CREATE INDEX IF NOT EXISTS time_blocks_range ON time_blocks (starts_at, ends_at);

-- Looking up the meetings around a time is the hot query of the clash check.
CREATE INDEX IF NOT EXISTS agenda_items_due_status ON agenda_items (due_at, status) WHERE kind = 'appointment';
