-- v2.3: what a meeting is for (topic), how it happens (mode: Google Meet, WebEx, phone call, in person…) and the contact's
-- name kept on the item, so the agenda survives deleting chats (logging out of WhatsApp used to take it along).
ALTER TABLE agenda_items ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE agenda_items ADD COLUMN IF NOT EXISTS mode TEXT;
ALTER TABLE agenda_items ADD COLUMN IF NOT EXISTS contact_name TEXT;
UPDATE agenda_items a SET contact_name = c.name FROM contacts c WHERE c.chat_id = a.chat_id AND a.contact_name IS NULL;
