# Knowledge Base Guide (for the owner)

Files whose names start with `_` (like this one) are ignored by the indexer.

## How retrieval works
The bot uses a **vectorless** retriever: each `##` (and `###`) section of every `NN_slug.md` file becomes one searchable unit, scored with BM25F. Matches in the heading and frontmatter `tags`/`aliases` count more than matches in the body. There are no embeddings and nothing to re-index. **Saving a file hot-reloads the index in milliseconds.**

## File format
```
---
title: Work Experience
tags: work, job, experience, company          <- English keywords, one line, comma-separated
aliases: naukri, kaam, नौकरी, exat             <- Hinglish, Devanagari, typos, abbreviations
---

# Work Experience

One to three sentences summarizing the whole file.

## Exxat — Product Analyst (March 2026 – Present)
Self-contained facts, written in first person ("I work at…", "my project…").
Name the company, project and dates in every section.
```

## Rules for good answers
1. **Write in first person.** eSumit speaks as Sumit, and these notes are given to it as its "My facts" block, so write "I built NeoAI", not "Sumit built NeoAI". Keep "Sumit"/"Sumit Soni" in the frontmatter `tags`/`aliases` (and in FAQ headings like `## Q: Where do I work? (Where does Sumit work?)`) so questions that use the name still match.
2. **One topic per `##` section**, roughly 40–250 words. Each section is retrieved on its own, so don't write "as above".
3. **Keyword-rich headings.** Write `## Faahhh Terminal Sound — VS Code Extension`, not `## Extension`.
4. **Only true facts.** The bot is told to answer strictly from these files.
5. Add Hinglish/Hindi words people actually type to `aliases`, and add query-expansion words to `synonyms.json` (`"term": ["english", "expansions"]`, with lowercase keys).
6. For common questions, add a `## Q: ...` entry to `10_faq.md`. FAQ entries are the fastest and most accurate retrieval path.
7. Never put private data (phone, personal email, address) here. Everything in these files can be sent to anyone who messages the bot.

