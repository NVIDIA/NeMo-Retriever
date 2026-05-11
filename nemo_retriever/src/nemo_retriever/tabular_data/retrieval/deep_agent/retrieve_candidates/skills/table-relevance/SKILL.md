---
name: table-relevance
description: How to filter the accumulated tables down to those genuinely needed to answer the question, applying business-context (acronyms, custom_prompts) when judging domain match.
---

# Table Relevance Filtering

After grounding is done, the accumulated tables include some retrieved
purely because their column names happened to match a search term. The
relevance filter prunes those out.

## Inputs

- `store.question` — the user's question.
- `store.accumulated_tables` — tables retrieved by `retrieve_for_entity`,
  each with `id`, `name`, optional `description`, and a `columns` list.
- `store.acronyms` and `store.custom_prompts` — caller-supplied business
  context that disambiguates the question's intent.

## Decision rules

1. Keep ONLY tables whose subject domain genuinely matches the question.
   Drop tables whose domain does not match, even if one of their columns
   matched a search term.
2. Apply the acronyms and custom-prompts rules when judging domain. If
   a domain rule says a generic term routes to a specific table, prefer
   that table over near-name matches in other domains.
3. Identify tables by their unique `id` (not name). The same table name
   can appear in different schemas, so always reference tables by `id`.
4. Be conservative: if a table is plausibly part of the join chain to
   the answer, keep it. Prefer false-positives over dropping a table the
   SQL author actually needs.

## Output discipline

Return only `id` values from the candidate list. Use the exact `id`
strings shown in the input. Never invent ids.
