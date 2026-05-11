---
name: entity-grounding
description: How to ground each entity to a database artifact via retrieve_for_entity, and how to fall back to synthesize_expression when no direct match exists.
---

# Entity Grounding

For each entity produced by the decomposer, ground it to a database
artifact using `retrieve_for_entity`. For entities the vector search
cannot cover, derive a SQL expression with `synthesize_expression`.

## Iteration discipline

- Call `retrieve_for_entity` ONCE for every entity in the store. Never
  skip one.
- Pass `entity_term` and `entity_type` only — accumulated state lives in
  the store; the tool reads/writes it automatically.
- The tool reports `COVERED` (custom analyses or relevant tables found)
  or `NOT COVERED`.

## Synthesis fallback

For each entity reported as `NOT COVERED`:

1. Call `synthesize_expression(entity_term)`.
2. The tool reads accumulated columns from the store; it does not need a
   column list argument.
3. Synthesis can produce a SQL expression that combines existing columns
   (e.g. `income - cost` for "profit"). Use ONLY columns the tool already
   has.
4. If synthesis fails, the entity is marked `unresolved` and processing
   continues.

## What NOT to do

- Never invent column names — `synthesize_expression` consumes only
  retrieved columns.
- Never write SQL queries here. The job is grounding only — Phase 2
  generates SQL from the produced `RetrievalContext`.
- Never call retrieval / synthesis twice for the same entity.
- Never stop after the first entity — every entity must be processed.
