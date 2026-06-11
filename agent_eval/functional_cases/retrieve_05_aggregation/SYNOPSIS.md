# Synopsis — RETRIEVE: aggregation (count / sum / list) across the retrieved corpus

**What user task this covers.** A user has loaded a pile of documents into NeMo Retriever and
wants to ask *aggregate* questions about the whole corpus — "**how many** documents are in my
knowledge base?", "**list** the distinct sources", "**sum** this metric across all the
groups". Success here has a sharp twist: the retriever must be **triggered** to build and
serve the index, but the actual count/sum/list must be **computed by the agent over the real
rows** — not a number the model *guesses* from a summary. A confident "looks like about three
documents" **fails** even if it happens to be right; the agent has to run an explicit
aggregation operator over the data and land on the exact answer.

**How we test it.** Five agent prompts, each a real aggregation question over a small real
corpus. We check the agent (1) fires the skill and drives `retriever ingest` to build the
LanceDB index (NRL triggered), and (2) computes the aggregate **deterministically over rows**
— using the skill's own canonical one-liners from `references/query.md`: count distinct
`source_name`, list distinct sources, `Counter` chunks per source, or `sum()` a table
column's cells. The grader looks for the aggregation **operator in the trace** (not an
`ls`/`find`/`wc` shortcut, not an LLM guess) **and** an exact match to ground truth.

**The five tests, simplest to hardest:**

1. **Count documents** — "how many documents are in my knowledge base?" Ingest a 3-PDF
   folder, count distinct sources over the index → **3**. The aggregation floor.
2. **List distinct sources** — same corpus, but now produce the exact filenames
   (`multimodal_test.pdf`, `table_test.pdf`, `woods_frost.pdf`), not just the count.
3. **Count chunks per source** — a group-by: how many chunks each document contributed
   (`Counter` over rows). Graded on shape — 3 keys whose counts sum to the total.
4. **Sum a numeric column** — from `table_test.pdf`, sum **Ted's** entire column across all 20
   years → **14806** (exact, agent-computed). Ted is used because its column has no `N/A`
   cells, so the total is unambiguous.
5. **Acceptance gate** — end to end: build the index, then answer *both* a distinct-document
   **count (3)** and the **Ted-column sum (14806)**, each computed over rows with the operator
   visible. The test the others build up to.

**Why this order.** Each rung adds exactly one new thing: first "can we count over the corpus
at all," then list-distinct, then grouping (per-source counts), then a numeric sum over
extracted table cells, then both a count and a sum composed end to end. Throughout, the
constant is the operational rule: **retriever triggered + aggregate computed over rows
(operator in the trace) + exact match** — never hallucinated from a summary.

**Ground truth (computed from the real fixtures).** 3-PDF corpus → `branch_summary pdf:3`,
**3 distinct documents**. `table_test.pdf` → 20 year rows; **Ted column sum = 14806** (full
column, no N/A); cross-checks James 2019 = 978, Susan 2023 = 970 both confirmed.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run` confirmed `pdf:3` /
`pdf:1`) and the skill's `references/query.md` corpus-aggregate one-liners; **not yet run
live**. The exact-match targets (count = 3, distinct-source list, Ted sum = 14806) are fixed
ground truth independent of a live run; only T3's per-source chunk counts depend on the live
chunking profile (graded on group-by shape). See `README.md` for the full spec and
`cases.json` for the machine-gradable definitions.
