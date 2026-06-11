# Functional test suite — RETRIEVE: single-source Q&A (answer from ONE named document)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **RETRIEVE** job in its **single-source** form: the user names **one**
document and wants a direct answer from it through the platform's RAG runtime. The defining
challenge is **scoping** — the answer must come from the named document and nothing else,
even when other documents are present.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Single-source Q&A: direct retrieval and answer from one
> document through the platform's RAG runtime." Time budget: **≤ 1 min**.

**This is an OPERATIONAL-pass row, not a RAGAS/graded row.** Per `CONVENTIONS.md`, functional
RETRIEVE tests use operational pass (correct grounded answer + right subcommand + right
scope), *not* the ≥80%/RAGAS≥0.75 judge gate — that lives in the separate performance-eval
suites.

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: **(1)** retrieval is **scoped to the ONE named document** — every hit used for the answer has that document's `source_name`, **no cross-doc bleed**; **(2)** the answer **matches the ground-truth value** (numeric/span) for the named doc. Scoping uses a **real CLI mechanism**, never a non-existent query-time `--where`. |
| Time | **RETRIEVE ≤ 1 min** per case (one small doc, one ingest + one scoped query; `hf` backends for a fast cold start). |
| Trigger rate | ≥ 95% — an "answer X from `<named doc>`" prompt must fire the skill. |
| Subcommand accuracy | ≥ 90% — `retriever ingest <paths>` (with `--table-name <doc>` when isolating one doc) then `retriever query … --table-name <doc>`, **or** the LanceDB `source_name` python filter on a shared table. **No `retriever query --where` flag** (it does not exist). |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"What's `<company>` saying about R&D in their 2022 10-K?"*
- *"Pull up the revenue numbers from `<doc>.pdf`."*
- *"What does `<doc>.docx` recommend for X, and what's the total amount?"*

---

## The single-source scoping mechanism (grounded in the CLI, not guessed)

The spec's validation references `vdb_kwargs={"where": "pdf_basename = '...'"}`. **That is a
library / python-API construct — it is NOT a `retriever query` CLI flag.** Verified:

- `retriever query --help` exposes **no** `--where`, `--source`, or `--pdf-basename`
  option; the only scoping-related flag is **`--table-name`** (and `--lancedb-uri`).
  (`retriever query --help | grep -c -- --where` → **0**.)
- A grep of `src/nemo_retriever` finds **no** `vdb_kwargs`, `pdf_basename`, or query-time
  `where` in the CLI path.

So single-source scoping on the shipped CLI is achieved by **one of two real mechanisms**:

**(a) Dedicated per-doc table.** Ingest **only** the named doc into its own table and query
that table — only that doc's rows can ever be returned:
```bash
$RETRIEVER ingest data/<doc> --table-name <doc> --overwrite
$RETRIEVER query "<question>" --table-name <doc> --top-k 5 \
  --query-embed-backend hf --reranker-backend hf --rerank
```

**(b) Shared table + `source_name` python filter.** Ingest the whole corpus into one table,
then filter LanceDB by `source_name` (the one-liner pattern from
`skills/nemo-retriever/references/query.md`):
```bash
$RETRIEVER_VENV/bin/python -c "import lancedb,json; \
df=lancedb.connect('./lancedb').open_table('nemo-retriever').to_pandas(); \
sub=[r for _,r in df.iterrows() if json.loads(r['source'])['source_name']=='<doc>']; \
print(len(sub),'of',len(df)); print('\n'.join(r['text'] for r in sub))"
```
Each hit's `source` is a JSON blob whose `source_name` is the document's basename; this is
the CLI-honest stand-in for the spec's `where pdf_basename = '...'` predicate.

---

## How the CLI behaves for this task (grounded in `--dry-run` + references)

- **Two subcommands only:** `retriever ingest <paths…>` and `retriever query "<text>"`.
  Format is **auto-detected** from the extension — **no `--input-type` flag**.
- **`--table-name`** (default `nemo-retriever`) names the LanceDB table on both ingest and
  query; the query table **must match** the ingest table. `--overwrite` (vs `--append`)
  resets a table on ingest.
- **`--use-table-structure`** (off by default) materializes `nemotron-table-structure-v1`
  row/col cells so numeric table cells (James 2019 = 978; Susan 2023 = 970) are queryable as
  `content_type=table` rows; used by the table-cell rungs (2, 4) with
  `--content-types text,table`.
- **DOCX routes through the pdf branch via libreoffice.** A single `.docx` `--dry-run`
  reports `branch_summary: pdf:1` — libreoffice converts it to PDF and the same pdf pipeline
  runs (T5 prereq: `sudo apt-get install -y libreoffice`).
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. `source` basename = doc_id; `source_name` inside the
  stored `source` blob is what the python filter matches.
- **`hf` backends** (`--query-embed-backend hf --reranker-backend hf --rerank`) give a fast
  single-query cold start (~20-30s), keeping each case under the ≤1-min RETRIEVE budget.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Scoping mechanism |
|---|---|---|---|
| 1 | `retrieve-ss-001` | **Floor.** Corpus has exactly ONE doc; query it. Single-source is trivially guaranteed (nothing else to bleed in). Establishes the ingest→query→grounded-answer loop. | none needed (one-doc table) |
| 2 | `retrieve-ss-002` | **Multi-doc on disk, ONE named doc.** Several PDFs present, user names one → ingest ONLY that doc into its own `--table-name` table and query it. | dedicated per-doc table |
| 3 | `retrieve-ss-003` | **Predicate verified.** All docs in ONE shared table; agent must PROVE the scope via the `source_name` predicate (distinct sources + scoped-row count) before answering. | `source_name` python filter |
| 4 | `retrieve-ss-004` | **Exact number under a distractor.** Two table-bearing docs in the workdir; scope to the named doc and return the EXACT cell — a sloppy unscoped query would grab the wrong table. | dedicated per-doc table |
| 5 | `retrieve-ss-005` | **Acceptance gate.** A named `.docx` (libreoffice→pdf) inside a genuinely multi-doc index; answer BOTH a span AND a number from it, with NO cross-doc bleed. | dedicated per-doc table (or `source_name` filter) |

The ladder adds exactly one dimension per rung: T1 single-doc index (trivial scope) → T2
multi-doc on disk, scope by dedicated table → T3 shared index, *prove* the scoping predicate
fired → T4 tighten to an exact number with a same-shaped distractor present → T5 compose:
non-PDF named source in a multi-doc index, span + number, no bleed.

---

### T1 — `retrieve-ss-001` · trivially single-source  *(complexity 1)*
- **Satisfies:** RETRIEVE single-source operational floor.
- **Data:** `cases/retrieve-ss-001/data/woods_frost.pdf` (only doc in the index).
- **Expected:** `RETRIEVER ingest data/woods_frost.pdf --overwrite` → non-zero rows; scoped
  query → **answer: the owner's house is "in the village"** (woods_frost.pdf p1). Scope is
  trivially satisfied because the table holds one doc.

### T2 — `retrieve-ss-002` · multi-doc, scope via a dedicated table  *(complexity 2)*
- **Satisfies:** "multi-doc index but answer must come from ONE named doc."
- **Data:** `table_test.pdf`, `woods_frost.pdf`, `multimodal_test.pdf` in `data/`; user names
  `table_test.pdf`.
- **Adds:** the SCOPING move — ingest **only** `table_test.pdf` into `--table-name table_test`
  and query that table. **Ground truth: James 2019 = 978**, cite `table_test.pdf` p1. The
  other two PDFs are physically absent from the table, so they cannot bleed in.

### T3 — `retrieve-ss-003` · scoping predicate verified  *(complexity 3)*
- **Satisfies:** "the scoping predicate/mechanism fired — only that doc's rows considered."
- **Data:** all three PDFs ingested into ONE shared `nemo-retriever` table.
- **Adds:** the agent must **prove** the scope — use the `source_name` python one-liner to
  report distinct sources (`['multimodal_test.pdf','table_test.pdf','woods_frost.pdf']`) and
  the scoped-row count, then answer from only the `multimodal_test.pdf` rows. **Ground truth:
  Table 1 Giraffe → Driving a car**, cite `multimodal_test.pdf` p1. This is the CLI-honest
  stand-in for the spec's `where pdf_basename = '...'` predicate.

### T4 — `retrieve-ss-004` · exact number under a distractor  *(complexity 4)*
- **Satisfies:** "answer matches the ground-truth value (numeric) for the named doc."
- **Data:** `table_test.pdf` **and** `multimodal_test.pdf` (both contain numeric tables).
- **Adds:** an exact-numeric bar with a same-shaped distractor in the workdir. Scope to
  `table_test.pdf` (dedicated table) + `--use-table-structure` + `--content-types text,table`.
  **Ground truth: Susan 2023 = 970**, cite `table_test.pdf` p1. A sloppy unscoped query could
  surface the multimodal table's numbers — the dedicated table prevents that by construction.

### T5 — `retrieve-ss-005` · acceptance gate (named .docx in a multi-doc index)  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the single-source RETRIEVE row.
- **Data:** `procurement_memo_q4.docx` + `woods_frost.pdf` + `table_test.pdf`.
- **Expected:** scope to `procurement_memo_q4.docx` (dedicated `--table-name`
  `procurement_memo_q4`, libreoffice→pdf branch), then one scoped query returning BOTH:
  - **span:** recommends procuring **512 additional H200 GPUs** (320 us-east-2, 192 eu-central-1);
  - **number:** total capital outlay **$14.8M** (payback 7 months).
- **Adds (the gate):** a non-PDF named source inside a real multi-doc index; answer must carry
  both a span and a number and exhibit **no cross-doc bleed** (no woods_frost/table_test
  content). Prereq: libreoffice.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** retrieval is **scoped** to the named document via a **real CLI mechanism** —
a dedicated `--table-name` table, or the LanceDB `source_name` python filter — **never** a
query-time `--where` flag (it does not exist); **(b)** **no cross-doc bleed** — every hit used
for the answer carries the named doc's `source_name`; **(c)** the answer matches the grounded
value (in the village; 978; Giraffe→Driving a car; 970; 512 H200 GPUs / $14.8M) with the
right citation; **(d)** for the `.docx` rung the file is auto-detected and routed via
libreoffice→pdf (no `--input-type`).

**Note on live runs (not run live).** Expected outputs are grounded in the CLI **source +
`--dry-run`** and the skill's `references/query.md`; the suite has **not** been run live yet.
A live run needs a reachable embedding/reranker backend (hosted `integrate.api.nvidia.com`
with `NVIDIA_API_KEY`, or a local GPU per the SETUP suites) and, for T5, the **libreoffice**
host package. Running live would capture the real per-table row counts (per doc), the scoped-
vs-total row split that proves the `source_name` predicate, query latencies (against the
≤1-min budget), and token baselines.

**Single-source-scoping mechanism note.** The shipped `retriever query` CLI has **no**
`--where` / source-filter flag (verified: `--help` shows only `--table-name`; no
`vdb_kwargs`/`pdf_basename` in the CLI source). The spec's
`vdb_kwargs={"where": "pdf_basename = '...'"}` is a **library/python-API** construct. This
suite scopes to a single named document using the two **real** mechanisms the CLI offers:
**(a)** a dedicated per-doc table (`retriever ingest <doc> --table-name <doc> --overwrite`,
then `retriever query … --table-name <doc>`), or **(b)** the LanceDB `source_name` python
filter from `references/query.md` over a shared table. We do **not** invent a `--where` flag.
