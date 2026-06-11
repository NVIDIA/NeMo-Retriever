# Functional test suite — DX user task #1 (end-to-end pipeline review)

An agent-driven functional test suite for the **NeMo Retriever Library skill**, built
against the real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` /
`retriever query`).

Unlike the **SETUP** suites (`setup_01_cpu_hosted`, `setup_02_local_gpu`), which test *one
host configuration* of *one* stage, this is a **Developer-Experience (DX) "review"** of the
**whole happy-path chain wired together**: **install → setup → ingest → query → cite**. The
question it answers is: *on a clean machine, does every stage succeed with no broken steps,
no stale model names, and a correct, timely, cited answer?* Each test is a self-contained
triple — a prompt, a per-case `data/` folder, and an expected output naming the correct
`retriever` subcommand(s).

---

## The user task under test

> **JTBD: DEVELOPER EXPERIENCE (DX).** "End-to-end review (install → setup → ingest → query
> → cite) succeeds on the platform with **no broken steps, no stale model names**, and a
> **correct, timely answer** [with a resolvable citation]." — **P0**

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: **every** stage of the chain succeeds with no manual repair — install → setup → ingest → query → citation surface; no broken steps; **no stale model names**; **no 404** on any referenced endpoint; final query returns a **correct** answer with a **resolvable** `(source, page_number)` citation |
| Time | **slow — full pipeline ≤ 10 min** on a small corpus; per-stage rungs (1–4) ≤ 2 min each |
| Trigger rate | ≥ 95% — a "walk me through the whole pipeline / run end-to-end on my docs" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — the happy-path chain `retriever --version` → `retriever ingest <corpus>/` → `retriever query "<q>" --top-k 5`; no `--input-type` flag (does not exist); model names current |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased in the prompts):
- *"Walk me through getting set up on a clean machine — install, load some docs, ask a question, surface a citation."*
- *"From scratch: load these PDFs, ask a question, get an answer with a source pointer. No skipped steps."*
- *"Run the whole pipeline on my docs and show me each stage."*

---

## How the CLI behaves for this task (grounded, not guessed)

Grounded by `skills/nemo-retriever/references/install.md`, the CLI source
(`cli/main.py`), and an offline `retriever ingest data/ --dry-run` run against the suite's
own fixtures:

- **Install (stage 1):** bootstrap per `references/install.md` — the recipe auto-detects a
  GPU + CUDA 13 and installs the `[local]` cu130 torch flavor, else a base install that
  uses hosted endpoints. `retriever --version` then prints a **current dev-stamped**
  version (e.g. `2026.06.10.devXXXX`) — *not* a stale placeholder. This suite is
  **host-agnostic**: the assertion is that each stage is wired and current, not *where* it
  runs.
- **No `--input-type` flag exists** — format is auto-detected from the extension. The two
  subcommands are just `retriever ingest <paths…>` and `retriever query "<text>"`.
- **Ingest (stage 2):** `retriever ingest data/` over the small PDF corpus. The dry-run
  confirms `branch_summary pdf:N`, `method pdfium`, `ocr_version v2`, `use_page_elements
  true`, and `extract_text/tables/charts/images` all true by default. The extraction
  pipeline uses the **current** names **nemotron-page-elements-v3 → nemotron-ocr-v2 →
  nemotron-table-structure-v1** (the last when `--use-table-structure` is set). Success
  line: `Ingested N file(s) → M row(s) in LanceDB lancedb/nemo-retriever.`
- **Query (stage 3) + citation (stage 4):** `retriever query "<q>" --top-k 5` prints a JSON
  array of hits, each exactly `{source, page_number, text}` with a **1-indexed** integer
  `page_number`. The **citation** is the `(source, page_number)` pair of the grounding hit;
  it must be non-null and **resolve** to the correct page of the correct file. Pinning
  `--embed-model-name nvidia/llama-nemotron-embed-1b-v2` in the dry-run surfaces
  `embed_model_name` + `embedding_endpoint` in the resolved plan — used here to confirm the
  chain references the **current** embedder with **no 404**.
- **Stale-name / 404 invariant (the distinct focus):** any legacy/unversioned extraction
  name, a non-`llama-nemotron` embedder, or a 404 on a referenced endpoint is a **fail** for
  this DX review even if an answer is somehow produced.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `dx-e2e-001` | **Floor — stage 1.** Install completes; `retriever --version` prints a **current** (non-stale) version. CLI entry point wired. | install, `--version` |
| 2 | `dx-e2e-002` | **+ stage 2 (ingest).** `ingest data/` over a 2-PDF corpus → non-zero rows; extraction stage wired with **current** names; no broken step. | `ingest` |
| 3 | `dx-e2e-003` | **+ stage 3 (query).** `query` returns a **correct grounded answer**; current embedder; no broken query/embedding step. | `ingest`, `query` |
| 4 | `dx-e2e-004` | **+ stage 4 (citation).** Answer's `(source, page_number)` **surfaces and resolves** to the right page; forces the table-structure path. | `ingest --use-table-structure`, `query --content-types text,table` |
| 5 | `dx-e2e-005` | **Acceptance gate.** ALL four stages in one end-to-end pass: no broken step, **no stale name**, **no 404**, correct cited answer, **≤ 10 min**. | install, `--version`, `ingest`, `query` |

The ladder adds exactly one stage of the chain per rung, then composes them: T1 proves the
install/CLI entry point; T2 lights up ingest; T3 lights up query; T4 lights up the
citation; T5 runs the whole pipeline as the row's real operational-pass gate. (Distinct
from the SETUP suites: those test *where* one stage runs; this tests that *every* stage is
wired correctly and uses current model names.)

---

### T1 — `dx-e2e-001` · version / install sanity  *(complexity 1)*
- **Satisfies:** the DX-review floor — stage 1 (install → CLI present, current version).
- **Data:** none (install + version only).
- **Expected:** install per `references/install.md` completes (prints `RETRIEVER_VENV=…`) →
  `RETRIEVER --version` → a current dev-stamped version (e.g. `2026.06.10.devXXXX`), **not**
  stale. No 404 during install.
- **Why it's the floor:** if the install or CLI entry point is broken or the version is
  stale, nothing downstream can be trusted.

### T2 — `dx-e2e-002` · ingest stage succeeds  *(complexity 2)*
- **Satisfies:** stage 2 — `retriever ingest <corpus>/` over a small 2-PDF corpus, non-zero
  rows, no broken extraction step.
- **Data:** `data/` = `woods_frost.pdf`, `table_test.pdf`.
- **Expected:** `RETRIEVER ingest data/ --dry-run` grounds the plan (`branch_summary
  pdf:2`, `method pdfium`, `ocr_version v2`, `use_page_elements true`) → `RETRIEVER ingest
  data/` → `Ingested 2 file(s) → N row(s) …` (N non-zero). Extraction references current
  names (`nemotron-page-elements-v3` → `nemotron-ocr-v2`).
- **Adds:** the ingest/extraction stage and the current-extraction-name check.

### T3 — `dx-e2e-003` · query stage returns a grounded answer  *(complexity 3)*
- **Satisfies:** stage 3 — `query` returns a **correct** grounded answer.
- **Data:** `data/woods_frost.pdf` (p1 contains "miles to go before I sleep").
- **Expected:** `RETRIEVER ingest data/woods_frost.pdf` → `RETRIEVER query "miles to go
  before I sleep" --top-k 5` → *"Stopping by Woods on a Snowy Evening"* by Robert Frost,
  citing `woods_frost.pdf` p1. Embedder name current (`nvidia/llama-nemotron-embed-1b-v2`).
- **Adds:** the query/embedding stage and a correctness check on the answer.

### T4 — `dx-e2e-004` · citation surfaces and resolves  *(complexity 4)*
- **Satisfies:** stage 4 — the `(source, page_number)` citation **surfaces and resolves**.
- **Data:** `data/table_test.pdf` (James 2019 = 978).
- **Expected:** `RETRIEVER ingest data/table_test.pdf --use-table-structure` → `RETRIEVER
  query "James value in 2019" --top-k 5 --content-types text,table` → **978**, with a
  grounding hit carrying `(source=table_test.pdf, page_number=1)` that resolves to p1.
- **Adds:** the resolvable-citation requirement (non-null source + page that points to the
  right page) and the table-structure path; catches a missing/null/wrong citation.

### T5 — `dx-e2e-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass DX row — all four stages, no broken step, no
  stale name, no 404, correct cited answer, ≤ 10 min.
- **Data:** `data/` = `woods_frost.pdf`, `table_test.pdf`, `multimodal_test.pdf`.
- **Expected:** install → `RETRIEVER --version` (current) → `RETRIEVER ingest data/`
  (`Ingested 3 file(s) → N row(s) …`, `branch_summary pdf:3`) → `RETRIEVER query "miles to
  go before I sleep" --top-k 5` → *"Stopping by Woods on a Snowy Evening"*, citing
  `woods_frost.pdf` p1.
- **Adds (the gate):** the whole pipeline in one pass with simultaneous assertions — every
  stage succeeds, all model names current (embedder `nvidia/llama-nemotron-embed-1b-v2`,
  reranker `llama-nemotron-rerank-1b-v2`, extraction `nemotron-page-elements-v3` →
  `nemotron-ocr-v2` → `nemotron-table-structure-v1`), no 404, resolvable citation, ≤ 10 min.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The checks unique to this suite vs. the SETUP suites:
**(a)** the full chain runs with **no broken step** (no manual repair between stages);
**(b)** every referenced model name is **current** (no stale names — verify against the
embedder/reranker/extraction names above); **(c)** **no 404** on any referenced endpoint;
**(d)** the final answer is correct and carries a **resolvable** `(source, page_number)`
citation; **(e)** the full pipeline completes in **≤ 10 min**.

**Note on live runs:** this suite has **not been run live yet**. The expected outputs are
grounded in the CLI source, `references/install.md`, and an offline `retriever ingest data/
--dry-run` (which confirmed `branch_summary pdf:2`, `method pdfium`, `ocr_version v2`,
`use_page_elements true`, and the current `nvidia/llama-nemotron-embed-1b-v2` embedder in
the resolved plan). A live run may install GPU/CUDA-13 wheels and/or hit billable hosted
endpoints, so it is deferred; running it live would capture real row counts, end-to-end
latencies (the ≤ 10 min gate), the resolved version string, and token baselines, and would
confirm no referenced endpoint 404s.
