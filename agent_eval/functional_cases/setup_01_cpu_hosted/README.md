# Functional test suite — SETUP user task #1 (CPU-only, hosted via build.nvidia.com)

This is the first suite in a planned set of agent-driven functional tests for the
**NeMo Retriever Library skill**, built directly against the real CLI in
`nemo_retriever/nemo_retriever/src` (`retriever ingest` / `retriever query`).

Each test is a self-contained triple: **a prompt** an agent receives, **a set of data
files** mounted alongside it, and **an expected output** — including the *correct
`retriever` subcommand(s)* (`ingest`, `query`, or both) and the flags appropriate for
this task. The five tests climb a single complexity ladder so we can see exactly where an
agent starts to drift.

---

## The user task under test

> **JTBD: SETUP — row 1.** "NeMo Retriever library setup on local **CPU-only** machine:
> no GPU detected (automatic detection) — configuration via **build.nvidia.com** (requires
> only an API key). Embedding, reranking, extraction reachable." — **P0**

**Success criteria for the row** (from the JTBD tab + the seed-queries tab):

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) clean install, (2) ingest a PDF hitting hosted endpoints, (3) a retrieval query returns grounded hits |
| Time | end-to-end (clean state → first successful retrieval) **≤ 10 min** |
| Trigger rate | ≥ 95% — a "set me up and load my docs" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — must run `retriever ingest` then `retriever query` with **CPU/hosted** flags (API key + hosted endpoints), **not** local-GPU flags |
| Token usage | tracked, not gated |

The seed queries this suite is derived from:
- *"I'm on a Mac with no GPU — set me up so I can run document Q&A using NVIDIA's hosted models."*
- *"Get document retrieval working with just my NVIDIA API key, no local GPU."*
- *"Configure my dev environment to call NVIDIA's hosted endpoints — I don't want to run any models locally."*

---

## How the CLI behaves on a CPU-only host (verified, not assumed)

Grounded by reading `src/nemo_retriever/cli/main.py` + `sdk_workflow.py` and running
`retriever ingest <pdf> --dry-run` (offline, no network):

- **No `--input-type` flag exists** — format is auto-detected from the file/dir. Both
  commands are just `retriever ingest <paths…>` and `retriever query "<text>"`.
- **Extraction NIMs** (page-elements, OCR v2, table-structure) default their `invoke_url`
  to the hosted `ai.api.nvidia.com` endpoints when unset; the **API key is read from
  `NVIDIA_API_KEY`** when `--api-key` is omitted (dry-run shows `api_key: <redacted>`).
- **Embedding** runs on the bundled **CPU HuggingFace** model by default. To force the
  *hosted* embedder (the "hosted models" intent of this row), pin
  `--embed-invoke-url https://integrate.api.nvidia.com/v1`
  `--embed-model-name nvidia/llama-nemotron-embed-1b-v2`. The dry-run then shows
  `embedding_endpoint = https://integrate.api.nvidia.com/v1` and **all `gpu_*` fields null**.
- **Reranking** is **off by default**; `--rerank` (no URL, CPU host) resolves to the hosted
  reranker `…/llama-nemotron-rerank-1b-v2/reranking`.
- **Defaults**: LanceDB at `lancedb`, table `nemo-retriever`, `--overwrite` on. `ingest`
  prints one line: `Ingested N file(s) → M row(s) in LanceDB lancedb/nemo-retriever.`
  `query` prints a JSON array of hits, each `{source, page_number, text}`.

Convention used in every command below: `RETRIEVER=/raid/nemo_retriever/.venv/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `setup-cpu-001` | **Baseline.** 1 trivial text PDF, default flags. Just close the ingest→query loop on a no-GPU box + prove the CLI is installed. | `--version`, `ingest`, `query` |
| 2 | `setup-cpu-002` | **Verify routing first.** Use `ingest --dry-run` to prove hosted endpoints + API key + no-GPU *before* the real run. | `ingest --dry-run`, `ingest`, `query` |
| 3 | `setup-cpu-003` | **Full hosted extraction stack.** Table-bearing PDF drives page-elements → OCR → table-structure (the "extraction reachable" clause); query a specific cell. | `ingest`, `query` |
| 4 | `setup-cpu-004` | **Folder + reranker.** 3-PDF folder in one `ingest`; `--rerank` exercises the hosted reranker (the "reranking reachable" clause) + best-match-first. | `ingest`, `query --rerank` |
| 5 | `setup-cpu-005` | **Acceptance gate.** Fresh **named** index (table name aligned across ingest *and* query), citation required, **no-GPU assertion**, end-to-end **≤ 10 min**. | `ingest`, `query` |

The ladder is deliberate: T1 proves the loop runs; T2 proves it runs *the right way*
(hosted, no GPU); T3 and T4 each light up one more required hosted model class
(extraction, then reranking) so "Embedding, reranking, extraction reachable" is covered
piece by piece; T5 composes everything into the row's actual operational-pass gate.

---

### T1 — `setup-cpu-001` · smoke loop  *(complexity 1)*
- **Satisfies:** SETUP operational-pass criteria 1–3 at their simplest.
- **Prompt:** "I'm on a Mac with no GPU. Set me up to run document Q&A using NVIDIA's
  hosted models, then confirm it works: load `data/t1_smoke/test.pdf` and tell me what the
  document says."
- **Data:** `data/t1_smoke/test.pdf` (1 page of placeholder text).
- **Expected:** `RETRIEVER --version` → version string · `RETRIEVER ingest
  data/t1_smoke/test.pdf` → `Ingested 1 file(s) → N row(s) in LanceDB
  lancedb/nemo-retriever.` · `RETRIEVER query "What does the document say?" --top-k 5` →
  JSON hits citing `test.pdf` p1.
- **Why it's the floor:** one file, trivial text, zero special flags. If this fails, the
  install or the CPU/hosted default path is broken.

### T2 — `setup-cpu-002` · prove hosted routing via dry-run  *(complexity 2)*
- **Satisfies:** validation-path bullets *"NVIDIA_API_KEY discovered"* and *"No GPU
  detected; no CUDA deps."*
- **Prompt:** "Before loading anything, prove my setup will use NVIDIA's hosted endpoints
  and my API key — not a local GPU. Then load `data/t2_hosted_plan/woods_frost.pdf` and
  tell me who owns the woods."
- **Data:** `data/t2_hosted_plan/woods_frost.pdf` (Robert Frost poem; owner's "house is in
  the village").
- **Expected:** `RETRIEVER ingest … --embed-invoke-url https://integrate.api.nvidia.com/v1
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --dry-run` → resolved plan with
  `embedding_endpoint = integrate.api.nvidia.com`, all `gpu_*` null, `api_key:
  <redacted>`, `vdb_kwargs {uri: lancedb, table_name: nemo-retriever}`. Then the same
  command **without** `--dry-run`, then `query "Who owns the woods?"`.
- **Adds:** a reasoning/verification step on the resolved plan before any billable call,
  plus explicit hosted-embed pinning + API-key passthrough.

### T3 — `setup-cpu-003` · hosted extraction stack  *(complexity 3)*
- **Satisfies:** the **"extraction reachable"** clause (page-elements → OCR v2 →
  table-structure on `ai.api.nvidia.com`).
- **Prompt:** "On my no-GPU box, load `data/t3_table/table_test.pdf` and tell me what value
  James had in 2019."
- **Data:** `data/t3_table/table_test.pdf` (Year × {Bill, Amy, James, Ted, Susan} grid).
- **Expected:** `RETRIEVER ingest data/t3_table/table_test.pdf` (defaults turn on tables +
  page-elements + OCR v2) · `RETRIEVER query "James value in 2019" --top-k 5
  --content-types text,table` → **978**, citing `table_test.pdf` p1.
- **Adds:** the multi-stage hosted extraction pipeline (not just text+embed) and a
  cell-level answer.

### T4 — `setup-cpu-004` · folder + hosted reranker  *(complexity 4)*
- **Satisfies:** the **"reranking reachable"** clause + multi-doc ingest.
- **Prompt:** "Set up on CPU with hosted models, load the whole `data/t4_folder/` folder,
  and find — best match first — which Robert Frost collection was published in 1923."
- **Data:** `data/t4_folder/` = `woods_frost.pdf` (collection table on p2 lists *New
  Hampshire — 1923*), `multimodal_test.pdf`, `table_test.pdf`.
- **Expected:** `RETRIEVER ingest data/t4_folder/` → `Ingested 3 file(s) → …` ·
  `RETRIEVER query "Robert Frost collection published in 1923" --rerank --top-k 5` →
  **New Hampshire (1923)**, citing `woods_frost.pdf` p2. `--rerank` resolves to the hosted
  reranker.
- **Adds:** one `ingest` over a whole folder (no per-file loop) **and** the third hosted
  model class via `--rerank`.

### T5 — `setup-cpu-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row — all four criteria + the validation
  path's no-GPU / hosted-only assertion + citations + the ≤ 10 min SLA.
- **Prompt:** "Do a clean end-to-end setup on this no-GPU machine using only my NVIDIA API
  key and hosted models: load `data/t5_acceptance/` into a fresh index called
  `setup_smoke`, then answer with a citation — in which poem does the speaker say they have
  'miles to go before I sleep'? Confirm no local GPU was used and that it finished in under
  10 minutes."
- **Data:** `data/t5_acceptance/` = `woods_frost.pdf`, `table_test.pdf`,
  `multimodal_test.pdf`.
- **Expected:** `RETRIEVER ingest data/t5_acceptance/ --lancedb-uri lancedb --table-name
  setup_smoke --embed-invoke-url https://integrate.api.nvidia.com/v1 --embed-model-name
  nvidia/llama-nemotron-embed-1b-v2` → `… LanceDB lancedb/setup_smoke.` · `RETRIEVER query
  "miles to go before I sleep" --table-name setup_smoke --rerank --top-k 5` → *"Stopping by
  Woods on a Snowy Evening" by Robert Frost*, citing `woods_frost.pdf` p1.
- **Adds (the trap):** a **custom `--table-name` that must match on both commands** (a
  common agent error is querying the default `nemo-retriever` after ingesting elsewhere),
  the no-GPU/hosted-only assertion, and the end-to-end timing gate.

---

## Running / grading

These cases are written to drop into the existing `agent_eval` harness style: mount each
test's `data/` folder into the agent workdir, give it the `prompt`, and grade against
`pass_when` in `cases.json`. T1–T4 grade as operational pass/fail (right subcommands +
correct grounded answer); T5 additionally checks the table-name alignment, citation
presence, no-GPU trace, and wall clock.

**Note on live runs:** every real (non-dry-run) command here calls **billable
build.nvidia.com hosted endpoints** with `NVIDIA_API_KEY`. The corpora are tiny (1–3 small
PDFs) so cost is negligible, but the suite has **not** been executed live yet — the
expected outputs are grounded in the CLI source and the offline `--dry-run` plan. Run live
to capture real row counts, latencies, and token baselines.
