# QA Evaluation Pipeline

**This document is the canonical QA evaluation guide** for the harness. The top-level [harness README](../../../../README.md) (`tools/harness/README.md`) only summarizes and links here so we avoid duplicating steps and env vars in two places.

The evaluation framework lives in **`nemo_retriever.evaluation`** (install `nemo-retriever[eval]` from PyPI, or **`uv pip install -e "./nemo_retriever[eval]"` from this repo root** so `graph_pipeline` and local changes resolve). The harness `utils/qa/` package only retains `TopKRetriever` (which depends on harness-specific recall utilities). All other evaluation components -- types, scoring, generators, judges, orchestrator, operators -- are imported from `nemo_retriever.evaluation`.

Measures LLM answer quality over a RAG pipeline: retrieve context from a VDB, generate answers with one or more LLMs, and score each answer against ground-truth references using multi-tier scoring and an LLM-as-judge.

**Pluggable retrieval:** The eval harness does not care how you retrieved chunks -- only that you produce a JSON file that matches the **[retrieval JSON specification](#retrieval-json-format-interface-contract)** expected by `run_qa_eval.py` / `FileRetriever`. Vector search, hybrid, agentic pipelines, or any custom system can plug in as long as the file format and query strings align with your chosen ground-truth dataset.

**Default ground truth:** Standalone runs default to **`data/bo767_annotations.csv`** at the repo root -- the **bo767 annotations subset** maintained for this benchmark (multi-modality Q&A over the bo767 PDFs). Override with `QA_DATASET` / `QA_CSV` or another registered loader when comparing different corpora.

Designed to be **plug-and-play** -- swap retrievers, generators, or judges independently via Python Protocols without touching the orchestrator.

## Table of Contents

- [Pipeline File Map and Data Flow](#pipeline-file-map-and-data-flow)
- [Reproducing the bo767 Run](#reproducing-the-bo767-run)
- [Quick Start (Harness CLI)](#quick-start-harness-cli)
- [Retrieval JSON Format (Interface Contract)](#retrieval-json-format-interface-contract)
- [Custom Datasets (CSV Loader)](#custom-datasets-csv-loader)
- [Retrieval Benchmarking](#retrieval-benchmarking)
- [Architecture](#architecture)
  - [Operator Graph Chain (single-model)](#operator-graph-chain-single-model)
  - [QAEvalPipeline (multi-model sweeps)](#qaevalpipeline-multi-model-sweeps)
  - [When to Use Which](#when-to-use-which)
  - [Protocol Interfaces](#protocol-interfaces)
  - [Files](#files-nemo_retrieverevaluation)
  - [Graph Framework Integration](#graph-framework-integration)
  - [Entry Points](#entry-points)
- [Configuration](#configuration)
  - [Harness CLI Config](#harness-cli-config-test_configsyaml)
  - [Eval Config File (YAML / JSON)](#eval-config-file-yaml--json)
- [Scoring System (Three-Tier Hierarchy)](#scoring-system-three-tier-hierarchy)
- [Adding a New Component](#adding-a-new-component)
- [Output Format](#output-format)
- [Dataset Limitations](#dataset-limitations)
- [Standalone Runner (Docker / CI)](#standalone-runner)

## Pipeline File Map and Data Flow

End-to-end bo767 + LanceDB + full-page markdown touches these **artifacts** and **library code**:

| Stage | Artifacts produced | Code / APIs involved |
|-------|-------------------|----------------------|
| **1. Ingest + embed** | `lancedb/<uri>/<table>/` (embedded sub-page chunks); optionally `data/bo767_extracted/*.parquet` | `python -m nemo_retriever.examples.graph_pipeline` (extract, embed, VDB upload to LanceDB via the operator graph). Add `--save-intermediate` to also save extraction Parquet for full-page markdown (recommended for best results). **Table name must match** `export_retrieval_nemo.py` (`LANCEDB_TABLE`, default `nv-ingest`). |
| **2. Full-page markdown index** | `data/bo767_page_markdown.json` (`source_id` -> page -> markdown) | `build_page_markdown_index.py` -> `nemo_retriever.io.markdown.build_page_index()` (which calls `to_markdown_by_page` per document); numpy list columns are coerced so structured content is not dropped. |
| **3. Retrieval export** | `data/test_retrieval/bo767_retrieval_fullpage.json` (or sub-page JSON) | `export_retrieval_nemo.py` -> `nemo_retriever.export.export_retrieval_json()` queries LanceDB; if `PAGE_MARKDOWN_INDEX` is set, hits are expanded/deduped by `(source_id, page)` and replaced with full-page markdown strings. |
| **4. Ground truth** | `data/bo767_annotations.csv` (repo root) | Questions/answers for export and eval; must align with **query string normalization** in `FileRetriever` (see retrieval JSON rules). |
| **5. Evaluation** | `qa_results_*.json` | `run_qa_eval.py` or operator graph chain -> `nemo_retriever.evaluation`: `RetrievalLoaderOperator >> QAGenerationOperator >> JudgingOperator >> ScoringOperator`, or `QAEvalPipeline` for multi-model sweeps. |

**Data flow (conceptual):** PDFs -> (A) **chunked embeddings in LanceDB** for similarity search; (B) **Parquet** for full-page reconstruction. **Export** runs search on (A), then **replaces** hit chunks with pages from (B) via the index. **Eval** never talks to LanceDB -- it only reads the retrieval JSON + ground-truth CSV.

```
 NeMo Retriever (steps 1-3)                            Universal (steps 4-5)
 ──────────────────────────                             ─────────────────────
 Step 1                         Step 2      Step 3
 Ingest + Embed                 Index       Export           QA Eval
+-----------------------------+ +--------+  +----------+  +------------------+
| graph_pipeline              | | Parquet|  | LanceDB  |  | RetrievalLoader  |
|  --lancedb-uri lancedb      | | -> page|->| queries  |->| >> Generation    |
|  [--save-intermediate <dir>]| | md idx |  | + pages  |  | >> Judging       |
| (always: LanceDB output)   | +--------+  | -> JSON  |  | >> Scoring       |
| (optional: Parquet output)  |             +----------+  +------------------+
+-----------------------------+       |          |               |
  |               |            page_md.json  retrieval.json  qa_results.json
lancedb/    *.parquet (opt.)

 Bring Your Own Retrieval    +---------+--------+
 ─────────────────────────   | Any pipeline that |
 Skip steps 1-3 entirely.   | outputs retrieval |
 Produce a JSON matching     | JSON (see spec)  |
 the interface contract. --> +---------+--------+
```

Steps 1-3 are one reference implementation (NeMo Retriever + LanceDB).
Any retrieval system that produces a conforming JSON can replace them.
Steps 4-5 can be re-run with different LLM configs without repeating retrieval.

**Harness CLI alternative:** `cases/qa_eval.py` + `test_configs.yaml` can drive the same eval with different defaults; align `qa_dataset` and `file_path` with the standalone flow when comparing results.

## Reproducing the bo767 Run

Exact commands to reproduce the full-page markdown QA evaluation from scratch.

**Working directory:** Step 1 (`graph_pipeline.py`) runs from the **repo root**; steps 2-5 run from `tools/harness/`.

**Debug:** Lance index build can hit `No space left on device` when `/tmp` is a tiny tmpfs; set `export TMPDIR=/raid/$USER/tmp` and `mkdir -p "$TMPDIR"` before step 1. If `extraction.parquet` was written but LanceDB failed, retry with `python -m nemo_retriever.examples.parquet_to_lancedb`; otherwise re-run `graph_pipeline`.

<details>
<summary><strong>Quick reference -- full-page markdown (all commands)</strong></summary>

```bash
# ── repo root ──
cd /path/to/nv-ingest

# 1. Ingest + embed + save Parquet in one pass (~45-90 min)
python -m nemo_retriever.examples.graph_pipeline /path/to/bo767 \
  --lancedb-uri lancedb \
  --save-intermediate tools/harness/data/bo767_extracted

# ── tools/harness ──
cd tools/harness

# 2. Build page markdown index (~5-10 min)
python build_page_markdown_index.py

# 3. Export retrieval results (~5-15 min)
export PAGE_MARKDOWN_INDEX=data/bo767_page_markdown.json
export OUTPUT_FILE=data/test_retrieval/bo767_retrieval_fullpage.json
python export_retrieval_nemo.py

# 4. Run QA evaluation (~1-2 hrs)
export NVIDIA_API_KEY="nvapi-..."
export RETRIEVAL_FILE=data/test_retrieval/bo767_retrieval_fullpage.json
export OUTPUT_FILE=data/test_retrieval/qa_results_bo767_annotations.json
export QA_MAX_WORKERS=8
python run_qa_eval.py
```

</details>

<details>
<summary><strong>Quick reference -- sub-page chunks (skip full-page markdown)</strong></summary>

If you do not need full-page markdown context and want the simplest path,
skip the Parquet/index steps entirely. This uses raw sub-page chunks as
retrieval context, which may produce lower scores for structured content
(tables, charts, infographics).

```bash
# ── repo root ──
cd /path/to/nv-ingest

# 1. Ingest + embed into LanceDB
python -m nemo_retriever.examples.graph_pipeline /path/to/bo767 \
  --lancedb-uri lancedb

# ── tools/harness ──
cd tools/harness

# 2. Export retrieval (sub-page chunks, no PAGE_MARKDOWN_INDEX)
python export_retrieval_nemo.py

# 3. Run QA evaluation
export NVIDIA_API_KEY="nvapi-..."
export RETRIEVAL_FILE=data/test_retrieval/bo767_retrieval.json
python run_qa_eval.py
```

</details>

### Bring your own retrieval (skip steps 1-3)

Steps 1-3 below are the **NeMo Retriever + LanceDB** reference implementation
for ingestion, extraction, indexing, and retrieval. If your team already has a
retrieval pipeline (agentic, hybrid, BM25, or any custom system), **skip
steps 1-3 entirely** and produce a retrieval JSON file that conforms to the
[Retrieval JSON Format (Interface Contract)](#retrieval-json-format-interface-contract).
Then proceed directly to [Step 5: Run QA evaluation](#step-5-run-qa-evaluation).

The only requirement is that your JSON contains a top-level `queries` object
mapping each ground-truth question string to `{ "chunks": ["...", ...] }`.
See the [interface contract](#retrieval-json-format-interface-contract) for the
full schema, required fields, and a worked example.

**Alternative: retrieval-bench integration.** If you want to benchmark dense
retrieval (or any retrieval-bench pipeline) against the LanceDB baseline on the
same bo767 dataset, see [Retrieval Benchmarking](#retrieval-benchmarking) for
ready-made scripts and exact commands.

### Python environment

Steps 1-3 (ingest, build index, export) require the **`nemo_retriever`** library with LanceDB, CUDA, and Ray support. Steps 4-5 (QA eval) additionally require **`litellm`**. These are **not** part of the minimal harness install (`uv pip install -e .`).

**Recommended setup:** create an isolated Python 3.12 virtual environment and install the **local** `nemo_retriever` checkout in editable mode (required when working in this repo so `python -m nemo_retriever.examples.graph_pipeline` resolves):

```bash
uv venv qa-retriever --python 3.12
source qa-retriever/bin/activate
cd /path/to/nv-ingest   # repo root
uv pip install -e "./nemo_retriever[eval]"
```

The `[eval]` extra installs `litellm` for LLM generation and judging. If you are not using this tree, you can instead `uv pip install "nemo-retriever[eval]"` from PyPI (package name uses a hyphen).

For the full harness install (includes Milvus-lite, nemotron models, etc.), see **Installation** in the [harness README](../../../../README.md) and [`tools/harness/pyproject.toml`](../../pyproject.toml).

**Eval-only path:** if you already have a retrieval JSON and only need to run `run_qa_eval.py`, an environment with editable `nemo_retriever[eval]` and the harness package (`cd tools/harness && uv pip install -e .`) is sufficient.

### Prerequisites (data and keys)

```bash
# bo767 PDFs (767 files)
ls /path/to/bo767/*.pdf | wc -l   # should be 767

# Ground truth: data/bo767_annotations.csv (1007 Q&A pairs across all modalities)
# Located at the repo root: <repo>/data/bo767_annotations.csv
```

### Step 1: Ingest and embed PDFs (NeMo Retriever)

`graph_pipeline.py` builds an operator graph (`AbstractOperator` nodes connected via `>>`) and
executes it through either a `RayDataExecutor` (batch mode, default) or `InprocessExecutor`
(single-process pandas). The pipeline extracts, embeds, and uploads chunks to LanceDB in a
single pass.

Run from the **repo root**. **Estimated time: ~45-90 min** (767 PDFs, GPU-accelerated
extraction + embedding).

**Recommended (full-page markdown):** pass `--save-intermediate <dir>` to also write the
extraction DataFrame as Parquet. That preserves table/chart/infographic columns for step 2
to reconstruct full pages and generally yields better results on structured content.

```bash
python -m nemo_retriever.examples.graph_pipeline /path/to/bo767 \
  --lancedb-uri lancedb \
  --save-intermediate tools/harness/data/bo767_extracted
```

Output:
- `lancedb/nv-ingest/` (~84k chunks) -- used by step 3 for retrieval queries.
- `tools/harness/data/bo767_extracted/*.parquet` -- used by step 2 for full-page markdown.

**Minimal (skip Parquet / full-page path):** omit `--save-intermediate` if you only need
LanceDB and will skip step 2. Step 3 will use raw sub-page chunks instead of full-page
markdown.

```bash
python -m nemo_retriever.examples.graph_pipeline /path/to/bo767 \
  --lancedb-uri lancedb
```

Output:
- `lancedb/nv-ingest/` (~84k chunks) -- used by step 3 for retrieval queries.

**Note:** `graph_pipeline.py` uses `--run-mode batch` (Ray Data) by default. For local testing
without Ray, pass `--run-mode inprocess`. Both modes produce the same output.

### Step 2: Build page markdown index (NeMo Retriever)

> **Requires** `--save-intermediate` from step 1. If you skipped it, you can
> skip this step too -- step 3 will fall back to raw sub-page chunks.

Steps 2-5 are run from `tools/harness/`.

Groups Parquet records by (document, page number) and renders each page via
`nemo_retriever.io.markdown.to_markdown_by_page()`. Outputs a JSON index
mapping `source_id -> page_number -> markdown`.
**Estimated time: ~5-10 min** (CPU-only, reads Parquet and renders markdown).

```bash
python build_page_markdown_index.py
```

Output: `data/bo767_page_markdown.json` (~180 MB, ~6k pages across 767 docs).

### Step 3: Export retrieval results (NeMo Retriever)

Queries LanceDB for each ground-truth question via `nemo_retriever.export.export_retrieval_json()`,
then looks up the full-page markdown for each hit's page. Multiple sub-page hits
from the same page are deduplicated into a single full-page chunk.
**Estimated time: ~5-15 min** (1005 LanceDB queries + page index lookup).

```bash
export PAGE_MARKDOWN_INDEX=data/bo767_page_markdown.json
export OUTPUT_FILE=data/test_retrieval/bo767_retrieval_fullpage.json
python export_retrieval_nemo.py
```

| Env Var | Default | Purpose |
|---------|---------|---------|
| `LANCEDB_URI` | `./lancedb` | LanceDB directory |
| `LANCEDB_TABLE` | `nv-ingest` | LanceDB table name |
| `TOP_K` | `5` | Chunks per query |
| `EMBEDDER` | `nvidia/llama-nemotron-embed-1b-v2` | Embedding model |
| `QA_CSV` | `data/bo767_annotations.csv` (repo root) | Ground-truth query/answer CSV |
| `PAGE_MARKDOWN_INDEX` | _(unset)_ | Set to enable full-page mode |
| `OUTPUT_FILE` | `data/test_retrieval/bo767_retrieval.json` | Output path |

Output: `data/test_retrieval/bo767_retrieval_fullpage.json` (~50 MB, 1005 queries).

**Sub-page chunk mode:** omit `PAGE_MARKDOWN_INDEX` (or leave it unset) to skip full-page
expansion. The export will use raw sub-page chunks directly from LanceDB, which requires
only step 1 without `--save-intermediate` (no Parquet or page index needed).

### Step 4: Export retrieval via harness stack (alternative)

If you used the harness CLI for ingestion (e.g., `--case=e2e --dataset=bo767` into Milvus),
use `retrieve_and_export.py` instead of `export_retrieval_nemo.py`. It queries Milvus or
LanceDB through the harness retrieval utilities and writes the same JSON format:

```bash
uv run python retrieve_and_export.py
```

Both exporters produce identical JSON conforming to the
[retrieval JSON specification](#retrieval-json-format-interface-contract).

### Step 5: Run QA evaluation

**Estimated time: ~15 min - 45 min** (1005 queries, ~12s per query for generation + judge, 8 concurrent workers).

```bash
export NVIDIA_API_KEY="nvapi-..."
export RETRIEVAL_FILE=data/test_retrieval/bo767_retrieval_fullpage.json
export OUTPUT_FILE=data/test_retrieval/qa_results_bo767_annotations.json
export QA_MAX_WORKERS=8
python run_qa_eval.py
```

| Env Var | Default | Purpose |
|---------|---------|---------|
| `RETRIEVAL_FILE` | _(required)_ | Retrieval JSON from step 3 or 4 |
| `NVIDIA_API_KEY` | _(at least one key required)_ | Fallback API key for both generator and judge |
| `GEN_API_KEY` | falls back to `NVIDIA_API_KEY` | API key for the generator model |
| `JUDGE_API_KEY` | falls back to `NVIDIA_API_KEY` | API key for the judge model |
| `QA_DATASET` | `csv:data/bo767_annotations.csv` (repo root) | Ground-truth dataset |
| `QA_TOP_K` | `5` | Chunks per query |
| `QA_MAX_WORKERS` | `4` | Concurrent API calls |
| `QA_LIMIT` | `0` (all) | Evaluate only first N queries |
| `OUTPUT_FILE` | auto: `qa_results_{dataset}_{model}_{timestamp}.json` | Where to write results JSON |
| `GEN_MODEL` | `nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5` | Generator (single) |
| `GEN_MODEL_NAME` | `generator` | Short label for the generator |
| `GEN_API_BASE` | _(unset)_ | Override endpoint URL for the generator |
| `GEN_MODELS` | _(unset)_ | Multi-model sweep: `name:model,...` (overrides `GEN_MODEL`) |
| `GEN_TEMPERATURE` | `0.0` | Sampling temperature for generator |
| `JUDGE_MODEL` | `nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1` | Judge model |
| `JUDGE_API_BASE` | _(unset)_ | Override endpoint URL for the judge |
| `LITELLM_DEBUG` | `0` | Set `1` for full request/response logging |
| `MIN_COVERAGE` | `0.0` | Abort if retrieval covers fewer queries (0.0-1.0, e.g. `0.8`) |

**API key resolution:** `GEN_API_KEY` and `JUDGE_API_KEY` each fall back to `NVIDIA_API_KEY` when unset. Set only `NVIDIA_API_KEY` if generator and judge share the same provider. Set individual keys when they use different providers.

**Auto-timestamped output:** When `OUTPUT_FILE` is not set, results are written to `data/test_retrieval/qa_results_{dataset}_{model}_{YYYYMMDD_HHMMSS}.json` so consecutive runs never overwrite each other.

**Switching inference providers:** To use a different provider for the generator, set `GEN_MODEL`, `GEN_API_BASE`, and `GEN_API_KEY`:

```bash
export GEN_MODEL="openai/my-org/my-model"
export GEN_API_BASE="https://your-openai-compatible-endpoint/v1"
export GEN_API_KEY="your-api-key"
export NVIDIA_API_KEY="nvapi-..."   # still used by the judge (NIM)
python run_qa_eval.py
```

LiteLLM routes by model prefix (`nvidia_nim/`, `openai/`, `huggingface/`). See [Model Strings](#model-strings) for supported providers. Each model can target a different endpoint/key.

**API cost:** Each query costs ~$0.01-0.02 (generation + judge) on NIM pay-as-you-go. A full 1005-query run is approximately $10-20. Set `QA_LIMIT` to cap during development.

**Multi-model sweeps and repeated runs:** To evaluate multiple generator/judge combos or run each configuration N times for statistical confidence, use a YAML config file instead of env vars:

```bash
python run_qa_eval.py --config eval_sweep.yaml
```

The config defines models once and composes evaluation combos with per-combo run counts.  See [Eval Config File](#eval-config-file-yaml--json) and `eval_sweep.yaml` for the full schema.

### Results (March 2026 -- full-page markdown, bo767_annotations.csv)

```
1005 queries evaluated (Nemotron Super 49B generator, Mixtral 8x22B judge)

Tier 1 - Retrieval Quality:
  Answer-in-Context rate:  88.2% (886/1005)

Tier 2 - Programmatic Answer Quality:
  generator            token_f1=0.120

Tier 3 - LLM Judge:
  generator            mean=3.74/5  scored=970  errors=35
                       dist: 1:251  2:44  3:28  4:26  5:621

Failure Breakdown:
  correct: 647  refused_missing_context: 95  refused_with_context: 37
  generation_miss: 89  partial: 65  retrieval_miss: 37
  thinking_truncated: 35
```

**Interpretation:** 88% of queries had the answer present in the retrieved
chunks (Tier 1). The generator answered correctly ~64% of the time. The gap
is primarily `refused_missing_context` (answer genuinely not in chunks) and
`refused_with_context` (model said "not found" when the answer was present),
plus `generation_miss` (model had the context but answered incorrectly).

### Sub-page chunk mode

To skip full-page markdown and use raw sub-page chunks instead, omit
`--save-intermediate` from step 1, skip step 2, and do not set
`PAGE_MARKDOWN_INDEX` in step 3. This produces smaller context windows and may
result in lower scores for queries that span structured content (tables, charts,
infographics). See the sub-page quick reference above.

## Quick Start (Harness CLI)

```bash
cd tools/harness

# 1. Ingest PDFs into VDB
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# 2. Export retrieval results for all ground-truth queries
uv run python retrieve_and_export.py

# 3. Run QA evaluation (requires NVIDIA_API_KEY)
export NVIDIA_API_KEY="nvapi-..."
export NVIDIA_NIM_API_KEY="$NVIDIA_API_KEY"
uv run nv-ingest-harness-run --case=qa_eval --dataset=bo767
```

Results are written to `artifacts/<run>/_test_results.json`.

## Retrieval JSON Format (Interface Contract)

The retrieval JSON is the **only interface** between your retrieval system and the
QA eval harness. Any retrieval method -- vector search, agentic retrieval, hybrid
pipelines, BM25, reranked, or a fully custom system -- can plug in by producing
a single JSON file that **matches this specification** (what `run_qa_eval.py` loads via `FileRetriever`). The harness takes it from there: generates answers with one
or more LLMs and scores them with the judge. If your JSON does not match, the
eval script will not load or align queries correctly.

### Minimal format (all you need)

```json
{
  "queries": {
    "What is the range of the 767?": {
      "chunks": ["First retrieved chunk text...", "Second chunk text..."]
    },
    "How many engines does it have?": {
      "chunks": ["The 767 is powered by two..."]
    }
  }
}
```

Rules:
- **`"queries"`** (required): dict mapping query strings to result objects.
- **`"chunks"`** (required per query): list of plain-text strings, one per retrieved passage. Order matters -- put the best/most relevant chunk first. The harness uses the first `qa_top_k` entries (default 5).
- **`"metadata"`** (optional per query): list of per-chunk provenance dicts (e.g. `{"source_id": "file.pdf", "page_number": 3}`). Carried through to the results JSON for traceability but not used for scoring.
- Top-level **`"metadata"`** (optional): free-form dict for your records (retrieval method, model, timing, etc.). Ignored by FileRetriever.
- Query matching is normalized (NFKC unicode, case-folded, whitespace-collapsed) so trivial formatting differences between the ground-truth CSV and the retrieval JSON don't cause misses.

### Full format example

```json
{
  "metadata": {
    "retrieval_method": "agentic_rag",
    "model": "nvidia/llama-nemotron-embed-1b-v2",
    "top_k": 5,
    "notes": "Used multi-step agent with query decomposition"
  },
  "queries": {
    "What percentage of infections occur without eyewear?": {
      "chunks": [
        "According to the infographic, 16% of infections...",
        "Protective eyewear reduces transmission by..."
      ],
      "metadata": [
        {"source_id": "1000360.pdf", "page_number": 3, "distance": 0.31},
        {"source_id": "1000360.pdf", "page_number": 3, "distance": 0.45}
      ]
    }
  }
}
```

### Using it with the harness

```bash
# Point the harness at your retrieval JSON and run
export RETRIEVAL_FILE="path/to/my_retrieval_results.json"
export NVIDIA_API_KEY="nvapi-..."
python run_qa_eval.py
```

This means you can compare retrieval strategies head-to-head by running
the same eval against different retrieval JSONs -- the generator and judge
stay constant, so any score difference is purely from retrieval quality.

## Custom Datasets (CSV Loader)

Bring your own Q&A dataset without writing code. Any CSV with `query` and `answer` columns works:

```csv
query,answer,category
"What is the capital of France?","Paris","geography"
"What year was Python released?","1991","tech"
```

Point the harness at it with the `csv:` prefix:

```bash
export QA_DATASET="csv:/path/to/my_questions.csv"
export RETRIEVAL_FILE="path/to/my_retrieval.json"
python run_qa_eval.py
```

All columns beyond `query` and `answer` are preserved as metadata in the output. Rows with empty query or answer are silently skipped.

Built-in datasets: `bo767_infographic`, `vidore/<hf_dataset_id>`, `csv:/path/to/file.csv`. The default dataset is `csv:data/bo767_annotations.csv` (1007 Q&A pairs across text, table, chart, and infographic modalities).

## Retrieval Benchmarking

Compare different retrieval strategies head-to-head by generating FileRetriever
JSON from alternative retrieval systems and running the same QA evaluation.
Three scripts in `retrieval_bench/` bridge the
[retrieval-bench](../../../../retrieval-bench/) framework with the QA eval
harness -- any pipeline (dense, agentic, hybrid, reranked) that produces a
conforming JSON can be evaluated without changing the generator or judge.

| Script | Purpose |
|--------|---------|
| `run_dense_retrieval_bo767.py` | Run **dense** (single-shot) retrieval on the bo767 corpus using sub-page chunks from extraction Parquet, then expand hits to full-page markdown. Enables apples-to-apples comparison with LanceDB retrieval (same chunk granularity, different embedding model). |
| `run_agentic_retrieval_bo767.py` | Run **agentic** retrieval on the same bo767 corpus. An LLM agent iteratively searches with multi-angle queries, query rewriting, and a selection agent -- then hits are expanded to full-page markdown. Same corpus/expansion as the dense script; the variable is the retrieval strategy. |
| `convert_traces_to_retrieval.py` | Convert retrieval-bench per-query trace files into FileRetriever JSON for any dataset (ViDoRe, BRIGHT, custom). |

**Prerequisites:** `retrieval-bench` must be installed (`pip install -e path/to/retrieval-bench`
or equivalent). For Path A, **both** the extraction Parquet (`data/bo767_extracted/`)
from [step 1](#step-1-ingest--embed-nemo-retriever) and the page markdown index
(`data/bo767_page_markdown.json`) from [step 2](#step-2-build-page-markdown-index-nemo-retriever)
are required.

### Path A: bo767 corpus (self-contained, recommended starting point)

Run dense retrieval on the bo767 sub-page chunks from extraction Parquet.
The script indexes the same chunks that nv-ingest stores in LanceDB, embeds
them with a retrieval-bench backend, retrieves the top scoring chunks per
query, then expands hits to full-page markdown (same dedup + expansion logic
as the LanceDB export). This path is self-contained -- no external
retrieval-bench run is needed beforehand:

```bash
cd tools/harness

python retrieval_bench/run_dense_retrieval_bo767.py \
  --backend llama-nv-embed-reasoning-3b \
  --top-k 5 \
  --retriever-top-k 100 \
  --parquet-dir data/bo767_extracted \
  --output data/test_retrieval/bo767_retrieval_dense.json
```

Then evaluate -- the generator and judge are the same as for LanceDB, so any
score difference is purely from retrieval quality:

```bash
export NVIDIA_API_KEY="nvapi-..."
export RETRIEVAL_FILE=data/test_retrieval/bo767_retrieval_dense.json
export QA_MAX_WORKERS=8
python run_qa_eval.py
```

To use a non-NIM generator (e.g., an OpenAI-compatible endpoint), add
`GEN_MODEL`, `GEN_API_BASE`, and `GEN_API_KEY` as described in
[Step 5](#step-5-run-qa-evaluation).

### Path A-2: bo767 agentic retrieval (LLM agent + dense backend)

Same corpus and post-retrieval expansion as Path A, but replaces the
single-shot dense retrieval with an **LLM-driven agent** that iteratively
searches from multiple angles, rewrites sub-queries for embedding model
compatibility, and re-ranks results via a selection agent.  This is the
same agentic pipeline that placed 1st on the
[ViDoRe V3 leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard).

Requires an OpenAI-compatible LLM endpoint for the agent (NVIDIA NIM,
vLLM, OpenAI, etc.).  The embedding backend is the same as Path A.

```bash
cd tools/harness

export NVIDIA_API_KEY="nvapi-..."

python retrieval_bench/run_agentic_retrieval_bo767.py \
  --backend llama-nv-embed-reasoning-3b \
  --llm-model nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --api-key "$NVIDIA_API_KEY" \
  --top-k 5 \
  --num-concurrent 4 \
  --output data/test_retrieval/bo767_retrieval_agentic.json
```

Then evaluate identically to Path A:

```bash
export RETRIEVAL_FILE=data/test_retrieval/bo767_retrieval_agentic.json
export QA_MAX_WORKERS=8
python run_qa_eval.py
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--backend` | `llama-nv-embed-reasoning-3b` | Embedding backend for the underlying dense retriever |
| `--llm-model` | _(required)_ | LiteLLM model string for the agent LLM |
| `--api-key` | `NVIDIA_API_KEY` / `OPENAI_API_KEY` env | API key for the agent LLM |
| `--base-url` | `OPENAI_BASE_URL` env | Override LLM endpoint URL |
| `--top-k` | `5` | Full-page chunks per query in output |
| `--retriever-top-k` | `500` | Sub-page candidates per agent `retrieve()` call |
| `--num-concurrent` | `1` | Concurrent agent queries (increase for throughput) |
| `--max-steps` | `200` | Max agent iterations per query |
| `--target-top-k` | `10` | Documents the agent selects via `final_results` |

**Estimated time:** ~2-4 hours for 1,005 queries at `--num-concurrent 4`,
depending on LLM latency. Each query involves multiple LLM round-trips.
The corpus embedding step reuses the same cache as Path A -- if you
already ran Path A with the same backend, indexing will load instantly.

**Three-way comparison:** Run Path A (dense), Path A-2 (agentic), and
the LanceDB baseline on the same bo767 corpus. All three use the same
sub-page chunks and full-page expansion.  The only variables are the
embedding model (LanceDB vs retrieval-bench) and retrieval strategy
(single-shot vs agentic).  Evaluate each with the same generator/judge
to isolate retrieval quality.

### Path B: ViDoRe dataset (requires retrieval-bench traces)

> **Prerequisite:** This path requires a completed `retrieval-bench` run that
> produced per-query trace files. You must run retrieval-bench separately on
> your chosen ViDoRe dataset **before** using the converter script below. If
> you have not set up retrieval-bench yet, use **Path A** above instead.

Convert the per-query traces into FileRetriever JSON:

```bash
cd tools/harness

python retrieval_bench/convert_traces_to_retrieval.py \
  --traces-dir /path/to/retrieval-bench/traces \
  --trace-run-name DenseRetrievalPipeline__llama-nv-embed-reasoning-3b \
  --dataset-name vidore/vidore_v3_finance_en \
  --top-k 5 \
  --output data/test_retrieval/vidore_finance_retrieval.json
```

Then evaluate with the standard QA pipeline:

```bash
export NVIDIA_API_KEY="nvapi-..."
export RETRIEVAL_FILE=data/test_retrieval/vidore_finance_retrieval.json
export QA_DATASET="vidore/vidore_v3_finance_en"
export QA_MAX_WORKERS=8
python run_qa_eval.py
```

### Extending to other retrieval systems

Any retrieval pipeline -- agentic RAG, hybrid search, BM25 + reranker, or a
fully custom system -- can be benchmarked through the same workflow. Produce a
JSON file conforming to the [retrieval JSON specification](#retrieval-json-format-interface-contract)
and point `RETRIEVAL_FILE` at it. The QA eval harness treats all retrieval
sources identically.

For the most direct comparison, use the **same dataset** (e.g., bo767) across
all retrieval methods. Path A above uses the same sub-page chunks, ground-truth
CSV, and page expansion logic as the LanceDB baseline from
[Reproducing the bo767 Run](#reproducing-the-bo767-run), so the only variable
is the embedding model. Run QA eval with the same generator and judge on
each retrieval JSON and compare the resulting scores side-by-side.

## Architecture

The evaluation framework provides **two execution paths** that share the same
underlying components. Both paths consume an identical retrieval JSON + ground-truth
dataset and produce the same multi-tier scored output.

### Operator graph chain (single-model)

Built from `AbstractOperator` nodes connected via the `>>` operator. Each operator
inherits from `EvalOperator` (which extends `AbstractOperator` + `CPUOperator`),
accepts a pandas DataFrame, validates required columns, and returns an enriched
DataFrame. The chain is executed by `InprocessExecutor` (DFS traversal).

```
RetrievalLoaderOperator >> QAGenerationOperator >> JudgingOperator >> ScoringOperator
        |                         |                      |                  |
  Loads ground truth +     Generates answers       LLM-as-judge        Programmatic
  retrieval JSON into      via LiteLLMClient       scoring (1-5)       metrics (F1,
  a DataFrame with         (ThreadPool, N          via LLMJudge        exact match,
  query, answer,           concurrent workers)                         failure modes)
  context columns
```

Construct with `build_eval_chain(config)` from `nemo_retriever.evaluation.config`
or manually:

```python
from nemo_retriever.evaluation.retrieval_loader import RetrievalLoaderOperator
from nemo_retriever.evaluation.generation import QAGenerationOperator
from nemo_retriever.evaluation.judging import JudgingOperator
from nemo_retriever.evaluation.scoring_operator import ScoringOperator

graph = (
    RetrievalLoaderOperator(retrieval_json="retrieval.json", ground_truth_csv="gt.csv")
    >> QAGenerationOperator(model="nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5")
    >> JudgingOperator(model="nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")
    >> ScoringOperator()
)
result_df = graph.execute(None)
```

### QAEvalPipeline (multi-model sweeps)

For evaluating multiple generators in a single run, `QAEvalPipeline` orchestrates
retrieval, generation, judging, and scoring across all configured models. It
accepts Protocol-based components (`RetrieverStrategy`, `LLMClient`, `AnswerJudge`)
and provides both a dict API (`evaluate()`) and a DataFrame API (`process()`).

```
nemo_retriever.evaluation.orchestrator (QAEvalPipeline)
    |
    |-- retriever  : RetrieverStrategy protocol
    |     |-- FileRetriever   (cached JSON -- recommended, in nemo_retriever.evaluation)
    |     |-- TopKRetriever   (live VDB query -- in nv_ingest_harness.utils.qa)
    |
    |-- llm_clients : dict[str, LLMClient protocol]
    |     |-- LiteLLMClient   (NVIDIA NIM, OpenAI, vLLM, Ollama)
    |
    |-- judge : AnswerJudge protocol
          |-- LLMJudge        (1-5 rubric via LLM-as-judge)
```

Construct with `build_eval_pipeline(config)` from `nemo_retriever.evaluation.config`
or manually via `QAEvalPipeline(retriever=..., llm_clients=..., judge=...)`.

### When to use which

| Criterion | Operator graph chain | QAEvalPipeline |
|-----------|---------------------|----------------|
| Number of models | Single generator | Multiple generators (sweep) |
| Entry point | `build_eval_chain(config)` | `build_eval_pipeline(config)` |
| Data interchange | pandas DataFrame flowing through `>>` | `evaluate(qa_pairs)` returns dict, `process(df)` returns DataFrame |
| Extensibility | Add operators to the chain with `>>` | Swap Protocol implementations |
| Config-driven | YAML/JSON via `load_eval_config()` | YAML/JSON via `load_eval_config()` |

### Protocol interfaces

All three pluggable interfaces are Python `Protocol` classes defined in
`nemo_retriever.evaluation.types`. Any object that implements the right method
signature works -- no inheritance or registration required.

| Protocol | Method | Default implementation |
|----------|--------|----------------------|
| `RetrieverStrategy` | `retrieve(query, top_k) -> RetrievalResult` | `FileRetriever` (cached JSON) |
| `LLMClient` | `generate(query, chunks) -> GenerationResult` | `LiteLLMClient` (NIM, OpenAI, vLLM) |
| `AnswerJudge` | `judge(query, reference, candidate) -> JudgeResult` | `LLMJudge` (1-5 rubric) |

### Files (`nemo_retriever.evaluation`)

The evaluation framework lives in `nemo_retriever/src/nemo_retriever/evaluation/`. The harness `utils/qa/` package only contains `TopKRetriever` (which depends on harness-specific recall utilities).

| Module | Purpose |
|--------|---------|
| `types.py` | Protocol definitions (`RetrieverStrategy`, `LLMClient`, `AnswerJudge`) and dataclasses (`RetrievalResult`, `GenerationResult`, `JudgeResult`, `QAQueryResult`) |
| `eval_operator.py` | `EvalOperator` base class for all QA operators -- extends `AbstractOperator` + `CPUOperator`, adds `required_columns` / `output_columns` validation |
| `retrieval_loader.py` | `RetrievalLoaderOperator` -- entry point for graph chains; loads ground truth + retrieval JSON into a DataFrame |
| `generation.py` | `QAGenerationOperator` -- wraps `LiteLLMClient` for concurrent batch generation (ThreadPoolExecutor) |
| `judging.py` | `JudgingOperator` -- wraps `LLMJudge` for concurrent batch scoring |
| `scoring_operator.py` | `ScoringOperator` -- wraps `score_dataframe()` for programmatic metrics |
| `config.py` | Eval config loader: `load_eval_config()`, `build_eval_chain()`, `build_eval_pipeline()`; supports YAML/JSON with `${VAR}` expansion |
| `retrievers.py` | `FileRetriever` (cached JSON with normalized query matching) |
| `generators.py` | `LiteLLMClient` -- unified LLM client via litellm (NIM, OpenAI, vLLM, HF) |
| `judges.py` | `LLMJudge` -- 1-5 scoring with key-term anchoring rubric |
| `scoring.py` | Programmatic scoring functions: `answer_in_context`, `token_f1`, `classify_failure`, `score_dataframe` |
| `orchestrator.py` | `QAEvalPipeline` -- multi-model orchestrator with `evaluate()` (dict API) and `process()` (DataFrame API) |
| `ground_truth.py` | Dataset loaders: `bo767_infographic`, `vidore/*`, and generic `csv:` loader |
| `text_utils.py` | Shared text processing (`strip_think_tags`) |

| Harness module | Purpose |
|----------------|---------|
| `nv_ingest_harness.utils.qa.TopKRetriever` | Live VDB retriever (Milvus/LanceDB, depends on harness recall utilities) |

### Graph framework integration

The eval operators integrate with the same graph framework used by the ingestion
pipeline (`nemo_retriever.graph`):

| Graph component | Role in eval |
|-----------------|-------------|
| `AbstractOperator` | Base class providing `preprocess` / `process` / `postprocess` lifecycle and `>>` chaining |
| `CPUOperator` | Mixin marking eval operators as CPU-only (no GPU resources required) |
| `Node` / `Graph` | Wraps operators into a traversable DAG; `Graph.execute()` runs DFS |
| `InprocessExecutor` | Runs the linear eval chain sequentially on pandas DataFrames |

The eval operators are **decoupled** from the ingestion graph. They consume
pre-computed artifacts (retrieval JSON + ground-truth CSV) and can be used
independently of any ingestion or retrieval pipeline.

### Entry Points

| Entry Point | Use Case |
|-------------|----------|
| `python -m nemo_retriever.examples.graph_pipeline` | Ingest PDFs into LanceDB (extract + embed + VDB upload via operator graph). |
| `build_page_markdown_index.py` | Build full-page markdown index from Parquet extraction results |
| `export_retrieval_nemo.py` | Export retrieval from NeMo Retriever LanceDB (supports full-page markdown) |
| `retrieve_and_export.py` | Export retrieval via harness stack (Milvus or LanceDB) |
| `run_qa_eval.py` | Standalone QA eval runner -- env-var-driven, uses `QAEvalPipeline` |
| `load_eval_config()` + `build_eval_chain()` | Config-driven single-model eval via operator graph chain |
| `load_eval_config()` + `build_eval_pipeline()` | Config-driven multi-model eval via `QAEvalPipeline` |
| `cases/qa_eval.py` | Harness CLI integration (`--case=qa_eval`) -- reads `test_configs.yaml` |
| `retrieval_bench/run_dense_retrieval_bo767.py` | Dense (single-shot) retrieval on bo767 sub-page chunks via retrieval-bench, with full-page expansion |
| `retrieval_bench/run_agentic_retrieval_bo767.py` | Agentic (LLM agent loop) retrieval on the same bo767 corpus, with full-page expansion |
| `retrieval_bench/convert_traces_to_retrieval.py` | Convert retrieval-bench traces to FileRetriever JSON (any dataset) |

## Configuration

There are **two configuration surfaces** depending on how you run the evaluation.

### Harness CLI config (`test_configs.yaml`)

When using the harness CLI (`--case=qa_eval`), all QA settings live in the
`qa_eval` section of `test_configs.yaml`:

```yaml
qa_eval:
  qa_dataset: csv:data/bo767_annotations.csv  # ground-truth dataset (or bo767_infographic, vidore/...)
  qa_top_k: 5                                 # chunks per query
  qa_retriever: file                           # "file" or "topk"
  qa_retriever_config:
    file_path: data/test_retrieval/bo767_retrieval_fullpage.json
  qa_max_workers: 8                            # concurrent threads

  qa_llm_configs:
    - name: nemotron_super_49b
      model: nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5
      api_key: ${NVIDIA_API_KEY}
    - name: qwen3_next_80b
      model: nvidia_nim/qwen/qwen3-next-80b-a3b-instruct
      api_key: ${NVIDIA_API_KEY}

  qa_judge_config:
    model: nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1
    api_key: ${NVIDIA_API_KEY}
```

### Eval config file (YAML / JSON)

For programmatic or CI use, `nemo_retriever.evaluation.config` provides a
standalone configuration layer.  Two schema formats are supported.

#### New format: `models` + `evaluations` (recommended)

Define models once, compose generator/judge combos with per-combo run counts.
Any model can serve as generator or judge.

```yaml
dataset:
  source: "csv:data/bo767_annotations.csv"

retrieval:
  type: "file"
  file_path: "data/test_retrieval/bo767_retrieval_fullpage.json"

models:
  generator-a:
    model: "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
    api_key: "${NVIDIA_API_KEY}"

  generator-b:
    model: "openai/my-org/my-model"
    api_base: "https://your-openai-compatible-endpoint/v1"
    api_key: "${GEN_API_KEY}"

  mixtral-judge:
    model: "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
    api_key: "${NVIDIA_API_KEY}"

evaluations:
  - generator: "generator-b"
    judge: "mixtral-judge"
    runs: 2

  - generator: "generator-a"
    judge: "mixtral-judge"
    runs: 5

execution:
  top_k: 5
  max_workers: 8
```

| Section | Purpose |
|---------|---------|
| `models` | Dict of model definitions keyed by short name. Fields: `model` (litellm string), `api_base`, `api_key`, `temperature`, `max_tokens`, `extra_params`, `num_retries`. |
| `evaluations` | List of generator+judge combos. Each entry references model names and has an optional `runs` count (default: `execution.runs`, then 1). Per-evaluation `temperature`/`max_tokens` overrides are supported. |
| `execution` | Shared settings: `top_k`, `max_workers`, `runs` (default for evaluations that omit it), `min_coverage` (abort if retrieval covers fewer queries than this fraction, default 0.0). |
| `output` | Optional `results_dir` (default: `data/test_retrieval`). |

**Config-driven sweep** -- run all evaluations with one command:

```bash
export GEN_API_KEY="your-api-key"    # for generators using an OpenAI-compatible endpoint
export NVIDIA_API_KEY="nvapi-..."    # for NIM models
python run_qa_eval.py --config eval_sweep.yaml
```

Each run produces a timestamped output file:
`qa_results_{dataset}_{generator}_{judge}_{run}of{total}_{YYYYMMDD_HHMMSS}.json`.
Failed runs are logged and skipped; a pass/fail summary prints at the end.

#### Legacy format: `generators` + `judge`

Still accepted and auto-normalised internally.  Every generator shares the
same judge.

```yaml
generators:
  - name: "nemotron"
    model: "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
    api_key: "${NVIDIA_API_KEY}"
    temperature: 0.0
    max_tokens: 4096

judge:
  model: "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
  api_key: "${NVIDIA_API_KEY}"

execution:
  top_k: 5
  max_workers: 8
```

Environment variables are expanded in string values: `${VAR}` resolves to
`os.environ["VAR"]` at load time. Secrets never live in the config file.

**Programmatic usage:**

```python
from nemo_retriever.evaluation.config import load_eval_config, build_eval_chain

config = load_eval_config("eval_config.yaml")
graph = build_eval_chain(config, model_name="nemotron")
result_df = graph.execute(None)
```

### Model Strings

LiteLLM routes by prefix:

| Prefix | Provider | Example |
|--------|----------|---------|
| `nvidia_nim/` | NVIDIA NIM (build.nvidia.com) | `nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| `openai/` | OpenAI or any OpenAI-compatible server | `openai/gpt-4o` |
| `huggingface/` | HuggingFace Inference Endpoints | `huggingface/meta-llama/Llama-3-70b-instruct` |

For local vLLM/Ollama, use `openai/<model>` with `api_base: http://localhost:8000/v1`.

### Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `NVIDIA_API_KEY` | Config expansion, fallback for `GEN_API_KEY`/`JUDGE_API_KEY` | Default API key for NIM models |
| `NVIDIA_NIM_API_KEY` | litellm's `nvidia_nim` provider | Alias -- set to same value as `NVIDIA_API_KEY` |
| `GEN_API_KEY` | `run_qa_eval.py`, config `${GEN_API_KEY}` | Generator API key (falls back to `NVIDIA_API_KEY`) |
| `JUDGE_API_KEY` | `run_qa_eval.py`, config `${JUDGE_API_KEY}` | Judge API key (falls back to `NVIDIA_API_KEY`) |

## Scoring System (Three-Tier Hierarchy)

Each (query, model) pair is scored by three independent tiers. Each tier tests a different layer of the RAG pipeline:

| Tier | Name | What it measures | Method | Key question |
|------|------|-----------------|--------|-------------|
| **1** | Retrieval quality | Did the retriever return chunks containing the answer? | Programmatic | "Are the reference answer's content words present in the retrieved context?" |
| **2** | Answer quality (token) | Does the generated answer overlap with the reference at the token level? | Programmatic | "How precisely did the model reproduce the expected facts?" |
| **3** | Answer quality (semantic) | Is the generated answer factually correct given the reference? | LLM-as-judge | "Did the model get the right facts, allowing paraphrasing and numeric equivalence?" |

### Tier 1: `answer_in_context` (retrieval signal)

Returns `True` if >= 50% of the reference answer's **content words** (stopwords removed, normalized) appear in the concatenated retrieved chunks. If `False`, the retriever failed to surface the answer -- any generation failure is a retrieval miss, not a model miss.

### Tier 2: `token_f1` (SQuAD-style token metrics)

After normalizing both reference and candidate (lowercase, strip punctuation, numeric equivalence like `16.00%` -> `16%`):

| Field | Formula | Meaning |
|-------|---------|---------|
| `precision` | `common_tokens / candidate_tokens` | Fraction of the model's output tokens that are relevant |
| `recall` | `common_tokens / reference_tokens` | Fraction of the reference's tokens the model captured |
| `f1` | `2 * P * R / (P + R)` | Harmonic mean of precision and recall |

### Tier 3: `judge_score` (LLM-as-judge, 1-5 scale)

An LLM scores the candidate against the reference using a structured rubric:

| Score | Label | Criteria |
|-------|-------|----------|
| **5** | Fully correct | All required facts present |
| **4** | Nearly correct | Nearly all facts present; one minor trivial difference |
| **3** | Mostly correct | Most facts present but at least one non-trivial fact missing or slightly wrong |
| **2** | Partially correct | Some facts present but core answer incomplete or has a significant error |
| **1** | Incorrect | None/almost none of the required facts; includes wrong answer, irrelevant response, or "context does not contain" when the answer exists |

<details>
<summary><strong>Current judge prompt</strong></summary>

**System:**

```
You are an expert evaluator for factual question answering.

You will receive a QUESTION, a REFERENCE answer, and a CANDIDATE answer.

Step 1 -- Identify required facts:
  Break the REFERENCE into its key terms: specific numbers, names, dates,
  percentages, units, or short phrases that constitute the factual core.
  Example: "16% of adults" -> required facts = ["16%", "adults"].

Step 2 -- Check each required fact in the CANDIDATE:
  - Allow numeric equivalence: "16.00%" = "16%", "1,000" = "1000".
  - Allow paraphrasing: "Peers" matches "Peers of those adults".
  - Allow additional correct detail: extra facts do NOT reduce the score.
  - Short but correct answers are fine: "Peers" is valid for "Peers".

Step 3 -- Score on a 1-5 scale based on the fraction of required facts present:
  5 - All required facts present. Answer is fully correct.
  4 - Nearly all required facts present. One minor fact may differ trivially.
  3 - Most required facts present but at least one non-trivial fact is missing
      or slightly wrong.
  2 - Some required facts present but the core answer is incomplete or has a
      significant factual error.
  1 - None or almost none of the required facts present. Includes: wrong answer,
      irrelevant response, or stating "the context does not contain this
      information" when the reference answer exists.

Respond ONLY with valid JSON:
{"score": <integer 1-5>, "reasoning": "<one sentence citing which required facts were matched or missed>"}

No text outside the JSON object.
```

**User:**

```
Question: {query}

Reference answer: {reference}

Candidate answer: {candidate}
```

</details>

### Output fields per query

| Field | Source | Type | Description |
|-------|--------|------|-------------|
| `query` | Input | `str` | The ground-truth question |
| `reference_answer` | Input | `str` | Ground-truth answer from the dataset |
| `context` | Tier 1 | `list[str]` | Retrieved chunks passed to the generator |
| `answer_in_context` | **Tier 1** | `bool` | >= 50% of reference content words found in chunks |
| `answer` | Generation | `str` | The model's generated answer |
| `gen_latency_s` | Generation | `float` | Generation API call latency (seconds) |
| `gen_error` | Generation | `str?` | Error string if generation failed, else `null` |
| `token_f1` | **Tier 2** | `float` | SQuAD-style F1 (0.0-1.0) |
| `judge_score` | **Tier 3** | `int?` | LLM judge score (1-5), `null` on error |
| `judge_reasoning` | **Tier 3** | `str` | One-sentence explanation from the judge |
| `judge_error` | Tier 3 | `str?` | Error string if judging failed, else `null` |
| `failure_mode` | Derived | `str` | Classification (see below) |

### Failure mode classification

Combines Tier 1 + Tier 3 to diagnose **where** a failure occurred:

| `failure_mode` | Condition | What it means |
|----------------|-----------|---------------|
| `correct` | `judge_score >= 4` | Model answered correctly |
| `partial` | `judge_score` 2-3, answer does not claim "no context" | Model gave an incomplete but relevant answer |
| `refused_missing_context` | `judge_score` 1-3, model claims "no context", AND `answer_in_context == False` | Model correctly reported that the answer was not in the retrieved chunks |
| `refused_with_context` | `judge_score` 1-3, model claims "no context", AND `answer_in_context == True` | Model falsely claimed the answer was missing despite it being in the chunks |
| `retrieval_miss` | `judge_score <= 1`, answer does NOT claim "no context", AND `answer_in_context == False` | Retriever failed to surface the answer |
| `generation_miss` | `judge_score <= 1`, answer does NOT claim "no context", AND `answer_in_context == True` | Retriever had the answer, model missed it (wrong answer, not a refusal) |
| `thinking_truncated` | `gen_error == "thinking_truncated"` | Reasoning model exhausted token budget before producing an answer |
| `judge_error` | `judge_score` is `null` AND `gen_error != "thinking_truncated"` | Judge API failed (timeout, parse error, rate limit); query could not be scored |

The **`refused_missing_context`** vs **`refused_with_context`** split replaces the old `no_context` bucket. Both indicate the model said "context does not contain..." but they differ on whether that was true: `refused_missing_context` = correct refusal (retrieval problem), `refused_with_context` = false refusal (generation problem).

### Extending the scoring system

All scoring is applied by `score_dataframe()` in `nemo_retriever.evaluation.scoring`. To add a custom metric:

1. **Add a scoring function** in `scoring.py` (or your own module) that takes a reference answer, candidate answer, and/or context and returns a value:

```python
def my_custom_metric(reference: str, candidate: str) -> float:
    """Example: character-level Levenshtein similarity."""
    # your logic here
    return similarity_score
```

2. **Wire it into `score_dataframe()`** by appending a new column in the iteration loop, or create your own post-processing function that takes the output DataFrame and adds columns:

```python
def add_custom_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["my_metric"] = df.apply(
        lambda row: my_custom_metric(row["reference_answer"], row["answer"]),
        axis=1,
    )
    return df
```

3. **Apply after the pipeline** -- `QAEvalPipeline.process()` returns a DataFrame. Chain your scoring on top:

```python
result_df = pipeline.process(input_df)
result_df = add_custom_scores(result_df)
```

The same pattern works for custom failure classifiers, alternative judge prompts (subclass `LLMJudge` and override the system prompt), or entirely new tiers. All components follow the Protocol pattern -- no inheritance or registration required.

## Adding a New Component

### Custom Retriever

```python
from nemo_retriever.evaluation.types import RetrieverStrategy, RetrievalResult

class MyRetriever:
    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        chunks = my_search(query, top_k)
        return RetrievalResult(chunks=chunks, metadata=[])
```

### Custom LLM Client

```python
from nemo_retriever.evaluation.types import LLMClient, GenerationResult

class MyClient:
    def generate(self, query: str, chunks: list[str]) -> GenerationResult:
        answer = my_llm(query, chunks)
        return GenerationResult(answer=answer, latency_s=0.0, model="my-model")
```

### Custom Judge

```python
from nemo_retriever.evaluation.types import AnswerJudge, JudgeResult

class MyJudge:
    def judge(self, query: str, reference: str, candidate: str) -> JudgeResult:
        score = my_scoring_logic(reference, candidate)
        return JudgeResult(score=score, reasoning="...")
```

No registration step needed -- pass the instance directly to `QAEvalPipeline`.

## Output Format

`_test_results.json` structure:

```json
{
  "dataset": "csv:data/bo767_annotations.csv",
  "retrieval_file": "data/test_retrieval/bo767_retrieval_fullpage.json",
  "top_k": 5,
  "qa_results": {
    "summary": {
      "total_submitted": 1005,
      "total_completed": 1005,
      "dropped_queries": 0
    },
    "tier1_retrieval": {
      "answer_in_context_rate": 0.8816,
      "answer_in_context_count": 886,
      "total": 1005
    },
    "tier2_programmatic": {
      "generator": {
        "mean_token_f1": 0.1196
      }
    },
    "tier3_llm_judge": {
      "generator": {
        "mean_score": 3.74,
        "score_distribution": {"1": 251, "2": 44, "3": 28, "4": 26, "5": 621},
        "mean_latency_s": 11.7,
        "scored_count": 970,
        "error_count": 35
      }
    },
    "failure_breakdown": {
      "generator": {
        "correct": 647, "partial": 65,
        "retrieval_miss": 37, "generation_miss": 89,
        "thinking_truncated": 35, "refused_missing_context": 95,
        "refused_with_context": 37
      }
    },
    "per_query": [
      {
        "query": "How much did Pendleton County spend out of their COVID-19 fund for the month of April 2021?",
        "reference_answer": "$205.43",
        "retrieved_chunk_count": 2,
        "answer_in_context": true,
        "token_f1": {"generator": {"f1": 0.057}},
        "failure_mode": {"generator": "correct"},
        "retrieved_chunks": ["## Page 2\n\n..."],
        "retrieval_metadata": [{"source_id": "1003421.pdf", "page_number": 2, "distance": 1.037}],
        "generations": {"generator": {"answer": "...$205.43...", "latency_s": 12.2}},
        "judgements": {"generator": {"score": 5, "reasoning": "The required fact '$205.43' is present..."}}
      }
    ]
  }
}
```

## Dataset Limitations

### bo767_annotations.csv (default)

The default ground truth (`data/bo767_annotations.csv`) contains 1007 Q&A pairs
across all modalities (text, table, chart, infographic) for 767 bo767 PDFs.

1. **Short factual answers**: Most reference answers are 1-5 words (e.g., "$205.43", "5"). Tier 2 programmatic metrics (exact match, F1) carry strong signal. For open-ended datasets, Tier 3 (LLM judge) becomes primary.

2. **Retrieval =/= QA quality**: A retrieval method that returns the correct page may still get a low QA score if the extracted text is garbled or incomplete. Always check Tier 1 first -- if `answer_in_context_rate` is low, the problem is retrieval or extraction, not the generator.

3. **Full-page markdown recommended**: Sub-page chunks may split structured content (tables, charts) across multiple records. The full-page markdown pipeline (step 2 in reproduction) reconstructs complete pages, matching the research team's approach and improving generation accuracy.

4. **Reasoning model truncation**: Models with extended thinking (e.g., Nemotron Super) may spend their token budget reasoning and never produce a final answer. The pipeline detects this (`thinking_truncated`) and nullifies the score.

5. **Model refusal failures**: The model sometimes responds "no information found" even when the answer is in the retrieved chunks. The failure breakdown splits these into `refused_missing_context` (answer genuinely absent -- retrieval problem) vs `refused_with_context` (answer present but model refused -- generator problem). Together they account for ~13% of queries in the reference run.

## Standalone Runner

For Docker or CI environments without the full harness CLI:

```bash
RETRIEVAL_FILE=/data/bo767_retrieval.json \
NVIDIA_API_KEY=nvapi-... \
python run_qa_eval.py
```

All config is via environment variables. See the docstring in `run_qa_eval.py` for
the full list.
