# QA Evaluation Pipeline

**This document is the canonical QA evaluation guide** for the harness. The top-level [harness README](../../../../README.md) (`tools/harness/README.md`) only summarizes and links here so we avoid duplicating steps and env vars in two places.

Measures LLM answer quality over a RAG pipeline: retrieve context from a VDB, generate answers with one or more LLMs, and score each answer against ground-truth references using multi-tier scoring and an LLM-as-judge.

**Pluggable retrieval:** The eval harness does not care how you retrieved chunks -- only that you produce a JSON file that matches the **[retrieval JSON specification](#retrieval-json-format-interface-contract)** expected by `run_qa_eval.py` / `FileRetriever`. Vector search, hybrid, agentic pipelines, or any custom system can plug in as long as the file format and query strings align with your chosen ground-truth dataset.

**Default ground truth:** Standalone runs default to **`data/bo767_annotations.csv`** at the repo root -- the **bo767 annotations subset** maintained for this benchmark (multi-modality Q&A over the bo767 PDFs). Override with `QA_DATASET` / `QA_CSV` or another registered loader when comparing different corpora.

Designed to be **plug-and-play** -- swap retrievers, generators, or judges independently via Python Protocols without touching the orchestrator.

## Table of Contents

- [Scripts (standalone)](#scripts-standalone)
- [Pipeline file map and data flow](#pipeline-file-map-and-data-flow)
- [Multi-Tier Scoring](#multi-tier-scoring)
- [Reproducing the bo767 Run](#reproducing-the-bo767-run)
  - [Bring your own retrieval (skip steps 1-4)](#bring-your-own-retrieval-skip-steps-1-4)
- [Python environment](#python-environment)
- [Quick Start (Harness CLI)](#quick-start-harness-cli)
- [Retrieval JSON Format (Interface Contract)](#retrieval-json-format-interface-contract)
- [Custom Datasets (CSV Loader)](#custom-datasets-csv-loader)
- [Pipeline Overview](#pipeline-overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Features](#features)
- [Adding a New Component](#adding-a-new-component)
- [Output Format](#output-format)
- [Dataset Limitations](#dataset-limitations)
- [Standalone Runner (Docker / CI)](#standalone-runner)

## Multi-Tier Scoring

A single LLM-as-judge score conflates retrieval quality, text extraction quality, and LLM comprehension into one number. Multi-tier scoring isolates each layer so you can pinpoint where the pipeline fails.

| Tier | What it measures | Cost |
|------|------------------|------|
| **Tier 1 -- Retrieval** | Reference answer key terms present in retrieved chunks? (`answer_in_context`) | Zero |
| **Tier 2 -- Programmatic** | Exact match and SQuAD-style token F1 vs reference | Zero |
| **Tier 3 -- LLM Judge** | 1-5 score with key-term anchoring rubric | One judge call per query |

### Tier 1 -- Retrieval Quality (zero LLM cost)

`answer_in_context`: Are >= 50% of the reference answer's content words present in the retrieved chunks? This catches retrieval misses before any LLM is involved.

### Tier 2 -- Programmatic Answer Quality (zero judge cost)

- `exact_match`: Does the generated answer contain the reference string (after normalization)?
- `token_f1`: SQuAD-style token-level precision/recall/F1 between generated and reference answers.

These metrics are fast, deterministic, and free. They provide a stable baseline that doesn't fluctuate with judge model behavior.

### Tier 3 -- LLM Judge

The existing 1-5 judge score, now with a key-term anchoring rubric that decomposes the reference into required facts and checks each one in the candidate. Short correct answers are not penalized.

### Per-Query Failure Classification

Each query is classified into one of:
- `correct`: judge score >= 4
- `partial`: judge score 2-3
- `retrieval_miss`: reference not in chunks AND score <= 2
- `generation_miss`: reference in chunks but score <= 2
- `thinking_truncated`: model hit token limit during reasoning
- `no_context`: candidate says "no information" when reference exists

### Multi-Model Sweep

Set `GEN_MODELS` to compare multiple generators in a single run:

```bash
export GEN_MODELS="nemotron:nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5,qwen3:nvidia_nim/qwen/qwen3-235b-a22b"
```

All tiers are computed per-model. The summary shows side-by-side comparisons.

## Scripts (standalone)

All of these live under `tools/harness/` and are configured with environment variables unless noted.

| Script | Purpose |
|--------|---------|
| `ingest_bo767.py` | Ingest PDFs into LanceDB (extract + embed + VDB upload) |
| `extract_bo767_parquet.py` | Extract-only ingest to Parquet (tables, charts, infographics) |
| `build_page_markdown_index.py` | Parquet to full-page markdown JSON index |
| `export_retrieval_nemo.py` | LanceDB top-k per query to FileRetriever JSON; optional full-page mode |
| `run_qa_eval.py` | Multi-tier QA eval from a retrieval JSON |
| `retrieve_and_export.py` | Export via harness VDB stack (Milvus or LanceDB), not NeMo Retriever-only |

### Pipeline file map and data flow

End-to-end bo767 + LanceDB + full-page markdown touches these **artifacts** and **library code**:

| Stage | Artifacts produced | Code / APIs involved |
|-------|-------------------|----------------------|
| **1. Vector index (retrieval)** | `lancedb/<uri>/<table>/` (embedded sub-page chunks) | `ingest_bo767.py` → `nemo_retriever` ingest (extract, embed, `vdb_upload` to LanceDB). **Table name must match** `export_retrieval_nemo.py` (`LANCEDB_TABLE`, default `nv-ingest`). |
| **2. Rich extract (parallel path)** | `tools/harness/data/bo767_extracted/*.parquet` | `extract_bo767_parquet.py` → same extract flags, **no** embed/VDB; `save_intermediate_results` preserves table/chart/infographic columns for rendering. |
| **3. Full-page markdown index** | `data/bo767_page_markdown.json` (`source_id` → page → markdown) | `build_page_markdown_index.py` → `nemo_retriever.io.markdown.to_markdown_by_page()`; numpy list columns are coerced so structured content is not dropped. |
| **4. Retrieval export** | `data/test_retrieval/bo767_retrieval_fullpage.json` (or sub-page JSON) | `export_retrieval_nemo.py` → `nemo_retriever.retriever.Retriever` queries LanceDB; if `PAGE_MARKDOWN_INDEX` is set, hits are expanded/deduped by `(source_id, page)` and replaced with full-page markdown strings. |
| **5. Ground truth** | `data/bo767_annotations.csv` (repo root) | Questions/answers for export and eval; must align with **query string normalization** in `FileRetriever` (see retrieval JSON rules). |
| **6. Evaluation** | `qa_results_*.json` | `run_qa_eval.py` → `nv_ingest_harness.utils.qa`: `FileRetriever`, `QAEvalPipeline`, `LiteLLMClient`, `LLMJudge`, `scoring.py` (tier 1–2 + failure modes). |

**Data flow (conceptual):** PDFs → (A) **chunked embeddings in LanceDB** for similarity search; (B) **Parquet** for full-page reconstruction. **Export** runs search on (A), then **replaces** hit chunks with pages from (B) via the index. **Eval** never talks to LanceDB—it only reads the retrieval JSON + ground-truth CSV.

**Harness CLI alternative:** `cases/qa_eval.py` + `test_configs.yaml` can drive the same eval with different defaults; align `qa_dataset` and `file_path` with the standalone flow when comparing results.

## Reproducing the bo767 Run

Exact commands to reproduce the full-page markdown QA evaluation from scratch.
All scripts below are run from `tools/harness/` unless noted.

### Bring your own retrieval (skip steps 1-4)

Steps 1-4 below are the **NeMo Retriever + LanceDB** reference implementation
for ingestion, extraction, indexing, and retrieval. If your team already has a
retrieval pipeline (agentic, hybrid, BM25, or any custom system), **skip
steps 1-4 entirely** and produce a retrieval JSON file that conforms to the
[Retrieval JSON Format (Interface Contract)](#retrieval-json-format-interface-contract).
Then proceed directly to [Step 5: Run QA evaluation](#step-5-run-qa-evaluation).

The only requirement is that your JSON contains a top-level `queries` object
mapping each ground-truth question string to `{ "chunks": ["...", ...] }`.
See the [interface contract](#retrieval-json-format-interface-contract) for the
full schema, required fields, and a worked example.

### Python environment

Steps 1-4 (ingest, extract, build index, export) require the **`nemo_retriever`** library with LanceDB, CUDA, and Ray support. Step 5 (QA eval) additionally requires **`litellm`**. These are **not** part of the minimal harness install (`uv pip install -e .`).

**Recommended setup:** create an isolated Python 3.12 virtual environment and install both stacks:

```bash
uv venv qa-retriever --python 3.12
source qa-retriever/bin/activate
uv pip install nemo_retriever litellm
```

For the full harness install (includes Milvus-lite, nemotron models, etc.), see **Installation** in the [harness README](../../../../README.md) and [`tools/harness/pyproject.toml`](../../pyproject.toml).

**Eval-only path:** if you already have a retrieval JSON and only need to run `run_qa_eval.py`, an environment with `litellm` and the harness package (`cd tools/harness && uv pip install -e .`) is sufficient.

### Prerequisites (data and keys)

```bash
# bo767 PDFs (767 files)
ls /path/to/bo767/*.pdf | wc -l   # should be 767

# Ground truth: data/bo767_annotations.csv (1007 Q&A pairs across all modalities)
# Located at the repo root: <repo>/data/bo767_annotations.csv
```

### Step 1: Ingest PDFs into LanceDB (NeMo Retriever)

Runs extraction, embedding, and VDB upload in a single pass.
**Estimated time: ~45-90 min** (767 PDFs, GPU-accelerated extraction + embedding).

```bash
cd tools/harness
# Use the same table name as export_retrieval_nemo.py (LANCEDB_TABLE, default nv-ingest).
python ingest_bo767.py --dataset-dir /path/to/bo767 --lancedb-table nv-ingest
```

Output: `lancedb/nv-ingest/` (~84k chunks).

### Step 2: Extract to Parquet (NeMo Retriever)

Runs extraction only (no embed/upload) and saves the full records including
table, chart, and infographic columns as Parquet files. These are needed by
step 3 to reconstruct full-page markdown.
**Estimated time: ~30-60 min** (extraction only, no embedding).

```bash
python extract_bo767_parquet.py --dataset-dir /path/to/bo767
```

Output: `data/bo767_extracted/*.parquet`

### Step 3: Build page markdown index (NeMo Retriever)

Groups Parquet records by (document, page number) and renders each page via
`nemo_retriever.io.markdown.to_markdown_by_page()`. Outputs a JSON index
mapping `source_id -> page_number -> markdown`.
**Estimated time: ~5-10 min** (CPU-only, reads Parquet and renders markdown).

```bash
python build_page_markdown_index.py
```

Output: `data/bo767_page_markdown.json` (~180 MB, ~6k pages across 767 docs).

### Step 4: Export retrieval results (NeMo Retriever)

Queries LanceDB for each ground-truth question, then looks up the full-page
markdown for each hit's page. Multiple sub-page hits from the same page are
deduplicated into a single full-page chunk.
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

### Step 5: Run QA evaluation

**Estimated time: ~1-2 hours** (1005 queries, ~12s per query for generation + judge, 8 concurrent workers).

```bash
export NVIDIA_API_KEY="nvapi-..."
export RETRIEVAL_FILE=data/test_retrieval/bo767_retrieval_fullpage.json
export OUTPUT_FILE=data/test_retrieval/qa_results_bo767_annotations.json
export QA_MAX_WORKERS=8
python run_qa_eval.py
```

| Env Var | Default | Purpose |
|---------|---------|---------|
| `RETRIEVAL_FILE` | _(required)_ | Retrieval JSON from step 4 |
| `NVIDIA_API_KEY` | _(required)_ | API key for NVIDIA NIM endpoints |
| `QA_DATASET` | `csv:data/bo767_annotations.csv` (repo root) | Ground-truth dataset |
| `QA_TOP_K` | `5` | Chunks per query |
| `QA_MAX_WORKERS` | `4` | Concurrent API calls |
| `QA_LIMIT` | `0` (all) | Evaluate only first N queries |
| `OUTPUT_FILE` | `/tmp/qa_results.json` | Where to write results JSON |
| `GEN_MODEL` | `nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5` | Generator (single) |
| `GEN_MODELS` | _(unset)_ | Multi-model sweep: `name:model,...` (overrides `GEN_MODEL`) |
| `JUDGE_MODEL` | `nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1` | Judge model |
| `LITELLM_DEBUG` | `0` | Set `1` for full request/response logging |

**API cost:** Each query costs ~$0.01-0.02 (generation + judge) on NIM pay-as-you-go. A full 1005-query run is approximately $10-20. Set `QA_LIMIT` to cap during development.

### Results (March 2026 -- full-page markdown, bo767_annotations.csv)

```
1005 queries evaluated (Nemotron Super 49B generator, Mixtral 8x22B judge)

Tier 1 - Retrieval Quality:
  Answer-in-Context rate:  88.2% (886/1005)

Tier 2 - Programmatic Answer Quality:
  generator            exact_match=0.0%  token_f1=0.120

Tier 3 - LLM Judge:
  generator            mean=3.74/5  scored=970  errors=35
                       dist: 1:251  2:44  3:28  4:26  5:621

Failure Breakdown:
  correct: 647  no_context: 132  generation_miss: 89
  partial: 65   retrieval_miss: 37  thinking_truncated: 35
```

**Interpretation:** 88% of queries had the answer present in the retrieved
chunks (Tier 1). The generator answered correctly ~64% of the time. The gap
is primarily due to `no_context` (model said "not found" when it was there)
and `generation_miss` (model had the context but answered incorrectly).

### Sub-page chunk mode

To skip full-page markdown and use raw sub-page chunks instead, omit steps 2-3
and do not set `PAGE_MARKDOWN_INDEX` in step 4. This produces smaller context
windows and may result in lower scores for queries that span structured content
(tables, charts, infographics).

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

## Pipeline Overview

```
 NeMo Retriever (steps 1-4)                        Universal (step 5)
 ──────────────────────────                         ──────────────────
 Step 1       Steps 2-3          Step 4
 Ingest       Build Index        Export               QA Eval
+-----------+ +-----------------+ +--------------+    +-----------------+
| PDFs      | | extract_parquet | | Query LanceDB|    | For each query: |
| extract + | | build_page_index| | + full-page  |--->|  generate(LLM)  |
| embed +   | | (Parquet -> JSON| | markdown     |    |  judge(LLM)     |
| VDB upload| |  page index)    | | -> JSON      |    |  score (3 tiers)|
+-----------+ +-----------------+ +--------------+    +-----------------+
     |               |                 |                      |
  lancedb/    page_markdown.json  retrieval.json       qa_results.json
                                       ^
                                       |
 Bring Your Own Retrieval    +---------+--------+
 ─────────────────────────   | Any pipeline that |
 Skip steps 1-4 entirely.   | outputs retrieval |
 Produce a JSON matching     | JSON (see spec)  |
 the interface contract. --> +---------+--------+
```

Steps 1-4 are one reference implementation (NeMo Retriever + LanceDB).
Any retrieval system that produces a conforming JSON can replace them.
Step 5 can be re-run with different LLM configs without repeating retrieval.

## Architecture

```
orchestrator.py (QAEvalPipeline)
    |
    |-- retriever  : RetrieverStrategy protocol
    |     |-- TopKRetriever   (live VDB query)
    |     |-- FileRetriever   (cached JSON -- recommended)
    |
    |-- llm_clients : dict[str, LLMClient protocol]
    |     |-- LiteLLMClient   (NVIDIA NIM, OpenAI, vLLM, Ollama)
    |
    |-- judge : AnswerJudge protocol
          |-- LLMJudge        (1-5 rubric via LLM-as-judge)
```

All three interfaces are Python `Protocol` classes defined in `types.py`.
Any object that implements the right method signature works -- no inheritance
required.

### Files

| File | Purpose |
|------|---------|
| `types.py` | Protocol definitions (`RetrieverStrategy`, `LLMClient`, `AnswerJudge`) and dataclasses |
| `retrievers.py` | `TopKRetriever` (live VDB) and `FileRetriever` (cached JSON with normalized matching) |
| `generators.py` | `LiteLLMClient` -- unified LLM client via litellm (NIM, OpenAI, vLLM, HF) |
| `judges.py` | `LLMJudge` -- 1-5 scoring with key-term anchoring rubric |
| `scoring.py` | Programmatic scoring: `answer_in_context`, `token_f1`, `classify_failure` (zero LLM cost) |
| `orchestrator.py` | `QAEvalPipeline` -- multi-tier orchestrator with failure classification |
| `ground_truth.py` | Dataset loaders: `bo767_infographic`, `vidore/*`, and generic `csv:` loader |

### Entry Points

| Entry Point | Use Case |
|-------------|----------|
| `ingest_bo767.py` | Ingest PDFs into LanceDB (extract + embed + VDB upload) |
| `extract_bo767_parquet.py` | Extract-only ingest, save rich records as Parquet |
| `build_page_markdown_index.py` | Build full-page markdown index from Parquet |
| `export_retrieval_nemo.py` | Export retrieval from NeMo Retriever LanceDB (supports full-page markdown) |
| `run_qa_eval.py` | Standalone QA eval runner -- reads env vars, no harness CLI needed |
| `retrieve_and_export.py` | Export retrieval via harness stack (Milvus or LanceDB) |
| `cases/qa_eval.py` | Harness CLI integration (`--case=qa_eval`) -- reads `test_configs.yaml` |

## Configuration

All QA settings live in the `qa_eval` section of `test_configs.yaml`:

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
| `NVIDIA_API_KEY` | Config expansion (`${NVIDIA_API_KEY}`) | API key for NIM models |
| `NVIDIA_NIM_API_KEY` | litellm's `nvidia_nim` provider | Alias -- set to same value as above |

## Features

**Pluggable components** -- Retriever, generator, and judge are swapped via config
without code changes. Implement the Protocol to add a custom component.

**Multi-LLM sweeping** -- Configure multiple LLMs in `qa_llm_configs`; each gets
its own score, latency, and distribution in a single run.

**Normalized query matching** -- FileRetriever uses NFKC unicode normalization,
case-folding, and whitespace collapsing so trivial formatting differences
between the ground-truth CSV and retrieval JSON don't cause silent misses.

**Pre-flight coverage check** -- Before the pipeline starts, FileRetriever
validates what percentage of ground-truth queries have retrieval results and
logs any misses.

**Concurrent execution** -- `ThreadPoolExecutor` with configurable worker count
(default 8). All work is I/O-bound (API calls), so threads are appropriate.

**Chunk text in results** -- Per-query output includes the actual retrieved chunk
text (truncated to 500 chars by default) for debugging, not just counts.

**Structured error reporting** -- Generation and judge errors are recorded per-query
with error strings; aggregates report error counts separately from scores.

**ViDoRe v3 support** -- Load any ViDoRe v3 dataset from HuggingFace by setting
`qa_dataset: vidore/<dataset_id>`.

## Adding a New Component

### Custom Retriever

```python
from nv_ingest_harness.utils.qa.types import RetrieverStrategy, RetrievalResult

class MyRetriever:
    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        chunks = my_search(query, top_k)
        return RetrievalResult(chunks=chunks, metadata=[])
```

### Custom LLM Client

```python
from nv_ingest_harness.utils.qa.types import LLMClient, GenerationResult

class MyClient:
    def generate(self, query: str, chunks: list[str]) -> GenerationResult:
        answer = my_llm(query, chunks)
        return GenerationResult(answer=answer, latency_s=0.0, model="my-model")
```

### Custom Judge

```python
from nv_ingest_harness.utils.qa.types import AnswerJudge, JudgeResult

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
        "mean_exact_match": 0.0,
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
        "thinking_truncated": 35, "no_context": 132
      }
    },
    "per_query": [
      {
        "query": "How much did Pendleton County spend out of their COVID-19 fund for the month of April 2021?",
        "reference_answer": "$205.43",
        "retrieved_chunk_count": 2,
        "answer_in_context": true,
        "token_f1": {"generator": {"exact_match": false, "f1": 0.057, "precision": 0.029, "recall": 1.0}},
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

3. **Full-page markdown recommended**: Sub-page chunks may split structured content (tables, charts) across multiple records. The full-page markdown pipeline (steps 2-3 in reproduction) reconstructs complete pages, matching the research team's approach and improving generation accuracy.

4. **Reasoning model truncation**: Models with extended thinking (e.g., Nemotron Super) may spend their token budget reasoning and never produce a final answer. The pipeline detects this (`thinking_truncated`) and nullifies the score.

5. **`no_context` failures**: The model sometimes responds "no information found" even when the answer is in the retrieved chunks. This accounts for ~13% of queries in the current run and is a generator behavior issue, not a retrieval problem.

## Standalone Runner

For Docker or CI environments without the full harness CLI:

```bash
RETRIEVAL_FILE=/data/bo767_retrieval.json \
NVIDIA_API_KEY=nvapi-... \
python run_qa_eval.py
```

All config is via environment variables. See the docstring in `run_qa_eval.py` for
the full list.
