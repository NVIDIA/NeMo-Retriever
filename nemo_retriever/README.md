# Quick Start for NeMo Retriever Library

NeMo Retriever Library is a retrieval-augmented generation (RAG) ingestion pipeline for documents that can parse text, tables, charts, and infographics. NeMo Retriever Library parses documents, creates embeddings, optionally stores embeddings in LanceDB, and performs recall evaluation.

This quick start guide shows how to run NeMo Retriever Library as a library all within local Python processes without containers. NeMo Retriever Library supports two inference options:
- Pull and run [Nemotron RAG models from Hugging Face](https://huggingface.co/collections/nvidia/nemotron-rag) on your local GPU(s).
- Make over the network inference calls to build.nvidia.com hosted or locally deployed NeMo Retriever NIM endpoints.

You’ll set up a CUDA 13–compatible environment, install the library and its dependencies, and run GPU‑accelerated ingestion pipelines that convert PDFs, HTML, plain text, audio, or video into vector embeddings stored in LanceDB (on local disk), with Ray‑based scaling and built‑in recall benchmarking.

## Prerequisites

Before starting, make sure your system meets the following requirements:

- The host is running CUDA 13.x so that `libcudart.so.13` is available.
- Your GPUs are visible to the system and compatible with CUDA 13.x.
​
If optical character recognition (OCR) fails with a `libcudart.so.13` error, install the CUDA 13 runtime for your platform and update `LD_LIBRARY_PATH` to include the CUDA lib64 directory, then rerun the pipeline. 

For example, the following command can be used to update the `LD_LIBRARY_PATH` value.

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

## Setup your environment

Complete the following steps to setup your environment. You will create and activate isolated Python and project virtual environments, install the NeMo Retriever Library and its dependencies, and then run the provided ingestion snippets to validate your setup.

1. Create and activate the NeMo Retriever Library environment

Before installing NeMo Retriever Library, create an isolated Python environment so its dependencies do not conflict with other projects on your system. In this step, you set up a new virtual environment and activate it so that all subsequent installs are scoped to NeMo Retriever Library.

In your terminal, run the following commands from any location.

For **local GPU inference** (Nemotron models running on your GPU), install with the `[local]` extra, which includes the model packages, transformers, and GPU tooling:

```bash
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install "nemo-retriever[local]==26.3.0" nv-ingest-client==26.3.0 nv-ingest==26.3.0 nv-ingest-api==26.3.0
```

For **remote NIM inference only** (no local GPU required), the base package is sufficient:

```bash
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install nemo-retriever==26.3.0 nv-ingest-client==26.3.0 nv-ingest==26.3.0 nv-ingest-api==26.3.0
```

This creates a dedicated Python environment and installs the `nemo-retriever` PyPI package, the canonical distribution for the NeMo Retriever Library.

2. Override Torch and Torchvision with CUDA 13 builds (local GPU only)

The `[local]` extra pulls PyTorch from PyPI, which defaults to a CPU build on Linux. Reinstall from the CUDA 13.0 wheel index to match the CUDA runtime required by the Nemotron model packages:

```bash
uv pip install torch==2.10.0 torchvision -i https://download.pytorch.org/whl/cu130
```

Skip this step if you are using remote NIM inference only.

## Quick Start (CLI)

Ingest once, then query as many times as you like. Reuse the same `--lancedb-uri` across both steps.

### Step 1 -- Ingest a corpus (`retriever pipeline run`)

The primary entry point. Owns the full ingestion graph (extract, embed, VDB upload) and writes a reusable LanceDB table.

```bash
export NVIDIA_API_KEY=nvapi-...

# Point it at any directory of PDFs -- bo767, your own ./docs/, or the
# test data under ../data/. LanceDB's default IVF index needs at least
# 16 chunks, so use a multi-document directory.
retriever pipeline run ./docs --lancedb-uri ./lancedb
```

### Step 2 -- Ask a question (`retriever answer`)

Smallest query-side example. Retrieves top-k chunks from the table built in Step 1 and generates an answer. For batch retrieval, evaluation, MCP, and every flag, see [CLI](#cli); for the codes agents and CI should branch on, see [Exit-code contract](#exit-code-contract).

```bash
retriever answer "What is this document about?" \
  --lancedb-uri ./lancedb \
  --model nvidia_nim/meta/llama-3.3-70b-instruct \
  --api-base https://integrate.api.nvidia.com/v1 \
  --top-k 5
```

## Run the pipeline

The [test PDF](../data/multimodal_test.pdf) contains text, tables, charts, and images. Additional test data resides [here](../data/).

> **Note:** `batch` is the primary intended run_mode of operation for this library. Other modes are experimental and subject to change or removal.

The examples below use default local GPU inference (no `invoke_url` specified) and require the `[local]` extra and the CUDA 13 torch override from the setup steps above. For remote NIM inference without a local GPU, see [Run with remote inference](#run-with-remote-inference-no-local-gpu-required).

### Ingest a test pdf
```python
from nemo_retriever import create_ingestor
from nemo_retriever.io import to_markdown, to_markdown_by_page
from pathlib import Path

documents = [str(Path("../data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")

# ingestion tasks are chainable and defined lazily
ingestor = (
  ingestor.files(documents)
  .extract(
    # below are the default values, but content types can be controlled
    extract_text=True,
    extract_charts=True,
    extract_tables=True,
    extract_infographics=True
  )
  .embed()
  .vdb_upload()
)

# ingestor.ingest() actually executes the pipeline
# results are returned as a ray dataset and inspectable as chunks
ray_dataset = ingestor.ingest()
chunks = ray_dataset.get_dataset().take_all()
```

### Ingest a test corpus (CLI)

`retriever pipeline run` is the supported ingestion entry point. Point it at a **directory** of PDFs to produce a ready-to-query LanceDB table.

> **Corpus size matters.** LanceDB's default IVF index needs at least 16
> chunks to train its 16 k-means partitions. Single-PDF ingestion will fail
> at the indexing step; point the command at a directory with enough
> documents to clear that threshold. Replace `/your-example-dir` below with
> the path to your own corpus.

```bash
retriever pipeline run /your-example-dir --lancedb-uri lancedb
```

Chunks land at `./lancedb/nv-ingest`, which matches the default `Retriever()`
constructor used in [Run a recall query](#run-a-recall-query) below. With the
`[local]` extra installed (see setup), defaults point at local-GPU extraction
and embedding. For a realistic retrieval corpus, see
[QA evaluation -- Step 1](./src/nemo_retriever/evaluation/README.md#step-1-ingest-and-embed-pdfs-nemo-retriever).

**No local GPU?** The same command accepts `--*-invoke-url` flags to route every
stage through [build.nvidia.com](https://build.nvidia.com/) NIMs. Run
`retriever pipeline run --help` for the authoritative flag surface; the
minimum full-remote recipe is:

```bash
export NVIDIA_API_KEY=nvapi-...

retriever pipeline run /your-example-dir \
  --lancedb-uri lancedb \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --graphic-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings
```

The default embedder is `nvidia/llama-nemotron-embed-vl-1b-v2`; override with `--embed-model-name` if you need a different embedder.

When you use the remote embedder, pair the `Retriever` with the matching
`embedder=` + `embedding_endpoint=` overrides shown in
[Run a recall query](#run-a-recall-query).

### Inspect extracts
You can inspect how recall accuracy optimized text chunks for various content types were extracted into text representations:
```python
# page 1 raw text:
>>> chunks[0]["text"]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose...'

# markdown formatted table from the first page
'| Table | 1 |\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |'

# a chart from the first page
>>> chunks[2]["text"]
'Chart 1\nThis chart shows some gadgets, and some very fictitious costs.\nGadgets and their cost\n$160.00\n$140.00\n$120.00\n$100.00\nDollars\n$80.00\n$60.00\n$40.00\n$20.00\n$-\nPowerdrill\nBluetooth speaker\nMinifridge\nPremium desk fan\nHammer\nCost'

# markdown formatting for full pages or documents:
# document results are keyed by source filename
>>> to_markdown_by_page(chunks).keys()
dict_keys(['multimodal_test.pdf'])

# results per document are keyed by page number
>>> to_markdown_by_page(chunks)["multimodal_test.pdf"].keys()
dict_keys([1, 2, 3])

>>> to_markdown_by_page(chunks)["multimodal_test.pdf"][1]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose. It contains some \r\nheadings and some placeholder text to fill the space. The text is not important and contains \r\nno real value, but it is useful for testing. Below, we will have some simple tables and charts \r\nthat we can use to confirm Ingest is working as expected.\r\nTable 1\r\nThis table describes some animals, and some activities they might be doing in specific \r\nlocations.\r\nAnimal Activity Place\r\nGira@e Driving a car At the beach\r\nLion Putting on sunscreen At the park\r\nCat Jumping onto a laptop In a home o@ice\r\nDog Chasing a squirrel In the front yard\r\nChart 1\r\nThis chart shows some gadgets, and some very fictitious costs.\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 1\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 1\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 2\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 2\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 3\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 3\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost'

# full document markdown also keyed by source filename
>>> to_markdown(chunks).keys()
dict_keys(['multimodal_test.pdf'])
```

Since the ingestion job automatically populated a lancedb table with all these chunks, you can use queries to retrieve semantically relevant chunks for feeding directly into an LLM:

### Run a recall query

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(
  # default values
  lancedb_uri="lancedb",
  lancedb_table="nv-ingest",
  embedder="nvidia/llama-3.2-nv-embedqa-1b-v2",
  top_k=5,
  reranker=False
)

query = "Given their activities, which animal is responsible for the typos in my documents?"

# you can also submit a list with retriever.queries[...]
hits = retriever.query(query)
```

If you ingested with the remote-NIM recipe above (no local GPU), point the
`Retriever` at the same embedding endpoint so query vectors are produced by the
same model that produced the stored chunk vectors:

```python
retriever = Retriever(
    lancedb_uri="lancedb",
    lancedb_table="nv-ingest",
    embedder="nvidia/llama-nemotron-embed-vl-1b-v2",
    embedding_endpoint="https://integrate.api.nvidia.com/v1/embeddings",
    top_k=5,
    reranker=False,
)
hits = retriever.query(query)
```

```python
# retrieved text from the first page
>>> hits[0]
{'text': 'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose. It contains some \r\nheadings and some placeholder text to fill the space. The text is not important and contains \r\nno real value, but it is useful for testing. Below, we will have some simple tables and charts \r\nthat we can use to confirm Ingest is working as expected.\r\nTable 1\r\nThis table describes some animals, and some activities they might be doing in specific \r\nlocations.\r\nAnimal Activity Place\r\nGira@e Driving a car At the beach\r\nLion Putting on sunscreen At the park\r\nCat Jumping onto a laptop In a home o@ice\r\nDog Chasing a squirrel In the front yard\r\nChart 1\r\nThis chart shows some gadgets, and some very fictitious costs.', 'metadata': '{"page_number": 1, "pdf_page": "multimodal_test_1", "page_elements_v3_num_detections": 9, "page_elements_v3_counts_by_label": {"table": 1, "chart": 1, "title": 3, "text": 4}, "ocr_table_detections": 1, "ocr_chart_detections": 1, "ocr_infographic_detections": 0}', 'source': '{"source_id": "/home/dev/projects/NeMo-Retriever/data/multimodal_test.pdf"}', 'page_number': 1, '_distance': 1.5822279453277588}

# retrieved text of the table from the first page
>>> hits[1]
{'text': '| Table | 1 |\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |', 'metadata': '{"page_number": 1, "pdf_page": "multimodal_test_1", "page_elements_v3_num_detections": 9, "page_elements_v3_counts_by_label": {"table": 1, "chart": 1, "title": 3, "text": 4}, "ocr_table_detections": 1, "ocr_chart_detections": 1, "ocr_infographic_detections": 0}', 'source': '{"source_id": "/home/dev/projects/NeMo-Retriever/data/multimodal_test.pdf"}', 'page_number': 1, '_distance': 1.614684820175171}
```

###  Generate a query answer using an LLM
The above retrieval results are often feedable directly to an LLM for answer generation.

To do so, first install the openai client and set your [build.nvidia.com](https://build.nvidia.com/) API key:
```bash
uv pip install openai
export NVIDIA_API_KEY=nvapi-...
```

```python
from openai import OpenAI
import os

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ.get("NVIDIA_API_KEY")
)

hit_texts = [hit["text"] for hit in hits]
prompt = f"""
Given the following retrieved documents, answer the question: {query}

Documents:
{hit_texts}
"""

completion = client.chat.completions.create(
  model="nvidia/nemotron-3-super-120b-a12b",
  messages=[{"role":"user","content":prompt}],
  stream=False
)

answer = completion.choices[0].message.content
print(answer)
```

Answer:
```
Cat is the animal whose activity (jumping onto a laptop) matches the location of the typos, so the cat is responsible for the typos in the documents.
```

### Live RAG SDK (`nemo_retriever.generation`)

`nemo_retriever.generation` is the composable RAG layer. `Retriever` stays narrow (query -> LanceDB hits); `generation.*` adds everything downstream. Every function takes a `Retriever` or a DataFrame and returns a `pandas.DataFrame` with additive columns, so the output of one step is always valid input to the next.

| Function | Signature | Columns added |
| --- | --- | --- |
| `generation.retrieve` | `(retriever, queries, *, top_k=None)` | `query`, `chunks`, `metadata` |
| `generation.answer` | `(retriever, queries, *, llm, reference=None)` | `answer`, `model`, `latency_s`, `gen_error` |
| `generation.score` | `(df, *, reference_column="reference_answer")` | `answer_in_context`, `token_f1`, `exact_match`, `failure_mode` |
| `generation.judge` | `(df, *, judge)` | `judge_score`, `judge_reasoning`, `judge_error` |
| `generation.eval` | `(retriever, queries, *, llm, reference, judge=None)` | union of all of the above |

Install the LLM extra once (`uv pip install "nemo-retriever[llm]"`, then `export NVIDIA_API_KEY=nvapi-...`). The full pipeline, end-to-end:

```python
from nemo_retriever import generation
from nemo_retriever.retriever import Retriever
from nemo_retriever.llm import LiteLLMClient, LLMJudge

retriever = Retriever(
    lancedb_uri="lancedb",
    lancedb_table="nv-ingest",
    embedder="nvidia/llama-nemotron-embed-vl-1b-v2",
    embedding_endpoint="https://integrate.api.nvidia.com/v1/embeddings",
    top_k=5,
)
llm = LiteLLMClient.from_kwargs(model="nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5")
judge = LLMJudge.from_kwargs(model="nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")

df = generation.eval(
    retriever,
    ["What is RAG?", "What is reranking?"],
    llm=llm,
    reference=["RAG combines retrieval with generation.", "Reranking re-scores retrieved passages."],
    judge=judge,
)
print(df[["query", "answer", "token_f1", "judge_score", "failure_mode"]])
```

Drop `judge` for Tier-1 + Tier-2 scoring only; swap `generation.eval` for `generation.answer` to skip scoring entirely. To persist intermediates, call `retrieve` / `answer` / `score` / `judge` individually and pass the DataFrame forward. Scoring-tier semantics and column definitions live in [`src/nemo_retriever/evaluation/README.md`](src/nemo_retriever/evaluation/README.md).

### CLI

The user-facing CLI surface is five verbs: `retriever pipeline run`, `retriever answer`, `retriever retrieve`, `retriever eval batch`, and `retriever mcp serve`. These cover the full ingest-to-query-to-agent workflow and are what a typical integration touches. Everything else listed in `retriever --help` (`retriever pdf`, `retriever html`, `retriever chart`, `retriever audio`, `retriever image`, `retriever benchmark`, `retriever harness`, `retriever vector-store`, `retriever recall`, `retriever compare`, `retriever local`, `retriever txt`) is stage-level or developer tooling; see `retriever <group> --help` or [docs/cli/](docs/cli/README.md) for that surface. New contributors can safely ignore the stage commands unless debugging a specific pipeline stage.

Three Typer subcommands expose the same `generation.*` entry points for shells, CI jobs, and agents:

| Subcommand | Backing function | Output |
| --- | --- | --- |
| `retriever retrieve` | `generation.retrieve` | JSONL, one row per query |
| `retriever answer` | `generation.answer` / `generation.eval` | Single JSON object (schema below) |
| `retriever eval batch` | `generation.eval` | JSONL, one row per query |

Install extras based on deployment: `uv sync --extra llm` for remote NIMs, add `--extra local` for a local embedder (omits `--embedding-endpoint`), add `--extra mcp` for the MCP server. `--embedding-api-key` / `--api-key` default to `$NVIDIA_API_KEY`.

`retriever retrieve` / `retriever answer` / `retriever eval batch` are strictly query-time commands: they read from an existing LanceDB table and do not ingest. Populate the target table once with `retriever pipeline run` (see `retriever pipeline --help`), then reuse its `--lancedb-uri` across every query command below. Keeping ingestion and retrieval behind separate verbs is intentional -- it mirrors the package boundary between `nemo_retriever.pipeline` (write side) and `nemo_retriever.generation` (read side).

```bash
# 0. Populate LanceDB once.  See `retriever pipeline --help` for the full flag surface
#    (dedup / caption / store / BEIR evaluation).
retriever pipeline run ./docs --lancedb-uri ./lancedb

# 1. Single-query live RAG (optionally score against --reference + --judge-model).
retriever answer "What is RAG?" \
  --lancedb-uri ./lancedb \
  --model nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --reference "RAG combines retrieved context with LLM generation." \
  --judge-model nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1 \
  --json - --quiet | jq '{answer, judge_score, failure_mode}'

# 2. Batch retrieval only -- accepts --query "..." or --queries file.jsonl ({"query": "..."} per line).
retriever retrieve \
  --lancedb-uri ./lancedb \
  --queries queries.jsonl \
  --top-k 5 \
  --output hits.jsonl

# 3. Batch evaluation -- requires --reference (or a JSONL file with "reference_answer" per row).
retriever eval batch \
  --lancedb-uri ./lancedb \
  --model nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --judge-model nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1 \
  --queries queries.jsonl \
  --output results.jsonl
```

The `retriever answer` JSON schema (`--json -` or `--json path.json`) is one row of the `generation.eval` DataFrame projected onto the historical `AnswerResult` keys: `query`, `answer`, `chunks`, `metadata`, `model`, `latency_s`, `error`, `judge_score`, `judge_reasoning`, `judge_error`, `token_f1`, `exact_match`, `answer_in_context`, `failure_mode`. Score columns are `null` unless `--reference` is supplied; judge columns additionally require `--judge-model`.

#### Exit-code contract

`retriever answer` publishes a stable exit-code contract so CI pipelines, agent runtimes, and MCP tool callers can branch on the failure class without parsing stdout. Codes are deliberately orthogonal: each value names exactly one root cause.

| Code | Meaning | Rationale |
| --- | --- | --- |
| `0` | Success. Answer generated; if `--min-judge-score` was set, judge passed. | Happy path. Emit the JSON payload and exit. |
| `2` | Answer generated but `judge_score < --min-judge-score`. | Answer is usable but below the caller's quality bar -- surface the judgment, let the caller decide whether to retry with a stronger model. |
| `3` | Retrieval returned zero chunks. No answer generated. | Root cause is the corpus/embedder, not the LLM. Re-ingest or broaden `--top-k` before retrying. Dominates `5` when both conditions hold -- a dry retrieval can't produce a generation. |
| `4` | Usage error (missing flag, `--judge-model` without `--reference`, `--min-judge-score` without `--judge-model`, unreadable `--lancedb-uri`, etc.). | Invalid invocation; fix the command line. |
| `5` | Generation failed (`AnswerResult.error` populated) on a non-empty retrieval. | LLM / transport failure after a healthy retrieval. Retry the LLM or fall back to a different `--model`. |

`retriever retrieve` and `retriever eval batch` use `0` / `4` only; they never run generation, so `2` / `3` / `5` are not applicable.

### MCP server (`retriever mcp serve`)

Agent runtimes (Cursor, Claude Desktop, Cline, Aider, ...) call `generation.answer` over stdio via the Model Context Protocol. Install `uv pip install "nemo-retriever[llm,mcp]"` and register the server in the runtime's `mcp.json`:

```json
{
  "mcpServers": {
    "nemo-retriever": {
      "command": "retriever",
      "args": [
        "mcp", "serve",
        "--lancedb-uri", "/path/to/lancedb",
        "--model", "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
      ],
      "env": {"NVIDIA_API_KEY": "nvapi-..."}
    }
  }
}
```

The server exposes one tool, `answer`, with inputs `question: string` (required), `top_k: integer`, and `reference: string`. Each call returns a single text content block containing the JSON payload described in the CLI schema above. Agent integrators should map the underlying [exit-code contract](#exit-code-contract) to tool-call errors; note that the judge tier is not wired into `retriever mcp serve` today, so MCP callers receive only Tiers 1-2 (`token_f1`, `exact_match`, `answer_in_context`) and no `judge_score` -- tracked as follow-up `fu-mcp-judge-parity`.

### Ingest other types of content:

For PowerPoint and Docx files, ensure libeoffice is installed by your system's package manager. This is required to make their pages renderable as images for our [page-elements content classifier](https://huggingface.co/nvidia/nemotron-page-elements-v3).

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y libreoffice
```

For SVG files, install the optional `cairosvg` dependency. SVG support is available in the NeMo Retriever Library, but not in the container deployment. `cairosvg` requires network access to install, so it will not work in air-gapped environments.
```bash
uv pip install "nemo-retriever[multimedia]"
# or to install only the SVG dependency:
uv pip install "cairosvg>=2.7.0"
```

Example usage:
```python
# docx and pptx files
documents = [str(Path(f"../data/*{ext}")) for ext in [".pptx", ".docx"]]
# mixed types of images
images = [str(Path(f"../data/*{ext}")) for ext in [".png", ".jpeg", ".bmp"]]
ingestor = (
  # above file types can be combined into a single job
  ingestor.files(documents + images)
  .extract()
)
```

*Note:* the `split()` task uses a tokenizer to split texts by a max_token length
### Render results as markdown

If you want a readable markdown view of extracted results, pass the full in-process result list
to `nemo_retriever.io.to_markdown`. The helper now returns a `dict[str, str]` keyed by input
filename, where each value is the document collapsed into one markdown string without per-page
headers, so both single-document and multi-document runs follow the same contract.

PDF text is split at the page level.

HTML and .txt files have no natural page delimiters, so they almost always need to be paired with the `.split()` task.

```python
# html and text files - include a split task to prevent texts from exceeding the embedder's max sequence length
documents = [str(Path(f"../data/*{ext}")) for ext in [".txt", ".html"]]
ingestor = (
  ingestor.files(documents)
  .extract()
  .split(max_tokens=5) #1024 by default, set low here to demonstrate chunking
)
results = ingestor.ingest()
markdown_docs = to_markdown(results)
print(markdown_docs["multimodal_test.pdf"])
```

Use `to_markdown_by_page(results)` when you want a nested
`dict[str, dict[int, str]]` instead, where each filename maps to its per-page markdown strings.
For audio and video files, ensure ffmpeg is installed by your system's package manager.

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y ffmpeg
```

```python
ingestor = create_ingestor(run_mode="batch")
ingestor = ingestor.files([str(INPUT_AUDIO)]).extract_audio()
```

### Store extracted images and text

Use `.store()` to persist extracted images, tables, charts, and text to local disk or object storage (S3, MinIO, GCS via fsspec). Stored URIs are written back to the DataFrame so downstream stages (embed, VDB upload) can reference them. By default, base64 payloads are stripped after writing to reduce memory pressure.

```python
ingestor = (
  ingestor.files(documents)
  .extract()
  .store(
    storage_uri="s3://my-bucket/citation-assets",  # or a local path
    storage_options={"key": "...", "secret": "..."},  # fsspec auth for S3/MinIO
    store_text=True,       # also write .txt files for page text and structured content
    strip_base64=True,     # free image payloads after writing (default)
  )
  .embed()
  .vdb_upload()
)
```

### Explore Different Pipeline Options:

You can use the [Nemotron RAG VL Embedder](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)

```python
ingestor = (
  ingestor.files(documents)
  .extract()
  .embed(
    model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
    #works with plain "text"s, "image"s, and "text_image" pairs
    embed_modality="text_image"  
  )
)
```

You can use a different ingestion pipeline based on [Nemotron-Parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) combined with the default embedder:
```python
ingestor = ingestor.files(documents).extract(method="nemotron_parse")
```

## Run with remote inference, no local GPU required:

For build.nvidia.com hosted inference, make sure you have NVIDIA_API_KEY set as an environment variable. 

```python
ingestor = (
  ingestor.files(documents)
  .extract(
    # for self hosted NIMs, your URLs will depend on your NIM container DNS settings
    page_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3",
    graphic_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1",
    ocr_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1",
    table_structure_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"
  )
  .embed(
    embed_invoke_url="https://integrate.api.nvidia.com/v1/embeddings",
    model_name="nvidia/llama-nemotron-embed-1b-v2",
    embed_modality="text",
  )
  .vdb_upload()
)
```

## Ray cluster setup

NeMo Retriever Library uses Ray Data for distributed ingestion and benchmarking. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

### Local Ray cluster with dashboard

To start a Ray cluster with the dashboard on a single machine use the following command.

```bash
ray start --head
```

Open `http://127.0.0.1:8265` in your browser for the Ray Dashboard, and run your NeMo Retriever Library pipeline on the same machine with `--ray-address auto` to attach to this cluster. [Connecting to a remote Ray cluster on Kubernetes](https://discuss.ray.io/t/connecting-to-remote-ray-cluster-on-k8s/7460)

### Single‑GPU cluster on multi‑GPU nodes

To restrict Ray to a single GPU on a multi‑GPU node use the following command.

```bash
CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1
```
Then run your pipeline as before with `--ray-address auto` so it connects to this single‑GPU Ray cluster. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

## Running multiple NIM instances on multi‑GPU hosts

### Resource heuristics (batch mode)

By default, batch mode computes resources using this order:

1. Auto-detected resources (Ray cluster if connected, otherwise local machine)
2. Environment variables
3. Explicit function arguments (highest precedence)

This means defaults are deterministic but easy to override when you need fixed behavior.

### Default behavior

- `cpu_count` / `gpu_count` are detected from Ray (`cluster_resources`) or local host.
- Worker heuristics:
  - `page_elements_workers = gpu_count * page_elements_per_gpu`
  - `detect_workers = gpu_count * ocr_per_gpu`
  - `embed_workers = gpu_count * embed_per_gpu`
  - minimum of `1` per stage
- Stage GPU defaults:
  - If `gpu_count >= 2` and `concurrent_gpu_stage_count == 3`, uses high-overlap values for page-elements/OCR/embed.
  - Otherwise uses `min(max_gpu_per_stage, gpu_count / concurrent_gpu_stage_count)`.

### Override variables

| Variable | Where to set | Meaning |
|---|---|---|
| `override_cpu_count`, `override_gpu_count` | function args | Highest-priority CPU/GPU override |

### Running multiple NIM service instances on multi-GPU hosts

### Start two stacks on separate GPUs

```bash
# GPU 0 stack
GPU_ID=0 \
PAGE_ELEMENTS_HTTP_PORT=8000 PAGE_ELEMENTS_GRPC_PORT=8001 PAGE_ELEMENTS_METRICS_PORT=8002 \
OCR_HTTP_PORT=8019 OCR_GRPC_PORT=8010 OCR_METRICS_PORT=8011 \
docker compose -p retriever-gpu0 up -d page-elements ocr

# GPU 1 stack
GPU_ID=1 \
PAGE_ELEMENTS_HTTP_PORT=8100 PAGE_ELEMENTS_GRPC_PORT=8101 PAGE_ELEMENTS_METRICS_PORT=8102 \
OCR_HTTP_PORT=8119 OCR_GRPC_PORT=8110 OCR_METRICS_PORT=8111 \
docker compose -p retriever-gpu1 up -d page-elements ocr
```

The `-p` project names create isolated stacks, `GPU_ID` pins each stack to a specific physical GPU, and distinct host ports avoid collisions between the services.  

### Check and tear down stacks

To verify that both stacks are running use the following command.

```bash
docker compose -p retriever-gpu0 ps
docker compose -p retriever-gpu1 ps
```

To stop and remove both stacks use the following command.

```bash
docker compose -p retriever-gpu0 down
docker compose -p retriever-gpu1 down
```

## ViDoRe Harness Sweep

The harness includes BEIR-style ViDoRe dataset presets in `nemo_retriever/harness/test_configs.yaml` and a ready-made sweep definition in `nemo_retriever/harness/vidore_sweep.yaml`.

The ViDoRe harness datasets are configured to:

- read PDFs from `/datasets/nv-ingest/vidore_v3_corpus_pdf/...`
- ingest with `embed_modality: text_image`
- embed at `embed_granularity: page`
- enable `extract_page_as_image: true` and `extract_infographics: true`
- evaluate with BEIR-style `ndcg` and `recall` metrics

To run the full ViDoRe sweep:

```bash
cd ~/nv-ingest/nemo_retriever
retriever-harness sweep --runs-config harness/vidore_sweep.yaml
```

The same commands also work under the main CLI as `retriever harness ...` if you prefer a single top-level command namespace.

### Harness with image/text storage

The harness can persist extracted images and text alongside other run artifacts. Set `store_images_uri` in `test_configs.yaml` (per-dataset or in `active:`) or via `--override`:

```bash
retriever harness run --dataset bo20 --preset single_gpu \
  --override store_images_uri=stored_images --override store_text=true
```

When `store_images_uri` is a relative path (like `stored_images`), it resolves to `artifact_dir/stored_images/` so each run is isolated. Absolute paths and fsspec URIs (e.g. `s3://bucket/prefix`) are passed through as-is.
