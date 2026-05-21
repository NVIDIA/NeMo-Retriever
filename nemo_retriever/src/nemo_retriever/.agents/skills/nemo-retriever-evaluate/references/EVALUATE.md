# Evaluate Reference

## Contents

- [Recall](#recall)
- [QA Evaluation](#qa-evaluation)
- [Eval From Environment](#eval-from-environment)
- [End-To-End QA Shortcut](#end-to-end-qa-shortcut)
- [LLM Requirements](#llm-requirements)

## Recall

Use recall when the task has labeled expected pages or document/page keys:

```bash
retriever recall vdb-recall run \
  --query-csv ./queries.csv \
  --lancedb-uri ./lancedb \
  --table-name nv-ingest \
  --top-k 10
```

The query CSV expects `query,pdf_page` or `query,pdf,page`. The command retrieves
at least 10 internally for recall@10 even when fewer hits are printed.

Remote query embedding options include:

- `--embedding-endpoint`
- `--embedding-http-endpoint`
- `--embedding-grpc-endpoint`
- `--embedding-model`
- `--embedding-api-key`

If you omit the embedding endpoint options, recall falls back to local
HuggingFace embeddings and may download a model. For quick remote-NIM runs,
pass the endpoint/model/API key explicitly.

## QA Evaluation

Preferred reproducible path:

```bash
retriever eval export \
  --lancedb-uri ./lancedb \
  --lancedb-table nv-ingest \
  --query-csv ./qa.csv \
  --output ./eval/retrieval.json

retriever eval run --config ./eval_sweep.yaml
```

`retriever eval export` writes the retrieval JSON contract consumed by
`retriever eval run` / `FileRetriever`. It can also use `--page-index` to replace
sub-page chunks with full-page markdown.

For local HuggingFace query embeddings:

```bash
retriever eval export \
  --lancedb-uri ./lancedb \
  --lancedb-table nv-ingest \
  --query-csv ./qa.csv \
  --output ./eval/retrieval.json \
  --top-k 5 \
  --embedder nvidia/llama-nemotron-embed-1b-v2 \
  --local-query-embed-backend hf \
  --local-hf-cache-dir "$HOME/models/huggingface" \
  --local-hf-device cuda
```

Minimal `eval_sweep.yaml` for an existing retrieval JSON:

```yaml
dataset:
  source: "csv:./qa.csv"

retrieval:
  type: "file"
  file_path: "./eval/retrieval.json"

models:
  generator:
    model: "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
    api_key: "${NVIDIA_API_KEY}"
  judge:
    model: "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
    api_key: "${NVIDIA_API_KEY}"

evaluations:
  - generator: "generator"
    judge: "judge"
    runs: 1

execution:
  top_k: 5
  max_workers: 8

output:
  results_dir: "./eval/results"
```

`retriever eval export` supports local-HF query embedding options, but it does
not currently expose a remote embedding endpoint flag. When the index must be
queried with a remote/self-hosted embedding endpoint, use the Python API to
create the same FileRetriever JSON contract:

```python
from nemo_retriever.export import write_retrieval_json
from nemo_retriever.retriever import Retriever

retriever = Retriever(
    top_k=5,
    vdb_kwargs={"uri": "./lancedb", "table_name": "nv-ingest"},
    embed_kwargs={
        "embedding_endpoint": "http://embed:8000/v1",
        "embed_invoke_url": "http://embed:8000/v1",
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
    },
    rerank=False,
)

all_results = {}
for row in queries:
    hits = retriever.query(row["query"])
    all_results[row["query"]] = {
        "chunks": [hit.get("text", "") for hit in hits],
        "metadata": [
            {
                "source_id": hit.get("source_id") or hit.get("source"),
                "page_number": hit.get("page_number"),
                "distance": hit.get("_distance"),
            }
            for hit in hits
        ],
    }

write_retrieval_json(all_results, "./eval/retrieval.json", {"vdb_backend": "lancedb"})
```

Build a page markdown index from ingestion Parquet:

```bash
retriever eval build-page-index \
  --parquet-dir ./processed_docs \
  --output ./page_markdown.json
```

## Eval From Environment

Run from an existing retrieval JSON:

```bash
export RETRIEVAL_FILE=./eval/retrieval.json
export QA_DATASET=csv:./qa.csv
export RESULTS_DIR=./eval/results
retriever eval run --from-env
```

Run live retrieval from LanceDB and optionally save the retrieval JSON for
repeatable reruns:

```bash
export LANCEDB_URI=./lancedb
export LANCEDB_TABLE=nv-ingest
export QA_DATASET=csv:./qa.csv
export RETRIEVAL_SAVE_PATH=./eval/retrieval.json
export RESULTS_DIR=./eval/results
retriever eval run --from-env
```

## End-To-End QA Shortcut

`retriever pipeline run` can ingest and run QA in one command:

```bash
retriever pipeline run ./data/corpus \
  --lancedb-uri ./lancedb \
  --evaluation-mode qa \
  --eval-config ./eval_sweep.yaml \
  --query-csv ./qa.csv \
  --retrieval-save-path ./eval/retrieval.json
```

Use this for development iteration. For benchmark comparisons, prefer the
separable export/run path so retrieval can be reused.

## LLM Requirements

QA generation and judging need the `nemo-retriever[llm]` extra and model/API
configuration in the eval config or environment. `NVIDIA_API_KEY` is commonly
used as a fallback for generator and judge keys.
