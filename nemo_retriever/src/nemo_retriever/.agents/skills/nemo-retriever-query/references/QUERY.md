# Query Reference

## Contents

- [CLI Query](#cli-query)
- [Python Query](#python-query)
- [Result Schema](#result-schema)
- [Answer Synthesis](#answer-synthesis)

## CLI Query

The root CLI command is:

```bash
retriever query "question" --top-k 5 --lancedb-uri ./lancedb --table-name nv-ingest
```

From the source checkout, prefix with the project environment when needed:

```bash
uv run --project nemo_retriever retriever query "question" \
  --top-k 5 \
  --lancedb-uri ./lancedb \
  --table-name nv-ingest
```

Observed from source/tests:

- Default `--top-k` is `10`.
- Default `--lancedb-uri` is `lancedb`.
- Default `--table-name` is `nv-ingest`.
- The CLI prints clean JSON on stdout on success.
- Reranking is off by default.
- `--rerank` enables local reranking. Any reranker URL/model/backend option also
  implicitly enables reranking.
- `EMBED_INVOKE_URL` and `RERANKER_INVOKE_URL` environment variables are used
  when the matching CLI flags are omitted.
- For local HuggingFace query embeddings, pass `--local-query-embed-backend hf`
  plus `--local-hf-cache-dir` and `--local-hf-device` when needed.

Useful summary:

```bash
retriever query "question" --top-k 5 \
  | jq -r 'to_entries[] | "rank=\(.key + 1) page=\(.value.page_number) source=\(.value.source_id // .value.path // .value.source) type=\(.value.metadata.type // .value.content_type // "?") text=\(.value.text[:200])"'
```

Do not use `fromjson` on `.metadata` for current `Retriever.query()` / root CLI
hits. The normalized API boundary returns `metadata` as a native dict.

## Python Query

For SDK use:

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(
    top_k=5,
    vdb_kwargs={"uri": "./lancedb", "table_name": "nv-ingest"},
)
hits = retriever.query("question")
```

Remote query embedding:

```python
retriever = Retriever(
    top_k=5,
    vdb_kwargs={"uri": "./lancedb", "table_name": "nv-ingest"},
    embed_kwargs={
        "embed_invoke_url": "https://integrate.api.nvidia.com/v1/embeddings",
        "embedding_endpoint": "https://integrate.api.nvidia.com/v1/embeddings",
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
    },
)
```

The equivalent CLI call must still carry the same LanceDB URI and table used at
ingest time:

```bash
retriever query "question" \
  --lancedb-uri ./lancedb \
  --table-name nv-ingest \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

Local HuggingFace query embedding:

```bash
retriever query "question" \
  --lancedb-uri ./lancedb \
  --table-name nv-ingest \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 \
  --local-query-embed-backend hf \
  --local-hf-cache-dir "$HOME/models/huggingface" \
  --local-hf-device cuda
```

The same settings in Python:

```python
from pathlib import Path

from nemo_retriever.params import ModelRuntimeParams
from nemo_retriever.retriever import Retriever

hf_cache_dir = str(Path.home() / "models/huggingface")
retriever = Retriever(
    top_k=5,
    vdb_kwargs={"uri": "./lancedb", "table_name": "nv-ingest"},
    embed_kwargs={
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "local_ingest_embed_backend": "hf",
        "runtime": ModelRuntimeParams(
            hf_cache_dir=hf_cache_dir,
            device="cuda",
        ),
    },
)
```

Use `run_mode="service"` only when you specifically need the CPU HTTP embedding
path to require an endpoint. It is not the same thing as the FastAPI ingest
service.

## Result Schema

Normalized hits may include:

- `text`: retrieved content.
- `metadata`: native dict with content metadata such as `type`, page fields, or
  stored image metadata.
- `source`, `source_id`, `path`: origin document path/name when known.
- `pdf_basename`: stem of the source PDF path.
- `page_number`: integer page number, 1-indexed when present.
- `pdf_page`: composite key like `<pdf_basename>_<page_number>`.
- `_distance`: vector distance. Lower is better within the same query/model.
- `_score` or `_rerank_score`: present for some backends/rerank paths.

## Answer Synthesis

- Prefer direct text evidence over chart/image transcriptions for exact numbers.
- Cite document/page when present.
- Preserve 1-indexed pages unless the task explicitly requests 0-indexing.
- When multiple entities, years, or categories are asked for, address each one
  explicitly, including "not found in retrieved evidence" where needed.
