# NeMo Retriever: `Retriever` class

The high-level **`Retriever`** runs **query → embed → vector search → optional rerank** through a small **operator graph** (see `nemo_retriever.retriever`). Configuration is grouped into three dicts that map to Pydantic / operator constructors.

## Constructor surface

| Parameter | Purpose |
|-----------|---------|
| **`run_mode`** | `"local"` (default): archetype batch embed (HTTP URL → CPU operator, else local model). `"service"`: **always** CPU HTTP embed; you must set an embedding URL in **`embed_kwargs`**. |
| **`top_k`** | Default number of hits per query (retrieval; rerank may use `refine_factor` to retrieve more candidates first). |
| **`rerank`** | If `True`, append **`NemotronRerankActor`** after **`RetrieveVdbOperator`**. |
| **`graph`** | Optional custom `Graph`. When set, **`embed_kwargs` / `vdb_kwargs` are not used to build the default graph**—you supply a fully wired pipeline. |
| **`embed_kwargs`** | Passed to **`EmbedParams`** (merged over library defaults). Controls model, endpoints, `input_type` (default `"query"`), batch sizes, `runtime` (device, HF cache), etc. |
| **`vdb_kwargs`** | Passed to **`RetrieveVdbOperator`**. Either nested `{"vdb_op": "lancedb", "vdb_kwargs": {"uri": "...", "table_name": "..."}}` or, for convenience, a **flat** Lance-only dict `{"uri": "...", "table_name": "..."}` is coerced to `vdb_op="lancedb"`. |
| **`rerank_kwargs`** | Forwarded to **`NemotronRerankActor`** (merged over defaults). Common keys: `model_name`, `invoke_url`, `api_key`, `batch_size`, `max_length`, `score_column`, `local_reranker_backend`. **`refine_factor`** (default `4`) multiplies `top_k` for retrieval when **`rerank`** is true; it is **not** passed to the actor. |

### Default pipeline

```text
_BatchEmbedActor >> RetrieveVdbOperator [>> NemotronRerankActor if rerank]
```

`RetrieveVdbOperator.preprocess` accepts an embedded **pandas `DataFrame`** from the embed step and converts it to query vectors before calling `VDB.retrieval`.

---

## Runnable snippets

Set `PYTHONPATH` to the `nemo_retriever` source tree (or use an installed wheel). Examples assume LanceDB already populated (e.g. after ingest).

### 1) Minimal local retrieval (default VL embed model, local backend)

```python
from nemo_retriever.retriever import Retriever

r = Retriever(
    vdb_kwargs={"uri": "./kb", "table_name": "nv-ingest"},
    embed_kwargs={
        "local_ingest_embed_backend": "hf",
        "runtime": {"hf_cache_dir": "~/.cache/huggingface"},
    },
    top_k=5,
)
hits = r.query("What is in this corpus?")
print(hits[0]["text"][:200])
```

### 2) Remote HTTP embeddings (CPU batch embed)

```python
import os
from nemo_retriever.retriever import Retriever

r = Retriever(
    vdb_kwargs={"vdb_op": "lancedb", "vdb_kwargs": {"uri": "./kb", "table_name": "nv-ingest"}},
    embed_kwargs={
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_invoke_url": os.environ["NIM_EMBED_URL"],  # OpenAI-compatible /v1
        "api_key": os.environ.get("NVIDIA_API_KEY", ""),
    },
)
print(r.query("hello")[0].keys())
```

### 3) `run_mode="service"` (force CPU URL embed)

Same as (2) but **`run_mode="service"`** requires a non-empty HTTP embedding URL in **`embed_kwargs`**; local GPU embed is not used for queries.

```python
from nemo_retriever.retriever import Retriever

r = Retriever(
    run_mode="service",
    embed_kwargs={
        "embed_invoke_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "...",
    },
    vdb_kwargs={"uri": "./kb", "table_name": "nv-ingest"},
)
```

### 4) Reranking

```python
from nemo_retriever.retriever import Retriever

r = Retriever(
    vdb_kwargs={"uri": "./kb", "table_name": "nv-ingest"},
    rerank=True,
    rerank_kwargs={
        "invoke_url": "http://localhost:8015",
        "model_name": "nvidia/llama-nemotron-rerank-1b-v2",
        "refine_factor": 4,
    },
    top_k=5,
)
hits = r.query("detailed question")
assert "_rerank_score" in hits[0]
```

### 5) Per-call overrides

```python
hits = r.query(
    "q",
    top_k=3,
    vdb_kwargs={"where": "content_type = 'text'"},
    embed_kwargs={"model_name": "nvidia/llama-nemotron-embed-1b-v2"},
)
```

### 6) Batch queries

```python
batch = r.queries(["q1", "q2"], top_k=5)
assert len(batch) == 2
```

### 7) Structured `RetrievalResult` (RAG)

```python
res = r.retrieve("What is RAG?", top_k=4)
print(res.chunks[0][:80], res.metadata[0])
```

### 8) Custom `graph`

Build any linear `Graph` whose first operator accepts a `DataFrame` whose text column matches **`EmbedParams.text_column`** (default `"text"`), and whose **last** node returns `list[list[dict]]` (or a rerank `DataFrame` when **`rerank=True`** on a default-style tail—prefer matching the default contract).

```python
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.retriever import Retriever

# my_graph: Graph = embed_op >> retrieve_op  # your operators
r = Retriever(graph=my_graph, top_k=5)
```

---

## gRPC embedding

The graph embed path targets **HTTP** (OpenAI-style) batch embedding. **`RecallConfig`** gRPC-only embedding previously used a separate client; with the graph **`Retriever`**, configure an **HTTP** embedding endpoint instead.

---

## Related types

- **`EmbedParams`**: `nemo_retriever.params.EmbedParams`
- **`RetrieveVdbOperator`**: `nemo_retriever.vdb.operators.RetrieveVdbOperator`
- **`NemotronRerankActor`**: `nemo_retriever.rerank.rerank.NemotronRerankActor`
