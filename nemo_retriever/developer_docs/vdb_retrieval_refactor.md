# VDB Retrieval Refactor Scope and Current Contract

**Module area:** `nemo_retriever.vdb`, `nemo_retriever.retriever`, graph pipeline recall

## Purpose

This PR is about making graph-pipeline upload and retrieval work across VDB
backends without turning Retriever into a new retrieval framework.

The intended mental model is:

> Old Retriever, but the LanceDB-only search/write assumptions become VDB
> operator boundaries that can support LanceDB and Milvus.

The immediate target is graph pipeline recall over locally embedded document
chunks stored in LanceDB or Milvus. The implementation should stay small,
reviewable, and centered on the graph pipeline path.

## Why This Exists

Historically, much of the system was LanceDB-first:

- pipeline CLI options and runtime summaries talked directly about LanceDB;
- graph ingestion wrote rows through LanceDB-specific utilities;
- recall assumed a LanceDB table;
- the older client VDB retrieval path owned query embedding through an endpoint
  or client-side retrieval helper.

That was reasonable when the nv-ingest system did not have the local HF
embedding path now used by NeMo Retriever. The newer Retriever path is different:
the graph pipeline creates document embeddings locally, and retrieval should
create matching query embeddings locally.

The goal is not to discard `nv_ingest_client.util.vdb.VDB`. That client VDB API
is the backend boundary we want to satisfy. For this PR, that means:

- `VDB.run(records)` uploads already-embedded graph records.
- `VDB.retrieval(vectors, **kwargs)` searches using already-embedded query
  vectors.

## Current Contract

For graph recall, the desired contract is:

1. The graph pipeline embeds document chunks with a local HF model.
2. The pipeline CLI materializes the graph result once.
3. `IngestVdbOperator` uploads those already-embedded records through
   `nv_ingest_client.util.vdb.VDB.run(...)`.
4. `Retriever` embeds query strings with the matching local HF model.
5. `RetrieveVdbOperator` passes those precomputed query vectors to
   `nv_ingest_client.util.vdb.VDB.retrieval(...)`.
6. Recall scoring consumes normalized Retriever hits.

For this PR, VDB-agnostic means backend-agnostic storage and vector search. It
does not mean embedding-owner-agnostic. Retriever remains the owner of local HF
query embedding unless we explicitly change that contract.

## In-Scope Flow: Graph Ingestion and VDB Upload

The graph upload path is currently CLI orchestration, not a graph node:

```text
pipeline run
  -> _build_ingestor(...)
  -> GraphIngestor.ingest()
  -> build_graph(...)
  -> RayDataExecutor/InprocessExecutor
  -> raw graph result
  -> _collect_results(run_mode, raw_result)
  -> _upload_vdb_records(records, vdb_op, vdb_kwargs)
  -> IngestVdbOperator(records)
  -> IngestVdbOperator.process(...)
  -> graph rows -> nv-ingest-client VDB record shape
  -> nv_ingest_client.util.vdb.<backend>.run(records)
```

Responsibilities:

- `pipeline/__main__.py` resolves `vdb_op` and opaque `vdb_kwargs`, owns result
  materialization, and invokes VDB upload.
- `GraphIngestor` only builds and executes the graph, then returns the graph
  result.
- `_collect_results(...)` is the single place that calls `take_all()` in batch
  mode.
- `_upload_vdb_records(...)` is the CLI handoff from materialized graph records
  to VDB upload.
- `IngestVdbOperator` is the boundary to nv-ingest-client VDB writers.
- Record conversion exists only because `VDB.run(...)` expects the
  nv-ingest-client nested record shape, while graph output rows are ordinary
  extraction/embed rows.

This path should not contain LanceDB-specific logic except for the default
`vdb_op="lancedb"` and whatever opaque kwargs are passed to the LanceDB client
VDB implementation.

### Why Upload Is Not a Graph Step Yet

`vdb_upload` is intentionally not recorded in `GraphIngestor._stage_order`
today. The previous post-graph hook made upload look like a graph stage while it
actually ran after graph execution and cached materialized Ray records for the
CLI. The current shape is more explicit:

```text
GraphIngestor
  -> graph construction/execution only

pipeline CLI
  -> materialize once
  -> upload once
  -> save/evaluate/summarize
```

When VDB upload is made part of the graph, it should be added deliberately as a
real graph sink/stage, likely by wiring `IngestVdbOperator` through graph
construction rather than reintroducing a hidden post-execution hook.

## In-Scope Flow: Recall Against a Populated VDB

The graph recall path is:

```text
pipeline run --query-csv ...
  -> _run_evaluation(...)
  -> RecallConfig(vdb_op=..., vdb_kwargs=...)
  -> retrieve_and_score(...)
  -> Retriever(vdb=..., vdb_kwargs=...)
  -> Retriever.queries(...)
  -> local HF query embedding
  -> RetrieveVdbOperator.process(query_vectors, ...)
  -> VDB.retrieval(query_vectors, ...)
  -> normalized Retriever hits
  -> recall@k scoring
```

Responsibilities:

- `RecallConfig` carries the selected VDB backend and kwargs into recall.
- `retrieve_and_score()` constructs `Retriever`.
- `Retriever.queries()` owns query text normalization, top-k handling, local HF
  query embedding, optional reranking, and returning Retriever-shaped hits.
- Backend-specific VDB search should live behind the client VDB retrieval
  boundary, not inline in `Retriever`.

The important invariant is that document vectors and query vectors are produced
by the same local HF embedding model family. This avoids depending on the older
hosted embedding path for graph recall.

## Retrieval Contract Change

Older client VDB retrieval implementations treated `VDB.retrieval(...)` as a
query-string API:

```text
query strings -> endpoint/service embedding -> vector search
```

That contract does not match graph recall, where Retriever must use the same
local HF embedding path as graph document embedding.

The approved direction for this PR is to redefine retrieval VDBs toward:

```text
query vectors -> vector search
```

That lets `Retriever.queries()` stay close to the upstream flow:

```text
resolve query strings
resolve top_k
embed query strings locally
call VDB retrieval with query vectors
rerank if configured
```

## Records and Hit Normalization

There are two separate conversion concerns that should not be confused:

1. **Upload conversion**
   - graph rows -> nv-ingest-client nested records
   - required while `VDB.run(records)` is the upload contract
2. **Retrieval normalization**
   - backend/client hits -> Retriever hit dictionaries
   - required while LanceDB and Milvus return different hit shapes

The upload conversion can be aggressively trimmed if we only support canonical
graph output rows. It does not need to support every historical input shape.

The retrieval normalization can also be narrowed, but PyMilvus `Hit.to_dict()`
handling must survive somewhere. Without that conversion, Milvus recall can
silently become zero because the hits are mapping-like but not plain dicts.

## Out of Scope

These are intentionally out of scope for this PR unless explicitly reopened:

- changing nv-ingest-client VDB behavior beyond the retrieval vector-in
  contract;
- moving local HF query embedding into a new adapter;
- making all legacy BEIR, harness, or outdated example pipelines fully
  VDB-agnostic beyond the compatibility fixes needed to keep current callers
  constructible;
- preserving deprecated LanceDB public shims in the graph pipeline CLI;
- adding a broad new retrieval framework.

## Review Objectives

The implementation should converge on:

- no first-class LanceDB concepts in the generic graph pipeline path;
- `vdb_op` plus opaque `vdb_kwargs` for backend selection/configuration;
- generic runtime language such as "VDB upload" instead of "LanceDB write";
- minimal record conversion needed by the graph pipeline upload path;
- one graph-result materialization point in the CLI before upload/evaluation;
- local HF query embedding remaining in Retriever;
- backend-specific vector search hidden behind VDB code, not spread through
  Retriever;
- nv-ingest-client VDB upload behavior unchanged.

## Validation Target

The PR is considered behaviorally sound when both LanceDB and Milvus can run the
same graph pipeline recall flow:

```text
ingest jp20
  -> upload 3147 embedded records
  -> retrieve 115 query-csv queries
  -> recall@1/5/10 matches between LanceDB and Milvus
```

The most recent JP20 validation showed:

```text
LanceDB:
  rows collected: 3154
  rows uploaded:  3147
  queries:        115
  recall@1/5/10: 0.6261 / 0.9043 / 0.9391

Milvus:
  rows collected: 3154
  rows uploaded:  3147
  queries:        115
  recall@1/5/10: 0.6261 / 0.9043 / 0.9391
```

Milvus upload is expected to be slower for this local deployment because the
current client backend uses MinIO-backed bulk import plus collection load/refresh
for datasets above the streaming threshold. That behavior is backend-specific
and outside this PR's scope unless Milvus upload performance is explicitly
reopened.
