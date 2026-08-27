# Experimental bounded Ray Data to LanceDB sink

Status: experimental evidence handoff. **Recommendation: do not adopt or
merge.** Gate A is not passed. Gate B was not justified and was not run. The
source baseline is `upstream/main` at
`41ab27533476e14683c1148b85c2326c5cb1b435` (`#2565`). This design is
independent of PR #2041.

Two independent blockers prevent Gate A from passing:

1. The source-frozen, partitioned BrowseComp replay with exactly 75,065 stored
   rows is not available on this host.
2. The production executor and full-result contract cannot establish the
   required total resident set size (RSS) bound. `GraphIngestor.ingest()`
   retains a corpus-sized pandas result, and the current NRB runner has no
   end-to-end production-executor arm that measures that retention together
   with Ray object-store pressure and process-tree RSS.

Restoring the replay does not resolve the second blocker. Component tests and
synthetic sink replays do not resolve either blocker. The candidate API is
therefore intentionally absent from the published product documentation.

## Candidate architecture decision

If investigation continues, use the public Ray `Dataset.iter_batches` pull
interface to feed one public LanceDB `RecordBatchReader` mutation. Do not make
each Ray block a LanceDB transaction, and do not use Ray's `Datasink`
Developer API.

The locked environment is Python 3.12.3, Ray 2.56.1, LanceDB 0.34.0,
PyArrow 25.0.0, and pandas 2.3.3. LanceDB 0.34.0 accepts an
`Iterable[pyarrow.RecordBatch]`, `RecordBatchReader`, Dataset, or Scanner for
table ingestion. The iterator path is the smallest supported mechanism that
can preserve one data mutation while pulling multiple bounded batches.

## Confirmed baseline

At the pinned base, `IngestVdbOperator.REQUIRES_GLOBAL_BATCH` makes the Ray
executor set `batch_size=None` and call `repartition(num_blocks=1)`. The
operator converts that whole pandas batch to nested Python records. The
LanceDB backend then calls `list(records)`, builds another complete Python row
list, performs one table mutation, and builds configured indexes.

The public batch result is the complete terminal pandas DataFrame. Preserving
that result means whole-process memory cannot be independent of corpus size.
This candidate bounds the canonical sink payload; it does not claim to bound
the retained public result.

## Prototype seam

For one first-party LanceDB `IngestVdbOperator`, `RayDataExecutor.ingest` stops
the lazy graph before the sink and consumes its native blocks with:

```python
dataset.iter_batches(
    batch_format=None,
    batch_size=None,
    prefetch_batches=policy.prefetch_batches,
)
```

The executor passes each native block to one sink coordinator. Arrow blocks
remain Arrow on the sink side. The same pass tees a separate pandas view to
preserve `GraphIngestor.ingest()`'s current return contract. If a node follows
the sink (currently the webhook), its Ray Dataset is rebuilt from those
retained result frames only after LanceDB finalizes. This preserves ordering
and one upstream execution without forcing a corpus-wide Ray materialization
before the sink can pull its first batch.

The coordinator performs these steps:

1. Resolve mode, schema, vector dimension, canonical columns, operation
   identity, and existing-table state.
2. Project graph rows one at a time through the existing NRL record and sidecar
   rules. It does not construct a whole-block Python record list. When
   `vector_dim` is omitted, a rare non-inferable prefix is written to an
   unlinked temporary file until the first nonempty list establishes the
   legacy dimension; the prefix is then replayed under the same bad-vector
   policy. Memory stays bounded, while temporary disk is proportional to that
   prefix.
3. Pack newly owned Arrow `RecordBatch` objects and verify
   `get_total_buffer_size() <= max_batch_bytes`. One oversized canonical row
   is rejected from exact buffer accounting before Arrow allocates it or a
   table mutation begins. NaN, null, shape, fill/drop/null, and Lance-coercible
   numeric behavior is normalized before the fixed-width Arrow conversion, so
   the sink digest and row count describe what Lance stores. A fixed 256-row
   projection window prevents Python canonical rows from accumulating up to
   corpus size when the Arrow payload is much smaller than its Python
   representation.
4. Pass one `RecordBatchReader` to one LanceDB `create_table` or `Table.add`
   call. Ray blocks never independently append, overwrite, index, or report
   success.
5. Reopen the table, validate exact fields and product metadata plus row count,
   build the vector and FTS indexes once, validate every requested index has
   zero unindexed rows, optionally optimize, validate version ancestry, and
   then mark success.

`PutVdbOperator`, collection writes, custom VDB backends, and in-process
execution retain their existing paths. `PutVdbOperator` explicitly opts out of
the bounded ingestion capability because its stable-key replacement semantics
are not append/overwrite ingestion semantics.

## Memory statement

The hard local invariant is only:

```text
each owned canonical RecordBatch <= max_batch_bytes
and at most 256 projected Python rows await Arrow conversion
```

`prefetch_batches` bounds only the iterator's local fetch queue. It does not
bound Ray producer tasks, streaming-executor queues, or total object-store
occupancy. An earlier non-versioned pinned-Ray probe with
`prefetch_batches=0`, 12 lazy input blocks, and a paused Lance consumer allowed
all 12 upstream blocks to complete. The retained diagnostic script adds a
2 MiB DeveloperAPI object-store resource limit. With approximately 1 MiB
blocks, three tasks had completed at the first sink pull, and five had
completed after a 2-second pause. The 12.58 MiB of pre-created input references
were outside that limit. These are diagnostic observations, not a hard bound
or qualification result. The sink report therefore separates configured
canonical bytes, observed canonical batch bytes, input block bytes, and
terminal-result bytes instead of turning the prefetch count into a memory
claim.

The following remain outside the hard byte bound and must be measured:

- the current and prefetched Ray source blocks;
- LanceDB/Rust writer and index buffers;
- the complete pandas result required by the public API;
- a post-finalization Ray Dataset rebuilt from the retained result when a
  post-sink node is present;
- temporary disk proportional to a non-inferable prefix when dense
  `vector_dim` is omitted (this is not resident memory).

Consequently this implementation alone cannot establish Gate A's total-RSS
claim. A summary-only return or a disk-backed terminal-result spool is a
separate product/API decision and belongs in qualification as an ablation.

## Failure, visibility, and retry contract

The full lifecycle is not atomic. Only the single Lance data mutation has the
tested all-or-nothing generator-failure boundary. Indexing, optimization, and
operation markers are later calls.

NRL retrieval checks Lance tags and creation metadata. It rejects a table
while data/finalization markers are incomplete. Direct LanceDB readers that do
not use NRL can observe a committed data version before finalization.

The coordinator resumes finalization only when a durable data marker, durable
success marker, or new-table schema identity proves the exact operation and
request fingerprint. A changed table version or an in-memory observation is
not enough. In particular, an append that advances the table without its
durable data marker always fails closed.

| Outcome | Retry behavior |
|---|---|
| Definite iterator failure before a commit | Remove the pending marker when the table version is unchanged; the prior table remains readable. |
| Data commit with a durable data marker, followed by validation, index, or optimize failure | Keep the data and pending markers. Identical input with the same operation ID resumes finalization without adding rows again. |
| Cleanup failure after a durable success marker | Success is durable, but NRL reads remain blocked. The same operation retry completes cleanup and returns `already_succeeded`. |
| Append advances the table but its data-marker acknowledgement is lost | Raise `CommitOutcomeUnknown` and never replay automatically. Public LanceDB 0.34.0 has no operation ID in `Table.add`, so automatic classification is impossible. |
| Overwrite acknowledgement is lost | This is not treated as a resume. Replaying identical input is content-idempotent, although it can add another table version. |
| Brand-new table creation acknowledgement is lost | Durable namespaced operation and request hashes in the committed Arrow schema identify the create. Identical input is boundedly rescanned, and finalization resumes without appending. |

An autogenerated operation ID is one-shot. Each `ingest()` call generates a
new value. `last_vdb_operation_id` and the exception note make that value
available for immediate diagnostics, but they do not persist external job
state. A recoverable external job must create and durably store an explicit
operation ID before its first attempt, then reuse it only for the identical
content and sink configuration.

Successful operation tags are retained to recognize acknowledged retries.
They pin their referenced Lance versions and need a retention/ledger policy
before broad production use. The existing `_write_lock` remains process-local;
the durable markers make incomplete work fail closed but are not a
cross-process lease.

## Paths evaluated

| Path | Finding | Decision |
|---|---|---|
| One LanceDB `RecordBatchReader` mutation | Public in LanceDB 0.34.0. A real generator failure left the prior version unchanged. A successful multi-batch call produced one data version. | Implemented as the component candidate. Fragment inventory remains an NRB audit responsibility, and the path still requires Gate A qualification. |
| Production Ray executor plus retained result | A real three-block focused test exercises `iter_batches`, one upstream execution, and one Lance mutation. It does not measure the fixed replay, production process-tree RSS, or corpus-sized result retention. | Component contract evidence only. It does not pass Gate A. |
| Ray `Datasink` | Source and API assessment only. `write_datasink` uses a Developer API and materializes its write-result Dataset. It does not naturally preserve the existing full pandas return or one driver-owned Lance mutation. | Reject for this candidate without claiming benchmark evidence. |
| Ray execution object-store limit | Per-Dataset `ExecutionOptions.resource_limits.object_store_memory` reduced producer read-ahead in a pinned-Ray pause probe, but the API is DeveloperAPI and the limit is soft: running tasks and block granularity overshot it, input refs are not counted, and blocking materializing operators bypass it. | Measure as Ray overhead; do not present it as the sink's hard bound. |
| Direct PyArrow Dataset to LanceDB | A one-off pinned-environment probe created two rows from two Parquet row groups in version 1, with one version, one fragment, and one data file. The probe bypassed the coordinated lifecycle. | Confirms LanceDB type support only. It is not a staged-arm measurement or gate evidence. |
| Staged canonical PyArrow Dataset/Scanner under the coordinator | NRL exposes no canonical Dataset lifecycle seam. Reimplementing prepare, retry, validation, indexing, and finalization in NRB would test different code. | Blocked and unmeasured. |
| Projected global control | The legacy operator accepts graph rows, not canonical stored rows. Selecting fewer graph columns does not reproduce the canonical projection. | Blocked and unmeasured. |
| Per-block `Table.add` | Adds lifecycle/version/fragment and duplicate-retry hazards per Ray block. | Ablation only. |
| Staging table plus clone/rename | OSS 0.34.0 has no atomic replace. Deep clone is unimplemented; deleting a shallow-clone source removes files required by the target. | Reject. |
| Lower-level distributed Lance fragments | More moving parts and bypasses the chosen LanceDB lifecycle without evidence the public iterator fails. | Defer. |

## Known limitations and adoption blockers

The following limitations must be resolved before an adoption review:

- Restore the exact 75,065-row source-frozen BrowseComp replay and its stored
  data identity.
- Add a production-executor qualification path that captures process-tree RSS,
  Ray object-store pressure, spills, and terminal-result retention. The
  alternative is an explicitly approved bounded-summary or disk-backed result
  contract.
- Resolve the `RayDataExecutor.build_dataset()` compatibility change. A driver
  must finalize the bounded sink, so the prototype raises for a graph that
  contains this sink and requires callers to use `ingest()`. The former lazy
  Dataset return cannot represent that lifecycle. This public-seam change has
  not been approved.
- Define a durable operation-ledger retention policy and a cross-process writer
  lease. Current success tags pin versions, and the current lock is local to
  one process.

## Component evidence does not pass Gate A

Focused tests cover strict retained-byte enforcement, multi-batch real Lance
writes, one final vector/FTS build, index coverage, overwrite and append
failures, all implemented retry windows, sidecar and heterogeneous stored-value
parity, schema/product metadata, representative dense/hybrid retrieval, and a
real three-block Ray pipeline with no `repartition(num_blocks=1)` and exactly
one upstream execution.

This evidence does not pass Gate A for two independent reasons. The pinned
75,065-row BrowseComp pre-embedded replay is not present on this host, and no
equivalent local artifact reconstructs its row shape. Separately, the current
production executor retains the complete pandas result, and there is no
qualifying production-executor run that measures that retention together with
the process tree and Ray object store. Synthetic runs validate instrumentation
and component behavior only. Gate B was not justified and was not run because
Gate A did not pass. The recommendation remains **do not adopt**.

## Exact local component evidence

The following commands ran on August 20, 2026, in the pinned environment. They
used the dirty experimental worktree and prove only the named component
contracts. They are not Gate A evidence.

The real LanceDB sink contract suite ran with this command:

```bash
cd /localhome/local-jioffe/NeMo-Retriever/.worktrees/bounded-vdb-sink
PYTHONPATH=nemo_retriever/src \
  nemo_retriever/.venv/bin/python -m pytest -q \
  nemo_retriever/tests/test_bounded_vdb_sink.py \
  nemo_retriever/tests/test_bounded_vdb_sink_failures.py \
  nemo_retriever/tests/test_bounded_vdb_sink_contract.py \
  nemo_retriever/tests/test_bounded_vdb_sink_parity.py \
  nemo_retriever/tests/test_bounded_vdb_sink_digest.py \
  nemo_retriever/tests/test_bounded_vdb_sink_retry.py \
  nemo_retriever/tests/test_bounded_vdb_sink_remaining_contracts.py \
  nemo_retriever/tests/test_bounded_vdb_sink_metrics.py \
  nemo_retriever/tests/test_bounded_vdb_sink_visibility.py \
  nemo_retriever/tests/test_bounded_vdb_sink_finalize_retry.py
```

Result: 58 tests passed. The failure suite uses real LanceDB generator errors
and retry. The contract and parity suites cover index finalization, stored-row
parity, and representative retrieval.

The pinned-Ray integration ran with its excluded-by-default marker selected:

```bash
cd /localhome/local-jioffe/NeMo-Retriever/.worktrees/bounded-vdb-sink
PYTHONPATH=nemo_retriever/src \
  nemo_retriever/.venv/bin/python -m pytest -q -m integration \
  nemo_retriever/tests/test_ray_vdb_sink_integration.py
```

Result: one test passed. The test rejects `repartition(num_blocks=1)` and
`Dataset.materialize()`, verifies one upstream execution, preserves a ragged
pandas object column through a post-sink node, and performs one real LanceDB
mutation. It does not measure a production workload or memory envelope.

Two exact diagnostic commands exercised executor ordering and Ray read-ahead:

```bash
cd /localhome/local-jioffe/NeMo-Retriever/.worktrees/bounded-vdb-sink
PYTHONPATH=nemo_retriever/src \
  nemo_retriever/.venv/bin/python /tmp/audit_executor_probe.py
PYTHONPATH=nemo_retriever/src \
  nemo_retriever/.venv/bin/python /tmp/audit_ray_readahead_probe.py
```

The executor probe stored two rows, let the downstream node observe both rows,
and made direct `build_dataset()` use fail before creating a table. The
read-ahead probe observed three completed tasks at the first pull and five
after the 2-second consumer pause; all 12 completed by the end. Both scripts
reside under `/tmp`, are not versioned, and are diagnostic evidence only.

The direct PyArrow Dataset compatibility probe ran with this command:

```bash
cd /localhome/local-jioffe/NeMo-Retriever/.worktrees/bounded-vdb-sink
PYTHONPATH=nemo_retriever/src nemo_retriever/.venv/bin/python - <<'PY'
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

with tempfile.TemporaryDirectory(prefix="bounded-vdb-pyarrow-dataset-") as tmp:
    root = Path(tmp)
    parquet = root / "parquet"
    parquet.mkdir()
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 2)),
        ]
    )
    rows = (
        {"id": "a", "vector": [1.0, 0.0]},
        {"id": "b", "vector": [0.0, 1.0]},
    )
    for part, row in enumerate(rows):
        pq.write_table(
            pa.Table.from_pylist([row], schema=schema),
            parquet / f"part-{part}.parquet",
        )
    source = ds.dataset(parquet, format="parquet")
    table = lancedb.connect(root / "lancedb").create_table(
        "chunks", data=source, mode="overwrite"
    )
    versions = table.list_versions()
    latest = versions[-1]
    data_files = list(
        (root / "lancedb" / "chunks.lance" / "data").glob("*.lance")
    )
    print(
        {
            "rows": table.count_rows(),
            "version": table.version,
            "version_count": len(versions),
            "fragments": latest.get("metadata", {}).get("total_fragments"),
            "data_files": len(data_files),
        }
    )
PY
```

Result: two rows, version 1, one version, one fragment, and one data file. This
probe confirms only that LanceDB 0.34.0 accepts a `pyarrow.dataset.Dataset`.
It bypasses the candidate lifecycle and does not measure the staged arm.

The NRB `staged` and `projected-global` arms are blocked before operator
construction. No measurement command was run for either arm, and neither has
timing, memory, version, fragment, or correctness evidence. Datasink,
clone/rename, and lower-level fragment paths received source or API assessment
only; they have no qualifying runtime artifact.

## Follow-up issues before reconsidering adoption

1. Recover or regenerate the source-frozen, partitioned 75,065-row replay and
   its expected stored-value digest; then run the checked-in NRB control and
   candidate matrix under explicit no-swap containment.
2. Decide whether batch ingest may return a summary or use a disk-backed result
   spool. The current full-DataFrame contract prevents a corpus-independent
   whole-process RSS claim even when the sink payload is bounded.
3. Add a stable Ray byte-backpressure surface if total producer/object-store
   read-ahead must follow the sink policy. Pinned Ray's per-Dataset resource
   limit is DeveloperAPI and soft, so it is measurement input, not a hard
   guarantee.
4. Replace retained success tags with a durable operation ledger and define a
   cross-process lease if independent writers must be supported. Current tags
   pin versions, and the existing lock is instance-local.
5. Add an NRL-owned canonical staging seam before qualifying the public
   `pyarrow.dataset.Dataset` path. Reimplementing lifecycle logic in NRB would
   not test the production contract.
