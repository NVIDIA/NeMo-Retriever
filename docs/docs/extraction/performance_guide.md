# Performance Guide

This page is a starting point for NeMo Retriever Library performance tuning guidance.

## Scope

Use this guide to document practical recommendations for:

- Extraction throughput and latency tuning
- Task-level settings (for example `extract`, `caption`, and `embed`)
- Deployment-specific tuning for library mode and Kubernetes/Helm
- NIM endpoint sizing and concurrency settings
- Benchmarking methodology and repeatable test setups

## Batch resource sizing

In batch mode, NeMo Retriever Library sizes unspecified Ray actor pools from Ray CPU and GPU resources. The library uses the resources that Ray reports as available immediately before it submits the pipeline. This prevents default extraction, OCR, and embedding pools from reserving more resources than the cluster can schedule.

For filesystem inputs, the library reserves CPU capacity for each Ray Data `ReadBinary` source task before it sizes actor pools. This reservation lets the input stage start instead of being blocked by persistent extraction actors. Inputs that are already Ray datasets, such as inline text rows, do not require this reservation.

When ingestion creates multiple extraction datasets, preflight also reserves 1 CPU per dataset for possible schema-normalization work. This keeps actor pools from occupying the CPUs that join-compatible branch schemas require.

Known text inputs use a direct `TxtSplitActor` graph that targets up to 8 actors, limited by the available CPUs. Shared preflight can reduce this default when the same batch job requires other actor pools or task reservations.

If you set `BatchTuningParams` worker counts or direct `node_overrides`, those requests and required task reservations must fit the available Ray CPU and GPU budget. The library validates the final plan before submitting work and raises an error when it is infeasible. Reduce `*_workers` or per-node concurrency, or wait for shared-cluster capacity before retrying.

For a direct Ray actor-pool tuple, such as `(minimum, maximum)` or `(minimum, maximum, initial)`, validation uses `maximum` as the requested pool size.

### Override worker counts

The library does not read environment variables to set worker counts or CPU and GPU totals. `CUDA_VISIBLE_DEVICES` still controls which GPUs Ray can see. To limit GPU count, start a Ray cluster with a restricted GPU set, for example `CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1`.

Set explicit worker counts with batch-mode CLI flags or with `BatchTuningParams` on `.extract()` and `.embed()`.

The following CLI example sets worker counts for a batch ingest:

```bash
retriever ingest batch ./data/pdf_corpus \
  --pdf-extract-workers 4 \
  --page-elements-workers 3 \
  --ocr-workers 3 \
  --embed-workers 2
```

The following Python example passes the same worker counts through `BatchTuningParams`:

```python
from pathlib import Path

from nemo_retriever import create_ingestor
from nemo_retriever.common.params import BatchTuningParams

documents = [str(Path("data/multimodal_test.pdf"))]

chunks = (
    create_ingestor(run_mode="batch")
    .files(documents)
    .extract(
        batch_tuning=BatchTuningParams(
            pdf_extract_workers=4,
            page_elements_workers=3,
            ocr_workers=3,
        )
    )
    .embed(
        batch_tuning=BatchTuningParams(
            embed_workers=2,
        )
    )
    .ingest()
)
```

Related batch-size, CPU, and GPU-per-actor flags are documented in the [CLI ingest options](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md).

Use the Ray dashboard to verify the available-resource snapshot and the planned worker allocation when you tune throughput.

## Shared preflight for custom Ray Data graphs

`GraphIngestor` reserves source capacity automatically. For custom graphs, declare source capacity before calling `preflight_executors(...)`. Set `source_cpu_reservation=1` on each `RayDataExecutor` that will receive a filesystem path or glob. `source_cpu_reservation` must be a finite, non-negative CPU value. An executor that only receives an existing Ray dataset can omit the reservation.

```python
file_executor = RayDataExecutor(graph, source_cpu_reservation=1)
inline_executor = RayDataExecutor(graph)
preflight_executors([file_executor, inline_executor], cluster_resources)
```

The shared preflight records these reservations. NeMo Retriever Library rejects a later filesystem input when its executor lacks the required reservation. It rejects the input before it starts Ray work. Construct a new executor with `source_cpu_reservation=1`, and include it in a new shared preflight instead.
