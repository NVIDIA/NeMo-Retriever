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

When ingestion combines multiple extraction datasets, preflight also reserves 1 CPU for schema-normalization work. This keeps actor pools from occupying all CPU capacity before the datasets have compatible schemas.

Ray Data runs repartition steps as short tasks that each reserve 1 CPU. Repartition steps run before PDF extraction, page elements detection, table structure, and embedding when the matching batch size and repartition control are set. When a graph has repartition steps, preflight stops growing automatically sized actor pools before they reserve every CPU. It keeps the following CPUs free:

- 1 CPU for each repartition step, and 2 for each stage that needs the whole dataset in one batch.
- On hosts with at least 32 available CPUs, a margin of 10% of the available CPUs, rounded up, with a maximum of 8 CPUs.
- 1 CPU for the Ray driver and system processes.

For example, the default PDF graph on a single-GPU node with 32 CPUs keeps 3 + 4 + 1 = 8 CPUs free. Without this reserve, actor pools can reserve every CPU and repartition tasks cannot start, so no rows reach the GPU stages. Pools that you size explicitly, for example with `pdf_extract_workers` or `node_overrides`, are not reduced. If explicit pools leave less than the reserve, preflight logs a warning.

Each PDF extraction worker reserves 1 CPU by default. With the default PDFium renderer, page rasterization is single-threaded. To change it, set `pdf_extract_num_cpus` in batch tuning or pass `--pdf-extract-cpus-per-task` to the CLI.

On hosts with 2 or more GPUs, the default page elements, OCR, and embedding pools give every GPU the same set of actors. This applies when the graph runs all three stages locally and has no other local GPU stage. Graphs with page elements off, captioning, table structure, Nemotron Parse, audio, or a remote endpoint for one of these stages keep the previous defaults. Any explicit `page_elements_workers`, `ocr_workers`, `embed_workers`, `gpu_page_elements`, `gpu_ocr`, or `gpu_embed` value also keeps the previous defaults for the other two stages. Ray places actors with fractional GPU requests on the first GPU with room, one pool after another. Each default GPU share fits exactly its actor count in the room that earlier pools leave, and the shares add up to 1.0 per GPU, so the pools spread across all GPUs:

| Smallest GPU memory | Page elements | OCR | Embedding |
|---|---|---|---|
| 72 GiB or more | 4 actors × 0.21 GPU | 2 actors × 0.06 GPU | 1 actor × 0.04 GPU |
| Less than 72 GiB, unknown, or GPUs on other Ray nodes | 1 actor × 0.51 GPU | 1 actor × 0.25 GPU | 1 actor × 0.24 GPU |

These pools apply only to the PDF graph. Text, HTML, image, audio, video, and automatic-detection graphs keep the previous defaults. GPU memory means the smallest free memory among the GPUs that Ray uses, as named by `CUDA_VISIBLE_DEVICES`. The library logs the chosen set and the reason at INFO. After the GPU actors start, it logs a WARNING if any of these pools has pending actors or an uneven number of actors on each GPU.

When these default pools apply, the page elements and embedding stages also skip their repartition steps, so rows arrive in the blocks that the previous stage produced. Ray Data runs too few of these repartition tasks to feed the larger GPU pools. Set `page_elements_repartition` or `embed_repartition` to keep a step.

A single GPU keeps 3 page elements and 3 OCR actors at 0.1 GPU each and 1 embedding actor at 0.2 GPU. Preflight grows each CPU-only pool to 2 actors per GPU or one eighth of its target, whichever is smaller, then grows the GPU pools, then shares the remaining CPUs.

The CPU stage that reshapes content before embedding (`CollapseContentToPageRows` or `ExplodeContentToRows`) targets 2 actors per GPU on hosts with 2 or more GPUs and 1 actor otherwise. Each reshape actor runs on one core, so a single actor can limit a multi-GPU run. Preflight grows the reshape pool after the GPU pools, so hosts with few CPUs per GPU keep 1 reshape actor. GPU shares only control placement; they do not limit GPU memory.

Known text inputs use a direct `TxtSplitActor` graph that targets up to 8 actors, limited by the available CPUs. Shared preflight can reduce this default when the same batch job requires other actor pools or task reservations.

Explicit `BatchTuningParams` worker counts and integer `node_overrides` concurrency values must fit the available Ray CPU and GPU budget, including required task reservations. For an explicit Ray actor-pool tuple, validation uses `minimum` for `(minimum, maximum)` or `initial` for `(minimum, maximum, initial)`. The library preserves the tuple's `maximum` so Ray can grow the pool as capacity becomes available.

Only automatically derived concurrency can be reduced. A node override that sets `batch_size` or other settings without `concurrency` leaves automatic concurrency sizing enabled. The library raises an error before submitting work if the plan is infeasible. Reduce `*_workers`, per-node concurrency, or a tuple's starting size, or wait for shared-cluster capacity before retrying.

### Override worker counts

The library does not read environment variables to set worker counts or CPU and GPU totals. `CUDA_VISIBLE_DEVICES` still controls which GPUs Ray can see. To limit GPU count, start a Ray cluster with a restricted GPU set, for example `CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1`.

Set explicit worker counts with batch-mode CLI flags or with `BatchTuningParams` on `.extract()` and `.embed()`.

The following CLI example sets worker counts for a batch ingest. Replace
`/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs \
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
            ocr_inference_batch_size=8,
        )
    )
    .embed(
        batch_tuning=BatchTuningParams(
            embed_workers=2,
            embed_reshape_workers=4,
        )
    )
    .ingest()
)
```

Related batch-size, CPU, and GPU-per-actor flags are documented in the [CLI ingest options](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md).

Use the Ray dashboard to verify the available-resource snapshot and the planned worker allocation when you tune throughput.

### Tune embedding preparation and remote requests

Embedding preparation, actor batches, and HTTP requests have separate concurrency and batch-size controls. Pass the following settings through `.embed(...)` in the Python API.

| Setting | Behavior |
| --- | --- |
| `batch_tuning.embed_reshape_workers` | Sets a positive concurrency limit for the CPU stages that collapse content into page rows or expand content into element rows before embedding. |
| `batch_tuning.embed_workers` | Sets the number of Ray embedding actors. |
| `batch_tuning.embed_batch_size` | Sets the number of rows in each Ray embedding actor batch. |
| `inference_batch_size` | Limits the number of inputs in each embedding inference batch. |
| `embed_inference_batch_size` | Adds a limit on the number of image-bearing inputs in each remote embedding HTTP request. |
| `nim_http_max_concurrent` | Caps concurrent remote embedding HTTP requests within each actor batch. |

Set `embed_reshape_workers` when CPU content preparation limits throughput. This setting applies to the named `CollapseContentToPageRows` and `ExplodeContentToRows` graph stages when present. When you do not set it, preflight sizes these stages from the GPU count and the CPU budget.

For remote image-bearing inputs, the request size is at most `min(inference_batch_size, embed_inference_batch_size)`. For example, `.embed(inference_batch_size=32, embed_inference_batch_size=8)` sends at most 8 image-bearing inputs per request. Select a multimodal model that your endpoint supports, as described in [multimodal embedding](embedding.md).

Benchmark these settings with representative documents. Monitor CPU capacity, endpoint GPU memory, HTTP errors, and output counts as you increase concurrency or batch sizes.

### Control row repartitioning

!!! note "Unreleased controls"

    These repartition controls are under development. They are not available
    in current NVIDIA `main` or a released package.

In batch mode, the following `BatchTuningParams` controls enable size-based Ray
repartitioning before each stage. Each control is on by default, except that `page_elements_repartition` and
`embed_repartition` are off when the default per-GPU pools apply. Set a control to `True` or `False` to
override its default.

| Stage | Control | Row-batch setting |
| --- | --- | --- |
| Page Elements | `page_elements_repartition` | `page_elements_batch_size` |
| Table Structure | `table_structure_repartition` | `table_structure_batch_size` |
| Embedding | `embed_repartition` | `embed_batch_size` |

When enabled and a stage has a positive batch size, the library uses that size
as `target_num_rows_per_block` before the stage runs. Set a control to `False`
to omit this derived block-size target. The stage still uses its configured
batch size and worker settings. Required grouping in other graph stages still
applies.

The following example disables these three block-size targets. Replace
`/path/to/your/document.pdf` with a PDF file that you supply.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import BatchTuningParams

chunks = (
    create_ingestor(run_mode="batch")
    .files(["/path/to/your/document.pdf"])
    .extract(
        batch_tuning=BatchTuningParams(
            page_elements_repartition=False,
            table_structure_repartition=False,
        )
    )
    .embed(
        batch_tuning=BatchTuningParams(
            embed_repartition=False,
        )
    )
    .ingest()
)
```

For typed ingest plans, set the extraction controls in
`IngestExtractBatchOptions` and the embedding control in
`IngestEmbedBatchOptions`. These optional fields default to `None`, which
resolves to the defaults described above: on, except that page elements and
embedding skip their repartition steps when the default per-GPU pools apply.

Compare throughput, memory use, errors, and extracted outputs against a
representative baseline after changing these controls.

### Tune PDF splitting separately from page extraction

PDF splitting runs on the CPU before page extraction. Each splitter actor
opens input PDF documents and emits single-page PDF packets. Page extraction
then interprets and renders those packets. The splitter and extractor have
separate worker and batch controls.

In native Ray batch mode, use the following splitter settings:

| Python batch tuning | CLI option | Default | Behavior |
| --- | --- | --- | --- |
| `pdf_split_workers` | `--pdf-split-workers` | `None` | Keeps the existing single splitter actor when unset. A positive value selects independent Ray splitter actors. |
| `pdf_split_batch_size` | `--pdf-split-batch-size` | `1` | Sets input PDF documents per splitter actor batch. This is not rendered pages per batch. |

Increase splitter workers when PDF input work queues up while page extraction
and GPU model stages wait for input. Independent actor processes can split
different documents concurrently. Increasing the document batch size alone
does not add splitter processes or distribute one large PDF across workers.

The following example selects four splitter actors with one document per
splitter batch, retaining the other pipeline defaults:

```bash
retriever ingest batch /path/to/your/pdfs \
  --pdf-split-workers 4 \
  --pdf-split-batch-size 1
```

The equivalent Python configuration passes splitter tuning to `.extract()`:

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import BatchTuningParams

chunks = (
    create_ingestor(run_mode="batch")
    .files(["report.pdf", "appendix.pdf"])
    .extract(
        extraction_mode="pdf",
        batch_tuning=BatchTuningParams(
            pdf_split_workers=4,
            pdf_split_batch_size=1,
        ),
    )
    .embed()
    .ingest()
)
```

More actors and larger document batches can increase CPU use, open document
state, buffered page packets, and Ray object-store memory. Leave capacity for
file loading and downstream stages. Monitor CPU scheduling, memory, queue
depths, completed page counts, and end-to-end throughput before increasing
these settings further. No particular worker count guarantees higher GPU
utilization or ingestion throughput.

Use `pdf_extract_workers` and `pdf_extract_batch_size` for downstream page
extraction. The splitter settings do not change CUDA page-stream concurrency
or model inference batch sizes. Typed ingest plans expose both splitter
settings in `IngestExtractBatchOptions`.

### Set the PDF extraction block size

In batch mode, the library repartitions pages before PDF extraction, targeting
`pdf_extract_batch_size` rows per Ray block. Each extraction task processes one
block. The target is best effort, so some blocks can hold fewer rows. To set the
block-size target separately from the batch size, use
`pdf_extract_rows_per_block` in `BatchTuningParams`. The default is `None`,
which keeps the target equal to `pdf_extract_batch_size`.

Larger blocks create fewer extraction tasks for the same pages. On long batch
jobs with many small tasks, fewer tasks can keep downstream stages busier. A
slow page costs more when it shares a larger block, because the other pages in
that task wait for it.

The following example keeps 4 pages per extraction batch and targets 16 pages
per extraction task. Replace `/path/to/your/document.pdf` with a PDF file that
you supply.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import BatchTuningParams

chunks = (
    create_ingestor(run_mode="batch")
    .files(["/path/to/your/document.pdf"])
    .extract(
        batch_tuning=BatchTuningParams(
            pdf_extract_batch_size=4,
            pdf_extract_rows_per_block=16,
        )
    )
    .embed()
    .ingest()
)
```

For typed ingest plans, set `pdf_extract_rows_per_block` in
`IngestExtractBatchOptions`.

## Evaluate GPU-resident PDF ingestion

The experimental `pdf_renderer="cuda_rdt"` mode moves page RGB directly between
renderer and local GPU model actors through Ray Direct Transport. CPU PDF
parsing and command preparation remain. Native shared-memory scene handles
avoid copying large command buffers into Python `bytes`. The path requires a
single Linux host, the same worker user, and a shared process namespace with
access to the scene owners' `/proc` file descriptors. Refer to
[GPU-resident Ray batch rendering](nemo-retriever-api-reference.md#gpu-resident-ray-batch-rendering)
for dependencies, configuration, and supported outputs.

Renderer children share their owning model pool's GPU reservation. These
reservations control scheduling, not GPU memory consumption. Rendered page
buffers remain owned until ingestion drains, so monitor GPU memory as well as
throughput when increasing page batch sizes or worker counts. Separate model
pools can also require GPU-to-GPU transfers.

Each GPU renderer producer shares a pool of 16 page workers across up to
16 concurrent batch requests. Each page worker uses its own nondefault
PyTorch CUDA stream. Before publishing a page tensor, the worker waits for
that page's stream to complete. This host wait does not copy RGB pixels to
CPU memory or synchronize every page. These internal limits are separate from
page batch size and do not guarantee simultaneous kernel execution.

Measure complete ingestion through index writing and retain the same PDF
inputs, model settings, and text embedding configuration. Record page and
indexed-row counts, ingestion pages per second, GPU memory, and stage timings.
GPU utilization alone does not identify the limiting stage.

The encoded renderers and GPU-resident mode present different model pixels
because the latter avoids JPEG compression. Compare extracted outputs and
retrieval accuracy before accepting a throughput result.

### Bound source ownership through final consumption

For native `cuda_rdt`, `CUDA_PDF_RGB_OWNER_BYTES` optionally enables explicit
producer ownership of published RGB references. Set it in the NRL driver
environment before ingestion. Unset or `0` disables owner leases. Positive
integer values must be at least 3,145,728 bytes (3 MiB); invalid values fail
setup. The driver forwards the value to renderer producers.

The value limits charged outstanding bytes per producer. Each page's RGB
byte size rounds up to a 3 MiB multiple, and a batch charges the sum of its
pages. This accounting applies even without a source RGB pool. The limit is
independent of `CUDA_PDF_RGB_POOL_BYTES` and
`CUDA_PDF_RGB_RECEIVER_POOL_BYTES`; it does not allocate another image slab
or bound all GPU memory used by models and transport.

Before rendering, the producer reserves its batch charge. A batch exceeding
the entire owner limit fails before rendering. Admission uses a first-in,
first-out (FIFO) queue of at most 1,024 pending requests per producer. Retries retain
the same request identity, batch charge, and page count, so newer requests
cannot bypass a queued request. Queue overflow raises an error.

If capacity is unavailable, admission returns busy immediately so actor threads
remain available for acknowledgements. The caller retries with bounded backoff.
A 120-second idle watchdog resets when the producer reports newly acknowledged
pages from final consumers. Duplicate acknowledgements do not count as progress.
Total capacity waiting can exceed 120 seconds while consumption progresses;
the watchdog does not limit an admitted batch's rendering duration. On error,
the caller attempts to cancel its queued request without masking the original
failure. Cancellation removes pending admission metadata, not published images.

After publication, the producer retains the actual source `ObjectRef`.
OCR, the final supported image consumer, waits for its scoped GPU work before
acknowledging page indices. The source batch releases its reference and charge
only after every page index has been acknowledged. This completion wait also
applies when receiver pooling is disabled. Source command-scene ownership
through replay remains separate and unchanged.

Producer status reports owner limits, active and peak charged bytes, active
and retained batches, pending pages, acknowledged pages, completed batches,
queued requests, and capacity rejections.
Use those counters to distinguish outstanding consumers from ordinary GPU
allocation pressure. No image is expired by age, and no unbounded reference
cache is introduced. Capacity or ownership failures do not select CPU
rasterization or host RGB transport.

Synthetic GPU ownership stress checks finish with zero retained owner entries,
charged bytes, and pending pages. Exact RGB gates also pass with GPU allocation
overflow, slot reuse, and terminated encoding actors before consumer fetch;
their source leases and owner entries drain. Four guarded Page Elements/OCR
pages pass without image tensor readback or encoding.

Normal pipeline completion additionally requires zero active owner batches
and queued admission requests; an undrained ledger raises an ownership error. This completion check does not
replace an original model failure. The experimental mode does not guarantee
replay or retry recovery after final acknowledgement releases a source.
A full native CLI trial with owner leases fails with illegal GPU-memory access
under NVIDIA Multi-Process Service (MPS). The trial uses stock NIXL transport,
an upstream Ray nightly, readiness-checked transport disabled, and a 64 GiB
owner limit per producer across eight GPUs. No capacity rejections are recorded,
so the failure does not indicate exhaustion of the configured owner limit.
The originating kernel remains unknown; the observations do not establish that
the ownership protocol caused or resolved the fault.

A later experimental candidate, revision `f1d09da5`, completes all 213,538
DC10K pages on eight GPUs without MPS. Ingestion takes 623.5897 seconds, with
zero page RGB readback or CPU raster calls. Owner, admission, and RGB lease
ledgers drain at completion. The following results compare this and a
subsequent candidate with the saved CPU PDFium renderer baseline. Inference
models run on GPUs in all cases.

| Metric | Saved PDFium baseline | Initial ownership candidate | Subsequent kernel candidate |
| --- | ---: | ---: | ---: |
| Ingestion pages per second | 385.9195 | 342.4335 | 351.6552 |
| Retrieval queries per second | 5.5287 | 4.9809 | 4.6714 |
| nDCG@10 | 0.63006314 | 0.63086199 | 0.63171757 |
| Recall@10 | 0.76169265 | 0.76391982 | 0.76391982 |

For the initial ownership candidate, retrieval against the completed index
succeeds after a telemetry failure requires a query-only retry. It completes all 1,347 queries with no unmapped
hits; model warmup of 45.4222 seconds is separate from query throughput. Neither aggregate score
decreases in this run, but that observation does not establish accuracy
equivalence. Ingestion remains slower than the saved baseline.

The initial natural run records zero busy admissions. A subsequent candidate
with a CUDA kernel optimization and unchanged NRL runtime completes all
213,538 pages on eight GPUs without MPS in 607.2369 seconds. It records 28,413
busy admission returns, exercising owner backpressure on natural input. All
source, owner, and admission ledgers drain, with zero page RGB readback or CPU
raster calls. Retrieval completes all 1,347 queries with no unmapped hits;
its separate model warmup takes 47.1631 seconds.

Both candidate snapshots remain below the saved CPU renderer ingestion
throughput. Their aggregate retrieval scores do not decrease relative to that
baseline, but no accuracy-equivalence margin has been established. The 90%
whole-CLI GPU utilization target remains unmet. These experimental snapshots
do not select a new default; the mode remains disabled by default. Natural
backpressure and successful drain do not establish general GPU memory safety,
liveness under all workloads, or retry recovery after final acknowledgement.
Earlier MPS faults and their originating kernel remain unresolved.

### Publish only completed RGB tensors

For native `cuda_rdt`, set `CUDA_PDF_READY_TRANSPORT=1` in the NRL driver
environment to enable experimental readiness-checked RGB publication. Unset or
`0` retains stock Ray NIXL transport. The driver accepts only `0` or `1` and
registers the selected transport before creating renderer actors.

This option requires the audited Ray version 2.56.1 exactly. Incompatible Ray
versions fail setup. The implementation registers `NRL_READY_NIXL`, a separate
transport that reuses Ray's NIXL transfers, descriptor ownership, and cleanup.
It checks the exact original contiguous CUDA `uint8` HWC RGB tensor and a
completed CUDA event on the same device before extracting publication metadata.
Missing, mismatched, or pending completion proofs raise errors without a
host-image fallback.

Each page records and completes its own CUDA event before publication. The
readiness-checked metadata path avoids the stock device-wide publication wait,
which can include unrelated streams. It does not eliminate page completion
waits or receiver model-scope synchronization. Original tensor and pool leases
must still remain valid until all consumers and transfers finish.

Source and receiver arenas register with the transport manager selected for
the transferred reference. Pool settings, model configuration, actor counts,
and the default renderer remain unchanged. Page RGB stays on the GPU.

Bounded checks pass 384 exact RGB renders and four real-model pages. One
full-corpus trial fails with illegal GPU-memory access that propagates to OCR
and renderer clients. A later trial, after publication-thread CUDA device
binding, fails with `RayReferenceCountingAssertionError` and a deleted GPU image
reference during Page Elements fetch. That trial logs no illegal GPU-memory
access, but it also does not complete.

Publication-thread context binding passes bounded fresh-thread checks. Those
checks do not establish broad memory or reference-lifetime safety. The
originating kernel and causal sources of the full-run failures remain unresolved.
This option is not validated for full ingestion. Neither failed trial provides
a valid full-corpus throughput or accuracy result. Stock transport remains the
default. Earlier DC10K results do not establish a speedup or GPU utilization
improvement for this transport.

### Reuse registered GPU RGB buffers

For `cuda_rdt`, the following independent settings reserve reusable RGB
storage. Set them in the NRL driver environment before ingestion. The driver
validates the values and passes them to the relevant actors in the supported
single-host native Ray path.

| Environment variable | Allocation scope | Purpose |
| --- | --- | --- |
| `CUDA_PDF_RGB_POOL_BYTES` | Each GPU renderer producer. | Reuses registered source RGB storage. |
| `CUDA_PDF_RGB_RECEIVER_POOL_BYTES` | Each Page Elements and OCR GPU actor. | Reuses registered destination RGB storage. |

Both settings default to disabled when unset or `0`. Positive values must be
at least 3,145,728 bytes (3 MiB); invalid values fail setup. Source and receiver
pooling can be enabled independently.

The pool divides its capacity into 3 MiB slots and rounds capacity down to a
whole number of slots. Each slot fits a 1,024 × 1,024 RGB `uint8` image or a
smaller image. For example, `100663296` bytes reserves 96 MiB, or 32 slots,
per configured actor. These capacities are not shared pipeline budgets.
Receiver memory multiplies by Page Elements and OCR actor counts. For example,
32 Page Elements actors and 16 OCR actors create 48 receiver pools, independent
of any source pools on eight GPU producers. Budget memory separately on each
GPU for its model actors, receiver pools, and producer pool.

NRL registers the slab with Ray NIXL once for its lifetime. Page tensors reuse
views of that storage, avoiding repeated per-page registration and associated
remote metadata changes. Pool ownership must remain valid until all GPU and
transport consumers finish. If a page exceeds one slot or no slot is free,
rendering uses an ordinary GPU allocation. This allocation fallback does not
copy RGB pixels to the CPU or select CPU rasterization.

Receiver actors allocate target tensors from complete CPU-side batch-shape
metadata and register the targets with Ray through `set_target_for_ref`
before fetching RGB. This reuses destination storage as well as source
storage. Each model scope retains the original target leases, then waits for
its GPU context before releasing them. This completion wait transfers no
image pixels to host memory. Missing shape metadata or invalid received
CUDA RGB tensors raise errors; these failures do not select a host-image path.
Receiver slot exhaustion or oversized pages use ordinary GPU allocations.

The NVTX ranges `nrl.cuda_rdt::receive_targets` and
`nrl.cuda_rdt::receive_scope_ready` identify receive-target preparation and
the scope's completion wait. These ranges can include allocation or waiting
time and do not by themselves measure GPU kernel execution.

Renderer status includes pool capacity, slot size, active and peak usage,
allocation and release counts, and counts of exhausted or oversized requests.
Monitor those counters alongside GPU memory and pipeline throughput. Increasing
the reserved capacity consumes memory that models and other GPU stages need.

Transport microbenchmarks do not establish complete ingestion performance.
Bounded pooled-output checks pass exact RGB comparisons and delayed-consumer,
slot-exhaustion, and slot-reuse tests without RGB readback or CPU rasterization.
A guarded local-model test also passes without image tensor readback.
Receiver pooling also passes bounded exact-pixel, cross-GPU lifetime and
slot-reuse checks, plus a guarded local-model test. A full DC10K receiver-pool
run completes ingestion and retrieval evaluation with zero CPU raster calls
and page RGB readback. These checks do not establish exact pixel equality
for every corpus page or identical model outputs. The historical DC10K
results below precede both pool options and describe that earlier revision.

### DC10K renderer comparison

#### Initial GPU-resident snapshot

A completed DC10K evaluation processes 10,000 PDFs containing 213,538 pages
through native NRL ingestion, text embedding, index writing, and retrieval.
The `cuda_rdt` run uses all eight GPUs and reports zero page RGB readback
and zero CPU raster calls. The evaluated sources are NRL `be908600` and
renderer `df7d8a5`.

The following results compare that run with the saved default PDFium renderer
baseline. Both pipelines use GPU model inference; CPU refers to rasterization
in the baseline, not to the entire pipeline.

| Metric | Default PDFium renderer | CUDA renderer with Ray Direct Transport |
| --- | ---: | ---: |
| Ingestion pages per second | 385.9195 | 347.2670 |
| Retrieval queries per second | 5.5287 | 5.3764 |
| nDCG@10 | 0.63006314 | 0.63241428 |
| Recall@10 | 0.76169265 | 0.76391982 |

CUDA ingestion is approximately 10% slower in this comparison. Observed
retrieval quality is similar, with slightly higher aggregate nDCG@10 and
Recall@10 for this run. These measurements do not establish exact accuracy
equivalence, statistical significance, or a speedup. Paired analysis covers
1,347 queries; 95% bootstrap confidence intervals for both quality differences
include zero. The baseline is a saved run rather than a freshly repeated
baseline measurement.

Rendering every input page does not imply that every page produces index
rows. The CUDA index contains 9,987 documents and 211,621 pages; the baseline
contains 9,986 documents and 211,603 pages. No baseline indexed document is
missing, but 15 baseline indexed pages lack CUDA rows, and 33 additional pages
have CUDA rows. These output differences require consideration alongside the
aggregate retrieval metrics.

Full-corpus completion and retrieval metrics are separate from the bounded
64-page exact RGB gate. These results do not establish pixel equality for
every DC10K page or resolve the previously observed concurrent-profiling
memory failures. Repeat validation after changing hardware, dependencies,
concurrency, or model settings.

#### Reference-lifetime fix snapshot

A later completed run evaluates NRL `4ea60d78` with renderer `17ee48e`,
including the native pandas metadata reference-lifetime fix. It processes all
10,000 PDFs and 213,538 pages with eight GPU producers, zero CPU raster calls,
and zero page RGB readback. The final ownership audit reports all source
buffer leases released.

The following measurements describe this revision separately from the initial
snapshot and use the same saved default PDFium renderer baseline. Both
pipelines run their inference models on GPUs.

| Metric | Saved default PDFium renderer | Reference-lifetime fix snapshot |
| --- | ---: | ---: |
| Ingestion pages per second | 385.919537 | 422.444051 |
| Retrieval queries per second | 5.5287 | 5.133720 |
| nDCG@10 | 0.63006313719 | 0.63221404349 |
| Recall@10 | 0.761692650334 | 0.763919821826 |

The whole ingestion CLI completes in 505.4823 seconds. Its ingestion throughput
is approximately 9.46% higher than the saved baseline. Retrieval evaluates all
1,347 queries in 262.3828 seconds; its separate warmup takes 46.0295 seconds.
The mean whole-CLI GPU utilization is 56.60%, calculated from time-weighted
one-second NVML samples across all eight GPUs. The 90% utilization target
remains unmet.

Observed aggregate retrieval scores remain similar. Paired 95% confidence
intervals for CUDA minus baseline include zero: [-0.00331831, 0.00757341]
for nDCG@10 and [-0.00371195, 0.00816630] for Recall@10. These results do not
establish exact accuracy equivalence or a statistically significant gain.

The index contains 306,930 rows from 9,987 documents and 211,621 pages. No
baseline indexed document is missing. However, 16 baseline indexed pages lack
CUDA rows, and 34 additional pages have CUDA rows. Rendering completion and
aggregate quality therefore do not establish identical extraction coverage.

An earlier pooled snapshot remains the fastest observed run at 461.659 pages
per second, with 61.09% mean whole-CLI GPU utilization. The reference-lifetime
fix run is approximately 8.5% slower than that snapshot. These separate runs
do not isolate an intrinsic performance effect of the metadata block format.

The exact 64-page RGB gate remains distinct from full-corpus processing and
retrieval validation. The previously observed illegal GPU-memory accesses and
native heap corruption during concurrent profiling remain unresolved. This
completed run does not establish production readiness or a 90% utilization
result.

## Profile batch ingestion with Nsight Systems

Batch mode emits NVIDIA Tools Extension (NVTX) ranges from the driver and Ray
actor processes through the `nvtx` Python package, which is installed with NeMo
Retriever Library. These host annotations do not require PyTorch or a GPU.
Use NVIDIA Nsight Systems to correlate these semantic stages
with CUDA and operating system activity. Start with a representative, bounded
input because profiler tracing can increase runtime and report size.

Run a batch ingest under `nsys profile` as follows:

```bash
nsys profile \
  --trace=nvtx,cuda,osrt \
  --sample=none \
  --wait=all \
  --output=nrl-batch-profile \
  -- retriever ingest batch /path/to/your/pdfs
```

The command writes `nrl-batch-profile.nsys-rep`. Open the report in the Nsight
Systems graphical interface, or run `nsys-ui nrl-batch-profile.nsys-rep` on a
workstation with the interface installed. Search the timeline for `nrl.batch::`,
`nrl.model::`, and `nrl.remote::` to find the following ranges.

| Range | Interpretation |
| --- | --- |
| `nrl.batch::pipeline.ingest` | One complete `RayDataExecutor` run on the driver, including `return_results=False` summary mode. Use this range for batch pipeline wall-clock time. Driver-side setup before the run, such as building and normalizing manifest branch datasets, is outside this range. |
| `nrl.batch::pdf_split.batch` | One PDF split actor call that splits a batch of documents into single-page PDFs. |
| `nrl.batch::pdf_extract.batch` | One PDF extraction actor call that extracts text and images from a batch of pages. |
| `nrl.batch::page_elements.startup` | One Page Elements actor constructor. A backend that loads lazily can perform more startup work in the first batch. |
| `nrl.batch::page_elements.batch` | One Page Elements actor processing call. |
| `nrl.batch::table_structure.startup` | One Table Structure actor constructor. A backend that loads lazily can perform more startup work in the first batch. |
| `nrl.batch::table_structure.batch` | One Table Structure actor processing call. |
| `nrl.batch::ocr.startup` | One OCR actor constructor. A backend that loads lazily can perform more startup work in the first batch. |
| `nrl.batch::ocr.batch` | One OCR actor processing call. |
| `nrl.batch::embedding.startup` | One embedding actor constructor. A local embedding model loads here, including its vLLM engine, so this range includes the model load time. |
| `nrl.batch::embedding.batch` | One embedding actor processing call. |
| `nrl.batch::ray.materialize` | Materialization of a terminal Ray dataset into a pandas DataFrame. This range can include lazy upstream execution. |
| `nrl.batch::pipeline.terminal_stream` | Consumption of the lazy upstream pipeline and streaming of its records to the vector database backend. This range is not pure vector database time. |
| `nrl.batch::result.concat` | Concatenation of retained frames when the graph ends at a streaming vector database sink and at least one batch was consumed. This range is absent when the sink has downstream nodes or the input has no batches. |

Local and remote (endpoint-backed) Page Elements, Table Structure, OCR, and embedding actors emit the same ranges; for remote actors, the batch range includes network requests to the endpoint.

Local model operations also emit stable, decorator-based ranges. These ranges
wrap existing high-level methods, so they can include CPU input preparation,
one or more backend calls, and output processing. Compare them with CUDA
activity in the Nsight Systems timeline to isolate device execution.

| Range | Interpretation |
| --- | --- |
| `nrl.model::llama_nemotron_embed_vl.embed` | One local document text embedding operation. |
| `nrl.model::llama_nemotron_embed_vl.embed_queries` | One local query embedding operation. |
| `nrl.model::llama_nemotron_embed_vl.embed_images` | One local image embedding operation. |
| `nrl.model::llama_nemotron_embed_vl.embed_text_image` | One local paired text-image embedding operation. |
| `nrl.model::ocr_v2.invoke` | One end-to-end Nemotron OCR v2 wrapper call, including input conversion and all backend calls. This is the model used by the current local OCR actor. |
| `nrl.model::page_elements.invoke` | One local Page Elements model call, including the initialization check and model execution. Output scaling runs in the separate `postprocess` call, outside this range. |
| `nrl.model::table_structure.invoke` | One local Table Structure operation, including input preparation, model execution, and output scaling. |
| `nrl.model::vlm_captioner.caption_batch` | One local visual language model batch captioning operation. |
| `nrl.model::parakeet_ctc.decode_batch` | One local Parakeet CTC chunk-batch operation, including feature preparation, model execution, and greedy decoding. |

Remote NVIDIA Inference Microservice (NIM) calls emit
`nrl.remote::nim.request`. This range covers one HTTP request and its retry and
backoff lifecycle. Compare it with the enclosing actor batch range to separate
remote service wait time from local preparation and response processing.

Ranges from concurrent Ray actors can overlap. Their durations include CPU
preparation, waits, and calls into inference backends. They do not represent
CUDA kernel time by themselves. In the timeline, compare range boundaries with
the CUDA activity rows to identify the stage that remains active near pipeline
completion and to find idle gaps between batches.

You can also print an aggregate NVTX summary:

```bash
nsys stats --report nvtx_sum nrl-batch-profile.nsys-rep
```

The summary adds durations from all matching ranges. With concurrent actors,
the summed duration can exceed wall-clock time, so use the timeline to reason
about overlap and the critical path.

## Tune remote OCR request batching

Remote OCR batches cropped regions across the page rows supplied to one OCR actor call. This behavior applies to in-process, batch, and service ingestion with a remote OCR NIM. It preserves page and region output order.

For in-process and batch ingestion, set `BatchTuningParams.ocr_inference_batch_size` through `.extract(batch_tuning=...)` to limit the cropped regions in each OCR HTTP request. This value takes precedence over `ExtractParams.inference_batch_size` for OCR. When you do not set it, OCR uses `ExtractParams.inference_batch_size`, which defaults to `8`.

Set `ExtractParams.remote_retry.remote_max_pool_workers` through `.extract(remote_retry=...)` to cap concurrent remote requests per actor. Each submitted group contains at most `ocr_inference_batch_size * remote_max_pool_workers` cropped regions, using the effective OCR batch size. Requests for different pages can overlap when a group contains multiple requests.

The NIM HTTP client applies the configured retry policy. If a group spans multiple pages and all concurrent failures are HTTP `400`, `413`, or `422`, OCR retries each page separately to isolate input-specific failures. Any authentication failure, exhausted rate-limit retry, transport failure, server error, or invalid response count prevents page-level retries for that group. If a remote request or response processing fails a page, OCR discards that page's results, including results from earlier groups, and reports the page error.

The execution mode determines how page rows reach the OCR actor and how many actors can submit requests.

| Mode | OCR batching and concurrency |
| --- | --- |
| `inprocess` | OCR batches across the page rows in the current graph stage, including pages of one PDF. It uses the remote request pool without Ray. |
| `batch` | In the dedicated PDF graph, Ray supplies page-row batches to OCR actors. `ocr_inference_batch_size` also sets this Ray row-batch size, while `ocr_workers` controls the number of actors. Remote request limits apply separately within each actor. |
| `service` | Each service worker runs an in-process graph. The whole-document route splits the PDF inside that graph, so OCR can batch across its pages. Service worker counts control concurrent work items. |

In service mode, use `.extract(inference_batch_size=...)` to set the request batch size. The default service policy does not accept client overrides for `batch_tuning` or `remote_retry`. The server controls NIM endpoints and credentials. Across Ray actors or service workers, total request concurrency can exceed the per-actor limit.

A smaller OCR inference batch size creates more, smaller HTTP requests. In Ray batch mode, it also reduces the number of page rows available to each actor call. A larger value creates fewer, larger requests. Benchmark both latency and throughput with representative documents, and monitor NIM GPU memory, HTTP `429` responses, errors, and output counts. Overlapping HTTP requests does not establish concurrent GPU execution; the NIM controls backend queueing and execution.

## Shared preflight for custom Ray Data graphs

`GraphIngestor` reserves source capacity automatically. For custom graphs, declare source capacity before calling `preflight_executors(...)`. Set `source_cpu_reservation=1` on each `RayDataExecutor` that will receive a filesystem path or glob. `source_cpu_reservation` must be a finite, non-negative CPU value. An executor that only receives an existing Ray dataset can omit the reservation.

```python
file_executor = RayDataExecutor(graph, source_cpu_reservation=1)
inline_executor = RayDataExecutor(graph)
preflight_executors([file_executor, inline_executor], cluster_resources)
```

The shared preflight records these reservations. NeMo Retriever Library rejects a later filesystem input when its executor lacks the required reservation. It rejects the input before it starts Ray work. Construct a new executor with `source_cpu_reservation=1`, and include it in a new shared preflight instead.
