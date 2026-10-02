# cuDF direct-transport study

Date: 2026-09-18
Host: one NVIDIA H100 NVL (95,830 MiB), Ray 2.56.1
cuDF environment: `.venv-cudf`, `cudf-cu13==26.8.1`
Baseline environment: `.venv` (the locked project environment was left unchanged)

## Unchanged NRL baselines

| Dataset | Files | Pages | Ingest seconds | Pages/s | Queries | p50 ms | p95 ms | NDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BO767 | 767 | 54,730 | 2,100.185 | 26.06 | 991 | 41.347 | 50.580 | 0.752057305 | 0.852674067 | 0.899091826 |
| ViDoRe computer science | 2 | 1,360 | ~314* | ~4.331* | 1,290 | 32.672 | 65.934 | 0.712831866 | 0.603659327 | 0.734666436 |

`*` ViDoRe's comparable complete harness ingest phase was approximately 314
seconds, measured from phase-event timestamps with one-second resolution. Its
internal Ray Data execution took 277.54 seconds. The initial offline query
phase found the dataset cache but not one Transformers dynamic-model file.
Evaluation was resumed against the unchanged index through the same NRL query
and BEIR functions after resolving that cache file; ingestion was not repeated.

The BO767 profile exposed the main opportunity: immediately before the writer,
Ray repartitioned the heterogeneous result into one 50.9 GiB pandas block. The
actual stored vector payload is about 0.68 GB. A narrow GPU island should move
only row IDs and vectors, leaving images, text, and metadata on the CPU side.

## Transport microbenchmark

Payload: 10,000 rows x 2,048 float32 dimensions plus int64 row IDs, 82.0 MB,
10 measured iterations after warm-up. Every result passed row-count and vector
checksum validation.

| Path | Median latency | p95 latency | Logical GB/s | Speedup vs Ray default |
|---|---:|---:|---:|---:|
| Ray default cuDF serialization | 284.576 ms | 346.828 ms | 0.288 | 1.0x |
| Narrow island, same GPU actor | 1.767 ms | 1.851 ms | 46.397 | 161.0x |
| Custom cuDF CUDA IPC through Ray RDT | 9.160 ms | 10.727 ms | 8.952 | 31.1x |

The custom transport uses `DataFrame.device_serialize()`, CUDA allocation IPC
handles plus per-frame offsets, and `DataFrame.device_deserialize()`. It sends
device buffers without a GPU-to-host copy. The smoke test also transported a
100,000-row cuDF frame losslessly between two Ray actors.

## Full-index roundtrip and accuracy

The real NRL indexes were split into a CPU sidecar plus a narrow `row_id +
vector` cuDF frame, passed through the selected GPU path, recombined without
changing the fixed-size vector schema, and rebuilt with the unchanged
`IVF_HNSW_SQ` settings.

| Dataset/path | Rows | Narrow/custom transfer | Total rebuild | NDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| BO767 baseline | 82,953 | n/a | n/a | 0.752057305 | 0.852674067 | 0.899091826 |
| BO767 narrow | 82,953 | 0.4971 s** | 14.3297 s | 0.752057305 | 0.852674067 | 0.899091826 |
| BO767 custom IPC | 82,953 | 0.01250 s | 13.9515 s | 0.752057305 | 0.852674067 | 0.899091826 |
| ViDoRe baseline | 1,360 | n/a | n/a | 0.712831866 | 0.603659327 | 0.734666436 |
| ViDoRe narrow | 1,360 | 0.00723 s** | 0.3957 s | 0.712878714 | 0.603659327 | 0.734764314 |
| ViDoRe custom IPC | 1,360 | 0.00751 s | 0.4614 s | 0.712877053 | 0.603659327 | 0.734764314 |

`**` The narrow timing includes conversion of the vector frame back to Arrow;
custom IPC timing ends when the receiving actor has reconstructed the cuDF
frame. Total rebuild time is the comparable end-to-end column-roundtrip value.

BO767 accuracy is exactly unchanged. ViDoRe Recall@5 is unchanged; its rebuilt
ANN indexes shifted NDCG@10 by at most 0.00004685 and Recall@10 by 0.00009788.
The narrow and custom results agree to within 0.00000167 NDCG@10, consistent
with approximate-index build nondeterminism rather than transport corruption.
Direct Arrow comparison also confirmed exact element-for-element vector-column
equality for all four rebuilt indexes versus their source indexes.

## Full end-to-end benchmark

Both approaches were then integrated into full, unmodified-dataset harness
runs. The stock Ray Data graph was retained through extraction. The remaining
embedding and LanceDB stages were replaced with a bounded streaming GPU tail:

- **Narrow GPU island:** one actor owns embedding and incremental LanceDB
  writes, avoiding transport of the heterogeneous dataframe.
- **Custom transport:** a GPU embedding actor sends only a cuDF `row_id +
  vector` frame over CUDA IPC to a colocated writer actor; CPU columns travel
  separately and are recombined at the writer.

Both paths flush every 2,048 records and build the same `IVF_HNSW_SQ` index
inside the measured ingest phase. Pages/s uses the complete harness ingest
time, not the narrower Ray Data execution timer.

| Dataset | Implementation | Rows | Ingest seconds | Pages/s | Change vs baseline | NDCG@10 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BO767 | Unchanged baseline | 83,340 | 2,100.185 | 26.060 | — | 0.752057305 | 0.852674067 | 0.899091826 |
| BO767 | Narrow GPU island | 83,017 | 1,693.089 | 32.326 | +24.05% | 0.752579790 | 0.854692230 | 0.901109990 |
| BO767 | Custom cuDF CUDA IPC | 82,954 | 1,695.962 | 32.271 | +23.83% | 0.753182644 | 0.854692230 | 0.900100908 |
| ViDoRe computer science | Unchanged baseline | 1,360 | ~314 | ~4.331 | — | 0.712831866 | 0.603659327 | 0.734666436 |
| ViDoRe computer science | Narrow GPU island | 1,360 | 319.149 | 4.261 | -1.62% | 0.713034807 | 0.603169862 | 0.735098646 |
| ViDoRe computer science | Custom cuDF CUDA IPC | 1,360 | 309.262 | 4.398 | +1.55% | 0.712237967 | 0.602200870 | 0.733994918 |

BO767 is the representative throughput result: the narrow island saved
407.096 seconds and custom transport saved 404.223 seconds, reducing total
ingest time by 19.38% and 19.25%, respectively. The custom implementation was
only 0.17% slower than the narrow island end to end. ViDoRe is too small to
amortize actor and indexing overhead reliably; both implementations remain
within about 1.6% of its timestamp-derived baseline.

BO767 accuracy did not regress: every reported metric increased slightly.
Across ViDoRe, all absolute accuracy shifts were below 0.0015. The small row
count differences between separate BO767 runs and the small metric variations
are attributable to extraction/model and approximate-index nondeterminism, not
transport corruption; the transport-specific roundtrip checks above were
element-for-element exact.

One preliminary custom BO767 attempt encountered an empty embedding produced
upstream. It is excluded from these results. The benchmark adapter was corrected
to apply the stock LanceDB writer's bad-vector policy before transport, and the
full retry shown above completed successfully.

## Instrumented streaming optimization

The experiment adapter was instrumented around model initialization, embedding,
record construction, cuDF packing, actor handoff/queueing, cuDF-to-host
conversion, every LanceDB flush, final index construction, and driver-side
stream/drain phases. The extraction `materialize()` barrier was then removed so
the Ray Data prefix could feed the GPU tail incrementally. Extraction retained
its observed 0.6-GPU reservation and the tail used the remaining 0.4 GPU.

An initial streaming smoke run reserved one additional CPU for the tail actor.
Ray Data's fixed actor pools already reserved the machine's 32 CPUs, leaving no
CPU for source tasks. That invalid run was stopped and excluded. Streaming tail
actors use no scheduler CPU reservation in the successful runs below.

| Dataset | Implementation | Ingest seconds | Pages/s | vs prior implementation | vs unchanged baseline | NDCG@10 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BO767 | Streaming narrow island | 1,459.847 | **37.490** | +15.97% | +43.86% | 0.749511536 | 0.851664985 | 0.898082745 |
| BO767 | Streaming custom cuDF IPC | 1,461.899 | **37.438** | +16.01% | +43.66% | 0.751624800 | 0.853683148 | 0.899091826 |
| ViDoRe computer science | Streaming narrow island | 238.516 | **5.702** | +33.82% | +31.66%* | 0.712933019 | 0.603169862 | 0.735253685 |
| ViDoRe computer science | Streaming custom cuDF IPC | 241.266 | **5.637** | +28.17% | +30.15%* | 0.711056831 | 0.600728002 | 0.732462987 |

`*` Relative to ViDoRe's approximately 314-second timestamp-derived baseline.

The BO767 narrow run saved 640.338 seconds versus baseline and 233.242 seconds
versus the previous materialized narrow island. Custom transport finished only
2.052 seconds behind narrow, a 0.14% throughput difference.

### BO767 measured bottleneck breakdown

The active timers below overlap and therefore must not be summed. They show
where work is performed while the 1,443--1,446 second input stream is active.

| Timer | Streaming narrow | Streaming custom |
|---|---:|---:|
| Time to first extracted batch | 22.037 s | 21.745 s |
| Extraction input stream | 1,442.782 s | 1,445.556 s |
| Active embedding | 723.197 s | 703.629 s |
| Final tail drain | 0.867 s | 0.929 s |
| LanceDB flushes, 41 total | 11.831 s | 12.655 s |
| Final ANN index | 11.612 s | 10.850 s |
| cuDF packing | n/a | 39.952 s |
| cuDF-to-host conversion | n/a | 84.033 s |
| Handoff plus writer queue | n/a | 31.412 s |

Full overlap stretches the former 1,262-second extraction phase to roughly
1,443 seconds because detector, OCR, and embedding kernels contend for the same
H100. It nevertheless wins end to end because embedding is almost completely
hidden beneath that stream: less than one second remains to drain after the
last extracted batch. Storage and index construction consume only about 23
seconds and are not the current limiting stages.

Across the 2,605 custom batches, host conversion is the largest transport-side
active cost, followed by cuDF packing and handoff/queueing. These costs overlap
the critical input stream, which is why custom finishes only two seconds behind
narrow despite their much larger summed active time.

The streaming BO767 accuracy shifts versus the unchanged baseline are at most
0.00255 absolute for narrow and 0.00101 for custom. ViDoRe shifts are at most
0.00059 for narrow and 0.00293 for custom. The exact transport roundtrip tests
remain lossless, so separate-run model and approximate-index nondeterminism is
the likely cause, but repeated seeded runs are required before setting a tighter
accuracy-regression threshold.

### Next optimization focus

1. Tune shared-GPU scheduling across page detection, OCR, and embedding. Ray's
   fractional GPU resources control placement, not kernel compute share. Test a
   delayed or duty-cycled embedding consumer, CUDA MPS limits where supported,
   and starting embedding after a controlled fraction of extraction completes.
2. Sweep tail batch size and the active model inference batch size above 32.
   The H100 retained more than 40 GiB of memory headroom, and larger batches
   would reduce the current 2,605 actor calls and conversions.
3. Keep the narrow island as the default. If the separate-actor boundary is
   required, replace `to_arrow().to_pylist()` and nested Python sidecars with a
   fixed-size-list Arrow path into LanceDB. The measured custom active costs
   make this the first transport-side optimization.
4. Do not prioritize LanceDB flush or ANN-index tuning yet; together they are
   less than 2% of BO767 ingest wall time and mostly overlap other work.

## Page admission and GPU scheduling experiments

Five additional experiment-only variants tested whether controlling page flow
ahead of embedding improves GPU saturation. Each variant was run end to end on
both BO767 and ViDoRe, including fresh LanceDB index construction and the full
retrieval evaluation:

- delayed embedding until 25%, 50%, or 75% of unique source pages had emerged
  from extraction;
- eight embedding batches followed by a one-second extraction-only duty window;
- incremental PDF splitting that yielded bounded 16-page pandas frames instead
  of building every single-page PDF for an input batch before returning.

Delayed batches were kept as Ray object references rather than physical page
files. The experiment therefore retained Ray's object-store backpressure and
source metadata while avoiding filesystem fan-out. The production splitter and
NRL source remained unchanged.

### Full benchmark results

| Dataset | Variant | Ingest seconds | Pages/s | vs immediate streaming | NDCG@10 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BO767 | Immediate streaming reference | 1,459.847 | 37.490 | -- | 0.749511536 | 0.851664985 | 0.898082745 |
| BO767 | 25% delayed admission | 1,462.449 | 37.424 | -0.18% | 0.752429727 | 0.852674067 | 0.899091826 |
| BO767 | **50% delayed admission** | **1,446.515** | **37.836** | **+0.92%** | 0.750452676 | 0.850655903 | 0.898082745 |
| BO767 | 75% delayed admission | 1,552.272 | 35.258 | -5.95% | 0.752693964 | 0.853683148 | 0.900100908 |
| BO767 | 8-batch / 1-second duty cycle | 1,458.462 | 37.526 | +0.10% | 0.752413495 | 0.855701312 | 0.899091826 |
| BO767 | Incremental 16-page splitter | 1,464.056 | 37.382 | -0.29% | 0.749894042 | 0.850655903 | 0.898082745 |
| ViDoRe computer science | Immediate streaming reference | 238.516 | 5.702 | -- | 0.712933019 | 0.603169862 | 0.735253685 |
| ViDoRe computer science | 25% delayed admission | 264.249 | 5.147 | -9.73% | 0.712881969 | 0.603169862 | 0.735098646 |
| ViDoRe computer science | 50% delayed admission | 276.754 | 4.914 | -13.82% | 0.713031187 | 0.603169862 | 0.735098646 |
| ViDoRe computer science | 75% delayed admission | 285.996 | 4.755 | -16.61% | 0.712947709 | 0.603255995 | 0.735046966 |
| ViDoRe computer science | 8-batch / 1-second duty cycle | 242.613 | 5.606 | -1.68% | 0.713020244 | 0.603266761 | 0.735098646 |
| ViDoRe computer science | Incremental 16-page splitter | 239.897 | 5.669 | -0.58% | 0.712876454 | 0.603169862 | 0.734943607 |

All runs passed the expected file, page, and query-count gates. Relative to the
immediate-streaming runs, ViDoRe's maximum absolute metric shift was 0.00031.
BO767's maximum shift was 0.00404, and its maximum shift relative to the
unchanged baseline was 0.00303. These values remain in the range observed
across the earlier separate model and approximate-index builds; the transport
roundtrip itself remains element-for-element exact.

### Admission-controller timings

| Dataset/variant | Admission delay | Buffered batches | Buffered rows | Input stream | Active embedding | Final drain |
|---|---:|---:|---:|---:|---:|---:|
| BO767 25% | 388.126 s | 664 | 21,248 | 1,445.025 s | 698.911 s | 0.633 s |
| BO767 50% | 719.199 s | 1,329 | 42,528 | 1,429.017 s | 635.508 s | 0.799 s |
| BO767 75% | 990.540 s | 1,951 | 62,432 | 1,534.793 s | 509.932 s | 0.788 s |
| ViDoRe 25% | 83.676 s | 11 | 352 | 248.337 s | 169.608 s | 11.002 s |
| ViDoRe 50% | 101.390 s | 22 | 704 | 260.984 s | 163.651 s | 10.816 s |
| ViDoRe 75% | 120.052 s | 32 | 1,024 | 270.322 s | 155.046 s | 10.796 s |

BO767's 50% gate is the only material throughput candidate. It shortened the
measured input stream by 13.765 seconds and reduced active embedding by 87.689
seconds versus immediate streaming, while retaining enough overlap to keep the
final drain below one second. At 75%, less embedding contention reduced active
embedding further, but draining 1,951 buffered batches through the bounded
four-request window stretched the input stream by 92.011 seconds. ViDoRe is too
small to hide any delayed work, so every delayed-start setting regressed
monotonically.

The duty cycle imposed 325.076 seconds of sleeps on BO767 and 5.000 seconds on
ViDoRe. Ray continued advancing upstream work during those driver pauses, so
most sleep time overlapped existing pipeline slack. The resulting BO767 gain
was only 0.10%, and ViDoRe lost 1.68%; a fixed cadence is therefore not a useful
default.

The incremental splitter preserved page order and metadata and passed both
full accuracy runs, but it did not improve throughput or time to first extracted
batch. BO767 changed by -0.29%, and ViDoRe changed by -0.58%. Ray Data's existing
block scheduling and backpressure already hide most of the splitter's
per-document burstiness for these datasets. Incremental emission could still
reduce peak memory for unusually large PDFs, but it is not a measured PPS
optimization here.

### Revised recommendation

Keep the immediate narrow GPU island as the general default. The 50% BO767 gate
is promising for large, known-size batch corpora, but its +0.92% single-run gain
is too small to claim beyond run-to-run variance without repetitions. If gating
is pursued, replace fixed percentages and sleeps with a small adaptive
high/low-watermark controller that observes extracted backlog and downstream
latency. It should begin embedding only after a useful backlog exists, then
avoid both starvation and the 75% experiment's oversized drain.

The next experiment therefore swept the tail and active model inference batch
sizes together. The active `EmbedParams.inference_batch_size` was 32; a legacy
adapter field with value 16 was not read by the embedding implementation.

## Embedding batch-size sweep

The immediate streaming narrow island was rerun with both the Ray tail batch
and `EmbedParams.inference_batch_size` increased from 32 to 64 and then 128.
All other transport, GPU allocation, dataset, indexing, and evaluation settings
were held constant. Every run passed the expected file, page, and query gates.

| Dataset | Tail / inference batch | Ingest seconds | Pages/s | vs 32/32 | Tail calls | NDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BO767 | 32 / 32 reference | 1,459.847 | **37.490** | -- | 2,605 | 0.749511536 | 0.851664985 | 0.898082745 |
| BO767 | 64 / 64 | 1,463.565 | 37.395 | -0.25% | 1,303 | 0.751742353 | 0.852674067 | 0.900100908 |
| BO767 | 128 / 128 | 1,459.885 | **37.489** | -0.00% | 652 | 0.748752842 | 0.848637740 | 0.896064581 |
| ViDoRe computer science | 32 / 32 reference | 238.516 | **5.702** | -- | 43 | 0.712933019 | 0.603169862 | 0.735253685 |
| ViDoRe computer science | 64 / 64 | 251.133 | 5.415 | -5.03% | 22 | 0.713076814 | 0.603299061 | 0.735179809 |
| ViDoRe computer science | 128 / 128 | 256.000 | 5.312 | -6.84% | 11 | 0.712610577 | 0.602964374 | 0.735197679 |

Larger batches fit comfortably and reduced BO767 active embedding time from
723.197 seconds at 32 to 669.877 seconds at 64 and 633.113 seconds at 128.
That work remained hidden under the 1,441--1,445 second extraction stream, so
end-to-end BO767 throughput did not improve. At 128, time to first batch rose
from 22.037 to 27.807 seconds and final drain rose from 0.867 to 2.264 seconds.

ViDoRe could not amortize the larger batches. Its time to first batch increased
from 48.887 seconds at 32 to 62.901 seconds at 64 and 73.615 seconds at 128;
final drain increased from 10.860 to 20.802 and 46.511 seconds. Active embedding
did fall modestly, from 184.132 to 176.697 seconds, but the latency bubbles
dominated the full run.

Accuracy shifts across the batch sweep were small and non-monotonic, consistent
with the separate-run model and approximate-index variability already observed.
No batch size produced a systematic accuracy regression, but BO767 128/128's
Recall@5 was 0.00303 below the 32/32 run and should be covered by repeated seeded
runs before adopting a tighter regression threshold.

Keep 32/32 as the default. Increasing both batch sizes is capacity-safe but does
not improve full-pipeline PPS: extraction is the BO767 critical path, while
larger batches directly hurt latency on small inputs. If embedding is revisited,
decouple the knobs and test inference 64 or 128 behind a tail batch of 32; that
can probe model efficiency without delaying Ray tail admission. The higher-value
system optimization remains extraction-side GPU scheduling and kernel overlap.

### Decoupled inference-batch sweep

The follow-up held Ray tail admission at 32 rows and changed only
`EmbedParams.inference_batch_size`. This preserves the 32-row streaming cadence
and isolates the model-side setting.

| Dataset | Tail / inference batch | Ingest seconds | Pages/s | vs 32/32 | Active embedding | NDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BO767 | 32 / 32 reference | 1,459.847 | 37.490 | -- | 723.197 s | 0.749511536 | 0.851664985 | 0.898082745 |
| BO767 | 32 / 64 | 1,463.362 | 37.400 | -0.24% | 712.787 s | 0.753066300 | 0.854692230 | 0.901109990 |
| BO767 | 32 / 128 | 1,456.705 | **37.571** | +0.22% | 717.531 s | 0.753170959 | 0.853683148 | 0.901109990 |
| ViDoRe computer science | 32 / 32 reference | 238.516 | **5.702** | -- | 184.132 s | 0.712933019 | 0.603169862 | 0.735253685 |
| ViDoRe computer science | 32 / 64 | 244.430 | 5.564 | -2.42% | 188.458 s | 0.713949357 | 0.603686658 | 0.735253685 |
| ViDoRe computer science | 32 / 128 | 240.832 | 5.647 | -0.96% | 185.405 s | 0.713097752 | 0.603365814 | 0.734986674 |

The 32-row tail means the embedder never receives more than 32 rows in one
call, so raising the internal maximum cannot create a larger physical model
batch. BO767 32/128's +0.22% coincided with a 3.158-second shorter extraction
stream while active embedding was slower than at 32/64. ViDoRe regressed at
both settings. These small, inconsistent movements provide no evidence that
changing the inference cap alone improves throughput.

The accuracy movements remain non-monotonic and within the variation observed
across the broader experiment. All four decoupled runs passed their file, page,
and query gates. The recommendation remains 32/32: increasing the inference cap
behind a 32-row tail is effectively inert, while increasing the tail itself adds
latency. Further optimization should target the extraction critical path and
shared-GPU scheduling rather than embedding batch size.

## Follow-up optimization matrix

The remaining recommended experiments were run as full benchmarks where a
screening result justified it. All reported full runs rebuilt the LanceDB table
and index, ran retrieval evaluation, and passed the expected file, page, and
query-count gates.

### Three-run variance check

BO767 was repeated three times for the immediate 32/32 reference, the 50%
admission gate, and tail 32 / inference 128.

| Variant | Run pages/s | Median pages/s | Range | vs reference median |
|---|---:|---:|---:|---:|
| Immediate 32/32 | 37.490, 37.371, 37.423 | 37.423 | 0.318% | -- |
| 50% admission gate | 37.836, 37.729, 37.732 | **37.732** | 0.284% | **+0.83%** |
| Tail 32 / inference 128 | 37.571, 37.434, 37.377 | 37.434 | 0.518% | +0.03% |

The gate's gain is small but reproducible. The inference-128 movement is noise.
The repeated accuracy results stayed within the non-monotonic variation already
seen from separate model executions and approximate index builds.

### Adaptive bounded backlog

| Backlog low/high/max rows | ViDoRe pages/s | NDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| 128 / 256 / 512 | 5.491 | 0.711021263 | 0.601005779 | 0.733071145 |
| 256 / 512 / 1024 | 4.885 | 0.713082331 | 0.603299061 | 0.735375856 |
| 512 / 1024 / 2048 | 4.644 | 0.713175530 | 0.603299061 | 0.735448974 |

The best controller was then run on BO767 and produced 37.504 pages/s with
NDCG@10 0.751702913, Recall@5 0.852674067, and Recall@10 0.899091826. It did
not improve on the immediate reference or the 50% gate. A bounded controller is
memory-safer than the fixed gate, but this implementation is not a throughput
optimization.

### Direct contention measurement

The materialized narrow run executed extraction in 1,262.630 seconds and
finished ingest in 1,693.089 seconds (32.326 pages/s). Immediate streaming
stretched the shared-GPU extraction stream to 1,442.782 seconds, a 14.27%
contention penalty, but finished ingest in 1,459.847 seconds (37.490 pages/s).
Overlapping extraction and embedding therefore saved 233.242 seconds, or
13.78% end to end, despite the kernel contention. Serializing the stages is not
a useful optimization.

### CUDA MPS screen

| Embedding active-thread limit | ViDoRe pages/s | NDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| 10% | 2.719 | 0.712494393 | 0.603856183 | 0.734473606 |
| 20% | 4.200 | 0.713016006 | 0.603460560 | 0.734943607 |
| 30% | 4.788 | 0.693736030 | 0.585316662 | 0.715355319 |

Every MPS setting regressed throughput, and 30% also caused a material accuracy
regression. MPS was rejected without consuming a BO767 full run.

### Extraction-side screens

ViDoRe screened page-element batch sizes 12 and 48, OCR batch sizes 16 and 64,
and fixed page-element/OCR worker counts of 2 and 4. Pages/s ranged from 5.439
to 5.767 versus 5.702 for the reference. The apparent OCR-64 gain (5.767) did
not reproduce (5.685), and no setting had a clear, repeatable improvement.
Keep the existing defaults and autoscaling; none qualified for BO767 promotion.

### Direct Arrow custom transport

The custom writer was changed to build fixed-size-list Arrow arrays directly,
eliminating the row-wise `to_arrow().to_pylist()` path.

| Dataset | Pages/s | NDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| ViDoRe | 5.618 | 0.713106619 | 0.602916433 | 0.735395804 |
| BO767 | 37.443 | 0.751407665 | 0.851664985 | 0.899091826 |

On BO767, active cuDF-to-host/Arrow construction fell from 84.033 seconds to
2.350 seconds, including 2.071 seconds of Arrow construction. Full throughput
was unchanged because conversion was already hidden beneath extraction. This is
the preferred custom implementation when the actor boundary is required, but it
does not displace the simpler narrow island.

### Final recommendation

Use immediate streaming with the narrow GPU island and 32/32 tail/inference
batches as the general default. For large, known-size batch corpora with enough
Ray object-store capacity, a 50% start gate provides a reproducible BO767 median
gain of 0.83%, but buffers about 42,528 rows and should not be used for small or
open-ended streams. Do not adopt MPS throttling, larger embedding batches, or
the screened extraction overrides.

The remaining performance work should target extraction model execution,
preprocessing, and kernel-level profiling. Transport, LanceDB flush/index work,
and driver scheduling are no longer material end-to-end bottlenecks in this
single-node configuration.

## Feasibility conclusion

Custom cuDF direct transport is feasible for Ray Core actors on the same node
and GPU. Ray 2.56.1's alpha Direct Transport API is actor-only; Ray Data's
`map_batches(batch_format="cudf")` converts output through `to_arrow()` and does
not invoke RDT. Therefore, the practical integration is a narrow Ray Core GPU
island around the vector-producing/consuming stages, not transporting NRL's
full heterogeneous dataframe as cuDF.

For this single-node pipeline, the narrow island is the preferred design: it
is simpler and marginally faster while producing the same practical accuracy.
The custom transport is justified when embedding and storage must remain
separate actors; it preserves essentially all of the full-pipeline throughput
gain while providing that modular boundary.

This experiment did not test multi-node transfer. That would need a different
backend (for example NIXL) and explicit topology/lifetime handling. The custom
CUDA IPC backend is intentionally same-node, same-GPU, and benchmark-scoped.
