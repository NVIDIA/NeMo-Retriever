# Retriever Ingest Effective Extraction

Status: release bridge for PR 2068

This note documents the current input-aware `retriever ingest` changes added to
address a PDF-only performance regression while expanding ingest beyond PDF
inputs. It is intentionally scoped as a release-safe bridge. The longer-term
architecture is described in [Retriever Ingest Manifest Router](../../developer_docs/retriever_ingest_manifest_router.md).

## Problem

The original product ask was for `retriever ingest` to handle the same broad set
of file types that pipeline workflows support:

- PDF and document inputs: PDF, DOCX, PPTX
- Text-like inputs: TXT, HTML
- Images: JPG, JPEG, PNG, TIFF, TIF, BMP, SVG
- Media: MP3, WAV, M4A, MP4, MOV, MKV

The first implementation direction made `auto` extraction the default so a
single ingest command could accept mixed folders. That fixed direct image
coverage for BMP and TIFF, but it also meant existing PDF-only consumers could
be routed through `MultiTypeExtractOperator` instead of the dedicated PDF graph.
That was a serious regression risk because the dedicated PDF graph exposes PDF
split, extraction, page-element detection, OCR, table structure, graphic
elements, embedding, and VDB upload as separate Ray Data operators with their
own scheduling and actor lifetimes.

The observed regression that motivated this bridge was measured on a single
H100 NVL with self-hosted NIM endpoints:

| Dataset | Pages | Pre-change dedicated PDF graph | `auto` through multitype | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `jp20` | 1,940 | 100 s, 19.37 pages/s | 259 s, 7.50 pages/s | 2.6x wall time |
| `bo767` | 54,730 | 1,834 s, 29.85 pages/s | 8,020 s, 6.82 pages/s | 4.4x wall time |

Quality was effectively unchanged in those runs, and the NIM call deltas
matched the dedicated PDF path. That points away from the inference endpoints as
the root cause and toward Ray graph shape, operator wiring, batching, or
per-batch work inside the multitype path. GPU utilization also dropped
materially in the large PDF-only run, consistent with the accelerator waiting on
upstream dispatch or actor work.

## Release Bridge

The bridge keeps broad `retriever ingest` input support while protecting the
common PDF-only path. The key idea is "effective extraction": keep `auto` as the
CLI default, expand directories to supported file types, inspect the resolved
file extensions, and choose the most specific extraction path that can handle
the observed inputs.

This is implemented in two places:

- `nemo_retriever/src/nemo_retriever/adapters/cli/sdk_workflow.py`
  - Adds `--input-type auto|pdf|doc|txt|html|image|audio|video`.
  - Expands bare directories according to the selected input type.
  - Validates file extensions before graph construction.
  - Resolves effective input type:
    - PDF plus DOC inputs use the dedicated PDF/document graph.
    - A single non-PDF family uses that family's graph.
    - Mixed families use `auto`.
    - The multitype `auto` fallback uses the same default media settings as
      typed root CLI audio/video ingest: env-aware ASR params with
      `segment_audio=False`, size-based audio splitting at `500000`, video
      frame extraction at `0.5` FPS, video text dedup enabled, and AV fusion
      enabled.
- `nemo_retriever/src/nemo_retriever/graph_ingestor.py`
  - Allows unified `.extract()` to infer the effective extraction mode from the
    configured input files.
  - Preserves explicit extraction modes for callers that want to force a path.
  - Initializes modality-specific params only when the effective input family
    needs them.

Supported file discovery lives in
`nemo_retriever/src/nemo_retriever/utils/input_files.py`.

## Current Routing

For `retriever ingest --input-type auto`, effective routing is:

| Observed input families | Effective path |
| --- | --- |
| none, unknown glob, or PDF/DOC only | `pdf` |
| image only | `image` |
| TXT only | `text` |
| HTML only | `html` |
| audio only | `audio` |
| video only | video extraction setup, currently represented through `auto` internally |
| mixed families | `auto` through `MultiTypeExtractOperator` |

For explicit `--input-type`, the CLI validates inputs against the requested
family. `--input-type doc` maps to the PDF/document path because DOCX and PPTX
are converted to PDF before extraction.

## Why This Works for Release

The release bridge fixes the high-risk regression without forcing a large graph
executor redesign:

- Existing PDF-only ingest does not pay for multitype dispatch.
- Directory ingest can now discover non-PDF supported files.
- The root CLI exposes modality selection without requiring users to switch to
  a different command.
- Batch tuning flags remain stage-specific for the dedicated PDF path.
- Mixed folders still have a correctness path through the existing multitype
  operator.

This bridge is intentionally conservative. It does not try to make the core
executor support real fan-out/fan-in in the same release window.

## Validation

The jp20 release gate compared PR 2068 against the merge-base on the same host,
using default local behavior and `--run-mode batch`:

| Metric | Baseline | PR 2068 |
| --- | ---: | ---: |
| Return code | 0 | 0 |
| Elapsed seconds | 177.81 | 171.84 |
| Pages/sec | 10.91 | 11.29 |
| Ray dataset execution seconds | 112.31 | 112.96 |
| LanceDB rows | 3,309 | 3,309 |
| Average GPU utilization | 26.57% | 26.46% |
| Peak GPU utilization | 100% | 100% |

Both runs used the dedicated PDF graph with `PDFExtractionCPUActor`, not
`MultiTypeExtractOperator`. This passed the gate: PR wall time was 96.6% of the
baseline wall time.

Local artifact:

```text
/raid/jioffe/tmp/retriever-ingest-perf/jp20-20260520_001607/report.md
```

Small mixed-modality smoke testing on the repo-root `data/` fixture covered PDF,
DOCX, PPTX, TXT, HTML, BMP, TIFF, JPEG, PNG, and SVG. The successful SVG run used
`--extra local --extra multimedia` because SVG rendering depends on `cairosvg`.

Local artifact:

```text
/raid/jioffe/tmp/retriever-ingest-perf/data-auto-20260520_131438/pr2068-multimedia-noaudio/pr2068-data-auto-multimedia-noaudio.summary.json
```

## Known Gaps

The bridge still has the same architectural limitation that caused concern in
the first place: truly mixed inputs use one composite multitype actor. That path
can preserve correctness, but it is not the right long-term scaling shape.

Known limitations:

- Mixed PDF/image/text/media folders still route through
  `MultiTypeExtractOperator`.
- `MultiTypeExtractOperator` repeats extraction wiring internally. The bridge
  keeps its audio/video defaults aligned with typed ingest, but duplicated graph
  code can still drift.
- Ray Data sees multitype extraction as one stage, which hides the natural
  stage boundaries needed for scheduling, concurrency, and model reuse.
- Dependency checks are only partially pruned. Effective extraction avoids
  audio/video setup for PDF-only inputs, but mixed `auto` still has to be
  careful not to initialize absent branches.
- Audio/video require system `ffmpeg` and `ffprobe` on `PATH`. Python packages
  such as `ffmpeg-python` and `nemo-retriever[multimedia]` do not provide those
  binaries.
- SVG requires the `multimedia` extra because `cairosvg` is optional.

## Intended Sunset

Effective extraction should remain in flight for the release because it protects
the PDF fast path and gives users broader ingest coverage immediately. The
longer-term fix should replace the scalar effective mode with an input manifest
and branch plan, described in
[Retriever Ingest Manifest Router](../../developer_docs/retriever_ingest_manifest_router.md).

