# Retriever Ingest Manifest Router

Status: architecture note

This note describes a proposed follow-up architecture for `retriever ingest`
that handles unknown folders and mixed file types without eagerly constructing
or initializing every possible modality pipeline. It builds on the release
bridge documented in
[Retriever Ingest Effective Extraction](../src/nemo_retriever/retriever_ingest_effective_extraction.md).

## Context

`retriever ingest` is moving from PDF-only ingestion toward "everything pipeline
supports" ingestion. That includes PDF, DOCX, PPTX, TXT, HTML, raster images,
SVG, audio, and video. The user experience should be simple:

```bash
retriever ingest /path/to/data --run-mode batch
```

The system should not require users to know all modalities in a folder ahead of
time, but it also must not eagerly initialize heavy dependencies or model actors
for file types that are not present.

The immediate release bridge uses effective extraction. It inspects input
extensions and collapses the result to one scalar extraction path, for example
`pdf`, `image`, `audio`, or `auto`. That is good enough to preserve the PDF fast
path, but it is not the final architecture for mixed folders.

## Core Problem

The current `auto` fallback relies on `MultiTypeExtractOperator`. This operator
is a composite actor: it receives a batch, groups rows by extension, and then
runs PDF, image, text, HTML, audio, or video sub-pipelines inside its own
`process()` method.

That shape is convenient but has several scaling and maintenance problems:

- It hides the natural pipeline stages from Ray Data. Ray sees one multitype map
  stage instead of separate PDF extraction, page-elements, OCR, table, image,
  audio, video, and text stages.
- Stage-specific scheduling is harder. The dedicated PDF graph can tune PDF
  extraction tasks, page-element actors, OCR actors, table actors, and embedding
  actors independently. A composite multitype actor collapses these boundaries.
- Actor/model reuse can suffer. Heavy sub-operators can be constructed inside
  multitype processing instead of having their own persistent Ray actor pool.
- Missing optional dependencies can be discovered late and in branches that
  should not matter for the actual input set.
- Extraction wiring is duplicated between the dedicated graph paths and
  `MultiTypeExtractOperator`, increasing drift risk. The release bridge aligns
  multitype audio/video defaults with typed ingest, but the composite actor can
  still fall behind as modality graphs evolve.
- Operational visibility is worse. Ray dashboards and logs show a broad
  multitype stage instead of pinpointing which modality stage is slow.

The measured PDF-only regression that motivated the release bridge is an
example of this architectural issue. When PDF-only inputs were routed through
`auto`, wall time regressed by 2.6x on `jp20` and 4.4x on `bo767` in a
self-hosted-NIM batch test, while quality and NIM call counts stayed comparable.
That points to graph structure and operator scheduling, not model inference.

## Desired Properties

The long-term design should satisfy these properties:

- Unknown folders work without user-provided modality knowledge.
- Only observed modalities are built and validated.
- PDF-only and document-only inputs use the dedicated PDF/document graph.
- Homogeneous non-PDF inputs use a specific graph, not a generic composite path.
- Mixed folders fan out to modality-specific branches and then fan in to the
  common post-extraction stages.
- Branches reuse the same operator-building helpers as dedicated graphs.
- Optional dependencies are branch-scoped:
  - no audio/video files means no `ffmpeg` or ASR requirement
  - no SVG files means no `cairosvg` requirement
  - no local OCR/page-element work means no local GPU model initialization
- Batch and in-process modes preserve consistent semantics.
- The common downstream contract remains unchanged for embedding and VDB upload.

## Proposed Architecture

Introduce an input manifest and extraction branch plan.

### InputManifest

The manifest is built before graph construction. It is cheap and should avoid
importing heavy model or media libraries.

Suggested fields:

```python
@dataclass(frozen=True)
class InputManifest:
    files_by_family: dict[str, tuple[str, ...]]
    unsupported_files: tuple[str, ...]
    explicit_globs: tuple[str, ...]
    has_buffers: bool
```

Families should match the ingest input families:

```text
pdf, doc, image, txt, html, audio, video
```

The manifest should preserve enough information to answer:

- Which branches are needed?
- Which explicit user constraints should be validated?
- Which optional dependencies should be checked?
- Which source files should each branch receive?

### ExtractionBranchPlan

The planner converts the manifest and user config into branch plans:

```python
@dataclass(frozen=True)
class ExtractionBranchPlan:
    family: str
    input_paths: tuple[str, ...]
    graph_kind: str
    params: object
    required_capabilities: frozenset[str]
```

Example branches:

| Family | Branch graph |
| --- | --- |
| `pdf`, `doc` | document conversion, PDF split, PDF extraction, page-elements, table/graphic, OCR, optional chunk |
| `image` | image load, page-elements, table/graphic, OCR, optional chunk |
| `txt` | text split/chunk |
| `html` | HTML split/chunk |
| `audio` | media chunk, ASR, optional chunk |
| `video` | video split, optional ASR, frame OCR, dedup/fusion, optional chunk |

### Branch Execution

For `run_mode=batch`, the executor can execute each branch as a Ray Dataset
pipeline over only that branch's files, then union the branch datasets after
schema normalization.

Conceptually:

```text
manifest
  -> plan branches
  -> for each present branch:
       read branch files
       apply branch extraction graph
       normalize extracted rows
  -> union extracted rows
  -> apply common post-extract graph
       dedup/caption/content reshape/embed/vdb/webhook
```

For `run_mode=inprocess`, the same branch plan can execute branch graphs on
pandas DataFrames, concatenate normalized outputs, and then apply the common
post-extract graph.

This does not require the core `Graph` object to become a fully general DAG on
day one. The branch plan can sit above the existing linear executor. That keeps
the change focused while still delivering fan-out/fan-in behavior.

## Avoiding Duplicate Graph Code

The goal is not to create separate hand-written graph builders that drift.
Instead, extract shared helpers from the current `build_graph()`:

```text
append_pdf_document_extraction(graph, params, split_config)
append_page_image_detection(graph, extract_params)
append_image_extraction(graph, params, split_config)
append_text_extraction(graph, text_params)
append_html_extraction(graph, html_params)
append_audio_extraction(graph, audio_params, asr_params, split_config)
append_video_extraction(graph, video_params, asr_params, extract_params, split_config)
append_common_post_extract_stages(graph, stage_order, embed/store/webhook params)
```

The current dedicated PDF path and the future PDF branch should call the same
PDF helper. The current image path and future image branch should call the same
image helper. `MultiTypeExtractOperator` should stop reimplementing those
subgraphs internally.

If the branch plan is adopted, `build_graph()` can remain as the entry point but
delegate to helpers based on the plan. The public API does not need a new graph
construction surface.

## Dependency Handling

Dependency checks should move from broad graph construction to branch
validation:

| Dependency | Required when |
| --- | --- |
| `cairosvg` | at least one SVG file is routed to the image branch |
| `ffmpeg` and `ffprobe` binaries | at least one audio or video file is routed to media branches |
| local page-elements model | PDF/image/video-frame OCR branches require local page-elements with no remote endpoint |
| local OCR model | OCR is enabled and no OCR endpoint is configured |
| local ASR model | audio/video ASR is enabled and no remote ASR endpoint is configured |

This keeps absent modalities invisible to the runtime. A PDF-only folder should
not validate media dependencies. A JPG-only folder should not validate SVG
rendering. An audio-free mixed folder should not initialize ASR.

## Impact on Current Architecture

The graph representation is already flexible enough to describe more than a
linear chain, but the current batch and in-process executors linearize graphs
and reject fan-out. A branch planner avoids requiring a full executor rewrite.

Likely changes:

- Add `InputManifest` and `ExtractionBranchPlan` near the ingest planning code.
- Replace scalar effective extraction resolution with manifest-to-branch
  planning.
- Refactor `build_graph()` into shared append helpers while preserving it as the
  main entry point.
- Add executor helpers that can apply a graph to an existing Ray Dataset without
  immediately materializing to pandas.
- Add a branch union step that validates and normalizes schemas before common
  stages.
- Keep explicit `--input-type` as a validation/filtering hint, not the primary
  architecture.
- Keep typed shortcuts such as `extract_audio()` and `extract_video()` as forced
  branch selection APIs.

## Migration Plan

1. Keep the release bridge in place, including aligned multitype media defaults.
2. Extract reusable graph-building helpers from the dedicated PDF, image, text,
   HTML, audio, and video paths.
3. Add `InputManifest` generation and tests for directory, glob, explicit file,
   and buffer inputs.
4. Add a branch planner that emits a single branch for homogeneous inputs and
   multiple branches for mixed inputs.
5. Add batch branch execution and union before common stages.
6. Add in-process branch execution with matching semantics.
7. Route mixed `auto` through the branch planner instead of
   `MultiTypeExtractOperator`.
8. Keep `MultiTypeExtractOperator` as a compatibility fallback until branch
   execution has parity tests.
9. Remove duplicated subgraph logic from `MultiTypeExtractOperator` or retire the
   operator once no public path depends on it.

## Test Plan

Recommended coverage:

- PDF-only directory uses the dedicated PDF graph.
- DOCX/PPTX-only directory uses the PDF/document graph.
- Image-only directory uses image graph and does not build audio/video branches.
- SVG-only image directory requires `cairosvg`; JPG-only image directory does
  not.
- Audio/video-free mixed directory does not validate `ffmpeg`.
- Audio and video directories report missing `ffmpeg`/`ffprobe` before actor
  startup.
- Mixed PDF/image/text/html folder produces the same normalized rows as running
  each modality separately and concatenating results.
- Common embed and VDB upload run once after branch union.
- Batch and in-process paths produce comparable source counts and metadata.
- Stage-level Ray execution plans expose modality stages rather than one
  monolithic multitype extraction stage.

## Open Questions

- Should branch execution run branches sequentially or launch independent Ray
  pipelines before union? Sequential branch execution is simpler; parallel branch
  execution may improve throughput for large mixed datasets.
- Should explicit globs be expanded before manifest planning, or should unresolved
  globs force the compatibility path?
- How should branch-level failures be reported when only one modality fails in a
  mixed ingest?
- What is the minimum normalized schema before union?
- Should audio and video ASR share one ASR actor pool when both branches are
  present?
- Should the public CLI expose a "strict modalities" option that fails if the
  manifest contains any unplanned family?

## Recommendation

Do not expand `MultiTypeExtractOperator` into a larger all-purpose router. Use
it only as a release bridge and compatibility fallback.

The core system should become multi-file and mixed-modality aware through input
manifest planning. That gives users the simple "ingest this folder" experience
without making absent modalities impose dependency, model, memory, or scheduling
costs.
