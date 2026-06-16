# Root Ingest CLI Design

This note explains the `retriever ingest` redesign for reviewers and future CLI
maintainers. It is intentionally scoped to the root ingest CLI. It does not
change `GraphIngestor`, `Retriever.query`, eval, harness logic, BEIR/QA/audio
recall, or pipeline reporting.

## Summary

`retriever ingest` is now the first-class ingest CLI:

```bash
retriever ingest DOCUMENTS...          # default local graph ingest
retriever ingest local DOCUMENTS...    # explicit local graph ingest
retriever ingest batch DOCUMENTS...    # Ray/batch graph ingest
retriever ingest service DOCUMENTS...  # remote service ingest
```

The public CLI no longer exposes a root `--run-mode` flag. Local and batch still
map to the same internal graph `run_mode` values used by
`create_ingestor(run_mode=...)`; the CLI chooses a command shape that keeps each
mode's valid option set separate.

## Public Contract

| CLI Command | Internal Owner | Request Type | Execution |
|---|---|---|---|
| `retriever ingest DOCUMENTS...` | graph ingest | `IngestPlanRequest` | `run_mode="inprocess"` |
| `retriever ingest local DOCUMENTS...` | graph ingest | `IngestPlanRequest` | `run_mode="inprocess"` |
| `retriever ingest batch DOCUMENTS...` | graph ingest | `IngestPlanRequest` | `run_mode="batch"` |
| `retriever ingest service DOCUMENTS...` | service ingest | `ServiceIngestPlanRequest` | `ServiceIngestor` client |

Behavior intentionally preserved:

- Local and batch use `resolve_ingest_plan(...)` and `run_ingest_workflow(...)`.
- Service uses `resolve_service_ingest_request(...)` and
  `run_service_ingest_workflow(...)`.
- Dry-run still prints the resolved request/plan JSON for the selected mode.
- Local and batch success summaries report files and LanceDB rows.
- Service success summaries report files, service URL, and service-returned row
  count when available.
- `retriever pipeline run` compatibility behavior is left in place.

Behavior intentionally changed:

- Bare `retriever ingest DOCUMENTS...` now means local/in-process graph ingest.
- Batch ingest is selected with `retriever ingest batch ...`, not
  `--run-mode batch`.
- Service ingest is selected with `retriever ingest service ...`, not
  `--run-mode service`.
- Service-local invalid options are parser-level unknown options instead of
  runtime-denied options.

## Why Not `--run-mode`

`run_mode` is still the correct Python API and core graph concept. It is not the
best user-facing CLI boundary because root ingest has two different ownership
families:

- graph ingest: local and batch runtime modes for `GraphIngestor`
- service ingest: a client for a remote `ServiceIngestor` service

A single command with `--run-mode` has to mix graph-owned options and
service-owned options in one help surface. Subcommands keep the ownership split
visible:

- `retriever ingest local ...` and `retriever ingest batch ...` share the graph
  ingest callback and typed graph request.
- `retriever ingest batch ...` maps to graph `run_mode="batch"`.
- `retriever ingest local ...` maps to graph `run_mode="inprocess"` and rejects
  batch-only Ray tuning before request construction.
- `retriever ingest service ...` has a separate parser surface and cannot accept
  LanceDB target flags, local NIM endpoint URLs, local embed backend flags, Ray
  tuning, `--ocr-lang`, or local audio/video controls.

This is separation of concerns, not loss of parity. The CLI maps to
`run_mode="inprocess"` or `run_mode="batch"` at the graph boundary. The Python
factory keeps its `run_mode` API for programmatic use.

## File Ownership

Root ingest CLI files:

| File | Responsibility |
|---|---|
| `cli/main.py` | Registers top-level Typer apps. It does not own ingest option construction. |
| `cli/ingest/app.py` | Creates the `retriever ingest` sub-app and default-local router. |
| `cli/ingest/graph_commands.py` | Owns local and batch graph-ingest CLI callbacks and builds graph `IngestPlanRequest`. |
| `cli/ingest/service.py` | Owns service CLI callback and builds `ServiceIngestPlanRequest`. |
| `cli/ingest/options.py` | Typer option metadata only. No request construction or policy. |
| `cli/ingest/shared.py` | Quiet output capture, CLI error handling, and success-summary helpers. |

Canonical ingest files:

| File | Responsibility |
|---|---|
| `ingest/plan.py` | Graph ingest policy, typed graph request dataclasses, plan resolution. |
| `ingest/execution.py` | Executes resolved graph ingest plans through `GraphIngestor`. |
| `ingest/service.py` | Service ingest request dataclasses, service request resolution, service execution. |

The Typer layer is adapter-only. Typer callbacks are private Python functions
(`_graph_ingest_command` and `_service_command`) because the public surface is
the shell command, not the callback symbol. Programmatic use should go through
the ingest plan/service APIs or `create_ingestor(...)`.

## Request Flow

Local/default flow:

```text
retriever ingest docs/
  -> nemo_retriever.cli.ingest.app routes to local
  -> nemo_retriever.cli.ingest.graph_commands builds IngestPlanRequest
  -> ingest.plan.resolve_ingest_plan(...)
  -> nemo_retriever.cli.ingest_workflow.run_ingest_workflow(...)
  -> ingest.execution.execute_ingest_plan(...)
  -> GraphIngestor
  -> local LanceDB
```

Batch flow:

```text
retriever ingest batch docs/
  -> nemo_retriever.cli.ingest.graph_commands builds IngestPlanRequest(run_mode="batch")
  -> ingest.plan.resolve_ingest_plan(...)
  -> nemo_retriever.cli.ingest_workflow.run_ingest_workflow(...)
  -> ingest.execution.execute_ingest_plan(...)
  -> GraphIngestor
  -> local LanceDB
```

Service flow:

```text
retriever ingest service docs/
  -> nemo_retriever.cli.ingest.service builds ServiceIngestPlanRequest
  -> ingest.service.resolve_service_ingest_request(...)
  -> nemo_retriever.cli.ingest_workflow.run_service_ingest_workflow(...)
  -> ServiceIngestor client
  -> remote retriever service
```

## Handling The Large Option Surface

The large number of flags is real public surface area, so the CLI keeps it
visible. The cleanup is how those values move inward:

- Typer command signatures declare the public knobs explicitly.
- `options.py` centralizes repeated Typer metadata only when the flag spelling,
  default, validation, and help text are identical.
- `graph_commands.py` immediately groups parsed values into typed `Ingest*Options`
  dataclasses. Exact-name fields are selected from the `ingest.plan` dataclass
  fields; semantic CLI-to-request mappings stay explicit.
- `service.py` immediately groups parsed values into typed
  `ServiceIngest*Options` dataclasses.
- `resolve_ingest_plan(...)` remains the only graph-mode planner.
- `GraphIngestor` receives only resolved execution parameters, not raw CLI
  options.

There is no `locals()` funnel, reflection, dynamic command generation, mode
registry, or generic option framework. The graph builders use one narrow
dataclass-field matcher for exact-name fields and keep semantic mappings
explicit:

- `_build_source_options`
- `_build_runtime_options`
- `_build_extract_batch_options`
- `_build_extract_options`
- `_build_media_options`
- `_build_caption_options`
- `_build_dedup_options`
- `_build_chunk_options`
- `_build_embed_batch_options`
- `_build_embed_options`
- `_build_image_store_options`
- `_build_storage_options`

The few name differences are semantic, not accidental. For example,
`--api-key` fans out to extract, caption, and embed endpoint API keys, and
`--store-images-uri` maps to the image-store `images_uri` field.

## Batch-Only Fields

Local and batch share one graph Typer callback so the duplicated public graph
contract stays in one signature. Batch-only fields are declared on that shared
callback, labeled as batch-only in help, and rejected for local/default mode
before building `IngestPlanRequest`:

- Ray runtime: `--ray-address`, `--ray-log-to-driver`
- extraction tuning: PDF split/extract, page elements, OCR, table structure,
  Nemotron Parse workers, batch sizes, CPUs, GPUs
- embedding tuning: embed workers, batch size, CPUs, GPUs

Service ingest never receives those dataclasses and does not expose those
options.

## Service Mode

`retriever ingest service` is a client for a running retriever service. The CLI
does not expose `--lancedb-uri` or `--table-name` because service persistence is
owned by the server deployment. The server decides its vector database through
service configuration.

Service mode exposes only the controls represented by
`ServiceIngestPlanRequest`: connection settings, source/profile, service-side
extract toggles, dedup, caption behavior, chunking, embed modality/granularity,
image-store URI, dry-run, and quiet output.

Service-backed query support belongs in the query CLI/service boundary, not in
the ingest CLI.

## Pipeline Compatibility

`retriever pipeline run` is not the future public ingest interface. It remains
the compatibility and development command for
pipeline-only behavior such as:

- intermediate Parquet artifacts
- pipeline reports and runtime metrics
- eval, recall, harness, BEIR/QA workflows
- legacy callers not yet migrated to root ingest/query

For graph ingest paths, pipeline compatibility should continue to reuse the
canonical ingest plan/execution layer instead of shelling out to root CLI
commands.

## Adding Or Changing A Flag

For an option that already exists in the graph ingest plan:

1. Add or reuse a Typer alias in `cli/ingest/options.py`.
2. Add the parameter to `_graph_ingest_command`.
3. If the CLI parameter name matches the request field, no mapping list is
   needed; the dataclass-field matcher picks it up from `ingest.plan`. If the
   CLI name differs, keep the semantic mapping explicit in `graph_commands.py`.
4. If the option is batch-only, add it to `_BATCH_ONLY_FLAGS` so local/default
   mode rejects it before request construction.
5. Add or update a focused root CLI test that asserts the typed request field.
6. Update CLI docs if the option is user-facing.

For an option that does not exist in the graph ingest plan:

1. Add the capability to the canonical graph layer first:
   `ingest/plan.py` dataclass, resolver, and execution mapping.
2. Then expose it through the CLI using the graph steps above.

For a service option:

1. Add the capability to `ServiceIngestPlanRequest` and
   `resolve_service_ingest_request(...)` if needed.
2. Add the Typer alias only if the public flag should exist.
3. Add it to `_service_command`.
4. Add or update service CLI request-construction tests.

Do not add a generic kwargs funnel to avoid writing these mappings. The mapping
is the ownership boundary that keeps CLI spelling, typed request fields, and
core ingest behavior reviewable.

## Reviewer Checklist

When reviewing root ingest changes, check:

- `cli/main.py` only registers top-level apps.
- Typer-specific code stays under `cli/ingest/`.
- Graph options build `IngestPlanRequest`; service options build
  `ServiceIngestPlanRequest`.
- Service mode does not regain local-only or batch-only flags.
- Batch-only tuning remains batch-only.
- The CLI does not shell out to itself.
- Feature policy stays in `ingest/plan.py` or `ingest/service.py`, not in the
  Typer layer.
- Tests assert mode-specific request construction and parser-level rejection of
  invalid service flags.
