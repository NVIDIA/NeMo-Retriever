# Retriever CLI — Replacement Examples for `nv-ingest-cli`

This folder contains `retriever` command-line examples that deliver the same
end-user outcomes as the `nv-ingest-cli` examples in
`nv-ingest/docs/`, `nv-ingest/api/`, `nv-ingest/client/`, and `nv-ingest/deploy/`.

The original `nv-ingest-cli` documentation is **not removed** — these files sit
alongside it as a new-CLI counterpart you can link to or migrate to.

## Key shape difference

`nv-ingest-cli` is a **single command that talks to a running REST service on
`localhost:7670`** and composes work via repeated `--task extract|split|caption|embed|dedup|filter|udf`.

`retriever` is a **multi-subcommand Typer app**. Most of the old CLI examples
map to `retriever pipeline run INPUT_PATH`, which runs the graph pipeline
locally (in-process or via Ray) and writes results to LanceDB and, optionally,
to Parquet / object storage. Other subcommands cover focused tasks:

| Old intent | New subcommand |
|------------|----------------|
| Extract + embed + store a batch of documents | `retriever pipeline run` |
| Run an ad-hoc PDF extraction stage | `retriever pdf stage` |
| Run an HTML / text / audio / chart stage | `retriever html run`, `retriever txt run`, `retriever audio extract`, `retriever chart run` |
| Upload stage output to LanceDB | `retriever vector-store stage` |
| Query LanceDB + compute recall@k | `retriever recall vdb-recall` |
| Run a QA evaluation sweep | `retriever eval run` |
| Serve / submit to the online REST API | `retriever online serve` / `retriever online stream-pdf` |
| Benchmark stage throughput | `retriever benchmark {split,extract,audio-extract,page-elements,ocr,all}` |
| Benchmark orchestration | `retriever harness {run,sweep,nightly,summary,compare}` |

## Contents

| New file | Replaces example(s) in |
|----------|------------------------|
| [`retriever_cli.md`](retriever_cli.md) | `nv-ingest/docs/docs/extraction/nv-ingest_cli.md` and the rebranded mirror `cli-reference.md` |
| [`quickstart.md`](quickstart.md) | Former `NeMo-Retriever/docs/docs/extraction/quickstart-guide.md` (Docker Compose quickstart; page removed from docs — see `deployment-options.md` / repo `docker-compose.yaml`) (the `nv-ingest-cli` section) |
| [`pdf-split-tuning.md`](pdf-split-tuning.md) | `nv-ingest/docs/docs/extraction/v2-api-guide.md` (CLI example) |
| [`smoke-test.md`](smoke-test.md) | `nv-ingest/api/api_tests/smoke_test.sh` |
| [`cli-client-usage.md`](cli-client-usage.md) | `nv-ingest/client/client_examples/examples/cli_client_usage.ipynb` |
| [`pdf-blueprint.md`](pdf-blueprint.md) | `nv-ingest/deploy/pdf-blueprint.ipynb` (CLI cell) |
| [`benchmarking.md`](benchmarking.md) | `nv-ingest/docs/docs/extraction/benchmarking.md` and `nv-ingest/tools/harness/README.md` |

## Gaps with no retriever-CLI equivalent (kept out of this folder)

The following `nv-ingest-cli` examples are **not** migrated here because the
new CLI does not yet expose an equivalent — continue to use `nv-ingest-cli`
for these cases:

- `--task 'udf:{…}'` — user-defined functions
  (`nv-ingest/docs/docs/extraction/user-defined-functions.md`,
  `nv-ingest/examples/udfs/README.md`). `retriever` does not expose UDFs.
- `--task 'filter:{content_type:"image", min_size:…, min_aspect_ratio:…, max_aspect_ratio:…}'`.
  The image scale/aspect-ratio filter stage is not reproduced in the new CLI.
- Bare service submission (`nv-ingest-cli --doc foo.pdf` with no extract tasks
  and full content-type metadata returned by the service). `retriever online submit`
  is currently a stub — only `retriever online stream-pdf` is implemented.
- `gen_dataset.py` dataset creation with enumeration and sampling.
- `--collect_profiling_traces --zipkin_host --zipkin_port`. Use
  `--runtime-metrics-dir` / `--runtime-metrics-prefix` instead for a different
  metrics flavor.

## Conventions used in the examples

- Input paths assume you invoke `retriever` from the `nv-ingest/nemo_retriever`
  directory (or point at absolute paths).
- `--save-intermediate <dir>` writes the extraction DataFrame as Parquet for
  inspection. LanceDB output goes to `--lancedb-uri` (defaults to `./lancedb`).
- `--store-images-uri <uri>` stores extracted images to a local path or an
  fsspec URI (e.g. `s3://bucket/prefix`).
- `--run-mode inprocess` skips Ray and is ideal for single-file demos and CI;
  `--run-mode batch` (the default) uses Ray Data for throughput.

Run `retriever pipeline run --help` for the authoritative flag list.
