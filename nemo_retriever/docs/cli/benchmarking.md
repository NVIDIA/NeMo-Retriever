# Benchmarking with the `retriever` CLI

This page is the `retriever`-CLI counterpart to
`nv-ingest/docs/docs/extraction/benchmarking.md` and
`nv-ingest/tools/harness/README.md`.

The old benchmarking workflow is driven by `tools/harness` and
`uv run nv-ingest-harness-run`. The `retriever` CLI exposes the harness (and
per-stage micro-benchmarks) as first-class subcommands, so you can run
benchmarks without `uv run` or a separate harness repo.

## Harness (end-to-end benchmarks)

Old:

```bash
cd tools/harness
uv sync
uv run nv-ingest-harness-run --case=e2e --dataset=bo767
uv run nv-ingest-harness-run --case=e2e --dataset=/path/to/your/data
```

New — the harness is a subcommand on the main CLI (full parity):

```bash
retriever harness run --case=e2e --dataset=bo767
retriever harness run --case=e2e --dataset=/path/to/your/data
```

Related commands (browse with `--help`):

```bash
retriever harness --help       # run, sweep, nightly, summary, compare
retriever harness run --help
retriever harness sweep --help
retriever harness nightly --help
retriever harness summary --help
retriever harness compare --help
```

### Harness with image / text storage

Old:

```bash
retriever harness run --dataset bo20 --preset single_gpu \
  --override store_images_uri=stored_images --override store_text=true
```

New (unchanged — this form is already the `retriever` CLI):

```bash
retriever harness run --dataset bo20 --preset single_gpu \
  --override store_images_uri=stored_images --override store_text=true
```

When `store_images_uri` is a relative path it resolves to
`artifact_dir/stored_images/` per run; absolute paths and fsspec URIs
(e.g. `s3://bucket/prefix`) are passed through unchanged.

## Per-stage micro-benchmarks

The new CLI also exposes stage-level throughput benchmarks that had no direct
counterpart in `nv-ingest-cli`:

```bash
retriever benchmark --help           # split, extract, audio-extract, page-elements, ocr, all
retriever benchmark split --help
retriever benchmark extract --help
retriever benchmark audio-extract --help
retriever benchmark page-elements --help
retriever benchmark ocr --help
retriever benchmark all --help
```

Example — benchmark the PDF extraction actor:

```bash
retriever benchmark extract ./data/pdf_corpus \
  --pdf-extract-batch-size 8 \
  --pdf-extract-actors 4
```

Each benchmark reports rows/sec (or chunk rows/sec for audio) for its actor.
Use these when you want focused numbers for a single stage instead of an
end-to-end run.

## Parity notes

- The harness use-cases in the old docs (`--case=e2e`, `--dataset=bo767`,
  `--dataset=/path/...`, `--override ...`) are preserved verbatim — only the
  launcher changes (`retriever harness run …` instead of
  `uv run nv-ingest-harness-run …`).
- If you have a repo-local `uv` environment, `uv run retriever harness run …`
  still works.
- Stage benchmarks (`retriever benchmark …`) are net-new relative to the old
  `nv-ingest-cli` examples — they are the recommended way to profile
  individual actors before tuning `pipeline run` flags.
