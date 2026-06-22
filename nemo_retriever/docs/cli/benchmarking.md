# Benchmarking with the `retriever` CLI

`retriever benchmark` and `retriever harness` are development and experimental subcommands
with no guarantees — see [Supported vs development / experimental subcommands](README.md#supported-vs-development--experimental-subcommands).

This page covers benchmark workflows for NeMo Retriever Library. See also
`docs/docs/extraction/benchmarking.md` for published extraction benchmark notes and
[`nemo_retriever/harness/HANDOFF.md`](../../harness/HANDOFF.md) for operator-oriented
notes on `retriever harness`.

Use `retriever harness` for benchmark orchestration and `retriever benchmark` for
per-stage micro-benchmarks.

## Harness (development / experimental)

Run from the repository root (or any directory; pass `--config` if needed). Uses
`--dataset` and `--preset` against `nemo_retriever/harness/test_configs.yaml`.

```bash
# Named dataset from nemo_retriever/harness/test_configs.yaml
retriever harness run --dataset bo767 --preset PE_GE_OCR_TE_DENSE

# Default active profile (jp20 + single_gpu in test_configs.yaml)
retriever harness run --dataset jp20

# Custom directory on disk
retriever harness run --dataset /path/to/your/data

# Override a single config key
retriever harness run --dataset bo767 --override run_mode=inprocess
```

Related commands:

```bash
retriever harness --help       # run, sweep, nightly, summary, compare, portal
retriever harness run --help
retriever harness sweep --help
retriever harness nightly --help
retriever harness summary --help
retriever harness compare --help
```

Sweep and nightly examples:

```bash
retriever harness sweep --runs-config nemo_retriever/harness/nightly_config.yaml
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml --dry-run
```

### Image storage

For normal ingest, configure image persistence on `retriever ingest` with
`--store-images-uri <uri>` (local path or fsspec URI). The harness does not
configure store directly; `retriever pipeline run --store-images-uri <uri>`
remains available for pipeline-specific compatibility workflows. Stored assets
follow `--embed-granularity` (page vs element images).

## Per-stage micro-benchmarks

Stage throughput benchmarks on the main CLI (no full harness required):

```bash
retriever benchmark --help           # split, extract, audio-extract, page-elements, ocr, all
retriever benchmark split --help
retriever benchmark extract --help
retriever benchmark audio-extract --help
retriever benchmark page-elements --help
retriever benchmark ocr --help
retriever benchmark all --help
```

Example — PDF extraction actor:

```bash
retriever benchmark extract ./data/pdf_corpus \
  --pdf-extract-batch-size 8 \
  --pdf-extract-actors 4
```

Each benchmark reports rows/sec (or chunk rows/sec for audio) for its actor.

## Notes

- **Configuration:** `retriever harness` uses `--dataset` / `--preset` /
  `--override KEY=VALUE` against
  `nemo_retriever/harness/test_configs.yaml`.
- **Launcher:** for internal benchmarking, `retriever harness run …` is the
  benchmark orchestration entry point (development / experimental; no guarantees).
- **Stage benchmarks:** `retriever benchmark …` is specific to the retriever CLI and
  covers per-stage throughput rather than full harness orchestration.
