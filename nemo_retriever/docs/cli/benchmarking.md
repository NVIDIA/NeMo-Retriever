# Benchmarking with the `retriever` CLI

This page covers benchmark workflows for NeMo Retriever Library. See also
`docs/docs/extraction/benchmarking.md`, [`tools/harness/README.md`](../../../tools/harness/README.md)
(legacy integration harness), and [`nemo_retriever/harness/HANDOFF.md`](../../harness/HANDOFF.md)
(the harness behind `retriever harness`).

There are two harness stacks:

| Stack | How you run it | Config |
|-------|----------------|--------|
| **Legacy** (`tools/harness/`) | `python -m nv_ingest_harness.cli.run` from `tools/harness/` | `tools/harness/test_configs.yaml` |
| **Retriever CLI** (`nemo_retriever.harness`) | `retriever harness run` | `nemo_retriever/harness/test_configs.yaml` |

The `retriever` CLI also exposes per-stage `retriever benchmark …` micro-benchmarks.

## Retriever harness (recommended)

Run from the repository root (or any directory; pass `--config` if needed). Uses
`--dataset` and `--preset` — there is no `--case` flag on this harness.

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

Image persistence is configured on `retriever pipeline run`, not on the harness.
Use `--store-images-uri <uri>` (local path or fsspec URI). Stored assets follow
`--embed-granularity` (page vs element images).

## Legacy `tools/harness` (nv_ingest_harness)

Unsupported developer tooling; kept for older CI and compose/helm managed runs.
After `cd tools/harness`, omit `--project tools/harness` — uv finds `pyproject.toml`
in the current directory.

```bash
cd tools/harness
uv sync
uv pip install -e .

uv run python -m nv_ingest_harness.cli.run --case=e2e --dataset=bo767
uv run python -m nv_ingest_harness.cli.run --case=e2e --dataset=/path/to/your/data
uv run python -m nv_ingest_harness.cli.run --case=e2e --dataset=bo767 --managed
```

From the **repo root** without `cd`, use the project flag:

```bash
uv run --project tools/harness python -m nv_ingest_harness.cli.run --case=e2e --dataset=bo767
```

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

## Parity notes

- **Not a drop-in flag map:** legacy harness uses `--case=e2e`; `retriever harness`
  uses `--dataset` / `--preset` / `--override KEY=VALUE` against
  `nemo_retriever/harness/test_configs.yaml`.
- **Datasets:** names like `bo767` and `jp20` exist in both configs but paths and
  defaults may differ; check the YAML for each stack.
- **Launcher:** prefer `retriever harness run …` for new work; use
  `nv_ingest_harness` only when you still depend on `--case` or `--managed` behavior
  documented in `tools/harness/README.md`.
- **Stage benchmarks:** `retriever benchmark …` is specific to the retriever CLI and
  has no legacy service-CLI equivalent.
