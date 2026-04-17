<!-- SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Nemo Retriever harness

Operator-oriented notes for `nemo_retriever` benchmark runs. Implementation details live in source; this file is the short map.

## Scope

- Harness is standalone under `nemo_retriever` (not based on `tools/harness`).
- It wraps `nemo_retriever.examples.graph_pipeline`.
- Primary use case is benchmark orchestration for local/cluster runs without Docker orchestration.
- Vector DB is LanceDB only.
- Recall gating is supported and enforced by config (`recall_required`).

## Graph Refactor Integration (Apr 2026)

- Upstream main includes `53f3a8c5` (`Refactor step (#1778)`), which removed mode-specific runtime files:
  - removed: `nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py`
  - removed: `nemo_retriever/src/nemo_retriever/examples/inprocess_pipeline.py`
  - removed: `nemo_retriever/src/nemo_retriever/ingest_modes/`
  - added: `nemo_retriever/src/nemo_retriever/examples/graph_pipeline.py`
  - added: `nemo_retriever/src/nemo_retriever/graph_ingestor.py`
  - added/active seam: `nemo_retriever/src/nemo_retriever/graph/ingestor_runtime.py`
- Any tuning or stability changes previously implemented in `batch_pipeline.py` or `ingest_modes/*` must now be ported to:
  - CLI arg wiring: `examples/graph_pipeline.py`
  - graph node override mapping: `graph/ingestor_runtime.py`
  - embed operators/runtime: `text_embed/operators.py`, `text_embed/runtime.py`
  - local model behavior: `model/local/llama_nemotron_embed_1b_v2_embedder.py`
- For embed OOM stabilization specifically, old `batch/inprocess` diffs should be treated as historical evidence only; the mergeable implementation surface is graph-only.

## Key Files

| Path | Role |
|------|------|
| `nemo_retriever/src/nemo_retriever/harness/run.py` | Run/sweep/nightly, metrics from `*.runtime.summary.json`, `results.json` |
| `nemo_retriever/src/nemo_retriever/harness/config.py` | YAML + env merge → `HarnessConfig` |
| `nemo_retriever/src/nemo_retriever/harness/artifacts.py` | Session dirs, `session_summary.json`, `latest_commit` helper |
| `nemo_retriever/src/nemo_retriever/harness/recall_adapters.py` | Query CSV adapters for recall |
| `nemo_retriever/harness/test_configs.yaml` | `active`, presets, datasets |
| `nemo_retriever/harness/nightly_config.yaml` | Sweep / nightly run list |

## Defaults (`test_configs.yaml`)

- Default dataset: `jp20` (recall workflow with query CSV).
- Default run mode: `batch`.
- `bo20`: ingestion-oriented (`recall_required: false`, no query CSV).
- Presets: `single_gpu`, `dgx_8gpu`.
- Datasets with adapters: `earnings` (`page_plus_one`), `financebench` (`financebench_json`, `pdf_only`), `bo10k` (recall off by default until queries are set).

## CLI (repo root)

```bash
source .retriever/bin/activate   # or retriever_runtime in container
uv pip install -e ./nemo_retriever   # if needed
```

```bash
retriever harness run --dataset jp20 --preset single_gpu
retriever harness run --dataset jp20 --preset single_gpu --tag nightly

retriever harness sweep --runs-config nemo_retriever/harness/nightly_config.yaml
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml --dry-run

retriever harness summary nemo_retriever/artifacts/<session_dir>
retriever harness compare <session_a> <session_b>
```

## Artifacts

Per run directory:

- **`results.json`** — authoritative record (outcome, `test_config`, `metrics`, `summary_metrics`, `runtime_summary`, paths).
- **`command.txt`** — replayable command line.
- **`runtime_metrics/`** — `*.runtime.summary.json` (metrics source for the harness).
- **`lancedb/`** — vector store (largest on-disk use).

Session:

- **`session_summary.json`** — one row per run (high level); use each run’s **`results.json`** for detail.

## `results.json` (short contract)

Prefer **`summary_metrics`** for dashboards (small set: pages, ingest timing, recall headline keys when present). **`metrics`** holds the fuller flat merge. **`test_config`** is the resolved harness configuration for that run (not the same as post-hoc “what the graph resolved” unless documented elsewhere). Shape is defined by `harness/run.py` and consumers should tolerate new optional keys.

## Operational notes

- `recall_required: true` without recall metrics → harness failure (`missing_recall_metrics`).
- Relative `query_csv` resolves next to the YAML file, then repo root.
- Dataset paths under `/datasets/nv-ingest/...` may resolve to `/raid/$USER/...` when the former is missing.
- Optional **store** options (`store_images_uri`, `store_text`, `strip_base64`) map to graph CLI flags; relative `store_images_uri` is resolved under the run artifact directory. See `config.py` for env overrides (`HARNESS_STORE_*`).

3. **TTY-backed subprocess retained**
   - Harness runs graph pipeline through a PTY so Ray progress remains rich/pretty by default.

Ideas not committed to code; pick up or trim as priorities change.

- **Config ergonomics:** preset inheritance or a scaling helper to cut duplicated numeric blocks in YAML.
- **Ops:** optional artifact retention command (prune old sessions by age or size).
- **Recall UX:** a single `recall_profile` (or similar) that maps to adapter + match mode to avoid invalid combinations.
- **Runs:** optional matrix expansion for `nightly_config`-style lists (keep explicit run list as default UX).
- **Reporting:** export session or `results.json` sets to CSV for spreadsheets.
- **Strictness:** optional mode that errors on unknown harness YAML keys.

## When changing the harness (checklist)

1. `config.py` + defaults in `test_configs.yaml` (and env keys if any).
2. `run.py` and artifact payloads (`results.json` / `session_summary.json` shape).
3. Unit tests under `nemo_retriever/tests/test_harness_*.py` and graph coverage via `test_batch_pipeline.py`.
4. `nemo_retriever/README.md` harness section if user-facing commands changed.
5. At least one local run or `--dry-run` on the affected CLI path.

**Done-ish bar:** tests green, schema changes intentional, README examples still true, session output not accidentally duplicated.

## Tests

```bash
pytest -q nemo_retriever/tests/test_harness_run.py \
  nemo_retriever/tests/test_harness_config.py \
  nemo_retriever/tests/test_harness_reporting.py \
  nemo_retriever/tests/test_harness_nightly.py \
  nemo_retriever/tests/test_harness_recall_adapters.py \
  nemo_retriever/tests/test_batch_pipeline.py
```

See `nemo_retriever/README.md` for broader retriever documentation.
