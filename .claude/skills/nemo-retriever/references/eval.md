# retriever eval

End-to-end QA evaluation: retrieval + generation + judge. Three
subcommands:

- `retriever eval run` — run a configured QA sweep.
- `retriever eval export` — turn a LanceDB table into FileRetriever JSON for
  use as a static retriever in an eval config.
- `retriever eval build-page-index` — build a page-level markdown index for
  full-page eval mode.

If flags below look stale, re-check `retriever eval <subcmd> --help`.

## When to use this

- You want a single number for "is this retrieval+generation setup good?"
  (judge score, per-question answers, etc.).
- You're comparing models or chunking strategies and need a controlled QA
  benchmark.

**Use a different command when:**

- You only need retrieval recall metrics → [[recall]].
- You want a single ad-hoc query → [[query]].
- You're tuning extraction quality, not QA → [[pipeline]] / [[pdf]].

## Canonical invocations

Run a sweep from a config file:

```bash
retriever eval run --config evaluation/eval_sweep.yaml
```

Run a sweep from environment (Docker/CI pattern):

```bash
export RETRIEVAL_FILE=out/retrieval.json
export QA_DATASET=path/to/qa.json
export GEN_MODEL=...
export JUDGE_MODEL=...
retriever eval run --from-env
```

Export LanceDB → FileRetriever JSON so eval can consume it:

```bash
retriever eval export \
  --lancedb-uri ./lancedb --lancedb-table nv-ingest \
  --query-csv evaluation/queries.csv \
  --output out/retrieval.json \
  --top-k 5
```

Build a page index for full-page eval mode:

```bash
retriever eval build-page-index \
  --parquet-dir out/extractions/ \
  --output out/page_index.json
```

## Inputs / Outputs

- **`run`** — config (YAML/JSON) or env vars; emits per-question results +
  aggregated metrics.
- **`export`** — needs a populated LanceDB + a query CSV; emits a
  FileRetriever JSON.
- **`build-page-index`** — needs a directory of extraction Parquets; emits
  a JSON mapping `(pdf, page) → markdown`.

## Key flags

`eval run`:

| Flag | Notes |
|---|---|
| `--config FILE` | YAML/JSON sweep config (exclusive with `--from-env`). |
| `--from-env` | Build config from env vars (`RETRIEVAL_FILE`, `QA_DATASET`, `GEN_MODEL`, `JUDGE_MODEL`, …). |

`eval export`:

| Flag | Default | Notes |
|---|---|---|
| `--lancedb-uri` | `lancedb` | DB path. |
| `--lancedb-table` | `nv-ingest` | Source table. **Note**: this is `--lancedb-table` (with `lancedb-` prefix), unlike [[ingest]] / [[query]] / [[recall]] / [[vector-store]] which use `--table-name`. Must point at the same table either way. |
| `--query-csv` | — | Required. `query` (+ optional `answer`) columns. |
| `--output` | — | Required output JSON path. |
| `--top-k` | `5` | Chunks per query. |
| `--embedder` | `nvidia/llama-nemotron-embed-1b-v2` | Must match ingest embedder. |
| `--page-index FILE` | — | Enables full-page mode using `build-page-index` output. |

## Common failure modes

- **`run --from-env` errors with "RETRIEVAL_FILE not set"** — set every env
  var the loader requires; `--from-env` is all-or-nothing.
- **`export` writes empty file** — embedder mismatch with the LanceDB table
  (different dim) or `--query-csv` lacks a `query` column.
- **`build-page-index` is slow / OOM** — parquet directory is huge. Run on
  a subset and merge JSONs, or run in a higher-memory environment.

## Related

- [[recall]] — retrieval-only metrics.
- [[harness]] — orchestrates `eval`/`recall` sweeps with sessions, tags, and
  Slack reporting.
- [[compare]] — diff two eval runs.
