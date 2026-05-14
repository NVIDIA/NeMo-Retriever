# retriever harness

Benchmark / eval orchestration. Wraps [[recall]] / [[eval]] /
[[benchmark]] / [[pipeline]] runs into named *sessions* with tags,
artifacts, and (optionally) a web portal + history DB + Slack reporting.

Subcommands:

| Subcommand | What it does |
|---|---|
| `run` | One configured run against a dataset. |
| `sweep` | Multiple runs from a sweep YAML. |
| `nightly` | Curated nightly sweep; can post results to Slack. |
| `summary` | Print summary for a session. |
| `compare` | Diff two sessions. |
| `portal` | Launch the web portal. |
| `backfill` | Import existing `results.json` artifacts into the history DB. |
| `runner` | Runner agent (registers with a portal manager). |

If flags below look stale, re-check `retriever harness <subcmd> --help`.

## When to use this

- You want reproducible, tagged eval/benchmark sessions you can come back
  to later.
- You're triaging nightly regressions and want the session+Slack flow.
- You want to compare two sessions visually or via CLI.

**Use a different command when:**

- One-off run, no session bookkeeping → [[recall]] / [[eval]] /
  [[benchmark]].
- You're tuning extraction directly → [[pipeline]].

## Canonical invocations

Single run against a named dataset (preset from the config):

```bash
retriever harness run \
  --dataset bo767 \
  --config nemo_retriever/harness/test-config.yaml \
  --run-name "baseline-2026-05-13" \
  --tag dataset=bo767 --tag model=llama-nemotron-embed-1b-v2
```

Sweep:

```bash
retriever harness sweep \
  --config nemo_retriever/harness/test-config.yaml \
  --runs-config nemo_retriever/harness/sweep-runs.yaml \
  --session-prefix sweep
```

Nightly with Slack:

```bash
retriever harness nightly \
  --config nemo_retriever/harness/test-config.yaml \
  --runs-config nemo_retriever/harness/nightly-runs.yaml
```

Replay a previous run to Slack without rerunning:

```bash
retriever harness nightly --replay runs/2026-05-12/session_summary.json
```

Compare two sessions:

```bash
retriever harness compare runs/baseline/ runs/candidate/
```

Print a session summary:

```bash
retriever harness summary runs/2026-05-13/
```

Launch the portal:

```bash
retriever harness portal --host 0.0.0.0 --port 8100
```

Backfill old artifacts into the history DB:

```bash
retriever harness backfill --artifacts-dir runs/ --db harness-history.db
```

## Key flags

`harness run`:

| Flag | Notes |
|---|---|
| `--dataset` | Required. Dataset name (from config) or direct path. |
| `--preset` | Override the preset selection. |
| `--config` | Harness test config YAML. |
| `--run-name` | Label persisted in artifacts. |
| `--override KEY=VALUE` | Per-run config override (repeatable). |
| `--tag` | Tag persisted in artifacts (repeatable). |
| `--recall-required/--no-recall-required` | Override the recall-required gate. |

`harness sweep` / `nightly`:

| Flag | Notes |
|---|---|
| `--runs-config` | YAML listing the runs to execute. |
| `--preset` | Force preset for all runs. |
| `--session-prefix` | Directory prefix (sweep only). |
| `--tag` | Session-level tag (repeatable). |
| `--dry-run` | Print the plan, don't execute. |
| `--skip-slack` | Don't post to Slack (nightly only). |
| `--replay PATH` | Replay an existing session to Slack (nightly only). |

## Outputs

- Session directory containing per-run subdirectories, each with
  `results.json`, configs, and logs.
- `session_summary.json` aggregating metrics.
- Optional rows in the history DB (`backfill` / `portal`).
- Optional Slack post (`nightly`).

## Common failure modes

- **`--dataset` not found** — name doesn't resolve in `--config`'s dataset
  registry. Pass an absolute path or fix the name.
- **`Slack post failed`** — env vars missing; pass `--skip-slack` or
  configure the webhook.
- **`portal` shows no runs** — history DB is empty. Run `backfill` once
  against an artifacts root.
- **`recall-required` gate fails** — a run's recall@k dropped below
  threshold; the session is marked failed. Investigate before overriding
  with `--no-recall-required`.

## Related

- [[recall]] / [[eval]] / [[benchmark]] — the underlying runners.
- [[compare]] — non-harness JSON-level diff.
