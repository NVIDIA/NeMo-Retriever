# Retriever Harness PRD: End-to-End Ingest/Query Benchmarks

Last updated: 2026-06-24

## Summary

Rebuild `retriever harness` as the internal benchmark runner for Retriever
engineers. The harness should run the new library direction end to end:

1. Ingest documents through the same shared code path as `retriever ingest`.
2. Query the resulting LanceDB table through the same shared code path as
   `retriever query`.
3. Run full BEIR-style evaluation.
4. Emit stable `summary_metrics` and machine-readable artifacts for humans,
   agents, and future nightly reporting.

This is a total revamp, not a compatibility wrapper around the current harness.
Assume `retriever pipeline run` is deleted in the next release. The current
`sweep`, `compare`, and legacy graph-pipeline command builder can be rewritten
or removed if they get in the way.

The most important design choice: use a small typed benchmark registry in code,
not a sprawling YAML configuration system. YAML/runfiles can exist as an escape
hatch for one-off experiments, but day-to-day benchmark definitions should live
in the repository as reviewed Python objects.

The second most important design choice: stdout is not an API. The harness can
print concise human summaries, but agents and orchestrators must rely on stable
files such as `status.json`, `summary_metrics.json`, `beir_metrics.json`, and
`query_results.jsonl`.

## Research Notes

The industry-standard pattern is not "one huge YAML that can do anything." The
useful patterns are:

- **Separate the benchmark runner from the system under test.** MLPerf
  Inference uses a load generator outside the submitted system under test, so
  timing, workload generation, logging, and validation are not buried inside the
  model/backend implementation. Retriever should mirror that boundary: the
  harness owns run lifecycle, timing, query iteration, metrics, and artifacts;
  Retriever ingest/query own retrieval behavior.
- **Keep benchmark suites close to code.** ASV benchmarks Python packages over
  time and treats benchmark definitions as part of the project rather than as a
  pile of external config files. For Retriever engineers, code-owned benchmark
  specs are easier to review, type-check, and refactor with the library.
- **Emit durable artifacts, not just console output.** Benchmark systems such as
  ASV and Phoronix Test Suite treat saved run artifacts, histories, and
  comparisons as first-class outputs. Retriever should make `results.json`,
  `summary_metrics`, BEIR metrics, runfiles, and per-query outputs the product
  of a run.
- **Use BEIR conventions for retrieval quality.** BEIR evaluates retrieval with
  NDCG@k, MAP@k, Recall@k, Precision@k, and custom metrics like MRR. Retriever
  harness should emit that full metric family, then choose a small subset for
  `summary_metrics`.
- **Avoid adopting Hydra-style composition unless we truly need it.** Hydra is
  powerful, but its value comes from config groups, launchers, sweepers, output
  directory patterns, and multi-file composition. That is more machinery than
  this internal harness needs right now.

Reference links are at the end of this document.

## Goals

- Make `retriever harness run <benchmark>` the standard internal entry point
  for end-to-end Retriever benchmarks.
- Default to local, in-process execution suitable for developer laptops,
  workstations, and agents.
- Run full BEIR evaluation for benchmark datasets that have qrels.
- Emit a compact, stable `summary_metrics` object every run.
- Emit detailed artifacts for debugging, reproducibility, and future reporting.
- Make benchmark definitions easy to discover, review, and extend.
- Avoid duplicating the `retriever ingest` and `retriever query` option surface.
- Avoid YAML sprawl and config inheritance chains.
- Make ablations simple enough for an agent to run without guessing paths or
  settings.
- Make every run inspectable by agents and orchestrators without parsing stdout.

## Non-goals

- Do not preserve the graph-pipeline harness path.
- Do not preserve current `sweep` or `compare` behavior for its own sake.
- Do not introduce a user-facing `--engine` flag.
- Do not adopt Hydra, MLflow Projects, W&B Sweeps, or another orchestration
  framework in phase one.
- Do not make this a public supported product API.
- Do not build Slack posting in phase one. Design artifacts so a Slack sink can
  be added later.
- Do not make CLI text formatting part of the run contract.
- Do not make pytest the validation strategy for the harness. This harness is
  itself a functional evaluation tool; success should be judged by harness
  command exit codes and artifact contracts.

## Users

- Retriever engineers validating ingest/query behavior.
- Performance owners running throughput and quality ablations.
- Agents asked to run a named benchmark or ablation.
- Future nightly automation that needs stable result files.

## Design Principles

- **Artifact-first:** every durable result is written to a documented file.
  Console output is for humans only.
- **Code-owned defaults:** recurring benchmarks and runsets live in reviewed
  Python registry entries.
- **Small overrides:** agents and engineers can apply targeted `--set` changes
  without creating new config files.
- **One real execution path:** the harness calls shared Retriever workflow code
  directly instead of shelling out to the CLI per phase.
- **Phase visibility:** long runs expose current phase, elapsed time, and
  partial artifacts as they progress.
- **Typed failures:** failures include phase, reason, retryability, and pointers
  to the relevant logs/artifacts.
- **Orchestration-neutral:** the harness has no dependency on MLflow, Airflow,
  Argo, Ray Tune, or similar systems, but its inputs and outputs are easy for
  those systems to wrap later.

## Recommended Shape

### One Harness Execution Path

The harness should run in the current Python process and call shared Retriever
workflow functions directly.

There should not be a user-facing `--engine inprocess|subprocess` flag. That
idea came from separating two implementation options:

- direct Python calls into shared ingest/query workflow code
- shelling out to the `retriever` CLI as subprocesses

For this harness, subprocess mode is not worth the extra concept. It would make
query benchmarks less honest by repeatedly measuring process startup and model
warmup unless carefully special-cased. It would also complicate artifacts and
error handling.

Instead:

- The harness always uses the shared Python workflow path.
- The harness writes replay hints that show equivalent `retriever ingest` and
  `retriever query` commands where useful.
- Console-script smoke tests can live elsewhere.

### Local vs Batch Is an Ingest Setting

There is still a real distinction between local and Ray-backed ingest:

- `local`: maps to `ingest.run_mode = "inprocess"`.
- `batch`: maps to `ingest.run_mode = "batch"` and may set Ray/tuning kwargs.

Expose that as benchmark intent, not as a generic execution engine:

```bash
retriever harness run jp20_smoke
retriever harness run jp20_beir --mode batch
```

Internally, `--mode batch` only changes the ingest/query benchmark spec fields
that need to change. Query execution should still use one constructed Retriever
object per run so measured query latency is not dominated by setup.

### Code-Owned Benchmark Registry

Create a canonical registry, for example:

```text
nemo_retriever/src/nemo_retriever/harness/
  benchmark_specs.py
  benchmark_registry.py
  runner.py
  beir_eval.py
  artifacts.py
```

Core types:

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    query_csv: str | None = None
    input_type: str = "pdf"
    beir_loader: str | None = None
    beir_doc_id_field: str = "pdf_page"
    beir_ks: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset: str
    ingest: Mapping[str, Any]
    query: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    summary_keys: tuple[str, ...]
    tags: tuple[str, ...] = ()
```

The registry should include named benchmarks such as:

- `jp20_smoke`
- `jp20_beir`
- `bo767_beir`
- `financebench_beir`
- `bo10k_beir_fast_text`
- `earnings_beir` once its query/qrels file is available in the repo or dataset
  mount

Commands:

```bash
retriever harness list
retriever harness show jp20_beir
retriever harness run jp20_beir
retriever harness run jp20_beir --set query.top_k=20
retriever harness run jp20_beir --set ingest.profile=fast-text
```

This is the default workflow. Engineers add benchmarks by editing reviewed
Python specs, not by dropping new YAML files around the repo.

### Narrow Runfiles

Support one optional runfile path for reproducible one-off runs, agent
handoffs, and orchestrator inputs:

```bash
retriever harness run --runfile nemo_retriever/harness/runfiles/jp20_beir.json
```

Runfiles should be intentionally small JSON/YAML objects:

```json
{
  "schema_version": 1,
  "name": "jp20_beir_expected",
  "benchmark": "jp20_beir",
  "mode": "local",
  "require": ["files==20", "pages==1940", "query_count==115"],
  "set": {
    "query.top_k": 10
  }
}
```

Rules:

- A runfile must extend a named registry benchmark.
- A runfile cannot define a new schema from scratch in phase one.
- The resolved benchmark spec is always written into artifacts.
- The source runfile payload is copied into `runfile.json` in the artifact
  directory.
- Runfiles are for reproducible run requests and agent instructions, not the
  default source of truth for recurring benchmark definitions.
- During `--dry-run`, gates for unavailable execution metrics are skipped and
  reported in `results.json`; static dataset gates are still evaluated.

### Ablations

Prefer explicit runsets in code for recurring ablations. Phase one runsets are
intentionally literal lists of named benchmarks; they do not expand matrices yet.

```python
RunSet(
    name="jp20_profile_x_rerank",
    runs=("jp20_beir", "jp20_beir_rerank"),
)
```

Commands:

```bash
retriever harness list --runsets
retriever harness run-set jp20_profile_x_rerank
```

For phase one, `run-set` can simply expand to individual benchmark runs and
write `expanded_runs.json`. We do not need a separate legacy `sweep` concept.

## Agent And Orchestration Contract

Agents will use this harness for continuous benchmark/performance loops. They
may inspect artifacts, compare metrics, choose the next ablation, and rerun.
That changes the contract: the harness must behave like a protocol, not just a
pretty CLI.

### Required Inputs

Every run should support:

```bash
retriever harness run <benchmark> \
  --run-id <stable-id> \
  --output-dir <artifact-dir> \
  --set query.top_k=20 \
  --dry-run
```

Rules:

- `--run-id` is optional for humans but required by orchestrated jobs that need
  deterministic artifact paths.
- `--output-dir` controls exactly where artifacts are written.
- `--dry-run` writes the resolved benchmark and planned artifacts without
  running ingest or query.
- All prompts are forbidden. The command must be non-interactive.

### Required Live State

Write `status.json` early and update it at phase transitions:

```json
{
  "run_id": "jp20_beir_20260623_120000",
  "benchmark": "jp20_beir",
  "status": "running",
  "phase": "query",
  "started_at": "2026-06-23T12:00:00Z",
  "updated_at": "2026-06-23T12:34:56Z",
  "artifact_dir": "/artifacts/jp20_beir_20260623_120000",
  "summary_metrics_path": null,
  "failure": null
}
```

Allowed statuses:

- `planned`
- `running`
- `complete`
- `failed`

Allowed phases:

- `resolve`
- `ingest_plan`
- `ingest`
- `query_plan`
- `query`
- `evaluate`
- `write_artifacts`

Also write append-only `events.jsonl` for phase changes and major milestones.
Agents can tail this file or poll `status.json`; they should never scrape
stdout.

### Failure Shape

On failure, `status.json` and `results.json` should include:

```json
{
  "failed_phase": "query",
  "failure_reason": "lancedb_table_missing",
  "retryable": false,
  "message": "LanceDB table nv-ingest was not found",
  "debug_artifacts": [
    "ingest_plan.json",
    "logs/query.log"
  ]
}
```

Failure reasons should be stable enough for agents to branch on. Examples:

- `invalid_benchmark`
- `invalid_override`
- `dataset_missing`
- `ingest_plan_failed`
- `ingest_failed`
- `query_plan_failed`
- `query_failed`
- `evaluation_failed`
- `metric_gate_failed`
- `artifact_write_failed`

### Exit Codes

Use coarse but stable exit codes:

- `0`: success
- `2`: invalid benchmark/config/override
- `3`: dataset or input missing
- `10`: ingest failure
- `11`: query failure
- `12`: evaluation failure
- `20`: metric gate failure
- `30`: artifact write failure
- `70`: unexpected internal error

Agents should prefer `status.json` for detail and exit codes for coarse process
control.

### Metric Gates

Support gates after metrics are stable:

```bash
retriever harness run jp20_beir \
  --require recall_5>=0.80 \
  --require ndcg_10>=0.70 \
  --require query_latency_p95_ms<=150
```

Gate failures should still write all artifacts and exit with
`metric_gate_failed`.

## Reuse Points

Use the current root CLI's shared implementation functions as the source of
truth:

- `nemo_retriever.adapters.cli.sdk_workflow.resolve_ingest_plan()`
- `nemo_retriever.adapters.cli.sdk_workflow.ingest_documents()`
- `nemo_retriever.adapters.cli.sdk_workflow.query_documents()`

Add one query planning helper so the harness can avoid rebuilding a Retriever
for every query:

```python
@dataclass(frozen=True)
class ResolvedQueryPlan:
    top_k: int
    lancedb_uri: str
    table_name: str
    embed_kwargs: dict[str, Any]
    rerank: bool
    rerank_kwargs: dict[str, Any]

    def create_retriever(self) -> Retriever: ...


def resolve_query_plan(...) -> ResolvedQueryPlan: ...
```

Then:

- `retriever query` remains a thin single-query CLI wrapper.
- `retriever harness` creates one Retriever per benchmark run and queries the
  full BEIR query set.

Validate nested `ingest` and `query` keys against the signatures of these
workflow helpers. Unknown keys should fail before execution with suggestions.
Do not copy Typer option definitions into the harness.

## Run Lifecycle

For each benchmark run:

1. Resolve a `BenchmarkSpec` from the registry plus CLI `--set` overrides or a
   tiny runfile.
2. Resolve dataset paths and query/qrels files.
3. Create a run artifact directory.
4. Set the run's LanceDB URI under the artifact directory unless explicitly
   overridden.
5. Dry-run ingest with `resolve_ingest_plan()` and write a redacted plan.
6. Execute ingest with `ingest_documents()` and measure wall-clock time.
7. Count input files, pages, and LanceDB rows.
8. Build one Retriever from `resolve_query_plan()`.
9. Execute optional warmup queries.
10. Execute the full measured BEIR query set.
11. Write a BEIR/TREC runfile and raw per-query hits.
12. Compute BEIR metrics.
13. Write `results.json` with `summary_metrics`.
14. Write `status.json` with `status = "complete"`.
15. Print a concise terminal summary for humans.

## BEIR Evaluation

BEIR-style evaluation is a phase-one requirement, not a later add-on.

For every BEIR benchmark, write:

- `beir_run.trec`: runfile suitable for reranking/debugging.
- `beir_metrics.json`: full metric family.
- `query_results.jsonl`: query text, latency, ranked hits, and resolved doc IDs.

Metric family:

- `ndcg@k`
- `map@k`
- `recall@k`
- `precision@k`
- `mrr@k` when supported

Default k values:

```python
(1, 3, 5, 10)
```

`summary_metrics` should include the small set that engineers and agents need
first:

```json
{
  "files": 20,
  "pages": 496,
  "rows_processed": 12345,
  "ingest_secs": 321.5,
  "pages_per_sec_ingest": 1.54,
  "query_count": 200,
  "query_latency_p50_ms": 42.1,
  "query_latency_p95_ms": 87.3,
  "ndcg_10": 0.72,
  "recall_5": 0.81,
  "recall_10": 0.86
}
```

The exact values above are illustrative. The key names should be stable.

## Artifact Contract

Per run:

- `results.json`: authoritative run result.
- `summary_metrics.json`: the same compact object from `results.json`, written
  separately for simple consumers.
- `status.json`: current and final phase/status state.
- `events.jsonl`: append-only phase changes and run milestones.
- `resolved_benchmark.json`: fully resolved benchmark spec.
- `ingest_plan.json`: redacted dry-run ingest plan.
- `beir_metrics.json`: full BEIR metrics.
- `beir_run.trec`: BEIR/TREC runfile.
- `query_results.jsonl`: per-query hits and latency.
- `environment.json`: git SHA, package version, Python, host, GPU count, CUDA
  driver, Ray version where available.
- `lancedb/`: default vector store.

Session/runset:

- `session_summary.json`: one row per run, centered on `summary_metrics`.
- `expanded_runs.json`: resolved run order for runsets.

Do not make `compare` phase-one critical. Comparing runs can be rebuilt later
from `summary_metrics.json` and `session_summary.json`.

`results.json` should include pointers to every artifact path so external
systems can ingest one file and discover the rest.

## CLI

Phase-one CLI:

```bash
retriever harness list
retriever harness show <benchmark>
retriever harness run <benchmark>
retriever harness run <benchmark> --run-id <id> --output-dir <dir>
retriever harness run <benchmark> --mode local
retriever harness run <benchmark> --mode batch
retriever harness run <benchmark> --set query.top_k=20
retriever harness run <benchmark> --set ingest.profile=fast-text
retriever harness run --runfile /tmp/ablation.yaml
retriever harness run-set <runset>
retriever harness diff <run-a-dir> <run-b-dir> --json
```

Notes:

- `--mode local` is the default and maps to Retriever ingest/query settings.
- `--mode batch` is benchmark intent, not a separate harness engine.
- `--set` parses values with YAML/JSON scalar semantics.
- `--dry-run` should be available on `run` and `run-set`.
- `--json` on read-only commands writes machine-readable output to stdout.
  Human formatting remains non-contractual.

Defer or remove:

- legacy `sweep`
- legacy `compare`
- legacy `nightly`
- legacy `portal`
- legacy `runner`

These can return after the new artifact contract is stable.

`diff` can be a small new command, not the legacy compare implementation. It
should read two artifact directories and emit changed `summary_metrics` plus
selected BEIR deltas. Agents need this primitive more than humans need a full
reporting UI.

## Functional Validation

The harness should be validated like an evaluation runner, not like a conventional
pytest-heavy library component. Developers should prove behavior by running the
harness commands and inspecting stable artifacts.

Minimum local validation commands:

```bash
retriever harness list --json
retriever harness show jp20_beir --json
retriever harness run jp20_beir --dry-run --output-dir /tmp/retriever-harness-dry-run
retriever harness run jp20_beir --dry-run --require 'files>=20'
```

Functional validation should assert:

- expected exit code
- parseable `--json` output for read-only commands
- `status.json` exists and has the expected final status
- `events.jsonl` exists and includes phase transitions
- `resolved_benchmark.json`, `summary_metrics.json`, and `results.json` exist
- `run.log` exists for non-dry execution runs and contains suppressed
  lower-level stdout/stderr
- invalid overrides exit with code `2`
- missing datasets exit with code `3`
- metric gate failures write artifacts and exit with code `20`

Longer validation runs should use real benchmark datasets and BEIR evaluation:

```bash
retriever harness run jp20_smoke --output-dir /tmp/retriever-harness-jp20-smoke
retriever harness run jp20_beir --output-dir /tmp/retriever-harness-jp20-beir
retriever harness run jp20_beir \
  --output-dir /tmp/retriever-harness-jp20-beir-gated \
  --require 'files==20' \
  --require 'pages==1940' \
  --require 'query_count==115' \
  --require 'recall_5>=0.85' \
  --require 'ndcg_10>=0.75'
```

`jp20_smoke` is a cheap fast-text ingest check over the JP20 corpus and does not
run BEIR queries. `jp20_beir` runs the full JP20 end-to-end path: ingest, BEIR
query iteration, BEIR runfile output, and summary recall/NDCG metrics.
Known dataset facts, benchmark result ranges, and suggested gates should live in
`nemo_retriever/harness/EXPECTED_RESULTS.md`, not in benchmark Python code.

The harness may include tiny no-GPU fixtures or fake benchmark specs to make
developer validation cheap, but the primary confidence signal is functional
execution through the CLI and artifact contract. Existing pytest-style harness
tests can be retired as this replacement matures.

## Implementation Plan

### Phase 1: New Core Runner

- Add typed `DatasetSpec`, `BenchmarkSpec`, and `RunSet` models.
- Add benchmark registry with a small set of named BEIR benchmarks.
- Add `resolve_query_plan()` to shared CLI workflow code.
- Implement `retriever harness list`, `show`, `run`, and `run-set`.
- Implement full BEIR query execution and metrics output.
- Implement stable `summary_metrics`.
- Implement `status.json`, `events.jsonl`, typed failure payloads, and stable
  exit codes.
- Persist suppressed non-dry execution logs in `run.log`.
- Support explicit `--require` gates.
- Add a cheap functional validation path that exercises artifact writing without
  requiring a large GPU run.
- Retire legacy pytest harness coverage that assumes the old graph-pipeline
  harness design. Functional CLI runs are the validation contract.

### Phase 2: Real Dataset Validation

- Run `jp20_smoke` locally.
- Run one full BEIR benchmark on expected hardware.
- Validate that `summary_metrics`, BEIR metrics, and runfiles are sufficient for
  debugging failures.
- Tune default benchmark registry entries.

### Phase 3: Reporting

- Rebuild `summary` around the new artifact contract.
- Build `diff --json` around `summary_metrics` and BEIR deltas.
- Rebuild richer `compare` only if the team needs it after stable artifacts
  exist.
- Add nightly runner and Slack sink later.

## Open Decisions

- Which named benchmarks should be phase-one defaults?
- Should `bo10k` be included immediately or wait until smaller BEIR datasets are
  stable?
- What is the canonical page/doc ID mapping for each dataset's BEIR qrels?
- Which query latency metric should gate regressions: p50, p95, or both?
- Do we want `--mode batch` to be accepted for all benchmarks or only for specs
  that declare batch-safe tuning defaults?

## Risks

- If we expose too much arbitrary config, this becomes config hell. Keep the
  registry as the source of truth and make runfiles extend named benchmarks.
- If query execution shells out per query, latency metrics become mostly startup
  noise. Build one Retriever per run.
- If BEIR ID mapping is inconsistent across datasets, summary quality metrics
  will be misleading. Dataset specs need explicit doc ID policy.
- If `summary_metrics` changes frequently, downstream reporting and agents will
  become brittle. Treat key names as a contract.
- If agents need to scrape stdout, the artifact contract failed. Add or fix a
  machine-readable file instead of documenting text output.

## Acceptance Criteria

- `retriever harness list` shows named built-in benchmarks.
- `retriever harness show jp20_beir --json` emits the resolved benchmark as
  machine-readable JSON.
- `retriever harness run jp20_beir --dry-run` resolves ingest, query,
  evaluation, and artifact paths without touching graph-pipeline code.
- `retriever harness run jp20_beir` runs ingest, queries the full BEIR query
  set, writes BEIR outputs, and emits stable `summary_metrics`.
- `retriever harness run-set <name>` expands a code-owned ablation and writes
  `expanded_runs.json`.
- Every run writes `status.json`, `events.jsonl`, `summary_metrics.json`, and
  `results.json`.
- Failed runs write typed failure data without requiring stdout inspection.
- Unknown `--set` keys fail before execution.
- The harness has no user-facing `--engine` flag.
- The harness does not duplicate Typer options from `retriever ingest` or
  `retriever query`.
- No phase-one code path invokes `retriever pipeline run` or
  `nemo_retriever.examples.graph_pipeline`.
- CLI text formatting is explicitly non-contractual; machine consumers use
  artifact files or `--json` read-only commands.
- Phase-one validation is functional and artifact-driven, not pytest-gated.

## References

- [MLPerf Inference benchmark suite](https://github.com/mlcommons/inference):
  benchmark suite for measuring inference speed across deployment scenarios.
- [MLPerf Inference paper](https://arxiv.org/abs/1911.02549): describes the
  LoadGen/SUT split, accuracy/performance modes, and reproducibility goals.
- [BEIR repository](https://github.com/beir-cellar/beir): heterogeneous IR
  benchmark and evaluation framework.
- [ASV documentation](https://asv.readthedocs.io/en/latest/): Python benchmark
  suites over time with saved results.
- [Phoronix Test Suite](https://www.phoronix-test-suite.com/): benchmark
  profiles, suites, batch operation, saved results, and comparisons.
- [Hydra configuration overview](https://hydra.cc/docs/configure_hydra/intro/):
  useful reference for why we should avoid importing a full multi-file config
  composition framework unless the harness truly needs it.
