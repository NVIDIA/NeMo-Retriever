# Retriever Harness Design History

> This design has been implemented. The
> [Retriever Harness README](../harness/README.md) is the current user and agent
> contract.

The harness was rebuilt as an internal end-to-end benchmark runner for NeMo
Retriever engineers. It ingests registered datasets, queries the resulting
index, evaluates retrieval quality, and writes durable artifacts.

## Decisions That Still Apply

| Decision | Consequence |
| --- | --- |
| Keep benchmark definitions in a typed Python registry. | Recurring benchmarks are reviewed code, not an open-ended YAML system. |
| Treat runfiles as concrete requests. | A runfile selects a registered benchmark; it cannot define a new one. |
| Keep one harness contract across execution targets. | Library and service runs emit the same results, gates, and artifacts. |
| Treat local and batch as library ingest modes. | Batch changes how ingest runs; it is not a separate harness. |
| Treat service as a system-under-test mode. | The harness uses service ingest and query APIs while retaining the same evaluation contract. |
| Keep Helm outside benchmark semantics. | `run-helm` provisions around `run-files`; Helm is not a runfile mode. |
| Make artifacts the API. | Callers use exit codes, `status.json`, `results.json`, and `session_summary.json`, not stdout. |
| Keep execution separate from reporting. | `post-slack` reads completed artifacts and never reruns or mutates them. |
| Keep scheduling outside the harness. | The nightly launcher and its caller own recurrence, locking, Git selection, and secrets. |
| Require explicit gates. | The harness records quality and performance but does not impose a global score. |

## Product Boundary

Use `retriever ingest` and `retriever query` for direct product workflows. Use
`retriever harness` for registered benchmark and evaluation work.

The harness calls the same library workflow code or corresponding service APIs
used by the product. It owns benchmark resolution, run lifecycle, evaluation,
metric gates, and artifacts. Retriever owns ingest and query behavior.

## Non-Goals

The harness is not:

- a public supported product API
- a scheduler, retry system, or secret distributor
- a benchmark history database or regression-policy engine
- a general deployment manager
- a compatibility wrapper for retired pipeline or sweep commands

Git history retains the original implementation PRD and its research notes.
