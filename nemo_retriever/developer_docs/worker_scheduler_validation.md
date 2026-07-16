# Worker scheduler local validation

Use `retriever-dev` to create and exercise the GPU development cluster. When
working from a worktree, always point the CLI at that checkout; otherwise it
defaults to `/workspaces/NeMo-Retriever` and can build a different branch.

```bash
export RETRIEVER_ROOT="$PWD"
RETRIEVER_ROOT="$PWD" retriever-dev doctor
export NGC_API_KEY=...  # required for a full NIM deployment
RETRIEVER_ROOT="$PWD" retriever-dev cluster create
RETRIEVER_ROOT="$PWD" retriever-dev deploy \
  --values nemo_retriever/helm/values-split-scheduler-test.yaml
RETRIEVER_ROOT="$PWD" retriever-dev cluster status
RETRIEVER_ROOT="$PWD" retriever-dev verify
```

The scheduler overlay leaves `ServiceMonitor` disabled because stock
`retriever-dev` clusters do not install that CRD. Enable it only when a
Prometheus Operator is present.

## Scheduler state boundary

The gateway scheduler is process-local. A gateway restart, rollout, eviction,
or node failure loses all accepted jobs, queued work, active leases, job-status
history, and SSE catch-up state owned by that process. Status requests for old
jobs return not found; old callbacks and heartbeats return `409`. Clients must
create and submit a new job. Worker deletion remains recoverable through lease
expiry as long as the gateway process remains alive.

Before upgrading from a durable scheduler checkpoint, drain the gateway. The
new implementation does not read or delete an older `gateway-state.sqlite3` or
payload spool on the general PVC. After rollback is no longer needed, those old
files may be removed manually.

## Dataset ladder

Use the smallest corpus that exercises the scenario:

| Corpus | Documents | Approximate size | Use |
|---|---:|---:|---|
| `~/datasets/jp20/corpus` | 20 | 188 MB | ingest and result-routing smoke |
| `~/datasets/financebench/corpus` | 369 | 727 MB | default result-routing regression |
| `~/datasets/earnings_consulting/corpus` | 514 | 1.29 GB | medium load |
| `~/datasets/bo767/corpus` | 767 | 6.31 GB | scale, loss boundary, churn, HPA tail, throughput |

Run the smoke test before longer fault scenarios:

```bash
retriever-dev test result-routing \
  --dataset ~/datasets/jp20/corpus --documents 20
```

## Fault acceptance

- Accept a test job, restart the gateway, and verify the old job is not found,
  its old lease callbacks and heartbeats return `409`, and a newly created job
  starts with zero scheduler demand.
- Delete an active worker without restarting the gateway and verify lease expiry
  requeues its work and the job completes.
- Run six batch pods with 48 execution slots each. Active batch leases never
  exceed the configured cluster cap of 48 and NIMs do not restart.
- During the active-only tail, confirm `nemo_retriever_work_queue_demand`
  remains nonzero even when raw queued work is zero.
- Exercise payload fetch and hash failures through three claims. The third
  failed delivery marks the document failed and removes its payload.
- Verify gateway startup removes orphan `*.payload` files and graceful shutdown
  removes all queued and leased payloads.

For the 8xA100 run, gateway restart is intentionally excluded from zero-loss
acceptance criteria. Focus scale validation on the 48-lease cap, worker churn,
NIM stability, active-only demand, retrieval quality, and throughput.

Before publishing capacity or throughput claims, repeat clean six/eight-pod
baseline and remediated BO767 cells three times. Preserve retrieval quality and
retain run metadata, pod distributions, GPU telemetry, fault logs, rendered
values, and retrieval metrics with the July 16 experiment artifact shape.
