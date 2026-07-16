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
  --values nemo_retriever/helm/values-split-durable-test.yaml
RETRIEVER_ROOT="$PWD" retriever-dev cluster status
RETRIEVER_ROOT="$PWD" retriever-dev verify
```

Keep the cluster alive across gateway restart tests: its PVC is the durability
boundary. `retriever-dev cluster delete` is teardown and removes cluster-local
state.

## Dataset ladder

Use the smallest corpus that exercises the scenario:

| Corpus | Documents | Approximate size | Use |
|---|---:|---:|---|
| `~/datasets/jp20/corpus` | 20 | 188 MB | ingest and result-routing smoke |
| `~/datasets/financebench/corpus` | 369 | 727 MB | default result-routing regression |
| `~/datasets/earnings_consulting/corpus` | 514 | 1.29 GB | medium load |
| `~/datasets/bo767/corpus` | 767 | 6.31 GB | scale, restart, churn, HPA tail, throughput |

Run the smoke test before longer fault scenarios:

```bash
retriever-dev test result-routing \
  --dataset ~/datasets/jp20/corpus --documents 20
```

## Fault acceptance

- Restart the gateway during BO767 ingest with queued and leased work. The
  original job remains queryable, accepts remaining uploads, rejects stale
  callbacks with `409`, and reaches 767 terminal documents without loss.
- Run six batch pods with 48 execution slots each. Active batch leases never
  exceed the configured cluster cap of 48 and NIMs do not restart.
- Delete an active worker and verify lease expiry requeues its work.
- During the active-only tail, confirm
  `nemo_retriever_work_queue_demand` remains nonzero even when raw queued work
  is zero.
- Exercise payload fetch and hash failures through three claims. The third
  failed delivery marks the document failed and removes its payload.

Before publishing capacity or throughput claims, repeat clean six/eight-pod
baseline and remediated BO767 cells three times. Preserve retrieval quality and
retain run metadata, pod distributions, GPU telemetry, fault logs, rendered
values, and retrieval metrics with the July 16 experiment artifact shape.
