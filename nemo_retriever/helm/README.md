# nemo-retriever Helm chart

A Kubernetes Helm chart for running the **service** mode of
[`nemo-retriever`](../README.md): a FastAPI document ingestion server that
streams uploads through five NVIDIA NIM microservices
(page-elements, graphic-elements, table-structure, OCR, embed) and exposes
result + status APIs over HTTP / SSE.

**Unsupported developer path:** ad-hoc **Docker Compose** workflows (not
chart-managed) are documented separately in [`../docker.md`](../docker.md).
Use **Helm** (this chart and/or the **additional Library charts** documented in the
[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/))
for supported NIM and service deployment.

The chart ships two deployable layers behind feature flags:

- **the service** — always on; one Deployment built from
  `nemo_retriever/Dockerfile --target service`.
- **the five NIMs** — optional, GPU-backed Deployments wired into the
  service config automatically when `nims.enabled=true`.

> **Persistence today is SQLite on a single ReadWriteOnce PVC**, which caps
> the service at one replica. The chart already exposes the HPA scaffolding
> so it's a one-line change once the planned PostgreSQL backend lands.

---

## Layout

```
nemo_retriever/helm/
├── Chart.yaml
├── values.yaml
├── README.md            <-- this file
├── .helmignore
└── templates/
    ├── _helpers.tpl
    ├── NOTES.txt
    ├── configmap.yaml          # renders retriever-service.yaml
    ├── deployment.yaml         # the service Deployment
    ├── service.yaml            # ClusterIP for the service
    ├── ingress.yaml            # optional Ingress
    ├── hpa.yaml                # optional HorizontalPodAutoscaler
    ├── servicemonitor.yaml     # optional Prometheus ServiceMonitor
    ├── serviceaccount.yaml
    ├── pvc.yaml                # SQLite database PVC
    ├── secret-nim-api-key.yaml # chart-managed NVIDIA_API_KEY
    ├── secret-pull.yaml        # chart-managed dockerconfigjson
    └── nims/
        ├── _nim.tpl                # named template per NIM
        ├── nims.yaml               # iterates the 5 NIMs
        └── secret-ngc-api-key.yaml # NGC_API_KEY for NIM containers
```

---

## Quick start

### 1. Service image

The chart defaults to the staging image published to NGC:

```
nvcr.io/nvstaging/nim/nemo-retriever-service:043020205-001
```

Pulling from `nvcr.io/nvstaging` requires an NGC pull secret — either set
`imagePullSecret.create=true` (see below) or reference a pre-existing one
via `imagePullSecrets`.

To run a locally built image instead, build and push it from the repo root,
then override `service.image.repository` / `service.image.tag`:

```bash
# from the repo root:
docker build \
    -f nemo_retriever/Dockerfile \
    --target service \
    -t <YOUR_REGISTRY>/nemo-retriever-service:<TAG> .
docker push <YOUR_REGISTRY>/nemo-retriever-service:<TAG>
```

### 2. Install with NIMs disabled (talks to external NIMs)

If you already have NIM endpoints reachable from the cluster, the smallest
useful install looks like:

```bash
helm install retriever ./nemo_retriever/helm \
  --set imagePullSecret.create=true \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set serviceConfig.nimEndpoints.pageElementsInvokeUrl=http://page-elements.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.graphicElementsInvokeUrl=http://graphic-elements.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.tableStructureInvokeUrl=http://table-structure.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.ocrInvokeUrl=http://ocr.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.svc:8000/v1/embeddings \
  --set nimApiKey.value=$NVIDIA_API_KEY
```

### 3. Install with the NIMs deployed by this chart

This requires GPU nodes and an NGC pull secret + API key:

```bash
helm install retriever ./nemo_retriever/helm \
  --set imagePullSecret.create=true \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set nims.enabled=true \
  --set nims.ngcApiKey.value=$NGC_API_KEY
```

The chart auto-wires the in-cluster URLs of the deployed NIMs into the
service's `nim_endpoints` block, so no further config is needed for the
common case.

---

## Values reference (highlights)

The full schema lives in [`values.yaml`](./values.yaml). Below is the
short list of knobs you'll touch first.

### Service

| Path                          | Default                            | Notes |
|-------------------------------|------------------------------------|-------|
| `service.image.repository`    | `nvcr.io/nvstaging/nim/nemo-retriever-service` | Staging NGC image; requires NGC pull secret. |
| `service.image.tag`           | `043020205-001`                    |       |
| `service.replicas`            | `1`                                | Hard cap = 1 while SQLite is the backend. |
| `service.resources.requests`  | `500m / 1Gi`                       | Tune in tandem with `serviceConfig.processing.numWorkers`. |
| `service.resources.limits`    | `4 / 8Gi`                          |       |
| `service.gpu.enabled`         | `false`                            | The service does **not** need a GPU. |

### Service configuration (rendered into `retriever-service.yaml`)

| Path                                              | Default | Notes |
|---------------------------------------------------|---------|-------|
| `serviceConfig.server.port`                       | `7670`  | Container + Service port. |
| `serviceConfig.processing.numWorkers`             | `16`    | Per-pod worker processes. |
| `serviceConfig.processing.batchSize`              | `16`    | Pages per NIM batch. |
| `serviceConfig.processing.batchTimeoutS`          | `2.0`   | Max wait before flushing a partial batch. |
| `serviceConfig.nimEndpoints.*InvokeUrl`           | `""`    | Used as-is when `nims.enabled=false`. |
| `serviceConfig.database.path`                     | `/var/lib/nemo-retriever/retriever-service.db` | Lives on the PVC. |

### NIM sub-stack (`nims.enabled=true`)

Every NIM block is **merged on top of `nims.defaults`**, so you only override
what differs (image, replica count, etc.).

| Path                                | Default                                                | Notes |
|-------------------------------------|--------------------------------------------------------|-------|
| `nims.enabled`                      | `false`                                                | Master switch for the GPU sub-stack. |
| `nims.defaults.replicas`            | `1`                                                    | Per-NIM. |
| `nims.defaults.resources.limits.nvidia.com/gpu` | `1`                                       | One GPU per NIM pod. |
| `nims.defaults.cache.size`          | `100Gi`                                                | Per-NIM model cache PVC. |
| `nims.<nim>.image.repository`       | nvcr.io/nim/nvidia/...                                 | Override per NIM. |
| `nims.<nim>.invokePath`             | `/v1/infer` (or `/v1/embeddings` for `embed`)          | Appended to the auto-generated URL. |
| `nims.ngcApiKey.value`              | `""`                                                   | Required when `nims.enabled=true`. |

### Persistence

| Path                       | Default                       | Notes |
|----------------------------|-------------------------------|-------|
| `persistence.enabled`      | `true`                        |       |
| `persistence.size`         | `20Gi`                        |       |
| `persistence.accessModes`  | `[ReadWriteOnce]`             | Required by SQLite. |
| `persistence.storageClass` | `""`                          | Use cluster default unless set. Use `"-"` to disable a `storageClassName`. |
| `persistence.mountPath`    | `/var/lib/nemo-retriever`     | Both DB and log file are written here. |

### Secrets

| Path                              | Default                  | Notes |
|-----------------------------------|--------------------------|-------|
| `nimApiKey.value`                 | `""`                     | Inline value; chart creates the Secret. |
| `nimApiKey.existingSecret`        | `""`                     | Or point at an existing Secret. |
| `imagePullSecret.create`          | `false`                  | When true, render an NGC pull secret. |
| `imagePullSecret.password`        | `""`                     | NGC API key (chart sets username automatically). |

### Optional features

| Feature           | Toggle                          | Default |
|-------------------|---------------------------------|---------|
| Ingress           | `ingress.enabled`               | `false` |
| Autoscaling (HPA) | `autoscaling.enabled`           | `false` (max=1 anyway) |
| ServiceMonitor    | `serviceMonitor.enabled`        | `false` (no `/metrics` endpoint exists yet) |

---

## Configuration recipes

### Mount a custom retriever-service.yaml verbatim

The chart renders `retriever-service.yaml` from structured values so you
shouldn't normally need to ship a verbatim file. If you really want to,
mount one via `service.extraVolumes` + `service.extraVolumeMounts` at
`/etc/nemo-retriever/retriever-service.yaml` (which silently overrides the
chart-managed ConfigMap because `subPath` mounts win).

### Use an externally managed Secret

```yaml
nimApiKey:
  existingSecret: my-team-nvidia-secret
  existingSecretKey: NVIDIA_API_KEY
```

The chart will skip Secret creation and inject `NVIDIA_API_KEY` from your
existing Secret.

### Disable a single NIM and supply an external URL for it

```yaml
nims:
  enabled: true
  embed:
    enabled: false  # don't deploy the embed NIM in-cluster

serviceConfig:
  nimEndpoints:
    embedInvokeUrl: https://integrate.api.nvidia.com/v1/embeddings
```

The chart's resolution order is **explicit URL → in-cluster URL → empty**,
so per-endpoint overrides Just Work.

### Roll the service after editing values

The `Deployment` carries a `checksum/config` annotation derived from the
ConfigMap, so `helm upgrade` automatically rolls the pod when any
`serviceConfig.*` value changes.

---

## Queue-depth autoscaling (split mode)

In `topology.mode: split` deployments the realtime and batch worker
pods scale horizontally based on **queue fill ratio** and
**95th-percentile processing latency**. Both signals come straight out
of the pods' `/metrics` endpoint — the publisher is always on (see
`nemo_retriever_pool_queue_depth_ratio` in
[`prometheus.py`](../src/nemo_retriever/service/services/prometheus.py)).
The only choice you have to make is **how the metrics get from
Prometheus into the Kubernetes HPA**.

### Why queue depth (and not CPU)

CPU-based HPA reacts to *the pod that has already saturated its work*.
For an ingest pipeline that fans out to remote NIM endpoints, the work
spends most of its time blocked on HTTP — CPU stays low even when the
queue is full. Queue depth measures *demand to be served*, which is
what we actually want to scale on. A 95th-percentile-latency signal
rides alongside to catch the inverse case (a single hot pod whose
queue is shallow but whose per-item processing has stalled).

### Backend choices

The chart's `autoscaling.queueDepth.backend` controls which path is
wired up. All three options leave the metrics publisher untouched:

| backend                | When to pick it                                                  | Cluster prerequisite              |
|------------------------|------------------------------------------------------------------|-----------------------------------|
| `prometheus-adapter` *(default)* | Production. One adapter feeds HPA + Grafana + future autoscalers. | Prometheus Operator + `prometheus-community/prometheus-adapter`. |
| `cpu`                  | Bootstrap / dev cluster without Prometheus.                      | None — built-in.                   |
| `keda`                 | Already standardised on KEDA org-wide.                           | KEDA operator (you install + apply your own `ScaledObject`). |

The chart-recommended path is `prometheus-adapter`. The reasoning is
documented in `values.yaml`; in short, it keeps a single Prometheus as
the source of truth, supports HPA's multi-metric arithmetic-mean
evaluation out of the box, and doesn't force the chart to bundle new
CRDs.

### Wiring up prometheus-adapter (recommended)

The chart renders a ConfigMap named
`<release>-nemo-retriever-prom-adapter-rules` containing PromQL rules
for the External Metrics API. You point your existing
prometheus-adapter at it:

```bash
helm upgrade prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --reuse-values \
  --set rules.existing=<release>-nemo-retriever-prom-adapter-rules
```

Then verify both metrics show up in the External Metrics API:

```bash
kubectl get --raw \
  "/apis/external.metrics.k8s.io/v1beta1/namespaces/$NS/nemo_retriever_pool_queue_depth_ratio_avg?labelSelector=pool%3Drealtime" \
  | jq .
```

Once that returns a non-empty `items` array, the HPAs rendered by this
chart will start consuming them. The HPA annotation
`nemo-retriever.nvidia.com/hpa-signals` documents the active set per
HPA, e.g. `queueRatio=true latencyP95=true cpu=false`.

### CPU fallback (no Prometheus required)

Set `autoscaling.queueDepth.backend: cpu` and enable the CPU metric
under each role:

```yaml
autoscaling:
  queueDepth:
    backend: cpu
topology:
  realtime:
    hpa:
      metrics:
        queueDepthRatio: { enabled: false }
        processingLatencyP95: { enabled: false }
        cpu: { enabled: true, targetUtilizationPercentage: 60 }
  batch:
    hpa:
      metrics:
        queueDepthRatio: { enabled: false }
        processingLatencyP95: { enabled: false }
        cpu: { enabled: true, targetUtilizationPercentage: 80 }
```

The legacy `topology.<role>.hpa.targetCPUUtilizationPercentage` field
still works and behaves as an alias for the `metrics.cpu` block.

### KEDA path

Set `autoscaling.queueDepth.backend: keda` and disable the chart-managed
HPAs:

```yaml
autoscaling:
  queueDepth: { backend: keda }
topology:
  realtime: { hpa: { enabled: false } }
  batch:    { hpa: { enabled: false } }
```

Then apply your own `ScaledObject` — example for the realtime pool:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: nemo-retriever-realtime
spec:
  scaleTargetRef:
    name: nemo-retriever-realtime
  minReplicaCount: 2
  maxReplicaCount: 8
  cooldownPeriod: 300
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: nemo_retriever_pool_queue_depth_ratio
        threshold: "0.5"
        query: |
          avg by (pool) (
            nemo_retriever_pool_queue_depth{pool="realtime"}
            /
            on(pool, instance) group_left()
            nemo_retriever_pool_max_queue_size{pool="realtime"}
          )
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: nemo_retriever_pool_processing_duration_p95
        threshold: "30"
        query: |
          histogram_quantile(
            0.95,
            sum by (le, pool) (
              rate(nemo_retriever_pool_processing_duration_seconds_bucket{pool="realtime"}[2m])
            )
          )
```

KEDA's biggest win is **scale-from-zero**, which we don't use today —
both `minReplicas` defaults are ≥ 1 because the realtime pod is on the
hot path for SSE consumers. If you do want scale-from-zero (e.g. a
nightly batch-only job tenant), KEDA is the right tool and this is the
escape hatch.

### Tuning the thresholds

Per-role tuning lives under `topology.<role>.hpa.metrics`:

```yaml
topology:
  realtime:
    hpa:
      metrics:
        queueDepthRatio: { enabled: true, target: "500m" }   # 0.5
        processingLatencyP95: { enabled: true, targetSeconds: "30" }
  batch:
    hpa:
      metrics:
        queueDepthRatio: { enabled: true, target: "700m" }   # 0.7 — batch can run hot
        processingLatencyP95: { enabled: true, targetSeconds: "120" }
```

Quantity-string conventions are k8s standard: `500m == 0.5`, `2`, `2k`,
etc. The `target` is **per-replica** because the HPA template uses
`type: AverageValue` for both External metrics — that's what makes
"scale up when *average* queue fill across pods exceeds 0.5" work
without baking the pod count into the publisher.

### Verifying it scales

```bash
# Cause realtime pressure (anything that submits to /v1/ingest/job/.../page).
# Then watch the HPA decide:
kubectl get hpa -w

# And watch the active signals on each HPA:
kubectl get hpa <release>-realtime -o jsonpath='{.metadata.annotations.nemo-retriever\.nvidia\.com/hpa-signals}'
```

The dashboard's *Worker Pool Capacity* card on the **Overview** page
mirrors the same signal Prometheus is seeing, so it's a quick eyeball
sanity check before opening Grafana.

---

## Roadmap

1. **PostgreSQL backend** — replace `service.db.engine.DatabaseEngine` with
   a SQLAlchemy/asyncpg-based engine, then bump the chart to deploy a
   PostgreSQL StatefulSet (or take a sub-chart dependency on Bitnami's
   chart) and lift `service.replicas` to N.
2. **NetworkPolicies** restricting the service Pod to the NIM Pods + DB
   only.
3. **Gateway autoscaling** on inflight-uploads (currently fixed
   `topology.gateway.replicas`) — sticky-routing story for SSE
   subscribers needs to land first.

---

## Validation

The chart is exercised in CI with `helm lint` and `helm template`. Run
locally:

```bash
helm lint nemo_retriever/helm
helm template r nemo_retriever/helm > /tmp/r.yaml          # NIMs off
helm template r nemo_retriever/helm --set nims.enabled=true > /tmp/r-nims.yaml
```

Both renders should succeed cleanly and parse as valid Kubernetes manifests
(`kubectl apply --dry-run=client -f /tmp/r.yaml`).
