# nemo-retriever Helm chart

A Kubernetes Helm chart for running the **service** mode of
[`nemo-retriever`](../README.md): a FastAPI document ingestion server that
streams uploads through a set of NVIDIA NIM microservices
(page-elements, table-structure, OCR, VLM embed by default) and exposes
result + status APIs over HTTP / SSE.

**Unsupported developer path:** ad-hoc **Docker Compose** workflows (not
chart-managed) are documented separately in [`../docker.md`](../docker.md).
Use **Helm** (this chart and/or the **additional Library charts** documented in the
[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/))
for supported NIM and service deployment.

The chart ships two deployable layers behind feature flags:

- **the service** — always on; one Deployment (standalone) or three
  Deployments (split topology: gateway / realtime / batch), built from
  `Dockerfile --target service`.
- **the NIMs** — optional, GPU-backed `NIMCache` + `NIMService` custom
  resources (`apiVersion: apps.nvidia.com/v1alpha1`) reconciled by the
  **NVIDIA NIM Operator**. The chart auto-wires the operator-managed
  Service URLs into the retriever-service config when the operator CRDs
  are present in the cluster.

> **NIM Operator prerequisite.** The NIM templates are gated on the
> `apps.nvidia.com/v1alpha1` API group. Install the NIM Operator before
> running `helm install`:
> https://docs.nvidia.com/nim-operator/
>
> Without the operator the chart still installs cleanly — every NIMCache /
> NIMService template short-circuits and the service falls back to
> external NIM URLs supplied via `serviceConfig.nimEndpoints.*`.

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
    ├── configmap.yaml                         # renders retriever-service.yaml
    ├── deployment.yaml                        # the service Deployment(s)
    ├── service.yaml                           # ClusterIP/NodePort for the service
    ├── ingress.yaml                           # optional Ingress
    ├── hpa.yaml                               # optional HorizontalPodAutoscaler
    ├── servicemonitor.yaml                    # optional Prometheus ServiceMonitor
    ├── serviceaccount.yaml
    ├── pvc.yaml                               # SQLite database PVC
    ├── secrets.yaml                           # ngc-secret + ngc-api
    └── nims/
        ├── nemotron-page-elements-v3.yaml     # NIMCache + NIMService
        ├── nemotron-table-structure-v1.yaml   # NIMCache + NIMService
        ├── nemotron-ocr-v1.yaml               # NIMCache + NIMService
        ├── llama-nemotron-embed-vl-1b-v2.yaml           # NIMCache + NIMService (VLM embed)
        ├── llama-nemotron-rerank-1b-v2.yaml   # NIMCache + NIMService (optional; enabled by default; not auto-wired)
        ├── nemotron-parse.yaml                # NIMCache + NIMService (optional; enabled by default; not auto-wired)
        ├── nemotron-3-nano-omni-30b-a3b-reasoning.yaml  # NIMCache + NIMService (optional; enabled by default; not auto-wired)
        └── audio.yaml                         # NIMCache + NIMService (optional; enabled by default; not auto-wired)
```

---

## Quick start

### 1. Service image

The chart defaults to the staging image published to NGC:

```
nvcr.io/nvstaging/nim/nemo-retriever-service:043020205-001
```

Pulling from `nvcr.io/nvstaging` requires an NGC pull secret — either set
`ngcImagePullSecret.create=true` (see below) or pre-create one in the
namespace named `ngc-secret`.

To run a locally built image instead, build and push it from the repo root,
then override `service.image.repository` / `service.image.tag`:

```bash
# from the repo root:
docker build \
    --target service \
    -t <YOUR_REGISTRY>/nemo-retriever-service:<TAG> .
docker push <YOUR_REGISTRY>/nemo-retriever-service:<TAG>
```

Audio and video extraction require the `ffmpeg` and `ffprobe` system
binaries inside the service container. The bundled service image can install
them at container startup when you set `service.installFfmpeg=true`, which
sets `INSTALL_FFMPEG=true` for the image entrypoint:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set service.image.repository=<YOUR_REGISTRY>/nemo-retriever-service \
  --set service.image.tag=<TAG> \
  --set service.installFfmpeg=true
```

Do not also set `INSTALL_FFMPEG` in `service.env`; the chart fails rendering
when both are configured so the rendered Pod does not contain duplicate
environment variables.

Runtime installation uses passwordless `sudo` scoped to installing the
`ffmpeg` package in the service image. The pod must have network egress to the
Ubuntu package repositories, a writable root filesystem, and a security policy
that allows sudo/setuid behavior. Do not set
`service.securityContext.allowPrivilegeEscalation: false` or
`service.securityContext.readOnlyRootFilesystem: true` for this path.

For air-gapped or locked-down clusters, use a custom image with ffmpeg/ffprobe
already installed — see [Air-gapped deployment](#air-gapped-deployment). On a
connected staging host you can extend the service image, for example:

```dockerfile
FROM <YOUR_REGISTRY>/nemo-retriever-service:<BASE_TAG>
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER nemo
```

### 2. Install with external NIM endpoints (operator not required)

If you already have NIM endpoints reachable from the cluster (e.g. another
namespace, or NVIDIA Build), turn the master switch off and supply the
URLs directly:

```bash
helm install retriever ./nemo_retriever/helm \
  --set nims.enabled=false \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set serviceConfig.nimEndpoints.pageElementsInvokeUrl=http://page-elements.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.tableStructureInvokeUrl=http://table-structure.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.ocrInvokeUrl=http://ocr.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.svc:8000/v1/embeddings
```

`ngcApiSecret` materialises an `ngc-api` Secret containing both
`NGC_API_KEY` and `NGC_CLI_API_KEY` keys; the service container reads it
via `optional: true` `secretKeyRef`, so the install still succeeds when
the secret is absent (useful for fully local NIM endpoints).

### 3. Install with the NIM Operator (in-cluster NIMs)

Install the [NIM Operator](https://docs.nvidia.com/nim-operator/) first so
the `NIMCache` / `NIMService` CRDs (`apps.nvidia.com/v1alpha1`) are
registered. Then run the default install — `nims.enabled` is `true` out
of the box, so every per-NIM block under `nimOperator.<key>.enabled: true`
(all eight by default) is reconciled:

```bash
helm install retriever ./nemo_retriever/helm \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY
```

The chart auto-wires the operator-managed in-cluster URLs of the four
"core" NIMs into the service's `nim_endpoints` block:

| key | operator-managed Service | invoke path |
| --- | ------------------------ | ----------- |
| `nimOperator.page_elements`   | `nemotron-page-elements-v3`   | `/v1/infer`      |
| `nimOperator.table_structure` | `nemotron-table-structure-v1` | `/v1/infer`      |
| `nimOperator.ocr`             | `nemotron-ocr-v1`             | `/v1/infer`      |
| `nimOperator.vlm_embed`       | `llama-nemotron-embed-vl-1b-v2` | `/v1/embeddings` |

Track operator reconciliation with:

```bash
kubectl get nimcache,nimservice -n <namespace>
kubectl describe nimservice nemotron-page-elements-v3 -n <namespace>
```

First-time NIMCache reconciliation downloads model weights to a PVC; the
NIMCache resources carry the `helm.sh/resource-policy: keep` annotation so
those downloads survive `helm uninstall`.

---

## Values reference (highlights)

The full schema lives in [`values.yaml`](./values.yaml). Below is the
short list of knobs you'll touch first.

### Service

| Path                          | Default                            | Notes |
|-------------------------------|------------------------------------|-------|
| `service.image.repository`    | `localhost:32000/nemo-retriever-service` | Override to a published image. |
| `service.image.tag`           | `latest`                           |       |
| `service.replicas`            | `1`                                | Hard cap = 1 while SQLite is the backend. |
| `service.installFfmpeg`       | `false`                            | Install `ffmpeg`/`ffprobe` at container startup by setting `INSTALL_FFMPEG=true`. Requires network egress, writable root filesystem, and sudo/setuid allowed. Not for air-gapped clusters — use a custom image instead. |
| `service.resources.requests`  | `16 / 16Gi`                        | Tune in tandem with `serviceConfig.pipeline.*Workers`. |
| `service.resources.limits`    | `96 / 96Gi`                        |       |
| `service.gpu.enabled`         | `false`                            | The service does **not** need a GPU. |

For audio and video extraction, set `service.installFfmpeg=true` when your
cluster allows runtime package installation. If your cluster blocks startup
package installation (for example air-gapped environments), use a custom
service image that already contains ffmpeg/ffprobe and set
`service.image.repository` and `service.image.tag`.

### Service configuration (rendered into `retriever-service.yaml`)

| Path                                              | Default | Notes |
|---------------------------------------------------|---------|-------|
| `serviceConfig.server.port`                       | `7670`  | Container + Service port. |
| `serviceConfig.pipeline.realtimeWorkers`          | `24`    | Per-pod realtime worker count. |
| `serviceConfig.pipeline.batchWorkers`             | `48`    | Per-pod batch worker count. See [Timeouts and alleviating ingest failures](#timeouts-and-alleviating-ingest-failures) if embed or pool errors appear under load. |
| `serviceConfig.nimEndpoints.*InvokeUrl`           | `""`    | Override the auto-resolved NIM Operator URL. |
| `serviceConfig.vectordb.lancedbUri`               | `/data/vectordb` | LanceDB on the vectordb Pod's PVC. |
| `serviceConfig.vectordb.embedModel`               | `nvidia/llama-nemotron-embed-vl-1b-v2` | Passed to vectordb + worker `embed_model_name`. |

### NIM Operator sub-stack

Each NIM block under `nimOperator.<key>` renders a `NIMCache` + `NIMService`
pair gated on three conditions ALL holding:

1. The `apps.nvidia.com/v1alpha1` CRDs are installed in the cluster.
2. The master switch `nims.enabled` is `true`.
3. The per-NIM `nimOperator.<key>.enabled` is `true`.

| Path                                   | Default | Notes |
|----------------------------------------|---------|-------|
| `nims.enabled`                         | `true`  | Master switch. Set false to render no NIM resources. |
| `nimOperator.page_elements.enabled`    | `true`  | Page-elements detector NIM. |
| `nimOperator.table_structure.enabled`  | `true`  | Table-structure detector NIM. |
| `nimOperator.ocr.enabled`              | `true`  | OCR NIM. |
| `nimOperator.vlm_embed.enabled`        | `true`  | Multimodal embedding NIM (also used by the vectordb Pod). |
| `nimOperator.vlm_embed.nimServiceName` | `llama-nemotron-embed-vl-1b-v2` | NIMService / in-cluster DNS name. |
| `nimOperator.vlm_embed.image`          | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` | Default VLM embed NIM image. |
| `nimOperator.rerankqa.enabled`         | `true`  | Reranker NIM. |
| `nimOperator.nemotron_parse.enabled`   | `true`  | Structured-parse NIM. |
| `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled` | `true` | Multimodal reasoning LLM (30B). |
| `nimOperator.audio.enabled`            | `true`  | ASR NIM. |
| `nimOperator.<key>.image.repository`   | `nvcr.io/nim/nvidia/...` | Per-NIM image. |
| `nimOperator.<key>.image.pullSecrets`  | `[ngc-secret]` | Referenced by the NIMService CR. |
| `nimOperator.<key>.authSecret`         | `ngc-api`      | NIM auth Secret name. |
| `nimOperator.<key>.storage.pvc.size`   | `25Gi` (50Gi for vlm_embed/rerankqa, 100Gi parse, 300Gi VL) | NIMCache PVC size. |
| `nimOperator.<key>.replicas`           | `1`     | Per-NIMService replica count. |
| `nimOperator.<key>.resources.limits.nvidia.com/gpu` | `1` | GPUs per NIM pod. |
| `nimOperator.<key>.expose.service.port` | `8000` (9000 for audio) | HTTP port. |
| `nimOperator.<key>.expose.service.grpcPort` | `8001` (50051 for audio) | gRPC port. |

> Only the four "core" NIMs (page_elements, table_structure, ocr, vlm_embed)
> are auto-wired into the retriever-service config. The other NIMs are
> reconciled by the operator but the retriever-service won't call them
> unless you wire your own pipeline to use them.

**Charts and captioning (26.05).** Charts and infographics use **page_elements**
and **ocr** (no `graphic_elements` operator NIM in this chart). For image
captioning, enable `nemotron_3_nano_omni_30b_a3b_reasoning` — see
[Image captioning (26.05)](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/#image-captioning-2605).

### Persistence

| Path                       | Default                       | Notes |
|----------------------------|-------------------------------|-------|
| `persistence.enabled`      | `true`                        |       |
| `persistence.size`         | `50Gi`                        |       |
| `persistence.accessModes`  | `[ReadWriteOnce]`             | Required by SQLite. |
| `persistence.storageClass` | `""`                          | Use cluster default unless set. Use `"-"` to disable a `storageClassName`. |
| `persistence.mountPath`    | `/var/lib/nemo-retriever`     | Both DB and log file are written here. |

### Secrets

| Path                              | Default        | Notes |
|-----------------------------------|----------------|-------|
| `ngcImagePullSecret.create`       | `false`        | Chart-managed dockerconfigjson Secret. |
| `ngcImagePullSecret.name`         | `ngc-secret`   | Name referenced by every Pod and every NIMService. |
| `ngcImagePullSecret.password`     | `""`           | NGC API key. |
| `ngcApiSecret.create`             | `false`        | Chart-managed Opaque Secret. |
| `ngcApiSecret.name`               | `ngc-api`      | Name referenced by NIMCache/NIMService `authSecret`. |
| `ngcApiSecret.password`           | `""`           | NGC API key (populates `NGC_API_KEY` + `NGC_CLI_API_KEY`). |
| `imagePullSecrets`                | `[]`           | Extra pre-existing pull secrets appended to every Pod. |

### Optional features

| Feature           | Toggle                          | Default |
|-------------------|---------------------------------|---------|
| Ingress           | `ingress.enabled`               | `true`  |
| Autoscaling (HPA) | `autoscaling.enabled`           | `false` (max=1 anyway) |
| ServiceMonitor    | `serviceMonitor.enabled`        | `false` (auto-enabled in split mode) |

---

## Configuration recipes

### Mount a custom retriever-service.yaml verbatim

The chart renders `retriever-service.yaml` from structured values so you
shouldn't normally need to ship a verbatim file. If you really want to,
mount one via `service.extraVolumes` + `service.extraVolumeMounts` at
`/etc/nemo-retriever/retriever-service.yaml` (which silently overrides the
chart-managed ConfigMap because `subPath` mounts win).

### Use externally managed Secrets

```yaml
ngcImagePullSecret:
  create: false        # don't render; reference an existing Secret
  name: my-org-ngc-pull
ngcApiSecret:
  create: false
  name: my-org-ngc-api
```

The chart will skip Secret creation. Make sure `my-org-ngc-pull` exists
as `kubernetes.io/dockerconfigjson` and `my-org-ngc-api` as `Opaque` with
an `NGC_API_KEY` key, in the release namespace.

### Disable one NIM and supply an external URL for it

```yaml
nimOperator:
  vlm_embed:
    enabled: false   # don't deploy the embed NIM in-cluster

serviceConfig:
  nimEndpoints:
    embedInvokeUrl: https://integrate.api.nvidia.com/v1/embeddings
```

The chart's resolution order is **explicit URL → operator-managed URL →
empty**, so per-endpoint overrides Just Work.

### Roll the service after editing values

The `Deployment` carries a `checksum/config` annotation derived from the
ConfigMap, so `helm upgrade` automatically rolls the pod when any
`serviceConfig.*` value changes.

---

## Timeouts and alleviating ingest failures

Batch ingest fans out extract and embed work to remote NIM HTTP endpoints.
Under heavy parallelism a single slow or overloaded NIM can cause timeouts,
and a worker process crash can surface as many simultaneous `failed`
document callbacks even though only one root cause occurred.

### What the chart configures

| Layer | Default | Where it is set |
|-------|---------|-----------------|
| Remote embed HTTP calls | **600 s** (10 min) | Service image (`EmbedParams.request_timeout_s`); not a Helm value today. |
| Gateway → realtime/batch proxy | **300 s** | Rendered `gateway.timeout_s` in `retriever-service.yaml` (split topology). |
| VLM embed model name | `serviceConfig.vectordb.embedModel` | Also copied into worker `nim_endpoints.embed_model_name` in the ConfigMap. |

Symptoms to look for in pod logs:

- `Embedding error occurred: timed out` or `httpx.ReadTimeout` on the **batch** pod.
- `Batch process pool broken (worker crash)` followed by many
  `BrokenProcessPool` failures on other in-flight documents.
- Embed NIM pod messages such as `failed to allocate pinned system memory`
  (GPU pressure from too many concurrent `/v1/embeddings` requests).

The **gateway** pod usually only logs `status=failed` callbacks; diagnose on
**batch** (and **realtime** for page-sized uploads), plus the embed NIM pod.

### Recommended mitigations

**1. Lower batch worker concurrency (first step).**

The default `serviceConfig.pipeline.batchWorkers` is `48`, which can saturate
a single in-cluster VLM embed NIM. If you see embed timeouts or pool crashes,
reduce batch parallelism to **16** and redeploy:

```bash
helm upgrade retriever ./nemo_retriever/helm \
  --reuse-values \
  --set serviceConfig.pipeline.batchWorkers=16
```

You can tune further (for example `8` on small GPU nodes), but **16** is a
reasonable starting point when moving off the default. Realtime workers
(`realtimeWorkers`, default `24`) are less likely to overload embed NIMs
because they handle smaller units of work; adjust them only if realtime
ingest shows the same timeout pattern.

**2. Confirm embed wiring.**

Ensure `nim_endpoints.embed_model_name` in the mounted config matches the
VLM embed NIM SKU (`serviceConfig.vectordb.embedModel`, default
`nvidia/llama-nemotron-embed-vl-1b-v2`). A model mismatch produces
HTTP 404 on `/v1/embeddings`, not a timeout, but is worth ruling out when
debugging failed ingests.

**3. Retry failed documents.**

Failures caused by a one-time pool restart are often transient. After lowering
`batchWorkers` and rolling the batch Deployment, resubmit documents that
failed with `rows=0`.

**4. Scale or isolate the embed NIM.**

If timeouts persist at `batchWorkers: 16`, add embed NIM replicas (when your
cluster has GPU capacity), point `serviceConfig.nimEndpoints.embedInvokeUrl`
at an external embed endpoint, or temporarily disable optional NIMs on
dev clusters to free GPU memory for `vlm_embed`.

**5. Client and ingress timeouts.**

Long batch jobs may exceed the gateway proxy timeout (300 s) or an Ingress
`proxy-read-timeout`. Increase ingress annotations if clients disconnect
while workers are still processing; see the commented example on
`ingress.annotations` in `values.yaml`.

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

## Air-gapped deployment { #air-gapped-deployment }

Use this section when NeMo Retriever runs in a network-isolated environment
(private container registry, offline model caches, no outbound calls to
hosted NVIDIA APIs). It complements
[Deployment options — Air-gapped and disconnected deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/#air-gapped-deployment)
in the product docs.

### What works offline

- **Default pipeline** — With `nims.enabled: true`, the four core NIMs
  (`page_elements`, `table_structure`, `ocr`, `vlm_embed`) and the
  retriever service can run entirely in-cluster after you mirror images and
  preload models per the [NIM Operator air-gap guide](https://docs.nvidia.com/nim-operator/latest/air-gap.html).
- **Optional NIMs** — The four additional operator-managed NIMs
  (`rerankqa`, `nemotron_parse`, `nemotron_3_nano_omni_30b_a3b_reasoning`,
  `audio`) are **enabled by default** in `values.yaml` and reconciled when
  `nimOperator.<key>.enabled` is `true` (set to `false` to omit from your
  mirror set). The retriever service does not call them until you wire your
  pipeline or `serviceConfig.nimEndpoints.*` overrides.

### Audio and video

Audio and video are **not air-gap-ready by default.** The service image omits
`ffmpeg` and `ffprobe`. Do **not** set `service.installFfmpeg=true` in the
enclave (startup install needs package-repo egress). Build a custom image on a
connected staging host ([Service image](#1-service-image)), mirror it, and set
`service.image.repository` / `service.image.tag`. Mirror Parakeet (`audio`) like
any other NIM. See
[Audio and video ingestion](https://docs.nvidia.com/nemo/retriever/latest/extraction/audio-video/).

### End-to-end workflow

1. **Inventory** — List every image your release enables. Start from
   [`values.yaml`](./values.yaml) (`service.image` plus each
   `nimOperator.<key>.image`). Disable optional NIMs you do not need
   (`nimOperator.<key>.enabled: false`) to shrink the mirror set.
2. **Mirror on a staging host** — `docker pull` / `skopeo copy` from
   `nvcr.io` (and any other upstream registries), retag to
   `<PRIVATE_REGISTRY>/...`, record digests, and push to the registry
   reachable inside the air gap. Log in with NGC credentials on the
   staging host only.
3. **Stage charts and wheels** — `helm pull` the chart `.tgz` (or vendor
   `./nemo_retriever/helm` from this repo), copy Python client wheels if
   needed, and transfer operator bundles your policy requires.
4. **Preload NIM models** — Follow
   [NIM Operator: Air-gapped environments](https://docs.nvidia.com/nim-operator/latest/air-gap.html)
   so `NIMCache` PVCs are populated before workloads start.
5. **Install with offline values** — Override repositories/tags,
   configure `imagePullSecrets` for your mirror, set
   `imagePullPolicy: IfNotPresent`, and ensure no runtime URL still points
   at `integrate.api.nvidia.com` or `ai.api.nvidia.com` unless you
   intentionally proxy them. Leave `service.installFfmpeg` at `false`; for
   audio/video, point `service.image` at your ffmpeg-enabled custom image.
6. **Validate** — `helm template` with your values, test image pulls from
   a jump host, then call `GET /v1/health` and run a minimal ingest job.

### Container images to mirror (26.05 chart defaults)

Verify tags on the Git branch or tag you ship (for example `26.05` or
`26.05-RC1`). Defaults below match
[`values.yaml`](./values.yaml) on the current chart.

| Role | `nimOperator` key | Default image (`repository:tag`) |
|------|-------------------|----------------------------------|
| Retriever service | — | `service.image.repository`:`service.image.tag` (override for production) |
| Page elements | `page_elements` | `nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0` |
| Table structure | `table_structure` | `nvcr.io/nim/nvidia/nemotron-table-structure-v1:1.8.0` |
| OCR | `ocr` | `nvcr.io/nim/nvidia/nemotron-ocr-v1:1.3.0` |
| VL embed | `vlm_embed` | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` |
| Reranker (optional) | `rerankqa` | `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0` |
| Nemotron Parse (optional) | `nemotron_parse` | `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant` |
| Omni caption (optional) | `nemotron_3_nano_omni_30b_a3b_reasoning` | `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant` |
| Parakeet ASR (optional) | `audio` | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0` |

Also mirror images for the vectordb sidecar, Redis, or other subcharts if
your values enable them.

### Helm values for a private registry

Example overrides (replace placeholders):

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  -f my-airgap-values.yaml
```

`my-airgap-values.yaml` should include at least:

```yaml
service:
  image:
    repository: <PRIVATE_REGISTRY>/nemo-retriever-service
    tag: <PINNED_TAG>
    pullPolicy: IfNotPresent

imagePullSecrets:
  - name: my-private-registry

ngcImagePullSecret:
  create: false   # use secrets that authenticate to YOUR mirror

nimOperator:
  page_elements:
    image:
      repository: <PRIVATE_REGISTRY>/nemotron-page-elements-v3
      tag: "1.8.0"
      pullPolicy: IfNotPresent
  # Repeat for table_structure, ocr, vlm_embed, and any optional keys you enable.
```

- Set `nimOperator.<key>.image.pullSecrets` to the Secret name your
  `NIMService` resources should use (defaults to `ngc-secret`).
- Leave `serviceConfig.nimEndpoints.*` empty when operator-managed NIMs
  are in-cluster; set explicit URLs only for external or mirrored services
  outside the chart.
- For **offline captioning**, enable
  `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning` and point the pipeline
  caption endpoint at the in-cluster NIM URL (see
  [Image captioning (26.05)](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/#image-captioning-2605)).

### Mirroring pattern

```bash
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
docker pull nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0
docker tag nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0 \
  <PRIVATE_REGISTRY>/nemotron-page-elements-v3:1.8.0
docker push <PRIVATE_REGISTRY>/nemotron-page-elements-v3:1.8.0
```

For bulk sync, prefer [skopeo](https://github.com/containers/skopeo) or
[crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md).
Record `repository@sha256:...` digests for regulated environments.

### Related documentation

- [Pre-Requisites & Support Matrix](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/)
- [NIM Operator — Air-gapped environments](https://docs.nvidia.com/nim-operator/latest/air-gap.html)
- [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)

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
helm template r nemo_retriever/helm > /tmp/r.yaml                                         # operator CRDs absent
helm template r nemo_retriever/helm --api-versions apps.nvidia.com/v1alpha1 > /tmp/r-op.yaml  # operator CRDs present
```

Both renders should succeed cleanly and parse as valid Kubernetes manifests
(`kubectl apply --dry-run=client -f /tmp/r.yaml`).
