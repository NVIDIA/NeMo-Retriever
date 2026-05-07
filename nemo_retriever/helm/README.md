# nemo-retriever Helm chart

A Kubernetes Helm chart for running the **service** mode of
[`nemo-retriever`](../README.md): a FastAPI document ingestion server that
streams uploads through five NVIDIA NIM microservices
(page-elements, graphic-elements, table-structure, OCR, embed) and exposes
result + status APIs over HTTP / SSE.

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

## Roadmap

1. **PostgreSQL backend** — replace `service.db.engine.DatabaseEngine` with
   a SQLAlchemy/asyncpg-based engine, then bump the chart to deploy a
   PostgreSQL StatefulSet (or take a sub-chart dependency on Bitnami's
   chart) and lift `service.replicas` to N.
2. **`/metrics` Prometheus endpoint** in the service so the bundled
   `ServiceMonitor` is useful out of the box.
3. **NetworkPolicies** restricting the service Pod to the NIM Pods + DB
   only.
4. **HPA on a custom queue-depth metric** instead of CPU once the shared
   DB unblocks horizontal scaling.

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
