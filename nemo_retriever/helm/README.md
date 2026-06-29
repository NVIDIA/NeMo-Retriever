# nemo-retriever Helm chart

A Kubernetes Helm chart for running the **service** mode of
[`nemo-retriever`](../README.md): a FastAPI document ingestion server that
streams uploads through a set of NVIDIA NIM microservices
(page-elements, table-structure, OCR, VLM embed by default) and exposes
result + status APIs over HTTP / SSE.

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

> For behavioral consistency between local HuggingFace deployments and Helm service deployments: 
> `results = ingestor.ingest(...return_results=True)
> return_results defaults to True. This incurs a significant performance and system memory usage cost. 
> Unless you know explicitly you need to fetch extraction results to the client, you should use:
> return_results=False
> If you must return results, you may need to increase pod memory specs to support the increased pod memory usage.

---

## Layout

```
nemo_retriever/helm/
├── Chart.yaml
├── values.yaml
├── README.md            <-- this file
├── openshift.md         <-- OpenShift restricted-v2 install guide
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
        ├── nemotron-ocr-v2.yaml               # NIMCache + NIMService (OCR)
        ├── llama-nemotron-embed-vl-1b-v2.yaml           # NIMCache + NIMService (VLM embed)
        ├── llama-nemotron-rerank-vl-1b-v2.yaml  # NIMCache + NIMService (optional; not auto-wired)
        ├── nemotron-parse.yaml                # NIMCache + NIMService (optional; not auto-wired)
        ├── nemotron-3-nano-omni-30b-a3b-reasoning.yaml  # NIMCache + NIMService (optional; not auto-wired)
        └── audio.yaml                         # NIMCache + NIMService (optional; not auto-wired)
```

---

## Quick start

### 1. Service image { #1-service-image }

The chart defaults to the GA image published to NGC:

```
nvcr.io/nvidia/nemo-microservices/nrl-service:26.5.0
```

Pulling from `nvcr.io` requires an NGC pull secret — either set
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

When `service.installFfmpeg=false` (the default), the service still starts
normally and processes PDF, image, text and HTML uploads. Audio / video
uploads are rejected up-front with **HTTP 501**:

```text
Audio and video ingestion require FFmpeg in the retriever service
container, but the following dependencies are missing: ffmpeg, ffprobe.
Re-deploy the Helm chart with `--set service.installFfmpeg=true` …
```

The retriever-service container also logs a `WARNING` at startup when
FFmpeg is missing so cluster operators can fix the deployment before
the first media upload arrives, instead of debugging a Ray worker
traceback (`RuntimeError: MediaChunkActor requires media dependencies;
missing: ffmpeg, ffprobe`) after the fact. The same WARNING is emitted
on every pod (gateway, realtime, batch) because all roles classify
uploads — flipping `service.installFfmpeg=true` updates them all.

Runtime installation uses passwordless `sudo` scoped to installing the
`ffmpeg` package in the service image. The pod must have network egress to the
Ubuntu package repositories, a writable root filesystem, and a security policy
that allows sudo/setuid behavior. Do not set
`service.securityContext.allowPrivilegeEscalation: false` or
`service.securityContext.readOnlyRootFilesystem: true` for this path.

For air-gapped or locked-down clusters, see
[Deployment options — Air-gapped and disconnected deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/#air-gapped-deployment).
On a connected staging host you can extend the service image, for example:

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
registered. A plain `helm install` reconciles the four core NIMs
(`page_elements`, `table_structure`, `ocr`, `vlm_embed`) — every other
NIM (the VL reranker `rerankqa`, Nemotron Parse, Omni 30B, and the
Parakeet `audio` ASR NIM) is **disabled by default** to honor the
"optional and disabled by default" contract in
[deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md);
refer to [Recommended minimal install (26.05)](#recommended-minimal-install-2605)
for the opt-in `--set` flags that turn any of them on.

```bash
helm install retriever ./nemo_retriever/helm \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY
```

### Recommended minimal install (26.05) { #recommended-minimal-install-2605 }

Deploy only the four core NIMs that the retriever service auto-wires (`page_elements`, `table_structure`, `ocr`, `vlm_embed`):

```bash
helm install retriever ./nemo_retriever/helm \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY
```

> The VL reranker (`rerankqa`), Nemotron Parse, the Nemotron 3 Nano Omni 30B caption NIM, the generic answer-generation LLM (`answer_llm`, Super-49B defaults), and the Parakeet `audio` ASR NIM are **all off by default** in 26.05 — they only reconcile when you explicitly opt in. Opt-in flags:
>
> * VL reranker — `--set nimOperator.rerankqa.enabled=true`
> * Nemotron Parse — `--set nimOperator.nemotron_parse.enabled=true`
> * Omni 30B captioner — `--set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true`
> * Answer generation LLM — `--set nimOperator.answer_llm.enabled=true`
> * Parakeet ASR — `--set nimOperator.audio.enabled=true` (also set `serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051` to wire ASR into the service, plus `service.installFfmpeg=true` if your image does not bundle ffmpeg)
>
> This matches the "optional and disabled by default" contract in [deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md) and avoids silently pulling ≈ 62 GiB of Omni weights, loading a large two-GPU LLM, or claiming extra dedicated GPUs on a "default" install. Refer to the [model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements) table for per-NIM GPU and disk costs.

The chart auto-wires the operator-managed in-cluster URLs of the four
"core" NIMs into the service's `nim_endpoints` block:

| key | operator-managed Service | invoke path |
| --- | ------------------------ | ----------- |
| `nimOperator.page_elements`   | `nemotron-page-elements-v3`   | `/v1/infer`      |
| `nimOperator.table_structure` | `nemotron-table-structure-v1` | `/v1/infer`      |
| `nimOperator.ocr`             | `nemotron-ocr-v2`             | `/v1/infer`      |
| `nimOperator.vlm_embed`       | `llama-nemotron-embed-vl-1b-v2` | `/v1/embeddings` |

Track operator reconciliation with:

```bash
kubectl get nimcache,nimservice -n <namespace>
kubectl describe nimservice nemotron-page-elements-v3 -n <namespace>
```

First-time NIMCache reconciliation downloads model weights to a PVC. By
default (`nimOperator.nimCache.keepOnUninstall: true`) every **NIMCache**
carries `helm.sh/resource-policy: keep` so those downloads survive
`helm uninstall`. **NIMService** CRs do not use `keep` and are removed by
Helm on uninstall.

### Why NIM resources still exist after `helm uninstall`

| What you see | Typical cause |
|--------------|----------------|
| `NIMCache` + PVC remain | **Expected** when `keepOnUninstall` is true (default). Helm intentionally skips deleting caches so you do not re-pull multi‑GiB weights. |
| `NIMService` CR remains | **Not expected** on a normal uninstall. Usually an **orphan** from a failed install/upgrade (release never recorded the resource, or the chart renamed a NIM). |
| Deployments / GPU pods still running | Often the operator workload for a **kept** `NIMCache`, or a stale `NIMService` that Helm did not own. Check `kubectl get nimservice,nimcache -n <ns>`. |
| `nemotron-*-job-*` pods in `Error` | The NIM Operator's **model-download Job** for a `NIMCache` (not the retriever service). Failed cache pulls retry and leave Error pods until the Job or `NIMCache` is deleted. Common after a failed `helm install` when the release is rolled back but `keep` retains the cache CR. |
| `helm uninstall` appears to do nothing | Release may be missing or failed (`helm list -n <ns> -a`). CRs created before a failed install can be left without a release to clean them up. |

**Full teardown** (dev cluster — deletes caches and PVCs Helm kept):

```bash
NS=retriever
REL=nemo-retriever

helm uninstall "${REL}" -n "${NS}" 2>/dev/null || true

# Orphans and kept NIMCaches (Helm keep does not block kubectl delete):
kubectl delete nimservice,nimcache -n "${NS}" --all
# Optional: drop model PVCs if you will re-pull from NGC
kubectl delete pvc -n "${NS}" -l 'app.kubernetes.io/managed-by=nvidia-nim-operator' 2>/dev/null || true
```

**Dev installs** that should not retain caches on uninstall:

```bash
helm upgrade --install "${REL}" ./nemo_retriever/helm -n "${NS}" \
  --set nimOperator.nimCache.keepOnUninstall=false \
  ...
```

---

## Values reference (highlights)

The full schema lives in [`values.yaml`](./values.yaml). Below is the
short list of knobs you'll touch first.

### Service

| Path                          | Default                            | Notes |
|-------------------------------|------------------------------------|-------|
| `service.image.repository`    | `nvcr.io/nvidia/nemo-microservices/nrl-service` | GA NGC image; override to pin a different build or use a local registry. |
| `service.image.tag`           | `26.5.0`                           |       |
| `service.replicas`            | `1`                                | Hard cap = 1 while SQLite is the backend. |
| `service.installFfmpeg`       | `false`                            | Install `ffmpeg`/`ffprobe` at container startup by setting `INSTALL_FFMPEG=true`. Requires network egress, writable root filesystem, and sudo/setuid allowed. Not for air-gapped clusters — use a custom image instead. |
| `service.resources.requests`  | `16 / 16Gi`                        | Tune in tandem with `serviceConfig.pipeline.*Workers`. |
| `service.resources.limits`    | `96 / 96Gi`                        |       |
| `service.gpu.enabled`         | `false`                            | The service does **not** need a GPU. |

For audio and video extraction, set `service.installFfmpeg=true` when your
cluster allows runtime package installation. **OpenShift restricted-v2** blocks
that path — use a prebuilt service image instead; refer to [Audio and video on restricted OpenShift](./openshift.md#audio-and-video-ffmpeg-on-restricted-openshift).
For air-gapped clusters, refer to [Deployment options — Air-gapped and disconnected deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/#air-gapped-deployment).

### Audio and video (Parakeet ASR) { #audio-video-parakeet }

To run self-hosted Parakeet for [audio and video extraction](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/audio-video.md):

1. Set `nimOperator.audio.enabled=true` (it is on by default; disable other optional NIMs you do not need per [Recommended minimal install (26.05)](#recommended-minimal-install-2605)).
2. Pin the ASR `NIMService` to a **dedicated GPU** with `nimOperator.audio.resources`, `nodeSelector`, or `tolerations` (refer to [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html)).
3. Confirm the GPU SKU in [Model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements) (footnote ⁴ lists Blackwell limitations).
4. Set `service.installFfmpeg=true` when the retriever service will process audio or video on clusters that allow runtime package install (refer to `service.installFfmpeg` above). On **OpenShift restricted-v2**, use a [prebuilt service image](./openshift.md#audio-and-video-ffmpeg-on-restricted-openshift) instead.

The retriever service picks up the in-cluster ASR endpoint when `nimOperator.audio` is enabled; refer to [NIM Operator sub-stack](#nim-operator-sub-stack).

### Service configuration (rendered into `retriever-service.yaml`)

| Path                                              | Default | Notes |
|---------------------------------------------------|---------|-------|
| `serviceConfig.server.port`                       | `7670`  | Container + Service port. |
| `serviceConfig.pipeline.realtimeWorkers`          | `24`    | Per-pod realtime worker count. |
| `serviceConfig.pipeline.batchWorkers`             | `48`    | Per-pod batch worker count. Refer to [Timeouts and alleviating ingest failures](#timeouts-and-alleviating-ingest-failures) if embed or pool errors appear under load. |
| `serviceConfig.nimEndpoints.*InvokeUrl`           | `""`    | Override the auto-resolved NIM Operator URL. Available knobs: `pageElementsInvokeUrl`, `tableStructureInvokeUrl`, `ocrInvokeUrl`, `embedInvokeUrl`, and `captionInvokeUrl` (refer to [Image captioning (Omni 30B)](#image-captioning-omni-30b)). |
| `serviceConfig.nimEndpoints.captionModelName`     | `""`    | Model id sent to the remote VLM. Auto-set to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` whenever a caption URL is resolved. |
| `serviceConfig.llm.enabled`                         | `false` | Enables `POST /v1/answer`. Auto-flips to true when `nimOperator.answer_llm` is enabled and the operator URL resolves. |
| `serviceConfig.llm.apiBase`                         | `""`    | OpenAI-compatible LLM base URL. Explicit value wins; otherwise `answer_llm` opt-in resolves to `http://answer-llm:8000/v1` by default. |
| `serviceConfig.llm.apiKeySecret.name`                | `""`    | Optional Secret name for external LLM credentials. Explicit values win; otherwise operator-managed `answer_llm` mounts its `authSecret` as `NEMO_RETRIEVER_LLM_API_KEY` so LiteLLM/OpenAI has a credential value without writing it to the ConfigMap. |
| `serviceConfig.llm.apiKeySecret.key`                 | `api_key` | Secret key for external LLM credentials. Operator-managed `answer_llm` uses `NGC_API_KEY` from `nimOperator.answer_llm.authSecret` when no explicit LLM Secret is set. |
| `serviceConfig.llm.model`                           | `""` | Optional explicit LiteLLM model id. Leave empty to inherit `nimOperator.answer_llm.model` when using the operator-managed answer LLM; set it for external endpoints. |
| `serviceConfig.llm.ragSystemPromptPrefix`           | `""` | Optional explicit RAG prompt prefix. Leave empty unless an endpoint needs model-specific prompt directives. |
| `serviceConfig.llm.reasoningEnabled`               | `true` | Request-level reasoning toggle for `/v1/answer`. Defaults to true for external OpenAI-compatible providers; set false for Nemotron endpoints that should receive portable no-reasoning controls. |
| `serviceConfig.vectordb.enabled`                  | `true`  | Deploy the LanceDB vectordb Pod. When `true` the chart **requires** a resolvable embed endpoint (refer to [VectorDB and the embed endpoint](#vectordb-and-the-embed-endpoint)); `helm install` / `helm upgrade` fails fast otherwise. |
| `serviceConfig.vectordb.lancedbUri`               | `/data/vectordb` | LanceDB on the vectordb Pod's PVC. |
| `serviceConfig.vectordb.embedModel`               | `nvidia/llama-nemotron-embed-vl-1b-v2` | Passed to vectordb + worker `embed_model_name`. |

#### VectorDB and the embed endpoint { #vectordb-and-the-embed-endpoint }

The vectordb Pod's `/v1/query` handler embeds the incoming query text
before searching LanceDB.  It needs a NIM embedding endpoint to do that,
and rendering the Deployment with an empty `--embed-endpoint` produces a
Pod that passes its `/v1/health` probe but answers every `/v1/query`
request with `HTTP 501 No embedding endpoint configured.` — a healthy
deployment that silently breaks retrieval.

To prevent this, the chart now refuses to render
`deployment-vectordb.yaml` when no embed endpoint can be resolved.
`helm install` / `helm upgrade --install` fails with a message listing
the three supported escape valves:

```
serviceConfig.vectordb.enabled=true but the embed endpoint could not be
resolved.  Pick one of:

  1. --set serviceConfig.nimEndpoints.embedInvokeUrl=http://<host>:<port>/v1/embeddings
  2. --set nimOperator.vlm_embed.enabled=true   # requires apps.nvidia.com/v1alpha1 CRDs
  3. --set serviceConfig.vectordb.enabled=false
```

Resolution order matches the rest of the chart (refer to [Mix and match NIM
sources](#3-install-with-the-nim-operator-in-cluster-nims)):

1. Explicit `serviceConfig.nimEndpoints.embedInvokeUrl` always wins.
2. Otherwise the operator-managed URL of
   `nimOperator.vlm_embed.nimServiceName` is used, provided
   `nimOperator.vlm_embed.enabled=true` **and** the
   `apps.nvidia.com/v1alpha1` CRDs are installed in the cluster.
3. Otherwise the chart fails the install.

#### Answer generation (operator-managed LLM) { #answer-generation-llm }

Enable the generic `answer_llm` NIM slot to add service-mode answer
generation on top of the VectorDB query path. The slot defaults to the
Super-49B NIM, but the image, model id, service name, resources,
profile filter, and environment can be overridden for another
OpenAI-compatible LLM NIM.

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.answer_llm.enabled=true
```

When the NIM Operator CRDs are present, the chart renders an `answer-llm`
NIMCache/NIMService by default and writes this block into
`retriever-service.yaml`:

```yaml
llm:
  enabled: true
  model: "openai/nvidia/llama-3.3-nemotron-super-49b-v1.5"
  api_base: "http://answer-llm:8000/v1"
  rag_system_prompt_prefix: null
  reasoning_enabled: true
```

The retriever service then exposes `POST /v1/answer`, which calls the
VectorDB pod's `/v1/query` endpoint for context and sends those chunks to
the configured LLM endpoint. The `answer_llm` NIM deployment leaves
reasoning defaults model-neutral; `/v1/answer` controls reasoning per
request. By default, `serviceConfig.llm.reasoningEnabled=true`, so requests
leave reasoning behavior to the LLM endpoint defaults and avoid sending
provider-specific `chat_template_kwargs` to external OpenAI-compatible
endpoints. Set `serviceConfig.llm.reasoningEnabled=false` for Nemotron
endpoints that should skip reasoning; the service then adds both `/no_think`
and `chat_template_kwargs.enable_thinking=false`. The default Super-49B NIMService
resources request two GPUs (`nvidia.com/gpu: 2`) to match the bundled
tensor-parallel NIM profile. Override `resources`, `modelProfile`, or
`env` for deployments that use a different profile or hardware topology.
When `answer_llm` is enabled and no explicit `serviceConfig.llm.apiKeySecret`
is set, the service also mounts `nimOperator.answer_llm.authSecret` as
`NEMO_RETRIEVER_LLM_API_KEY`; OpenAI-compatible clients require a
credential value even for in-cluster NIM endpoints, and the key is never
rendered into the ConfigMap.

For example, to try Nemotron 3 Nano as the answer LLM on an A100 80GB
node, override the operator-managed slot instead of adding a second
hard-coded LLM service:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.answer_llm.enabled=true \
  --set nimOperator.answer_llm.nimServiceName=nemotron-3-nano \
  --set nimOperator.answer_llm.image.repository=nvcr.io/nim/nvidia/nemotron-3-nano \
  --set nimOperator.answer_llm.image.tag=1.7.0-variant \
  --set nimOperator.answer_llm.model=openai/nvidia/nemotron-3-nano-30b-a3b \
  --set-json nimOperator.answer_llm.modelProfile='{"profiles":["5f89f01a0af587fd8bae50c611b1f358f92effdb9fb29362e1af0a986e5561c3"]}' \
  --set-json nimOperator.answer_llm.resources='{"limits":{"nvidia.com/gpu":1},"requests":{"nvidia.com/gpu":1}}' \
  --set nimOperator.answer_llm.env[0].name=NIM_HTTP_API_PORT \
  --set-string nimOperator.answer_llm.env[0].value=8000 \
  --set nimOperator.answer_llm.env[1].name=NIM_SERVED_MODEL_NAME \
  --set-string nimOperator.answer_llm.env[1].value=nvidia/nemotron-3-nano-30b-a3b \
  --set nimOperator.answer_llm.env[2].name=NIM_TENSOR_PARALLEL_SIZE \
  --set-string nimOperator.answer_llm.env[2].value=1
```

Use the repository and tag available in your NGC environment; staging
registries can use the same override shape with `nvstaging` image names
or tags. `nimOperator.answer_llm.model` is the LiteLLM model id used by
the retriever service; for an OpenAI-compatible in-cluster NIM, keep the
`openai/` prefix there and set `NIM_SERVED_MODEL_NAME` to the raw model
name advertised by the NIM. Replace the default Super-49B `modelProfile`,
`resources`, and `env` when the target model requires a different
GPU/profile setup. Leaving `modelProfile` empty preserves NIM
Operator auto-discovery, but for Nano it can cache every advertised
profile on first reconciliation; pin a known-compatible profile when you
know the target GPU topology.

`serviceConfig.llm.apiBase` and `serviceConfig.llm.model` can be set
explicitly to point `/v1/answer` at an external OpenAI-compatible LLM
instead of deploying an answer LLM in-cluster. For external credentials,
create a Kubernetes Secret and set `serviceConfig.llm.apiKeySecret.name`
plus `serviceConfig.llm.apiKeySecret.key`; Helm mounts the Secret as an
environment variable instead of writing the key into the ConfigMap.

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
| `nimOperator.ocr.image`              | `nvcr.io/nim/nvidia/nemotron-ocr-v2:1.4.0` | Default OCR NIM image. |
| `nimOperator.vlm_embed.enabled`        | `true`  | Multimodal embedding NIM (also used by the vectordb Pod). |
| `nimOperator.vlm_embed.nimServiceName` | `llama-nemotron-embed-vl-1b-v2` | NIMService / in-cluster DNS name. |
| `nimOperator.vlm_embed.image`          | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` | Default VLM embed NIM image. |
| `nimOperator.rerankqa.enabled`         | `false` | VL reranker NIM (optional; not auto-wired). Set `true` to opt in. Default `false` so 26.05 installs honor the "optional and disabled by default" contract in [deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md) and do not silently provision an extra ≈ 3.1 GiB GPU NIM. The image points at the **VL** SKU (`llama-nemotron-rerank-vl-1b-v2`) per [prerequisites-support-matrix.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#default-helm-nims) — the text-only `llama-nemotron-rerank-1b-v2` silently degrades multimodal reranking and is not the documented POR. |
| `nimOperator.nemotron_parse.enabled`   | `false` | Structured-parse NIM (optional). Set `true` when using `extract_method="nemotron_parse"`. Default `false` so 26.05 installs honor the "optional and disabled by default" contract in [deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md). Image tag follows the [image tag conventions](#image-tag-conventions). |
| `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled` | `false` | Omni 30B caption NIM (optional). Set `true` to enable image captioning — refer to [Image captioning (Omni 30B)](#image-captioning-omni-30b). Default `false` so 26.05 installs do not silently pull ≈ 62 GiB of BF16 weights or claim a second dedicated GPU. Image tag follows the [image tag conventions](#image-tag-conventions). |
| `nimOperator.answer_llm.enabled`       | `false` | Generic answer-generation LLM NIM (optional; Super-49B defaults). Set `true` to enable `/v1/answer` — refer to [Answer generation (operator-managed LLM)](#answer-generation-llm). Default `false` so installs do not silently claim answer-generation GPUs. |
| `nimOperator.answer_llm.model`         | `openai/nvidia/llama-3.3-nemotron-super-49b-v1.5` | LiteLLM/OpenAI model id inherited by `serviceConfig.llm.model` when the operator-managed answer LLM is enabled and no explicit service model is set. |
| `nimOperator.answer_llm.ragSystemPromptPrefix` | `""` | Optional prompt prefix inherited by `serviceConfig.llm.ragSystemPromptPrefix` only when explicitly set. Leave empty to keep the operator-managed LLM model-neutral and use `serviceConfig.llm.reasoningEnabled` for request-level reasoning control. |
| `nimOperator.audio.enabled`            | `false` | Parakeet ASR NIM (optional). Set `true` for audio/video transcription; pair with `serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051` so the retriever-service can reach it. |
| `nimOperator.<key>.image.repository`   | `nvcr.io/nim/nvidia/...` | Per-NIM image. |
| `nimOperator.<key>.image.pullSecrets`  | `[ngc-secret]` | Referenced by the NIMService CR. |
| `nimOperator.<key>.authSecret`         | `ngc-api`      | NIM auth Secret name. |
| `nimOperator.<key>.storage.pvc.size`   | `25Gi` (50Gi for vlm_embed/rerankqa, 100Gi parse, 300Gi VL) | NIMCache PVC size. |
| `nimOperator.<key>.replicas`           | `1`     | Per-NIMService replica count. |
| `nimOperator.nimServiceGpuLimit`       | `1`     | Default `nvidia.com/gpu` limit on every NIMService when per-NIM `resources` is `{}`. Set to `null` for operator-only reconciliation (not reliable on all NIM Operator versions — refer to [GPU limits and `helm upgrade`](#gpu-limits-and-helm-upgrade)). |
| `nimOperator.<key>.resources`          | `{}`    | Per-NIM override of the whole `resources` block. Empty uses `nimServiceGpuLimit`; non-empty replaces the chart default (may require `--force-conflicts` on later `helm upgrade`). |
| `nimOperator.modelProfile`             | `{}`    | Chart-wide NIMCache GPU/profile filter. Applied to every NIMCache that does not have its own override. Refer to [Filtering cached GPU profiles](#filtering-cached-gpu-profiles). |
| `nimOperator.<key>.modelProfile`       | `{}`    | Per-NIM NIMCache GPU/profile filter. Non-empty values REPLACE the chart-wide default (no merge). Refer to [Filtering cached GPU profiles](#filtering-cached-gpu-profiles). |
| `nimOperator.<key>.expose.service.port` | `8000` (9000 for audio) | HTTP port. |
| `nimOperator.<key>.expose.service.grpcPort` | `8001` (50051 for audio) | gRPC port. |

> Only the four "core" NIMs (page_elements, table_structure, ocr, vlm_embed)
> are auto-wired into the retriever-service config. Optional NIMs may reconcile
> when `nimOperator.<key>.enabled` is `true` in `values.yaml`, but the
> retriever-service won't call them unless you wire your pipeline to use them.
> For 26.05, prefer the [minimal install](#recommended-minimal-install-2605) overrides.

#### Filtering cached GPU profiles { #filtering-cached-gpu-profiles }

Every NIMCache the chart renders supports the NIM Operator's
`spec.source.ngc.model` block, which restricts which model profiles the
cache job downloads. The chart exposes this through two values:

| Path | Scope | Behaviour |
| ---- | ----- | --------- |
| `nimOperator.modelProfile` | Chart-wide | Applied to every NIMCache that doesn't carry its own override. |
| `nimOperator.<key>.modelProfile` | Per-NIM | When non-empty, **REPLACES** the chart-wide default (no merge). |

Both default to `{}`. With both empty the chart emits no `model:`
block and the NIM Operator falls back to its "cache every profile
applicable to the detected GPUs" default — fine on a single-GPU
laptop, but on heterogeneous clusters (or any cluster with ≥ 3 NIMs)
this wastes tens of GiB of PVC storage, NGC bandwidth, and cache-job
runtime.

The mapping is rendered verbatim under `spec.source.ngc.model`, so the
shape lines up 1:1 with the [NIMCache CRD](https://docs.nvidia.com/nim-operator/latest/reference-nimcache.html).
Two filter dimensions are supported (use whichever fits your cluster;
`gpus` is the common case):

```yaml
nimOperator:
  modelProfile:
    gpus:
      # NIMCache only downloads profiles compatible with at least one
      # of these GPU selectors. Each selector is {ids: [...], product: ...}.
      - ids: ["26B5"]                       # PCI device ID(s)
        product: "NVIDIA-H100-80GB-HBM3"    # NVIDIA marketing name
    # profiles:
    #   # Alternative: list of exact profile UUIDs from `ngc registry
    #   # model list-profiles <repo>/<image>:<tag>`.
    #   - "11111111-2222-3333-4444-555555555555"
```

Equivalent overrides via `--set`:

```bash
# Homogeneous H100 80 GB cluster — every NIMCache only pulls the H100 profile:
helm upgrade --install retriever ./nemo_retriever/helm \
  --set 'nimOperator.modelProfile.gpus[0].ids[0]=26B5' \
  --set 'nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3'

# Restrict only the page_elements NIMCache to a specific profile UUID, leave the rest alone:
helm upgrade --install retriever ./nemo_retriever/helm \
  --set 'nimOperator.page_elements.modelProfile.profiles[0]=11111111-2222-3333-4444-555555555555'

# Chart-wide H100 default plus a per-NIM override (the override REPLACES the global; it does NOT merge):
helm upgrade --install retriever ./nemo_retriever/helm \
  --set 'nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3' \
  --set 'nimOperator.vlm_embed.modelProfile.profiles[0]=22222222-3333-4444-5555-666666666666'
```

Tips:

- Run `ngc registry model list-profiles nvcr.io/nim/nvidia/<image>:<tag>` to enumerate the available profiles for any chart-pinned NIM image and pick the smallest profile that matches your GPU.
- Filter mismatches surface as `NIMCache` events such as `NoCompatibleProfile`; check with `kubectl describe nimcache <name>`.
- The chart's defaults (`{}`) preserve operator behaviour, so adding `modelProfile` is a strict opt-in — existing releases keep working unchanged.

#### Image tag conventions { #image-tag-conventions }

Every NIM in this chart pins an exact NGC image tag in `values.yaml`
— there is no `:latest` floating reference. Two tag families show up:

| Family | Example | Meaning |
| ------ | ------- | ------- |
| Plain semver | `nemotron-page-elements-v3:1.8.0` | A standard NIM release, identical bytes on every pull. Used by the four core NIMs and the reranker / ASR NIMs. |
| `<semver>-variant` | `nemotron-parse-v1.2:1.7.0-variant`, `nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant` | The Nemotron Parse and Nemotron 3 Nano Omni 30B builds that ship per-GPU TensorRT engine variants the NIM Operator selects from at reconciliation time (see the Omni and Parse rows in the [model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements) table). The `-variant` suffix is the NGC tag that ships alongside the 26.05 chart and matches footnote ³ of the support matrix. |

For air-gapped mirror pipelines: mirror the *exact* tag — both the
plain semver and the `-variant` form — and do not substitute `:latest`.
Substituting `:latest` would pin to a moving target that may not match
the engine plans the NIM Operator profile expects for a given GPU.

If you want a different NIM build, override the tag explicitly:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true \
  --set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.image.tag=<your-tag>
```

and validate against the same release of the retriever service before
production rollout.

**Charts and captioning (26.05).** Charts and infographics use **page_elements**
and **ocr** (no `graphic_elements` operator NIM in this chart). For image
captioning, set `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true` — see
[Image captioning (Omni 30B)](#image-captioning-omni-30b) for the
chart-side wiring and
[Image captioning (26.05)](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/#image-captioning-2605)
for the product matrix.

#### Image captioning (Omni 30B) { #image-captioning-omni-30b }

The Nemotron 3 Nano Omni VLM is the canonical image-caption NIM for
26.05.  When you enable it,

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true \
  ...
```

the chart now auto-wires two fields into the rendered
`retriever-service.yaml` ConfigMap:

```yaml
nim_endpoints:
  caption_invoke_url: "http://nemotron-3-nano-omni-30b-a3b-reasoning:8000/v1/chat/completions"
  caption_model_name: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
```

The service derives `caption_enabled=true` from a non-null
`caption_invoke_url`, so the ingestion pipeline routes caption work to
the in-cluster Omni Pod with no manual ConfigMap edits.

Resolution order mirrors every other NIM endpoint (see the
[NIM Operator sub-stack](#nim-operator-sub-stack) section):

1. Explicit `serviceConfig.nimEndpoints.captionInvokeUrl` always wins
   (use this to point at a hosted endpoint, e.g.
   `https://integrate.api.nvidia.com/v1/chat/completions`).
2. Otherwise the operator-managed URL of
   `nemotron-3-nano-omni-30b-a3b-reasoning` is used, provided
   `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true`
   **and** the `apps.nvidia.com/v1alpha1` CRDs are installed.
3. Otherwise `caption_invoke_url` stays `null` and the caption stage
   is disabled.

`serviceConfig.nimEndpoints.captionModelName` follows the same order —
it defaults to the canonical Omni remote model id
(`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, matching
`nemo_retriever.common.modality.caption.model_profiles.OMNI_REMOTE_MODEL_ID`) whenever
the chart resolves any caption URL. Override only when pointing at a
different VLM SKU.

#### GPU limits and `helm upgrade` { #gpu-limits-and-helm-upgrade }

The chart defaults to **`nimOperator.nimServiceGpuLimit: 1`**, which
renders `spec.resources.limits.nvidia.com/gpu: 1` on every NIMService
unless a per-NIM `resources` map overrides it. This is required on
NIM Operator **v3.1.1** (and other versions tested on A100/H100): when
the chart omits the `resources` block entirely, the operator often
**does not** populate GPU limits from the model profile, and NIM pods
start without GPU access (`The NVIDIA Driver was not detected`).

**Trade-off:** Helm and the NIM Operator may both server-side-apply
`spec.resources.limits.nvidia.com/gpu`. A later `helm upgrade --install`
can then fail with:

```
Error: UPGRADE FAILED: conflict occurred while applying object
  <ns>/<nim> apps.nvidia.com/v1alpha1, Kind=NIMService:
  Apply failed with 1 conflict:
  conflict with "manager" using apps.nvidia.com/v1alpha1:
    .spec.resources.limits.nvidia.com/gpu
```

**Operator-only mode** (omit GPU limits from Helm — only if your NIM
Operator version reliably reconciles them):

```yaml
nimOperator:
  nimServiceGpuLimit: null
```

**If upgrades hit SSA conflicts** after the operator has reconciled GPU
limits, use one of:

1. `helm upgrade --install … --force-conflicts --server-side`
2. `kubectl -n <ns> edit nimservice <name>` to set GPU limits outside Helm

To pin a non-default GPU count chart-wide, set `nimServiceGpuLimit: 2`
(or set per-NIM `resources.limits.nvidia.com/gpu`).

### OCR NIM configuration { #ocr-nim-configuration }

The core OCR NIM is configured under [`nimOperator.ocr`](./values.yaml) (the `ocr:`
block). Confirm `image.repository` and `image.tag` before you upgrade.

| Path | Role |
|------|------|
| `nimOperator.nimCache.keepOnUninstall` | When `true`, NIMCache CRs survive `helm uninstall` (`helm.sh/resource-policy: keep`). NIMService CRs are always removed. Set `false` for dev clusters that should fully tear down on uninstall. |
| `nimOperator.ocr.enabled` | Reconcile the OCR `NIMService` |
| `nimOperator.ocr.image.repository` | NIM image (default `nvcr.io/nim/nvidia/nemotron-ocr-v2`) |
| `nimOperator.ocr.image.tag` | Pin the image tag for reproducible upgrades |

Override the auto-wired in-cluster URL with `serviceConfig.nimEndpoints.ocrInvokeUrl`
when the OCR service runs outside the operator sub-stack.

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

## Tracing and Zipkin

Helm installs the chart-owned OpenTelemetry Collector and Zipkin backend on by
default. This is intentional: the legacy 26.1.2 Helm chart shipped with a
managed Zipkin deployment enabled, so the new chart keeps a default trace
backend available for functional parity. Pod trace export is also enabled by
default for retriever service pods and chart-managed NIMs:

```yaml
topology:
  otel:
    enabled: true
  zipkin:
    enabled: true

service:
  otel:
    enabled: true

nimOperator:
  otel:
    enabled: true
```

Because Zipkin is chart-owned by default, an upgrade with default values can
create a Zipkin Deployment and Service. Set `topology.zipkin.enabled=false`
before upgrading if your deployment uses an external backend or should not run
chart-owned Zipkin.

With default values, retriever service pods and chart-managed NIMs emit OTLP to
the chart's OpenTelemetry Collector, which exports traces to the chart-owned
Zipkin service. Set `service.otel.enabled=false` or
`nimOperator.otel.enabled=false` to opt out by surface. Open a job and read the
Zipkin lookup key from either the JSON body or the `x-trace-id` response header:

```bash
kubectl port-forward svc/tracing-smoke-nemo-retriever 7670:80

curl -s -D headers.txt -o job.json \
  -X POST http://localhost:7670/v1/ingest/job \
  -H 'content-type: application/json' \
  -d '{"expected_documents":1}'

TRACE_ID=$(jq -r .trace_id job.json)
grep -i x-trace-id headers.txt
```

Port-forward Zipkin and query the trace directly:

```bash
kubectl port-forward svc/tracing-smoke-nemo-retriever-zipkin 9411:9411
curl "http://localhost:9411/api/v2/trace/${TRACE_ID}"
```

Common opt-out and override knobs:

```yaml
topology:
  zipkin:
    enabled: false                 # do not deploy chart-owned Zipkin
    exporter:
      enabled: false               # keep Zipkin deployed, but do not export traces to it
      endpoint: http://external-zipkin:9411/api/v2/spans

service:
  otel:
    enabled: false                 # do not inject service pod instrumentation env

nimOperator:
  otel:
    enabled: false                 # do not inject inherited NIM OTLP env
  page_elements:
    otel:
      enabled: false               # per-NIM opt-out
  ocr:
    otel:
      env:
        TRITON_OTEL_RATE: "10"     # per-NIM Triton OTel override
```

Set `topology.zipkin.exporter.endpoint` when you run your own Zipkin-compatible
collector. Set `topology.otel.enabled=false` to disable the chart-owned collector
and all chart-rendered collector wiring.

---

## OpenShift deployment { #openshift-deployment }

OpenShift install procedures, **restricted-v2** / PSA **restricted** value overrides, prebuilt `ffmpeg` images, internal registry pull secrets, optional NIM `LD_LIBRARY_PATH` tuning, and install examples are in **[OpenShift deployment](./openshift.md)**. Pass `-f openshift-restricted.yaml` from that guide when you install on OpenShift.

---

## Air-gapped deployment { #air-gapped-deployment }

Refer to [Deployment options — Air-gapped and disconnected deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/#air-gapped-deployment) for overview and workflow. Chart-specific reference for mirroring:

### Container images to mirror (26.05 chart defaults)

Verify tags on the Git branch or tag you ship (for example `26.05` or
`26.5.0`). Defaults below match
[`values.yaml`](./values.yaml) on the current chart.

| Role | `nimOperator` key | Default image (`repository:tag`) |
|------|-------------------|----------------------------------|
| Retriever service | — | `service.image.repository`:`service.image.tag` (override for production) |
| Page elements | `page_elements` | `nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0` |
| Table structure | `table_structure` | `nvcr.io/nim/nvidia/nemotron-table-structure-v1:1.8.0` |
| OCR | `ocr` | `nvcr.io/nim/nvidia/nemotron-ocr-v2:1.4.0` |
| VL embed | `vlm_embed` | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` |
| VL reranker (optional) | `rerankqa` | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:1.10.0` |
| Nemotron Parse (optional) | `nemotron_parse` | `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant` |
| Omni caption (optional) | `nemotron_3_nano_omni_30b_a3b_reasoning` | `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant` |
| Answer LLM (optional, Super-49B default) | `answer_llm` | `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:2.0.5` |
| Parakeet ASR (optional) | `audio` | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0` |

GPU SKU support for `audio` is in [Model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements).

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
  create: false
  name: ""   # Explicitly empty — clears the default "ngc-secret"

nimOperator:
  page_elements:
    image:
      repository: <PRIVATE_REGISTRY>/nemotron-page-elements-v3
      tag: "1.8.0"
      pullPolicy: IfNotPresent
  # Repeat for table_structure, ocr, vlm_embed, and any optional keys you enable.
```

- Set `nimOperator.<key>.image.pullSecrets` to your mirror pull secret
  (for example `my-private-registry`; chart default is `ngc-secret`).
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

# Operator CRDs present: vectordb resolves vlm_embed via the operator URL.
helm template r nemo_retriever/helm \
  --api-versions apps.nvidia.com/v1alpha1 > /tmp/r-op.yaml

# Operator CRDs absent: vectordb has no operator URL to fall back to, so
# either disable vectordb or supply an explicit embed endpoint.
helm template r nemo_retriever/helm \
  --set serviceConfig.vectordb.enabled=false > /tmp/r.yaml
#   or:
# helm template r nemo_retriever/helm \
#   --set serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.svc:8000/v1/embeddings \
#   > /tmp/r.yaml
```

Both renders should succeed cleanly and parse as valid Kubernetes manifests
(`kubectl apply --dry-run=client -f /tmp/r.yaml`). Refer to [VectorDB and the
embed endpoint](#vectordb-and-the-embed-endpoint) for why
`helm template r nemo_retriever/helm` without flags is rejected as a
misconfiguration.
