# Docker Compose for local development (unsupported)

> **Warning — unsupported developer tooling.** This page describes **Docker Compose** workflows for **local development** and **internal experimentation**. This path is **unsupported** and should be treated as **developer tooling**, not a supported deployment method.
>
> **This is not a supported deployment method for NIM containers** or for production NeMo Retriever services.

**Supported NIM and service deployment:** use **Kubernetes** and **Helm**:

- **NeMo Retriever Helm chart** (retriever **service** + optional in-cluster **NIMs**): [`nemo_retriever/helm/README.md`](helm/README.md)
- **Published Library Helm charts** and install procedures: [NeMo Retriever Library — prerequisites & getting started](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) (chart names and versions track the NeMo Retriever Library release you run)

For **library-only** Python workflows (no local NIM containers), follow [`README.md`](README.md) instead of this page.

---

## What this file covers

| Topic | Here |
|-------|------|
| Compose layout, auth, `.env`, runtime, profiles | This page |
| Building the retriever **service** image for Helm | [`helm/README.md` § Service image](helm/README.md#1-service-image) |
| Kubernetes values, secrets, upgrades | [`helm/README.md`](helm/README.md) |
| Harness SKU overrides (`docker-compose.<sku>.yaml`) | [`../tools/harness/plans/SERVICE_MANAGER.md`](../tools/harness/plans/SERVICE_MANAGER.md) (from **NeMo-Retriever** repository root) |

---

## Repository and `docker-compose.yaml` location

The canonical Compose file for NeMo Retriever extraction development lives at the root of the **[NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever)** repository:

- Upstream: [`NVIDIA/NeMo-Retriever` — `docker-compose.yaml`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docker-compose.yaml)
- In a monorepo checkout, run **all** `docker compose` commands from the directory that **contains** that file (the **repository root**), **not** from `nemo_retriever/` alone.

Optional GPU SKU merges use additional files in the same directory, for example `docker-compose.a10g.yaml`, `docker-compose.l40s.yaml`, `docker-compose.a100-40gb.yaml` (see the harness plan above for how automation merges `-f` flags).

---

## Prerequisites (Compose host)

- **Linux host** with NVIDIA drivers and a supported GPU (see [Pre-Requisites & Support Matrix](https://nvidia.github.io/NeMo-Retriever/extraction/prerequisites-support-matrix/) in the published docs site, or the in-repo equivalent under `docs/docs/extraction/`).
- **Docker Engine** with the [**Compose V2** plugin](https://docs.docker.com/compose/install/linux/) (`docker compose version`).
- [**NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so Compose can use `runtime: nvidia` / GPU reservations from `docker-compose.yaml`.

---

## Registry authentication (`nvcr.io`)

NIM images pull from `nvcr.io`. Authenticate **before** `docker compose up` (use an [NGC personal API key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html); same class of credential as used for Helm image pulls):

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

If login fails, image pulls will error with **401/403** — re-check the key and that your account can access the referenced image URIs.

---

## Environment variables and `.env`

Compose reads variables from your shell and from a **`.env` file in the project directory** (the directory you pass to `docker compose`, i.e. the **NeMo-Retriever repository root**). The stack expects (at minimum) valid NVIDIA / NGC credentials for NIM containers, for example:

- `NGC_API_KEY` — used by services as `NGC_API_KEY` / `NIM_NGC_API_KEY` fallbacks in `docker-compose.yaml`
- `NVIDIA_API_KEY` — some paths use this for build.nvidia.com–compatible keys inside the ingest runtime (see the file for exact wiring)

**Do not commit** `.env` or real keys. Use your secret manager / CI secrets for anything beyond a disposable dev machine.

Other commonly tuned variables include **`GPU_ID`** (physical GPU index passed through to Compose `device_ids`), image overrides (`*_IMAGE`, `*_TAG`), and port overrides (`PAGE_ELEMENTS_HTTP_PORT`, `OCR_HTTP_PORT`, …).

---

## Default developer stack (no extra profiles)

From the **NeMo-Retriever** repository root, after `docker login` and exporting keys (or populating `.env`):

```bash
docker compose up -d
```

This starts the **default** services defined **without** a `profiles:` gate (for example core layout NIMs, the **ingest microservice runtime** (see `docker-compose.yaml` for the exact service name on your revision), and bundled observability sidecars — the exact graph is authoritative in `docker-compose.yaml` for your revision).

### Bring up only selected services

Example — core layout + OCR + embedding NIMs (adjust service names to match your checkout):

```bash
docker compose up -d page-elements graphic-elements table-structure ocr embedding
```

### Compose **profiles** (optional components)

Several services are behind Compose **profiles** (see `profiles:` in `docker-compose.yaml`). Activate them with `--profile <name>`, for example:

| Profile (examples) | Typical use |
|----------------------|-------------|
| `reranker` | Reranking NIM |
| `nemotron-parse` | Nemotron Parse NIM |
| `vlm` | Vision-language NIM |
| `audio` | Parakeet / audio ASR NIM |
| `retrieval` | Milvus / MinIO / etcd stack for `vdb_upload` experiments |
| `graph` | Neo4j for graph / tabular experiments |

Example:

```bash
docker compose --profile audio up -d
```

---

## GPU SKU override files

To merge the base file with a GPU-specific override:

```bash
docker compose -f docker-compose.yaml -f docker-compose.a10g.yaml up -d
```

Replace `a10g` with the SKU file that matches your hardware (`docker-compose.l40s.yaml`, `docker-compose.a100-40gb.yaml`, etc., when present in the repo).

---

## Validate the stack

```bash
docker compose ps
```

Follow logs for a failing service:

```bash
docker compose logs -f page-elements
```

Health endpoints and ready checks vary by service; the ingest runtime exposes HTTP on host port **7670** by default (see the ingest runtime service mapping in `docker-compose.yaml`).

---

## Multi-GPU: isolated Compose projects on one host

Use distinct **Compose project names** (`-p`) and **non-colliding host ports** so two stacks can pin to different GPUs via `GPU_ID`.

### Start two stacks on separate GPUs

```bash
# GPU 0 stack
GPU_ID=0 \
PAGE_ELEMENTS_HTTP_PORT=8000 PAGE_ELEMENTS_GRPC_PORT=8001 PAGE_ELEMENTS_METRICS_PORT=8002 \
OCR_HTTP_PORT=8019 OCR_GRPC_PORT=8010 OCR_METRICS_PORT=8011 \
docker compose -p retriever-gpu0 up -d page-elements ocr

# GPU 1 stack
GPU_ID=1 \
PAGE_ELEMENTS_HTTP_PORT=8100 PAGE_ELEMENTS_GRPC_PORT=8101 PAGE_ELEMENTS_METRICS_PORT=8102 \
OCR_HTTP_PORT=8119 OCR_GRPC_PORT=8110 OCR_METRICS_PORT=8111 \
docker compose -p retriever-gpu1 up -d page-elements ocr
```

### Check and tear down

```bash
docker compose -p retriever-gpu0 ps
docker compose -p retriever-gpu1 ps
```

```bash
docker compose -p retriever-gpu0 down
docker compose -p retriever-gpu1 down
```

---

## Docker-specific troubleshooting (short)

| Symptom | Things to check |
|---------|-----------------|
| Image pull **401/403** | `docker login nvcr.io` with a valid `NGC_API_KEY`; entitlements for private/staging tags |
| Container **unhealthy** | `docker compose logs <service>`; GPU visible inside container (`nvidia-smi` in container if installed); `GPU_ID` matches available devices |
| **No GPU** in container | NVIDIA Container Toolkit installed; Docker default runtime or Compose `runtime: nvidia` as in the file; daemon restarted after toolkit install |
| Wrong compose file picked | `pwd` must be the directory that contains `docker-compose.yaml`; pass `-f` explicitly if you use overrides |

For **application-level** ingestion errors (not Docker wiring), use [Troubleshooting](https://nvidia.github.io/NeMo-Retriever/extraction/troubleshoot/) and [Environment variables](https://nvidia.github.io/NeMo-Retriever/extraction/environment-config/) on the docs site.

---

## Related developer docs (not Helm substitutes)

| Topic | Location |
|-------|----------|
| Neo4j via Compose (`graph` profile + Python connection) | [`src/nemo_retriever/tabular_data/neo4j/SETUP.md`](src/nemo_retriever/tabular_data/neo4j/SETUP.md) |
| Custom pipeline stages / UDFs | [`src/nemo_retriever/graph/README.md`](src/nemo_retriever/graph/README.md#nemo-retriever-graph) |
