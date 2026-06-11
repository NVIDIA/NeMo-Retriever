# Functional test suite — SETUP user task #3 (K8s GPU, NIMs via Helm)

Third suite in the agent-driven functional tests for the **NeMo Retriever Library
skill**, built against the real CLI in `nemo_retriever/nemo_retriever/src`
(`retriever ingest` / `retriever query`) and the repo's **Helm chart** at
`nemo_retriever/helm` (`nemo-retriever`, chart **26.5.0**).

Where suite 1 routes to **hosted** build.nvidia.com endpoints and suite 2 loads
models **directly on a local GPU**, this suite is the **Kubernetes** flavor: the
Extraction / Embedding / Reranking NIMs are deployed onto a **GPU node via Helm**
(as NIM Operator `NIMCache` + `NIMService` custom resources), and the CLI is then
pointed at the **in-cluster NIM Service URLs** for a smoke ingest + query. Each test
is a self-contained triple — a prompt, a per-case `data/` folder, and an expected
output naming the correct commands (helm/kubectl + `retriever`) and the in-cluster
flags.

---

## The user task under test

> **JTBD: SETUP — row 3.** "NeMo Retriever library setup on **K8s GPU** machine:
> **Extraction, Embedding, Reranking NIMs helm charts downloaded and deployed.**
> (On request, the agent should validate **GPU, Kubernetes, and Helm** prerequisites,
> **ask before installing** missing K8s/Helm, then **deploy NRL via Helm**.)" — **P1**

**Success criteria for the row (operational pass):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) clean library install, (2) deploy NIMs helm charts on a GPU node and pods/services healthy **≤ 30 min**, (3) ingest 1 PDF against the in-cluster NIMs, (4) retrieve a grounded query |
| Time | Helm deploy of the NIMs **≤ 30 min**; full end-to-end (clean → deploy → ingest → cited query) **≤ 40 min** |
| Trigger rate | ≥ 95% — a "deploy NeMo Retriever on my GPU Kubernetes cluster with Helm" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — validate prereqs + **ask before installing** missing k8s/helm; deploy via `helm install/upgrade`; then `retriever ingest`/`query` with **in-cluster NIM invoke-urls** (not hosted defaults) |
| Token usage | tracked, not gated |

**Seed queries.** None were provided for this row in the spec, so the five prompts are
**synthesized** from the user-task wording and the defining behaviors:

- *"I have a GPU Kubernetes cluster — deploy the NeMo Retriever NIMs with Helm and prove ingest + query works."*
- *"Before doing anything, check I'm ready (GPU node, kubectl, Helm) and ask me before installing anything that's missing."*
- *"Deploy extraction, embedding, AND reranking NIMs to my cluster, then point the CLI at the in-cluster endpoints."*

---

## How the deploy + CLI run on K8s (grounded against the repo Helm chart)

Grounded by `nemo_retriever/helm/{Chart.yaml,values.yaml,README.md,templates/NOTES.txt}`
and the CLI source (`cli/main.py`). The skill's `references/` do **not** yet document
the Helm path, so all Helm/K8s specifics below come from the chart itself.

- **Prerequisites.** A reachable Kubernetes cluster (chart `kubeVersion >= 1.25.0`),
  Helm 3, and at least one node with `nvidia.com/gpu` allocatable (NVIDIA device plugin
  / GPU Operator). Validate with `kubectl version` / `kubectl get nodes` / `helm version`.
- **NIM Operator prerequisite.** The NIM templates are gated on the
  `apps.nvidia.com/v1alpha1` API group (the **NVIDIA NIM Operator**). Without it the
  chart still installs, but every `NIMCache`/`NIMService` short-circuits and the
  service falls back to **external NIM URLs only**. Operator install (only **after
  asking the user**): `helm install k8s-nim-operator nvidia/k8s-nim-operator
  --namespace nim-operator --create-namespace`.
- **Ask-before-install.** The defining behavior of this row: if kubectl / Helm / the NIM
  Operator / the GPU device-plugin is missing, the agent **reports** it and **asks**
  before installing — it must **not** silently install Kubernetes/Helm/operator
  components.
- **Deploy.** `helm upgrade --install retriever ./nemo_retriever/helm -n <ns>
  --create-namespace` with the NGC secrets (`--set ngcImagePullSecret.create=true
  --set ngcImagePullSecret.password=$NGC_API_KEY --set ngcApiSecret.create=true
  --set ngcApiSecret.password=$NGC_API_KEY`). A plain install reconciles the **four
  core NIMs**; the **reranker is optional** and must be turned on with
  `--set nimOperator.rerankqa.enabled=true` (see caveat below).
- **In-cluster NIM Service URLs** (auto-wired into the service; also what the CLI points at):

  | model class | NIM | in-cluster URL |
  |---|---|---|
  | extraction (page elements) | `nemotron-page-elements-v3` | `http://nemotron-page-elements-v3:8000/v1/infer` |
  | extraction (table structure) | `nemotron-table-structure-v1` | `http://nemotron-table-structure-v1:8000/v1/infer` |
  | extraction (OCR) | `nemotron-ocr-v1` | `http://nemotron-ocr-v1:8000/v1/infer` |
  | embedding (VLM) | `llama-nemotron-embed-vl-1b-v2` | `http://llama-nemotron-embed-vl-1b-v2:8000/v1/embeddings` |
  | reranking (VL, opt-in) | `llama-nemotron-rerank-vl-1b-v2` | `http://llama-nemotron-rerank-vl-1b-v2:8000` |

- **Verify.** `kubectl get nimcache,nimservice -n <ns>` and `kubectl get pods -n <ns> -w`
  until Ready; the retriever service listens on **7670** (`kubectl port-forward
  svc/retriever 7670:7670 -n <ns>` → `curl http://localhost:7670/v1/health`). First-time
  `NIMCache` reconciliation downloads model weights to a PVC — allow several minutes.
- **CLI routing.** The CLI is pointed at the in-cluster NIMs via
  `--page-elements-invoke-url / --ocr-invoke-url / --table-structure-invoke-url /
  --embed-invoke-url / --reranker-invoke-url` (the embed/rerank model names are the **VL**
  SKUs: `nvidia/llama-nemotron-embed-vl-1b-v2`, `llama-nemotron-rerank-vl-1b-v2`). From
  outside the cluster, port-forward each NIM Service and substitute `http://localhost:<port>`;
  the requirement is that the URLs are the **deployed in-cluster NIMs**, not the hosted
  `ai.api.nvidia.com` / `integrate.api.nvidia.com` defaults.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Commands |
|---|---|---|---|
| 1 | `setup-k8s-001` | **Prereq validation only.** Detect GPU node + Kubernetes + Helm; report readiness; **ASK** before installing anything missing. No deploy/ingest. | `kubectl version`, `kubectl get nodes`, `helm version` |
| 2 | `setup-k8s-002` | **Helm deploy.** `helm upgrade --install` the chart so extraction+embedding+**reranking** NIMs land on the GPU node; verify pods/Services Ready (≤ 30 min). | `helm upgrade --install`, `kubectl get nimservice/pods/svc` |
| 3 | `setup-k8s-003` | **Smoke ingest.** Ingest 1 PDF with the CLI pointed at the **in-cluster** extraction+embed NIM URLs (not hosted). | `ingest` |
| 4 | `setup-k8s-004` | **Retrieve query.** Close ingest→query against the in-cluster embed **and reranker** NIMs; surface a table cell. | `ingest`, `query --rerank` |
| 5 | `setup-k8s-005` | **Acceptance gate.** Clean → validate (ask-before-install) → Helm deploy (≤ 30 min) → ingest into a named index → cited query, all in-cluster, end-to-end ≤ 40 min. | all of the above |

The ladder: T1 proves the agent validates prereqs and honors the ask-before-install
discipline; T2 adds the Helm deploy of all three NIM classes onto the GPU node; T3 adds
the first real CLI work (ingest against the in-cluster NIMs); T4 closes the loop with a
reranked query through the deployed embed+rerank NIMs; T5 composes everything into the
row's operational-pass / acceptance gate.

---

### T1 — `setup-k8s-001` · prerequisite validation only  *(complexity 1)*
- **Satisfies:** the "validate GPU, Kubernetes, and Helm prerequisites" + "ask before
  installing" clauses, in isolation.
- **Data:** `data/test.pdf` (present but not yet used).
- **Expected:** `kubectl version` (server ≥ 1.25), `kubectl get nodes -o wide` (≥ 1 Ready
  node with `nvidia.com/gpu`), `helm version` (Helm 3), and a NIM Operator CRD check. The
  agent emits a per-prereq present/missing verdict and, for anything missing, **asks**
  before installing (e.g. naming the NIM Operator install command but not running it).

### T2 — `setup-k8s-002` · Helm deploy + health  *(complexity 2)*
- **Satisfies:** the "Extraction, Embedding, Reranking NIMs helm charts downloaded and
  deployed" clause + the ≤ 30 min Helm SLA.
- **Data:** `data/test.pdf` (deployment target; not ingested in this rung).
- **Expected:** `helm upgrade --install retriever ./nemo_retriever/helm -n nrl
  --create-namespace …NGC secrets… --set nimOperator.rerankqa.enabled=true`, then
  `kubectl get nimcache,nimservice -n nrl` and `kubectl get pods -n nrl -w` until the
  three NIM classes (extraction page-elements/table-structure/OCR, embedding, reranking)
  and the retriever service pod are Ready; Services present; health OK.

### T3 — `setup-k8s-003` · smoke ingest against in-cluster NIMs  *(complexity 3)*
- **Satisfies:** success-criterion 3 (ingest 1 PDF) against the deployed NIMs.
- **Data:** `data/woods_frost.pdf`.
- **Expected:** one `retriever ingest data/woods_frost.pdf` with
  `--page-elements-invoke-url`/`--ocr-invoke-url`/`--table-structure-invoke-url`/
  `--embed-invoke-url` pointing at the in-cluster Service DNS names and
  `--embed-model-name nvidia/llama-nemotron-embed-vl-1b-v2`. Summary line reports 1 file;
  no hosted endpoints touched.

### T4 — `setup-k8s-004` · retrieve query against embed + rerank NIMs  *(complexity 4)*
- **Satisfies:** success-criterion 4 (retrieve a query) + the reranking-NIM clause.
- **Data:** `data/table_test.pdf` (James 2019 = 978).
- **Expected:** ingest (in-cluster extraction+embed) → `retriever query "James value in
  2019" --top-k 5 --content-types text,table --embed-invoke-url … --reranker-invoke-url
  http://llama-nemotron-rerank-vl-1b-v2:8000 --reranker-model-name
  llama-nemotron-rerank-vl-1b-v2 --rerank` → **978**, citing `table_test.pdf` p1.

### T5 — `setup-k8s-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row (clean → validate w/ ask-before-install
  → deploy ≤ 30 min → ingest → cited query ≤ 40 min, all in-cluster, zero hosted egress).
- **Data:** `data/` (3 PDFs).
- **Expected:** prereq validation (ask before installing anything missing) → `helm upgrade
  --install … --set nimOperator.rerankqa.enabled=true` (pods Ready ≤ 30 min) → ingest
  `data/` into `--table-name k8s_smoke` via the in-cluster NIMs → `query "miles to go
  before I sleep" --table-name k8s_smoke … --rerank` → *"Stopping by Woods on a Snowy
  Evening"*, citing `woods_frost.pdf` p1.
- **Adds (the traps):** ask-before-install honored; custom `--table-name` aligned across
  ingest **and** query; all-three-model NIM-readiness; zero-egress assertion; the two SLAs.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The checks unique to this suite vs. suites 1 & 2:
**(a)** the agent validates GPU + Kubernetes + Helm and **asks before installing** anything
missing (no silent install of k8s/helm/operator); **(b)** the NIMs are deployed via the
`nemo_retriever` **Helm chart** (extraction + embedding + reranking) onto a GPU node and
pods/Services reach Ready; **(c)** the CLI is pointed at the **in-cluster NIM Service URLs**
(`http://nemotron-*:8000/...`, `http://llama-nemotron-embed-vl-1b-v2:8000/v1/embeddings`,
`http://llama-nemotron-rerank-vl-1b-v2:8000`), not the hosted `ai.api.nvidia.com` /
`integrate.api.nvidia.com` defaults.

**Known chart caveats (built into the tests):**
- The **Reranking NIM is optional** and disabled by default (`nimOperator.rerankqa`, not
  auto-wired). Because the row explicitly names reranking, the deploy must pass
  `--set nimOperator.rerankqa.enabled=true`; a default install would omit it and fail the
  reranking clause (T2/T4/T5 enforce this).
- The NIM templates need the **NVIDIA NIM Operator** (`apps.nvidia.com/v1alpha1`); without
  it the chart installs but the NIMs do not reconcile and the service degrades to external
  URLs — a silent gap T5's acceptance gate is designed to catch.

**Note on live runs.** These expected outputs are **not yet run live** — they are grounded
in the CLI source + the repo's Helm chart (`nemo_retriever/helm`), not executed end-to-end.
Beyond the usual reasons (live ingest/query may hit billable endpoints or need a GPU), **this
row additionally requires a real GPU Kubernetes cluster + Helm 3 + the NVIDIA NIM Operator**
(and NGC pull credentials to download the NIM images/weights). A live run would capture: the
actual NIMCache/NIMService reconciliation + image-pull times (the dominant term in the ≤ 30
min Helm SLA), pod-Ready timings, real ingest row counts, query latencies, the end-to-end
≤ 40 min wall clock, and confirmation that no traffic egressed to ai.api.nvidia.com /
integrate.api.nvidia.com / build.nvidia.com.
