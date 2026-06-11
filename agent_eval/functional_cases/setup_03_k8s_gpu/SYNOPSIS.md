# Synopsis — SETUP #3: Kubernetes GPU setup (NIMs deployed via Helm)

**What user task this covers.** A developer with a **GPU Kubernetes cluster** wants to run
NeMo Retriever on it: the **Extraction, Embedding, and Reranking NIMs** deployed onto a GPU
node **via Helm**, then proven out with a real ingest + query. Crucially, before deploying,
the agent should **validate the prerequisites** — a GPU node, a working Kubernetes cluster,
and Helm — and if something is missing it should **ask the user before installing it**, not
silently install Kubernetes or Helm. Success means: the library installs cleanly, the NIMs
helm charts deploy on a GPU node in **under 30 minutes**, one PDF ingests through the
in-cluster NIMs, a query comes back with a grounded citation, and the whole thing finishes
**under 40 minutes**.

**How we test it.** Five agent prompts that each hand the agent small PDFs and check that it
drives the right tools in the right order: `kubectl`/`helm` to validate and deploy, then the
`retriever` CLI (`ingest` then `query`) pointed at the **in-cluster NIM Service URLs**
(`http://nemotron-page-elements-v3:8000/v1/infer`, …, the embed and reranker NIMs) — *not*
NVIDIA's hosted cloud endpoints. The deploy uses the repo's `nemo_retriever` Helm chart, and
because the row explicitly names reranking, the reranker NIM (optional in the chart) must be
turned on.

**The five tests, simplest to hardest:**

1. **Prerequisite validation only** — check for a GPU node, a reachable cluster, and Helm;
   report what's present and missing; **ask before installing** anything. No deploy yet.
2. **Helm deploy** — `helm upgrade --install` the chart so extraction, embedding, and
   reranking NIMs land on the GPU node; verify the pods and services are healthy (≤ 30 min).
3. **Smoke ingest** — load one PDF through the in-cluster extraction + embedding NIMs and
   confirm rows were written.
4. **Retrieve query** — close ingest → query against the in-cluster embedding **and
   reranker** NIMs to surface a specific table value.
5. **Acceptance gate** — a clean end-to-end run: validate (ask before installing) → Helm
   deploy ≤ 30 min → ingest into a custom-named index → cited answer, all through the
   in-cluster NIMs with no cloud calls, end-to-end ≤ 40 min. This is the test the others
   build up to.

**Why this order.** Each rung adds exactly one thing: first "is the cluster ready, and does
the agent ask before installing"; then the Helm deploy of all three NIM classes; then the
first real CLI work (ingest); then the query that also exercises the reranker; then
everything composed into the row's real pass/fail gate.

**Relationship to suites 1 & 2.** This is the **Kubernetes** sibling of
`setup_01_cpu_hosted` (hosted) and `setup_02_local_gpu` (on-device). Same small PDFs and
same answers on purpose — the variable under test here is **where the NIMs run and how they
get there**: deployed onto a GPU K8s node via Helm, reached at in-cluster Service URLs.

**Status.** Tests are authored and grounded in the real CLI plus the repo's Helm chart
(`nemo_retriever/helm`, chart 26.5.0); **not yet run live**. Live runs need an actual GPU
Kubernetes cluster, Helm 3, and the NVIDIA NIM Operator (plus NGC credentials for the NIM
images). See `README.md` for the full spec and `cases.json` for the machine-gradable
definitions.
