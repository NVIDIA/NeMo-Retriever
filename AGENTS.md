# Agent instructions: NeMo Retriever

Use this file when the user asks to set up, deploy, or troubleshoot **NeMo Retriever** with little or no prior context. Do not treat **Helm**, **library mode**, and **NIM** as three interchangeable top-level products.

## Concepts (read this first)

- **NeMo Retriever Library** — Python package and pipelines (`nemo_retriever`). Runs in your process: local GPU inference, calls to **hosted** model APIs, or calls to **self-hosted** NIM URLs. See [`nemo_retriever/README.md`](nemo_retriever/README.md).
- **Helm / Kubernetes** — Deploys ingestion and NIM microservices as cluster workloads (often with the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html)). See [`helm/README.md`](helm/README.md) and the narrative quickstart [`docs/docs/extraction/quickstart-guide.md`](docs/docs/extraction/quickstart-guide.md).
- **NIM** — **Inference serving** (NVIDIA-hosted at [build.nvidia.com](https://build.nvidia.com/) or self-hosted containers). It is not a separate “mode” from library vs Helm; both library and Helm setups **invoke** NIMs one way or another.

Human-facing comparison: [`docs/docs/extraction/deployment-options.md`](docs/docs/extraction/deployment-options.md).

## Before you propose steps

Infer or ask, in order:

1. **Kubernetes** — Is there a cluster with GPU node pools and (for self-hosted NIMs) the NIM Operator installed and supported?
2. **Local NVIDIA GPU** — Is CUDA 13–compatible stack available if the user wants **local** library inference (`[local]` extra and CUDA torch wheels per library README)?
3. **Data and compliance** — Must document or query content stay on-network or air-gapped? If yes, **hosted** NIMs are usually wrong unless security has explicitly approved them.
4. **Goal** — Embed in a Python app / notebook vs run shared multi-tenant services on a platform?

## Routing (default choices)

| Environment signal | Prefer |
|--------------------|--------|
| No Kubernetes; laptop, single VM, or simple CI | **Library** — [`nemo_retriever/README.md`](nemo_retriever/README.md); library deploy doc pointer: [`docs/docs/extraction/quickstart-library-mode.md`](docs/docs/extraction/quickstart-library-mode.md) |
| No local GPU (or no local CUDA 13 stack for Nemotron) | **Library, remote inference only** — base `pip`/`uv` install without `[local]`; GPU not required for NVIDIA-hosted inference (see [`docs/docs/extraction/prerequisites.md`](docs/docs/extraction/prerequisites.md)) |
| Kubernetes + GPUs + operator-supported deployment | **Helm** — [`helm/README.md`](helm/README.md); prerequisites [`docs/docs/extraction/prerequisites.md`](docs/docs/extraction/prerequisites.md) |
| Air gap or strict data residency | **Self-hosted** NIMs + private registries; follow Helm and [NIM Operator air-gap](https://docs.nvidia.com/nim-operator/latest/air-gap.html) guidance; see hosted vs self-hosted in [`docs/docs/extraction/deployment-options.md`](docs/docs/extraction/deployment-options.md) |
| Fastest try and policy allows cloud inference | **Hosted NIMs** + API keys — [`docs/docs/extraction/api-keys.md`](docs/docs/extraction/api-keys.md) and deployment-options “hosted NIMs” section |

**Avoid** presenting Helm, NIM, and library as a flat pick-one menu without the table above.

## Operational reminders

- Align **versions** across `nemo-retriever`, `nv-ingest*`, client packages, and Helm chart/image tags to the same release family the docs or README pin for that branch.
- **Python** for library and clients: **3.12+** (see prerequisites); older Python often fails dependency resolution.
- Helm first startup can take **many minutes** while NIM images pull and models load; set expectations and check pod readiness and GPU processes before declaring failure.
- Single-GPU clusters may need **values overrides** (see `helm/overrides/` and Helm README) if default memory or concurrency does not fit VRAM.

## Suggested user clarification (when stuck)

If context is missing after the checks above, ask briefly: whether they have Kubernetes and the NIM Operator, whether they have a local GPU suitable for library `[local]` mode, and whether content may be sent to NVIDIA-hosted APIs.
