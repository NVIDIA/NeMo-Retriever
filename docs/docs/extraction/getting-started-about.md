---
title: "Start here: get access, deploy, and run examples"
content_type: tutorial
audience: beginner
journey_stage: try
product: nemo-retriever-library
features: []
technologies: []
prerequisites:
  - extraction/overview.md
duration_minutes: 45
surface: published-docs
status: published
---

# Start here: get access, deploy, and run examples

This page is the recommended entry path for NeMo Retriever Library (NRL).
Complete the steps in order, or jump to a step that matches your goal.

| Step | Type | Time | Action |
|------|------|------|--------|
| 1 | How-to | ~10 min | [Get API keys](api-keys.md) |
| 2 | Reference | ~15 min | [Prerequisites and support matrix](prerequisites-support-matrix.md) |
| 3 | How-to | ~15 min | [Choose a deployment option](deployment-options.md) |
| 4 | How-to | ~25 min | [Ingest documents into a searchable collection](workflow-document-ingestion.md) |
| 5 | Tutorial | ~20 min | [Run the retriever CLI quickstart](../reference/retriever-cli-quickstart.md) |
| 6 | Tutorial | varies | Explore [Jupyter notebooks](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md) for end-to-end examples |

If you are new to the product, read [NeMo Retriever Library Overview](overview.md) and [Key concepts](concepts.md) under **Introduction** first.

Confirm the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for your OS, GPU, and software stack. Local GPU inference requires Linux. Remote NIM workflows can use the base package on Windows x64 and macOS Apple Silicon (arm64) as well. macOS Intel (x86_64) is not supported.

Helm, Docker, the full CLI reference, and graph customization live in the [NeMo-Retriever GitHub repository](external-documentation-map.md). For Helm, complete the [persistent storage prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#persistent-storage-prerequisite) and the [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite) before `helm install`.

The NeMo Retriever Library and its Helm chart are not supported under NVIDIA AI Enterprise (NVAIE). For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](overview.md#nvidia-ai-enterprise-nvaie-support).
