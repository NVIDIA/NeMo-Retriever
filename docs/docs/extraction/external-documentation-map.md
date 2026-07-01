---
title: "External documentation map"
content_type: reference
audience: intermediate
journey_stage: try
product: nemo-retriever-library
features:
  - deployment
  - api
technologies:
  - helm-kubernetes
  - docker
  - cli
prerequisites:
  - extraction/getting-started-about.md
surface: published-docs
status: published
---

# External documentation map

Several procedures in the NeMo Retriever documentation navigation point to the [NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever) GitHub repository. Use this page to find the canonical location for each topic.

| Topic | Canonical location | Surface | Owns |
|-------|-------------------|---------|------|
| Library quickstart and package source | [nemo_retriever/](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever) | `github-readme` | Install, package layout, first code examples |
| Helm chart (Kubernetes) | [nemo_retriever/helm/README.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) | `github-readme` | Chart values, NIM enablement, deploy procedures |
| OpenShift deployment | [nemo_retriever/helm/openshift.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/openshift.md) | `github-readme` | OpenShift-specific Helm configuration |
| Docker service image | [nemo_retriever/docker.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docker.md) | `github-readme` | Service container build and run |
| CLI reference | [nemo_retriever/docs/cli](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) | `github-cli-docs` | All `retriever` subcommands and flags |
| Graph and custom stages | [nemo_retriever graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph) | `github-readme` | Custom pipeline stages and UDF-style operations |

For published extraction guides that stay on docs.nvidia.com, start from [Start here](getting-started-about.md).
