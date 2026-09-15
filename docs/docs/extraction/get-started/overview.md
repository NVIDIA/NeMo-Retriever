# About getting started

This section walks you from **access and prerequisites** through **first deployment** and **hands-on notebooks**.

Typical order:

1. [Get your API key](api-keys.md) (NGC / API access as required by your workflow).
2. Confirm the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for your OS, GPU, software stack, and Kubernetes persistent storage and GPU scheduling if you use Helm. Local GPU inference requires Linux; remote NIM workflows can use the base package on Windows x64 and macOS Apple Silicon (arm64) as well. macOS Intel (x86_64) is not supported.
3. Choose a path in [Deployment options](../deploy/deployment-options.md). You can use the local library, hosted NIMs, the Helm chart for Kubernetes, or a standalone Docker service. For Helm, complete the [persistent storage prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#persistent-storage-prerequisite) and the [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite) before `helm install`.
4. Explore [Jupyter Notebooks](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md) for end-to-end examples.

The NeMo Retriever Library and its Helm chart are not supported under NVIDIA AI Enterprise (NVAIE). For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](../about/overview.md#nvidia-ai-enterprise-nvaie-support).

If you are new to the product, read [NeMo Retriever Library overview](../about/overview.md) and [Key concepts](../about/concepts.md) under **About** first.

## Next steps

Continue to [Ingest and extract content](../ingest/index.md) to run an ingestion workflow.
