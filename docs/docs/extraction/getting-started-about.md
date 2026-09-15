# About getting started

This section walks you from **access and prerequisites** through **first deployment** and **hands-on notebooks**.

If you are new to the library, read [NeMo Retriever Library Overview](overview.md) and [Concepts](concepts.md) under **Introduction** first. The library is the evaluation and integration layer for Nemotron models. It is not a turnkey RAG platform.

Typical order:

1. Confirm the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for your OS, GPU, software stack, and Kubernetes persistent storage and GPU scheduling if you use Helm. Local GPU inference requires Linux; remote NIM workflows can use the base package on Windows x64 and macOS Apple Silicon (arm64) as well. macOS Intel (x86_64) is not supported.
2. For a local GPU and Hugging Face checkpoints, start with the [package quick start](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever). You do not need to deploy the four default Helm NIMs for that path.
3. For hosted or self-hosted NIMs, Kubernetes, or a standalone Docker service, choose a path in [Deployment options](deployment-options.md). For Helm, complete the [persistent storage prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#persistent-storage-prerequisite) and the [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite) before `helm install`.
4. [Get your API key](api-keys.md) when your workflow calls NGC or hosted NVIDIA endpoints. Local Hugging Face checkpoints do not require a hosted API key.
5. Explore [Jupyter Notebooks](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md) for additional examples.
6. Measure quality on your documents with [Evaluate on your data](evaluate-on-your-data.md).

The NeMo Retriever Library and its Helm chart are not supported under NVIDIA AI Enterprise (NVAIE). For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](overview.md#nvidia-ai-enterprise-nvaie-support).
