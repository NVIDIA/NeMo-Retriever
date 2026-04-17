# When to self-host NIMs

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed NeMo Retriever Library.


Self-hosted NIMs run on your GPUs or air-gapped hardware, typically with Kubernetes and the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html).

Consider self-hosting when:

- You need an air gap, strict data residency, or customer data must not leave your network.
- You run at large scale where dedicated capacity can cost less than hosted API usage.
- You must meet latency or locality requirements that hosted regions cannot satisfy.

**GPU sharing.** The NIM Operator supports time-slicing and MIG so multiple NIM workloads can share GPUs. A NIM used with NeMo Retriever Library does not always need a full dedicated GPU when the operator and GPU profile are set correctly. For scheduling and GPU partitioning, see the [NIM Operator documentation](https://docs.nvidia.com/nim-operator/latest/index.html).

**Related**

- [Deploy (Helm Chart)](helm.md)
- [Support matrix](support-matrix.md)
- [Compare deployment options](choose-your-path.md)
