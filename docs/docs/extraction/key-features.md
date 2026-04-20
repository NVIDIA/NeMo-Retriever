# Key features

!!! note

    This documentation describes NeMo Retriever Library.


NeMo Retriever Library is built for **accuracy**, **throughput**, and **flexible deployment** in enterprise retrieval pipelines.

## Extraction and understanding

- **Multimodal extraction** — Text, tables, charts, and infographics from documents and media formats supported by the [support matrix](support-matrix.md).
- **Multiple extraction backends** — Trade off throughput and accuracy per document type (for example PDF via pdfium and [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse)).
- **Chunking and enrichment** — Splitting, transforms, filtering, embedding, and optional offload to storage.

## Operations and scale

- **High throughput** — Parallel extraction, embedding, and indexing designed for large document corpora.
- **Configurable pipelines** — Jobs are described as JSON; you choose tasks per payload rather than a single fixed pipeline.
- **Deployment options** — [Library mode](quickstart-library-mode.md), [Helm](helm.md), and [notebooks](notebooks.md) for different environments.

## Enterprise-oriented behavior

- **Composable services** — Use modules that fit your environment and security boundaries.
- **Clear metadata** — Structured JSON results with extraction metadata, annotations, and trace data for observability.

## Related topics

- [Concepts](concepts.md) — Core ideas (jobs, pipelines, metadata).
- [Choose your path](choose-your-path.md) — Pick a deployment and learning path.
- [Overview](overview.md) — Full product description.
