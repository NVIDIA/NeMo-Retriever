# Release Notes for NeMo Retriever Library

This documentation contains the 26.05 Release Notes (26.5.0) for [NeMo Retriever Library](overview.md).

## 26.05 Release Notes (26.5.0)

NVIDIA® NeMo Retriever Library version 26.05 builds on the 26.03 foundation with a graph-based ingest architecture, expanded multimodal and tabular capabilities, production-oriented service deployment, and documentation aligned to a Helm-first supported path.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

Highlights for the 26.05 release include:

- **Graph-based ingest pipeline** — `graph_pipeline` and the graph stage registry are the canonical ingestion path; mode-specific example scripts are consolidated around this model  
- **Root CLI** — `retriever ingest` and `retriever query` with NIM URL flags, batch tuning, and LanceDB controls (overwrite/append)  
- **Retriever Service v2** — scalable multi-pod architecture with gateway, process isolation, and VectorDB integration  
- **Nemotron OCR v2** — default OCR engine with CLI language selectors and unified OCR actors  
- **VLM image captioning** — image captioning via vLLM (including Omni caption model profiles); addresses the capability deferred in 26.03  
- **vLLM inference stack** — vLLM-backed text and vision-language embedders, multimodal VL reranker, and torch 2.11 stack for local GPU installs  
- **Video retrieval pipeline** — frame extraction, OCR, audio-visual fusion, and text deduplication for video corpora  
- **Text-to-SQL** — agent graph and tabular tooling for structured data retrieval  
- **Live RAG SDK** — `Retriever.answer()` and optional batch operator graph via LiteLLM (`[llm]` extra)  
- **Vector database** — VDB operators integrated directly in the pipeline; custom metadata support; LanceDB hybrid search guidance updated  
- **Evaluation** — BEIR-centric evaluation overhaul; `retriever skill-eval` benchmark CLI for the NeMo Retriever skill  
- **Packaging** — optional install extras (`[local]`, `[multimedia]`, `[llm]`, `[tabular]`, `[nemotron-parse]`, `[service]`, and others) including slim remote/NIM-only installs on Mac and Windows  
- **Audio** — long-audio Parakeet chunking with time-aligned segments; punctuation-based audio segmenting  
- **`allow_no_gpu`** — option to skip GPU requirement during ingest for CPU-only experimentation  
- **Chunking API** — text splitting moved into `.extract(split_config=...)`  
- **Documentation** — Helm-first deployment story; [Docker Compose for local development](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docker.md) documented as **unsupported** developer tooling (not a production NIM deployment path)  
- **Documentation** — duplicate user-defined stages page removed; UDF and custom stages guidance consolidated in the [graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph)  
- **Documentation** — consolidated extraction concepts, ingest workflow, embeddings, and audio/video guides  

## Release Notes for Previous Versions

| [26.03](https://docs.nvidia.com/nemo/retriever/26.3.0/extraction/releasenotes/)
| [26.1.2](https://docs.nvidia.com/nemo/retriever/26.1.2/extraction/releasenotes/)
| [26.1.1](https://docs.nvidia.com/nemo/retriever/26.1.1/extraction/releasenotes/)
| [25.9.0](https://docs.nvidia.com/nemo/retriever/25.9.0/extraction/releasenotes/) 
| [25.6.3](https://docs.nvidia.com/nemo/retriever/25.6.3/extraction/releasenotes/) 
| [25.6.2](https://docs.nvidia.com/nemo/retriever/25.6.2/extraction/releasenotes/) 
| [25.4.2](https://docs.nvidia.com/nemo/retriever/25.4.2/extraction/releasenotes/) 
| [25.3.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes/) 
| [24.12.1](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes/#release-24121) 
| [24.12.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes/#release-2412) 

## Related Topics

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Deployment options](deployment-options.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
