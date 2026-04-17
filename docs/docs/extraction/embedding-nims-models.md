# Embedding NIMs and models

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed NeMo Retriever Library.


Embeddings turn extracted text and multimodal content into vectors for semantic search. NeMo Retriever Library integrates with NVIDIA NIM microservices for embedding. Model names and compatibility vary by release; see the [Support matrix](support-matrix.md) and the [NVIDIA NIM catalog](https://build.nvidia.com/).

For multimodal or VLM embeddings, see [Multimodal embeddings (VLM)](vlm-embed.md).

After embedding, content is stored in a vector database; see [Vector databases](data-store.md). RAG-style collections are created and populated through your pipeline configuration and harness runs. For details, see [Benchmarking](benchmarking.md) and the [data store](data-store.md) documentation for your backend.
