# Embedding NIMs and models

Embeddings turn extracted text and multimodal content into vectors for semantic search. NeMo Retriever Library integrates with NVIDIA NIM microservices for embedding. Model names and compatibility vary by release; refer to the [Support matrix](support-matrix.md) and the [NVIDIA NIM catalog](https://build.nvidia.com/).

For multimodal or VLM embeddings, refer to [Multimodal embeddings (VLM)](embedding.md).

After embedding, content is stored in a vector database; refer to [Vector databases](vdbs.md). RAG-style collections are created and populated through your pipeline configuration and harness runs. For details, refer to [Benchmarking](benchmarking.md) and the [data store](vdbs.md) documentation for your backend.
