# NeMo Retriever API Reference

## PDF pre-splitting for parallel ingest

Server-side PDF splitting supports configurable page chunking. Use `.pdf_split_config(pages_per_chunk=...)` in the Python client, or use the equivalent PDF split page count option in the CLI. See this API guide and the [CLI reference](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) for parameter tables and examples.

::: nemo_retriever.ingestor

## Retriever

The `Retriever` class (LanceDB query helpers, optional reranking, and related utilities) lives in the NeMo Retriever Python package. For the authoritative interface—including method signatures, parameters, and inline examples—see [`nemo_retriever/retriever.py` on GitHub](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/retriever.py).

::: nemo_retriever.params
