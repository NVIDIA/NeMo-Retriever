# NeMo Retriever API Reference

## PDF pre-splitting for parallel ingest

Server-side PDF splitting uses ingest `api_version` `v2`: set `message_client_kwargs={"api_version": "v2"}` on the Python client (and optional `.pdf_split_config(pages_per_chunk=...)`), or pass `--api_version v2` with `--pdf_split_page_count` on the CLI. See the [Python API](python-api-reference.md) and [CLI reference](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) for parameter tables and examples.

::: nemo_retriever.ingestor

::: nemo_retriever.retriever

::: nemo_retriever.params
