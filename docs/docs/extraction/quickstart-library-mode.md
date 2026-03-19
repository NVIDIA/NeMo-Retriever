# NeMo Retriever Library

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed to the NeMo Retriever Library.

Use the [Quick Start for NeMo Retriever Library](https://github.com/NVIDIA/NeMo-Retriever/blob/26.03/nemo_retriever/README.md) to set up and run the NeMo Retriever Library locally, so you can build a GPU‑accelerated, multimodal RAG ingestion pipeline that parses PDFs, HTML, text, audio, and video into LanceDB vector embeddings, integrates with Nemotron RAG models (locally or via NIM endpoints), which includes Ray‑based scaling plus built‑in recall evaluation.

!!! note "Imports: library mode vs remote client"

    **In-process (`nemo_retriever` package):** there is no `nemo_retriever.client` module. Use APIs such as `create_ingestor` and `Retriever` (requires the full library stack — Ray, GPU models, etc.), for example:

    ```python
    from nemo_retriever import create_ingestor
    from nemo_retriever.retriever import Retriever
    ```

    See the linked README for setup and usage.

    **Remote client (Docker Compose / managed deployment):** use `nv_ingest_client`:

    ```python
    from nv_ingest_client.client.interface import Ingestor
    from nv_ingest_client.client import NvIngestClient
    ```

    Do not use a `NemoRetrieverClient` class — use **`NvIngestClient`**.