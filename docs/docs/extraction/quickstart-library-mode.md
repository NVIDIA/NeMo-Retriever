# Deploy Without Containers (Library Mode) for NeMo Retriever Library

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed NeMo Retriever Library.

Use the [Quick Start for NeMo Retriever Library](https://github.com/NVIDIA/NeMo-Retriever/blob/26.03/nemo_retriever/README.md) to set up and run the NeMo Retriever Library locally, so you can build a GPU‑accelerated, multimodal RAG ingestion pipeline that parses PDFs, HTML, text, audio, and video into LanceDB vector embeddings, integrates with Nemotron RAG models (locally or via NIM endpoints), which includes Ray‑based scaling with built‑in recall evaluation. Python 3.12 or later is required (see [Prerequisites](prerequisites.md)).

## `run_pipeline` API notes

For `run_pipeline(...)`, the following parameters are optional and should be treated as `Required: No`:

- `pipeline_config: Optional[PipelineConfigSchema] = None`
- `run_in_subprocess: bool = False`
- `block: bool = True`

In non-blocking subprocess mode (`run_in_subprocess=True` and `block=False`), the function returns
`RayPipelineSubprocessInterface` (not `RayPipelineInterface`).
