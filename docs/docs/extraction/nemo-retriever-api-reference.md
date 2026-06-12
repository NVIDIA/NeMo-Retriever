# NeMo Retriever API Reference

## PDF pre-splitting for parallel ingest

Large PDFs are split into page batches before Ray processing so extraction can run in parallel. This happens on the default ingest path; you do not need extra configuration for typical workloads.

To tune splitter throughput from the CLI, use `--pdf-split-batch-size` (Ray actor batch size for the splitter stage). See [Text chunking and PDF page batches](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli#text-chunking-and-pdf-page-batches) in the CLI reference.

**Python client (`pdf_split_config`):** Only `create_ingestor(run_mode="service")` implements `.pdf_split_config(pages_per_chunk=...)`, which records page-chunking settings in the request pipeline spec for the remote gateway. Local graph ingest (`run_mode="inprocess"` or `"batch"`) raises `NotImplementedError` if you call this method; PDFs are split automatically on the default ingest path without client-side configuration.

::: nemo_retriever.ingestor
    options:
      filters:
        - "!^pdf_split_config$"

::: nemo_retriever.graph.retriever

::: nemo_retriever.common.params
