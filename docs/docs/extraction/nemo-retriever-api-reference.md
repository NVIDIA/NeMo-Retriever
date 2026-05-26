# NeMo Retriever API Reference

## PDF pre-splitting for parallel ingest

Large PDFs are split into page batches before Ray processing so extraction can run in parallel. This happens on the default ingest path; you do not need extra configuration for typical workloads.

To tune splitter throughput from the CLI, use `--pdf-split-batch-size` (Ray actor batch size for the splitter stage). See [Text chunking and PDF page batches](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli#text-chunking-and-pdf-page-batches) in the CLI reference.

::: nemo_retriever.ingestor
    options:
      filters:
        - "!^pdf_split_config$"

::: nemo_retriever.retriever

::: nemo_retriever.params
