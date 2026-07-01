---
title: "Quickstart: retriever CLI"
content_type: tutorial
audience: beginner
journey_stage: try
product: nemo-retriever-library
features:
  - ingestion
technologies:
  - cli
prerequisites:
  - extraction/api-keys.md
  - extraction/getting-started-about.md
duration_minutes: 20
surface: published-docs
status: published
---

# Quickstart: retriever CLI

Use the `retriever` CLI to ingest documents locally and query the resulting LanceDB index. This quickstart covers a minimal local PDF ingest and query. For all flags and subcommands, refer to the [CLI reference on GitHub](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli).

## Before you begin

1. Install the NeMo Retriever package (for example `pip install nemo-retriever` from [PyPI](https://pypi.org/project/nemo-retriever/) or an editable install from the [nemo_retriever](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever) tree).
2. Confirm your environment against the [Prerequisites and support matrix](../extraction/prerequisites-support-matrix.md).
3. If you call [NVIDIA-hosted NIMs](https://build.nvidia.com/), set `NVIDIA_API_KEY` as described in [Get API keys](../extraction/api-keys.md).

## Ingest a PDF locally

From a checkout of NeMo-Retriever (or any directory that contains your PDF), run:

```bash
retriever ingest ./data/multimodal_test.pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --use-table-structure \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

By default, local ingest writes to `lancedb/nemo-retriever`. The [test PDF](https://github.com/NVIDIA/NeMo-Retriever/blob/main/data/multimodal_test.pdf) in the repository contains text, tables, charts, and images.

## Query the index

```bash
retriever query "What is in this document?" \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

`retriever query` reads from the same LanceDB URI and table name used during ingest unless you override `--lancedb-uri` or `--table-name`.

## Next steps

- For a Python-based ingest workflow, refer to [Ingest documents into a searchable collection](../extraction/workflow-document-ingestion.md).
- To route ingest to hosted NIM endpoints, export `NVIDIA_API_KEY` and pass `--page-elements-invoke-url`, `--ocr-invoke-url`, and related flags — refer to the [CLI reference — hosted endpoints](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli).
- For batch Ray ingest or service ingest, use `retriever ingest batch` or `retriever ingest service` — refer to the full [CLI reference](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli).
