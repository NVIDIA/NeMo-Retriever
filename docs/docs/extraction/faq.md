---
title: "FAQ"
content_type: faq
audience: beginner
journey_stage: discover
product: nemo-retriever-library
features:
  - support
technologies: []
prerequisites:
  - extraction/overview.md
surface: published-docs
status: published
---

# Frequently Asked Questions for NeMo Retriever Library

This documentation contains the Frequently Asked Questions (FAQ) for [NeMo Retriever Library](overview.md).

## Is the NeMo Retriever Library supported under NVIDIA AI Enterprise (NVAIE)?

No. The NeMo Retriever Library, including its container image and Helm chart artifacts, is not supported under NVIDIA AI Enterprise (NVAIE).

Some NIM microservices and models that the library calls may be individually covered by NVAIE. That coverage does not extend to the NeMo Retriever Library or its end-to-end extraction workflow. For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](overview.md#nvidia-ai-enterprise-nvaie-support).

## What if I already have a retrieval pipeline? Can I just use NeMo Retriever Library? 

You can use the CLI or Python APIs to perform extraction only, and then consume the results.
Using the Python API, `results` is a list object with one entry.
For code examples, refer to the Jupyter notebooks [Multimodal RAG with LlamaIndex](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/llama_index_multimodal_rag.ipynb) 
and [Multimodal RAG with LangChain](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/langchain_multimodal_rag.ipynb).

## Where does NeMo Retriever Library ingest to?

NeMo Retriever Library supports extracting text representations of various forms of content,
and ingesting to a vector database. **[LanceDB](https://lancedb.com/)** stores vectors as local Lance files on disk for the supported ingestion path.
You can ingest to other data stores; however, you must configure other data stores yourself.
For more information, refer to [Vector databases](vdbs.md).

## How would I process unstructured images?

For images that `nemoretriever-page-elements-v3` does not classify as tables, charts, or infographics,
you can use our VLM caption task to create a dense caption of the detected image. 
That caption is then embedded along with the rest of your content. 
For chart-labeled PDF regions and other caption scope limits, refer to [Are PDF chart or figure regions captioned when Omni is enabled?](#are-pdf-chart-or-figure-regions-captioned-when-omni-is-enabled). For more information, refer to [Extract Captions from Images](nemo-retriever-api-reference.md).

## Are PDF chart or figure regions captioned when Omni is enabled?

No. Chart-labeled PDF regions are not routed through Omni captioning. Refer to [Charts and infographics](multimodal-extraction.md#charts-and-infographics) and [Image captioning](multimodal-extraction.md#image-captioning) for caption scope and validation.

## When should I consider advanced visual parsing?

For scanned documents, or documents with complex layouts,
you can use [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) as an alternate PDF extraction method by setting `extract_method="nemotron_parse"`.
Nemotron Parse does not produce chart modality rows. For chart detection and chart-filtered retrieval, use the default **pdfium** layout path instead (refer to [Charts and infographics](multimodal-extraction.md#charts-and-infographics)).
For more information, refer to [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse).

## Why are the environment variables different between library mode and self-hosted mode?

### Self-Hosted Deployments

For [self-hosted deployments](deployment-options.md#when-to-self-host-nims), you should set the environment variables `NGC_API_KEY` and `NIM_NGC_API_KEY`.
For more information, refer to [Authentication and API keys](api-keys.md).

### Library Mode

For production environments, you should use the provided Helm charts. When you run the NeMo Retriever Library from Python without those charts, set `NVIDIA_API_KEY` only when you call [build.nvidia.com](https://build.nvidia.com/) hosted inference—it is not required for locally deployed Hugging Face models or self-hosted NIM endpoints. For more information, refer to [Deployment options](deployment-options.md) and [Authentication and API keys](api-keys.md).

For advanced scenarios, you might want to use library mode with self-hosted NIM instances. 
You can set custom endpoints for each NIM. 
For examples of `*_ENDPOINT` variables, refer to [Environment variables](environment-config.md) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

When you explicitly configure remote NIM endpoints in Python library mode, graph ingestion raises a `GraphIngestionError` if a stage reports row-level connection or inference errors. This makes unreachable services visible to callers instead of returning a DataFrame that looks successful. To intentionally keep partial results with row-level error payloads, pass `error_policy="collect"` to `GraphIngestor` or `create_ingestor`.

## What parameters or settings can I adjust to optimize extraction from my documents or data? 

Refer to [Evaluate on your data](evaluate-on-your-data.md) for extraction tuning and optimization guidance.

You can configure the `extract`, `caption`, and other tasks—including which content types to extract—using the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `GraphIngestor`). For PDF element selection, refer to [Extract Specific Elements from PDFs](nemo-retriever-api-reference.md).

To generate captions for images, use code similar to the following.
For more information, refer to [Extract Captions from Images](nemo-retriever-api-reference.md).

```python
from pathlib import Path

from nemo_retriever import create_ingestor

documents = [str(Path("data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")
ingestor = ingestor.files(documents).extract().caption().embed()
```
