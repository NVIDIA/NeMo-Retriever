# Use Multimodal Embedding with NeMo Retriever Library

!!! note "Text-only NeMo Retriever embedding NIM"

    You can still use the NeMo Retriever text embedding NIM (OpenAI-compatible embeddings for passage and query vectors) alongside or instead of the multimodal flows on this page. Product and deployment details are in the [NeMo Retriever Text Embedding NIM documentation](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html). In library and CLI pipelines, route embedding to that NIM with your configured embed endpoint and model name (refer to the [graph pipeline examples](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/README.md) for environment-based remote inference).

This documentation describes how to use [NeMo Retriever Library](../understand/overview.md) 
with the multimodal embedding model [Llama Nemotron Embed VL 1B v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2).

The **Llama Nemotron Embed VL 1B v2** model is optimized for multimodal question-answering retrieval. 
The model can embed documents in the form of an image, text, or a combination of image and text. 
Documents can then be retrieved given a user query in text form. 
The model supports images that contain text, tables, charts, and infographics.

## Example with Default Text-Based Embedding { #example-with-default-text-based-embedding }

When you use the multimodal model, by default, all extracted content (text, tables, charts) is treated as plain text. 
The following example provides a strong baseline for retrieval.

- The `embed` method is called with no arguments.

For parameter details, refer to the [Python API guide](../reference/nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed()  # Default behavior embeds all content as text
)
results = ingestor.ingest()
```


## Example with Embedding Structured Elements as Text + Images { #example-with-embedding-structured-elements-as-text-images }

It is common to process PDFs by embedding standard text as text and embed visual elements such as tables and charts as images. 
The following example enables the multimodal model to capture the spatial and structural information of the visual content.

- The `embed` method is configured with `embed_modality="text_image"` to embed the extracted tables and charts as images.
- This configuration is more accurate than text only, with a performance cost.

For parameter details, refer to the [Python API guide](../reference/nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_modality="text_image",
    )
)
results = ingestor.ingest()
```


## Example with Embedding Entire PDF Pages as Images { #example-with-embedding-entire-pdf-pages-as-images }

For documents where the entire page layout is important (such as infographics, complex diagrams, or forms), 
you can configure NeMo Retriever Library to treat every page as a single image.
The following example extracts and embeds each page as an image.

- Set `embed_modality="image"` to use the rendered page image as the embedding input.
- Set `embed_granularity="page"` to create one result row for each PDF page.

These arguments work together. When you set both arguments, the pipeline
enables page-image rendering during extraction, creates one row for each page,
and embeds the full rendered page image. Either argument alone does not enable
the complete page-as-image workflow.

For parameter details, refer to the [Python API guide](../reference/nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_modality="image",
        embed_granularity="page",
    )
)
results = ingestor.ingest()
```

## Related Topics { #related-topics }

- [Pre-Requisites & Support Matrix](../get-started/prerequisites-support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](../support/troubleshoot.md)
- [Use the Python API](../reference/nemo-retriever-api-reference.md)
