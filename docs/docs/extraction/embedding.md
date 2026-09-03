# Use Embedding with NeMo Retriever Library

NeMo Retriever Library defaults to the text-only [Nemotron 3 Embed 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) model. The logical model ID is `nvidia/nemotron-3-embed-1b`.

For local inference, the default `vllm` backend resolves that logical ID as follows:

- On Blackwell systems where every visible CUDA device has compute capability 10.0 or later, it selects `nvidia/Nemotron-3-Embed-1B-NVFP4`.
- On older GPUs, unknown devices, or systems without visible CUDA devices, it selects `nvidia/Nemotron-3-Embed-1B-BF16`.
- When you explicitly select the `hf` backend, it selects the BF16 checkpoint because the NVFP4 checkpoint requires vLLM.

The BF16 and NVFP4 checkpoints share an embedding space. Validate retrieval quality on representative data before you change the precision used by an existing index.

Helm deploys `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2`. The NIM selects a supported NVFP4 or BF16 engine automatically. For NIM product details, refer to the [NeMo Retriever Text Embedding NIM documentation](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html).

## Example with Default Text-Based Embedding { #example-with-default-text-based-embedding }

By default, all extracted content, including tables and charts, is treated as plain text.
The following example provides a strong baseline for retrieval.

- The `embed` method is called with no arguments.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

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

It is common to process PDFs by embedding standard text as text and visual elements such as tables and charts as images.
The following example enables the multimodal model to capture the spatial and structural information of the visual content.

- Set `embed_model_name="nvidia/llama-nemotron-embed-vl-1b-v2"` to override the text-only default with the legacy multimodal model.
- The `embed` method is configured with `embed_modality="text_image"` to embed the extracted tables and charts as images.
- This configuration is more accurate than text only, with a performance cost.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
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
- Set `embed_model_name="nvidia/llama-nemotron-embed-vl-1b-v2"` to use the legacy multimodal model.

These arguments work together. When you set both arguments, the pipeline
enables page-image rendering during extraction, creates one row for each page,
and embeds the full rendered page image. Either argument alone does not enable
the complete page-as-image workflow.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_modality="image",
        embed_granularity="page",
    )
)
results = ingestor.ingest()
```

## Related Topics { #related-topics }

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use the Python API](nemo-retriever-api-reference.md)
