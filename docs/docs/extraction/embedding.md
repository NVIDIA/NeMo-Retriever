# Use Multimodal Embedding with NeMo Retriever Library

This documentation describes how to use [NeMo Retriever Library](overview.md) 
with the multimodal embedding model [Llama 3.2 NeMo Retriever Multimodal Embedding 1B](https://build.nvidia.com/nvidia/llama-3_2-nemoretriever-1b-vlm-embed-v1).

The `Llama 3.2 NeMo Retriever Multimodal Embedding 1B` model is optimized for multimodal question-answering retrieval. 
The model can embed documents in the form of an image, text, or a combination of image and text. 
Documents can then be retrieved given a user query in text form. 
The model supports images that contain text, tables, charts, and infographics.


## Example with Default Text-Based Embedding

When you use the multimodal model, by default, all extracted content (text, tables, charts) is treated as plain text. 
The following example provides a strong baseline for retrieval.

- The `embed` method is called with no arguments.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract()
    .embed() # Default behavior embeds all content as text
)
results = ingestor.ingest()
```


## Example with Embedding Structured Elements as Text + Images

It is common to process PDFs by embedding standard text as text, and embed visual elements like tables and charts as images. 
The following example enables the multimodal model to capture the spatial and structural information of the visual content.

- The `embed` method is configured with `embed_modality="text_image"` to embed the extracted tables and charts as images.
- This configuration is more accurate than text only with a performance cost

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_modality="text_image",
    )
)
results = ingestor.ingest()
```


## Example with Embedding Entire PDF Pages as Images

For documents where the entire page layout is important (such as infographics, complex diagrams, or forms), 
you can configure NeMo Retriever Library to treat every page as a single image.
The following example extracts and embeds each page as an image.

- The `embed method` processes the page images.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_modality="image",
        embed_granularity="page"
    )
)
results = ingestor.ingest()
```

## Related Topics

- [Support Matrix](support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use the Python API](nemo-retriever-api-reference.md)
- [Extract Captions from Images](nemo-retriever-api-reference.md)
