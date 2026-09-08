# Use Multimodal Embedding with NeMo Retriever Library

!!! note "Text-only NeMo Retriever embedding NIM"

    You can still use the NeMo Retriever text embedding NIM (OpenAI-compatible embeddings for passage and query vectors) alongside or instead of the multimodal flows on this page. Product and deployment details are in the [NeMo Retriever Text Embedding NIM documentation](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html). In library and CLI pipelines, route embedding to that NIM with your configured embed endpoint and model name (refer to the [graph pipeline examples](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/README.md) for environment-based remote inference).

This documentation describes how to use [NeMo Retriever Library](overview.md) 
with the multimodal embedding model [Llama Nemotron Embed VL 1B v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2).

The **Llama Nemotron Embed VL 1B v2** model is optimized for multimodal question-answering retrieval. 
The model can embed documents in the form of an image, text, or a combination of image and text. 
Documents can then be retrieved given a user query in text form. 
The model supports images that contain text, tables, charts, and infographics.

## Example with Default Text-Based Embedding { #example-with-default-text-based-embedding }

When you use the multimodal model, by default, all extracted content (text, tables, charts) is treated as plain text. 
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

## Text inputs that exceed the model limit { #text-input-overflow }

Before it embeds text, NeMo Retriever Library checks the complete formatted
input against the embedding model's token limit. The check includes the model's
document or query prefix and special tokens. The default configured runtime
limit is 8,192 tokens. If the checkpoint declares a smaller supported limit,
the checkpoint limit takes precedence.

For a registered revision-pinned model, an explicitly revision-pinned model,
or a local checkpoint, the library loads the tokenizer and prompt configuration
for that exact model version. If the text does not fit, the library splits it
into deterministic contiguous token ranges that fit. This split does not
truncate text and occurs before either local or remote embedding.
If the exact tokenizer, selected prompt prefix, or checkpoint-supported token
limit is unavailable, embedding stage setup fails before inference rather than
guessing an admission policy.

Each split row preserves the source, page, element, bounding box, and existing
document chunk metadata from its parent. The library adds one
`metadata["embedding_split"]` mapping so you can identify and order the
embedding-specific children:

- `parent_id` identifies the parent content and provenance.
- `chunk_id` identifies one deterministic child.
- `chunk_index` and `chunk_count` describe the child order.
- `start_token` and `end_token` describe the source token range.
- `content` preserves the child's exact text, including whitespace.

The returned `DataFrame` can therefore contain more rows than the embedding
stage received. Existing fields such as `chunk_index` and the physical page
number keep their original meaning.

Local and remote embedding use the same prepared rows. When this client-side
policy is active for a remote endpoint, the request uses `truncate="NONE"` so
the endpoint cannot silently replace the client decision. If a backend still
rejects a prepared batch, the library reports a batch failure rather than
guessing from an HTTP status or exception that one document is invalid. The VDB
boundary refuses a mixed partial write when searchable rows are missing
embeddings.

For an unpinned custom remote model, the library does not guess its tokenizer
or input limit. Embedding stage setup fails with an actionable error. Use a
registered model, a local checkpoint, or an immutable model revision so the
library can enforce deterministic client-side admission.

The embedding stage records per-row counts in
`embedding_v1_counts_by_label`. When a batch contains an overlength or failed
row, it also logs a summary with `input_rows`, `output_rows`, `overlength`,
`split`, `split_children`, `truncated`, `failed`, `embedded`, and `unembedded`.
The deterministic split policy reports `truncated=0`.

## Example with Embedding Structured Elements as Text + Images { #example-with-embedding-structured-elements-as-text-images }

It is common to process PDFs by embedding standard text as text and embed visual elements such as tables and charts as images. 
The following example enables the multimodal model to capture the spatial and structural information of the visual content.

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

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

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

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use the Python API](nemo-retriever-api-reference.md)
