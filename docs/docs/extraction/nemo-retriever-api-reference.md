# NeMo Retriever API Reference

## PDF pre-splitting for parallel ingest

Large PDFs are split into page batches before Ray processing so extraction can run in parallel. This happens on the default ingest path; you do not need extra configuration for typical workloads.

To tune splitter throughput from the CLI, use `--pdf-split-batch-size` (Ray actor batch size for the splitter stage). See [Text chunking and PDF page batches](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli#text-chunking-and-pdf-page-batches) in the CLI reference.

**Python client (`pdf_split_config`):** Only `create_ingestor(run_mode="service")` implements `.pdf_split_config(pages_per_chunk=...)`, which records page-chunking settings in the request pipeline spec for the remote gateway. Local graph ingest (`run_mode="inprocess"` or `"batch"`) raises `NotImplementedError` if you call this method; PDFs are split automatically on the default ingest path without client-side configuration.

## One-shot text generation

`TextGenerationOperator` is the reusable base for synchronous, one-request-per-row text generation. It is a provisional text-only API: it does not support tool calls, agent loops, streaming, multiple choices, or structured domain results.

Concrete operators construct an immutable `TextGenerationTask` and provide reconstructible constructor state. Runtime task and client objects must not be included in graph constructor arguments. A custom completion client must be safe for concurrent calls or report that it does not support concurrent calls so the operator serializes access.

Embedding and captioning remain separate operator families because they use modality grouping, native batching, and specialized CPU/GPU lifecycles.

### Generic generation and summarization

Both operators consume a pandas DataFrame and add text, latency, model, and error columns without changing the input rows:

```python
import pandas as pd

from nemo_retriever.common.params import TextGenerationParams
from nemo_retriever.operators.generation import GenericGenerationOperator, SummarizationOperator

summary_params = TextGenerationParams.from_kwargs(
    model="openai/gpt-4o-mini",
    api_key="os.environ/OPENAI_API_KEY",
    temperature=0.0,
    max_tokens=512,
)
summaries = SummarizationOperator(summary_params).run(
    pd.DataFrame({"text": ["A long document to summarize."]})
)

prompt_params = TextGenerationParams.from_kwargs(
    model="openai/gpt-4o-mini",
    api_key="os.environ/OPENAI_API_KEY",
    prompt="Write a {tone} title for: {text}",
)
titles = GenericGenerationOperator(
    prompt_params,
    input_columns={"tone": "style", "text": "document"},
    output_column="title",
).run(pd.DataFrame({"style": ["concise"], "document": ["Quarterly results"]}))
```

`SummarizationOperator` defaults to `text`, `summary`, `summary_latency_s`, `summary_model`, and `summary_error`. `GenericGenerationOperator` maps each named prompt placeholder to a physical DataFrame column and derives the metadata column names from `output_column`. Prompt contracts are validated when the operator is constructed, before any provider request runs.

To define another one-request/one-text-result task, subclass `TextGenerationTask`, declare `required_inputs`, and implement `build_request()`. Then construct it from a `TextGenerationOperator` subclass with explicit logical-input-to-DataFrame-column mappings. This abstraction is intentionally text-only; use a separate operator family for embeddings, captioning, tools, streaming, or structured domain results.

Generation failures are collected per row using stable error codes: `empty_input`, `request_error`, `transport_error`, `unsupported_response`, `parse_error`, `empty_output`, and the RAG-specific `thinking_truncated`. Raw provider exceptions and credentials are not written to DataFrame outputs.

## Persisted graphs are trusted configuration

Graph loading imports operator classes and invokes their constructors. Load graph JSON only from trusted sources; do not expose graph payloads, callable references, or class names as model- or user-controlled agent tools.

Version 2 graph files preserve shared-node DAG identity and reject cycles. Constructor state must consist of supported JSON-native values, typed Pydantic models, paths, sets and tuples, or importable type/callable references. Runtime data such as DataFrames and opaque client objects is not persistable.

API keys are never written into graph JSON. Use an explicit environment reference in persisted configuration:

```python
QAGenerationOperator(
    model="openai/gpt-4o-mini",
    api_key="os.environ/OPENAI_API_KEY",
)
```

Serializing a graph containing a literal API key fails with a contextual error instead of guessing which provider credential should be used on a worker.


::: nemo_retriever.ingestor
    options:
      filters:
        - "!^pdf_split_config$"

::: nemo_retriever.graph.retriever

::: nemo_retriever.common.params
