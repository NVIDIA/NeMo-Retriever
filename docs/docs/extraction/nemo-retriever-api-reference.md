# NeMo Retriever API Reference

## Error and failure contract { #error-and-failure-contract }

The Python API does not define a separate set of numeric NeMo Retriever
extraction error codes. Depending on the run mode and failing stage, callers
observe one or more of the following:

- Python configuration or dependency exceptions, such as `ValueError`,
  `ImportError`, or `RuntimeError`.
- `GraphIngestionError` for row-level failures from explicitly configured
  remote NIM stages in `run_mode="inprocess"` or `"batch"` when
  `error_policy="raise"` (the default).
- HTTP status codes or gRPC errors returned by a remote NIM or by the
  Retriever service. These are transport or upstream-service statuses, not
  NeMo Retriever-specific error codes.
- Per-document failures in `ServiceIngestResult.failures` when
  `run_mode="service"`.

The generated API signatures and parameter models below are the API contract.
Exception text and upstream response bodies can change between releases; do
not parse them as stable codes. The stable text-generation codes documented in
[One-shot text generation](#one-shot-text-generation) apply to generation
operator output columns, not to document extraction.

### Choose raise or collect behavior

For graph run modes, `error_policy="raise"` raises `GraphIngestionError` when
an explicitly configured remote NIM stage reports a row-level error. The
exception retains the underlying records in `exc.records`. When available, its
message identifies the stage, invoke URL, and HTTP status in a form similar to
`[stage=OCR NIM url=https://... http=503]`, followed by a troubleshooting hint.

Use `error_policy="collect"` when partial results are useful and your
application inspects the error fields in every returned row. Alternatively,
pass `return_failures=True` to `.ingest()` to receive a `(result, failures)`
tuple. When no remote invoke URL is configured, `return_failures=True` scans
all output columns for row-level error fields so local failures are still
visible. In service mode, failures are also available from
`ServiceIngestResult.failures`.

### What `error_policy="raise"` covers

The strict policy applies only to stages where you explicitly configure a
remote NIM invoke URL. It does not raise for local-only PDFium parsing,
caption, audio/video, or ASR failures, even when those stages populate
row-level error fields.

| Configured invoke URL | DataFrame column scanned | Stage label in messages |
| --- | --- | --- |
| `page_elements_invoke_url` | `output_column` (default `page_elements_v3`) | Page Elements NIM |
| `ocr_invoke_url` | `ocr` | OCR NIM |
| `table_structure_invoke_url` | `table_structure_ocr_v1` | Table Structure NIM |
| `nemotron_parse_invoke_url` or `invoke_url` | `nemotron_parse_v1_2` | Nemotron Parse NIM |
| `embed_invoke_url` or `embedding_endpoint` | `output_column` (default `text_embeddings_1b_v2`) | Embedding NIM |

Caption and ASR use remote endpoints but are outside this raise path today.
Remote caption failures can abort the whole ingest instead of returning a
partial DataFrame. ASR failures can omit affected rows while logging a
warning, which can look like an empty transcript unless you inspect logs.

### Row-level error payloads

Most extraction stages write errors into the result row instead of raising
immediately. The common nested shape is:

```json
{
  "error": {
    "stage": "ocr_page_elements",
    "type": "HTTPError",
    "message": "HTTP 503 from https://example/v1/infer: ...",
    "traceback": "..."
  }
}
```

The `stage` string is a semi-stable operator identifier (for example
`remote_inference`, `nemotron_parse_pages`, or `split_pdf`). It is not a
product-wide error-code enum. HTTP status codes usually appear inside
`message` text rather than as a separate `status_code` field; when a
structured status is present, `GraphIngestionError` can include it in the
rendered exception.

```python
import os

from nemo_retriever import GraphIngestionError, create_ingestor
from nemo_retriever.common.params import ExtractParams

pipeline = (
    create_ingestor(run_mode="inprocess", error_policy="raise")
    .files(["document.pdf"])
    .extract(
        ExtractParams(
            method="ocr",
            ocr_invoke_url=os.environ["OCR_INVOKE_URL"],
        )
    )
)

try:
    result = pipeline.ingest()
except GraphIngestionError as exc:
    # Records can contain source paths, endpoint details, and upstream
    # response text. Extract only the known-safe diagnostic fields before
    # logging or sending to your support workflow.
    for record in exc.records:
        error_info = record.get("error", {}) if isinstance(record, dict) else {}
        print({
            "stage": error_info.get("stage"),
            "type": error_info.get("type"),
            "message": error_info.get("message"),
        })
```

For a support-oriented mapping of extraction paths, error signals, corrective
actions, and escalation criteria, refer to
[Python API error triage](troubleshoot.md#python-api-error-triage).

!!! note "Version-specific behavior"

    This reference describes the current NeMo Retriever Library. Older
    NV-Ingest releases, including `25.4.2`, can use different exception text
    and result shapes and might not include enriched `GraphIngestionError`
    diagnostics. When troubleshooting an older deployment, use the package and
    container versions from that deployment and include them in the support
    case.

## PDF pre-splitting for parallel ingest { #pdf-pre-splitting-for-parallel-ingest }

Large PDFs are split into page batches before Ray processing so extraction can run in parallel. This happens on the default ingest path; you do not need extra configuration for typical workloads.

To tune splitter throughput from the CLI, use `--pdf-split-batch-size` (Ray actor batch size for the splitter stage). Refer to [Text chunking and PDF page batches](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli#text-chunking-and-pdf-page-batches) in the CLI reference.

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
