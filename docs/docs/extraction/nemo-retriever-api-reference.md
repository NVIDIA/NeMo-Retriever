# NeMo Retriever API Reference

## Service-mode async ingest jobs

Use `create_ingestor(run_mode="service")` when you want the Python client to submit
documents to a running NeMo Retriever service instead of executing the pipeline in
the local process or Ray cluster. Service mode exposes three ingest surfaces:

- `ingest()` blocks until all submitted documents finish and returns a
  `ServiceIngestResult` with per-document completion events, `job_id`,
  `document_ids`, `failures`, `job_status`, `elapsed_s`, and, by default, a
  combined `dataframe` of completed results.
- `ingest_stream()` is a synchronous generator for callers that want progress
  events without managing an event loop.
- `aingest_stream()` is an asynchronous generator for applications that already
  run inside an event loop.

### Synchronous streaming example

```python
from nemo_retriever import create_ingestor


ingestor = (
    create_ingestor(
        run_mode="service",
        base_url="http://localhost:7670",
        documents=["docs/alpha.pdf", "docs/beta.pdf"],
    )
    .extract()
    .embed()
)

for event in ingestor.ingest_stream():
    match event["event"]:
        case "job_created":
            print(
                f"created job {event['job_id']} "
                f"for {event['expected_documents']} documents"
            )
        case "upload_complete":
            print(f"uploaded {event['filename']} as {event['document_id']}")
        case "upload_failed":
            print(f"could not upload {event['filename']}: {event['error']}")
        case "document_complete":
            if event["status"] == "completed":
                print(f"{event['document_id']} completed with {event['result_rows']} rows")
            else:
                print(f"{event['document_id']} failed: {event.get('error')}")
        case "job_progress":
            print(f"{event['completed']} completed, {event['failed']} failed")
        case "job_finalized" | "job_partial" | "job_failed":
            print(f"job {event['job_id']} finished with status {event['status']}")
```

Use this form from scripts, notebooks, CLI commands, or worker processes that are
otherwise synchronous but still need live job and document progress.

### Async streaming example

```python
import asyncio

from nemo_retriever import create_ingestor


async def main() -> None:
    ingestor = (
        create_ingestor(
            run_mode="service",
            base_url="http://localhost:7670",
            documents=["docs/alpha.pdf", "docs/beta.pdf"],
        )
        .extract()
        .embed()
    )

    async for event in ingestor.aingest_stream():
        if event["event"] == "job_created":
            print(f"created job {event['job_id']}")
        elif event["event"] == "upload_complete":
            print(f"uploaded {event['filename']}")
        elif event["event"] == "upload_failed":
            print(f"could not upload {event['filename']}: {event['error']}")
        elif event["event"] == "document_complete":
            print(f"{event['document_id']}: {event['status']}")
        elif event["event"] in {"job_finalized", "job_partial", "job_failed"}:
            print(f"job {event['job_id']}: {event['status']}")


asyncio.run(main())
```

Use this form from async web services, task runners, or notebooks that need to
keep other async work moving while ingestion is in flight.

### Event shapes

The streaming APIs yield dictionaries. Check the `event` key first, then read the
fields that apply to that event type:

| Event | Meaning | Key fields |
| --- | --- | --- |
| `job_created` | The service created one aggregate job for the submitted document set. | `job_id`, `expected_documents` |
| `upload_complete` | One local file uploaded and was assigned a service document ID. | `job_id`, `filename`, `document_id` |
| `document_complete` | One document reached a terminal document state. | `job_id`, `document_id`, `status`, `result_rows`, `elapsed_s`, `error` |
| `upload_failed` | One local file could not be uploaded. | `job_id`, `filename`, `error` |
| `job_started` | At least one document in the job started processing. | `job_id`, `status`, `expected_documents`, `counts`, `completed`, `failed`, `remaining`, `progress_pct`, `started_at` |
| `job_progress` | The job reached a progress reporting milestone. | `job_id`, `status`, `expected_documents`, `counts`, `completed`, `failed`, `remaining`, `progress_pct`, `elapsed_s` |
| `job_finalized` | All documents completed successfully. | `job_id`, `status`, `expected_documents`, `counts`, `completed`, `failed`, `remaining`, `progress_pct`, `elapsed_s`, `finalized_at` |
| `job_partial` | Some documents completed and some failed. | `job_id`, `status`, `expected_documents`, `counts`, `completed`, `failed`, `remaining`, `progress_pct`, `elapsed_s`, `finalized_at` |
| `job_failed` | Every document in the job failed. | `job_id`, `status`, `expected_documents`, `counts`, `completed`, `failed`, `remaining`, `progress_pct`, `elapsed_s`, `finalized_at` |

`document_complete` uses `status="completed"` or `status="failed"`. Job terminal
events use aggregate statuses: `job_finalized` reports `status="completed"`,
`job_partial` reports `status="partial_success"`, and `job_failed` reports
`status="failed"`.

Use `ingest(return_results=False)` when you only need the final job metadata and
document IDs. The default `ingest()` behavior fetches result rows for each
completed document so it can populate `result.dataframe`; streaming callers can
avoid that materialization and handle each event as it arrives.

## PDF pre-splitting for parallel ingest

Large PDFs are split into page batches before Ray processing so extraction can run in parallel. This happens on the default ingest path; you do not need extra configuration for typical workloads.

To tune splitter throughput from the CLI, use `--pdf-split-batch-size` (Ray actor batch size for the splitter stage). See [Text chunking and PDF page batches](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli#text-chunking-and-pdf-page-batches) in the CLI reference.

**Python client (`pdf_split_config`):** Only `create_ingestor(run_mode="service")` implements `.pdf_split_config(pages_per_chunk=...)`, which records page-chunking settings in the request pipeline spec for the remote gateway. Local graph ingest (`run_mode="inprocess"` or `"batch"`) raises `NotImplementedError` if you call this method; PDFs are split automatically on the default ingest path without client-side configuration.

::: nemo_retriever.ingestor
    options:
      filters:
        - "!^pdf_split_config$"

::: nemo_retriever.retriever

::: nemo_retriever.params
