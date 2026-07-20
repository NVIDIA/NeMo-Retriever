# Troubleshoot NeMo Retriever Library

Use this documentation to troubleshoot issues that arise when you use [NeMo Retriever Library](overview.md).

## Python API error triage { #python-api-error-triage }

NeMo Retriever Library does not assign product-specific numeric error codes to
each extraction method. The Python API surfaces Python exception types,
row-level failure records, and HTTP or gRPC statuses from the service or
upstream NIM. Treat the exception type, failing stage, upstream status, and
response detail together as the error identifier.

The current graph API enriches `GraphIngestionError` for explicitly configured
Page Elements, OCR, Table Structure, Nemotron Parse, and embedding endpoints.
When the upstream payload contains an HTTP status, the message includes the
stage, configured URL, and status. For example:

```text
Graph ingestion detected row-level errors from an explicitly configured
remote NIM endpoint. row 0, column ocr
[stage=OCR NIM url=https://example.invalid/v1/infer http=503], path error: ...
Troubleshooting: OCR NIM ... returned a 5xx server error ...
```

This enrichment is not a new error-code namespace. The `503` in this example
is the upstream HTTP status. Exception wording and upstream response bodies are
not stable API fields and should not be parsed programmatically.

In many row-level payloads, the HTTP status appears only inside
`error.message` (for example `HTTP 503 from …`), not as a separate
`status_code` field. Parse `exc.records` or inspect the failing DataFrame
column when troubleshooting.

### Coverage limits of `error_policy`

`error_policy="raise"` scans only remote NIM stages with an explicitly
configured invoke URL: Page Elements, OCR, Table Structure, Nemotron Parse,
and embedding. It does **not** automatically raise for:

- Local-only pipelines (`pdfium` without remote URLs), even when rows contain
  `metadata.error` or column-level error payloads.
- Caption or remote VLM stages. Missing credentials fail at actor setup;
  inference failures can abort the entire ingest.
- Audio or video ASR over gRPC/HTTP. Failures can drop individual rows and
  log warnings instead of raising `GraphIngestionError`.

For those paths, use `error_policy="collect"`, `return_failures=True`, or
inspect row columns and service logs directly.

### Error signals and first response

| Signal | Typical meaning | L1/L2 response |
| --- | --- | --- |
| `ValueError` or Pydantic validation error before execution | An unsupported run mode, parameter value, protocol, or parameter combination | Compare the call with the current [Python API reference](nemo-retriever-api-reference.md). Remove unknown parameters and reproduce with the smallest valid pipeline. |
| `ImportError`, `ModuleNotFoundError`, or a missing-dependency `RuntimeError` | The selected local extraction path requires a package or executable that is not installed | Install the documented package extra or system dependency. Confirm that the Python environment running the worker, not only the client shell, contains it. |
| `GraphIngestionError` with no HTTP status | The named remote stage returned a row-level error, its error payload omitted a status, or the endpoint was unreachable | Check DNS, routing, TLS, the endpoint URL, and the NIM readiness endpoint from the worker or service pod. Inspect `exc.records` after removing secrets and document content. |
| HTTP `401` or `403` from a NIM | Missing, expired, or unauthorized credentials | Verify `NVIDIA_API_KEY`, `NGC_API_KEY`, or the stage-specific credential in the environment that makes the request. Do not attach API keys to a support case. |
| HTTP `403` from the Retriever service | Authentication failure or a deployment policy that disallows the requested endpoint, stage, sink, or override | Read the response `detail`. Verify the service token and compare the requested pipeline with `/v1/ingest/pipeline-config`. |
| HTTP `404` or `410` while opening a service ingest job | The Python SDK and Retriever service can be on incompatible API versions | A current client raises `RetrieverServiceCompatibilityError`. Align the Python package, service image, and Helm chart versions. |
| Other HTTP `4xx` | The upstream service rejected the request | Check file type, rendered page or image size, model name, endpoint path, and request schema. For `413` or `422`, reduce the payload or image size and verify the endpoint's input limits. |
| HTTP `429` | The remote service is rate-limiting requests | Reduce concurrency or batch size and retry with backoff. Escalate only if throttling persists within the service quota. |
| HTTP `5xx`, including `503` | The upstream NIM is unavailable, overloaded, not ready, or failed during inference | Check readiness, pod restarts, GPU memory, server logs, and request volume. Retry a minimal input after the NIM is healthy. |
| Timeout, connection reset, DNS, TLS, or gRPC transport error | The client could not complete transport to the service or NIM | Test connectivity from the process or pod that runs the stage. Verify protocol, port, certificate trust, proxy, and network policy. Preserve the gRPC status and details when present. |
| A per-document entry in `ServiceIngestResult.failures` | Upload or pipeline processing failed after a service job was created | Correlate the document ID with the job ID and service logs. Other documents in the same result can still have succeeded. |
| Successful ingest with fewer rows than inputs (caption or ASR enabled) | Caption inference failed before row collection, or ASR dropped failed rows and logged warnings | Re-run with logging enabled. For caption, verify endpoint credentials and payload limits. For ASR, verify gRPC endpoint, `function_id`, and `NVIDIA_API_KEY`. |
| OOM, worker exit, or pod restart | Host or GPU resources were exhausted, or an orchestrator terminated the worker | Reduce batch size or concurrency, use smaller document groups, and inspect host, Ray, Kubernetes, and NIM resource telemetry. |

The service can retry some transient transport, `429`, and `5xx` failures.
Report the final status returned after retries, not an intermediate warning.

### Representative extraction paths

Use the failing stage—not only the top-level `method` value—to select the
troubleshooting path. A single document can pass through several stages.

| API path | Components that can fail | Representative signals |
| --- | --- | --- |
| `ExtractParams(method="pdfium")` | File loading, PDF splitting, PDFium parsing, page rendering; optionally Page Elements and Table Structure when enabled | Malformed or encrypted input, `pypdfium2` import failure, local Python exception, or remote-stage `GraphIngestionError` when an invoke URL is explicitly configured |
| `ExtractParams(method="pdfium_hybrid")` | PDFium plus Page Elements, OCR, and optionally Table Structure | The local PDF signals above, or a row-level/HTTP failure attributed to Page Elements, OCR, or Table Structure |
| `ExtractParams(method="ocr")` | Page rendering, Page Elements, and the local or remote OCR backend | Missing local model dependencies, invalid image payload, authentication/transport status, or OCR row-level failure |
| `ExtractParams(method="nemotron_parse")` | PDF rendering and local Nemotron Parse model or configured Nemotron Parse NIM | Missing `open_clip`, missing local model configuration, unsupported image input, or Nemotron Parse row-level/HTTP failure |
| `.caption(...)` | Local caption model or remote VLM endpoint | `ValueError` at setup when credentials or endpoint/protocol are invalid; remote inference failures can abort the whole ingest rather than populate a row error column |
| `.embed(...)` | Local embedding model or configured embedding NIM | Model/dependency error, input-size or schema rejection, authentication/transport status, or embedding row-level failure; `GraphIngestionError` when a remote embed URL is configured |
| Audio or video extraction | `ffmpeg`/`ffprobe`, media decoding, frame/chunk creation, and local or remote ASR | Missing executable, malformed media, codec failure, gRPC status, or credential error; ASR failures may omit rows and log warnings instead of raising, so verify logs when output is unexpectedly empty |

`pdfium` itself is primarily a local parser, so a Page Elements, Table
Structure, OCR, caption, or embedding HTTP status comes from an enabled
downstream stage rather than from PDFium.

### Collect diagnostics safely

Before escalating, collect:

1. The exact Python package version, service/container image tag, Helm chart
   version when applicable, and `run_mode`.
2. The exception class and complete sanitized message. For
   `GraphIngestionError`, include sanitized `exc.records`. For row-level
   failures, include the `stage`, `type`, and `message` fields from the
   affected column when present.
3. The extraction method and enabled stages, plus endpoint hostnames with
   credentials and signed query parameters removed.
4. The HTTP or gRPC status, response detail, job ID, document ID, trace ID,
   and timestamp when available.
5. Whether the endpoint readiness check succeeds from the worker or service
   pod.
6. A minimal non-confidential input that reproduces the issue, or input
   characteristics such as format, page count, dimensions, and size.
7. Relevant client, service, Ray, NIM, and Kubernetes logs around the same
   timestamp.

Never include API keys, bearer tokens, document contents, or unredacted signed
URLs in logs or support cases.

Escalate to NVIDIA L3 when the failure is reproducible on a supported,
version-aligned configuration after L1/L2 has verified input validity,
credentials, endpoint readiness, connectivity, and resource availability.
Escalate immediately for repeatable crashes, incorrect successful output, or a
`5xx` from a healthy NVIDIA-owned NIM with a minimal valid input. Keep
configuration, dependency, customer network, quota, and malformed-input issues
with L1/L2 unless the documented behavior is incorrect.

!!! note "Older NV-Ingest releases"

    Error text and result shapes differ by release. NV-Ingest `25.4.2`
    predates some current enriched diagnostics. Do not assume that a field
    shown in current NeMo Retriever Library output exists in `25.4.2`; include
    the exact old exception and logs when escalating.

## Can't process long, non-language text strings

NeMo Retriever Library is designed to process language and language-length strings.
If you submit a document that contains extremely long, or non-language text strings,
such as a DNA sequence, errors or unexpected results occur.

## Can't process malformed input files

When you run a job you might see errors similar to the following:

- Failed to process the message
- Failed to extract image
- File may be malformed
- Failed to format paragraph

These errors can occur when your input file is malformed.
Verify or fix the format of your input file, and try resubmitting your job.

## Audio or video extraction reports missing media dependencies { #audio-or-video-extraction-reports-missing-media-dependencies }

When you run audio or video extraction, you might see an error similar to one
of the following:

```text
Audio extraction requires media dependencies; missing: ffmpeg.
VideoFrameActor requires media dependencies; missing: ffprobe.
```

The `ffmpeg-python` wrapper and `nemo-retriever[multimedia]` do not install the
`ffmpeg` or `ffprobe` binaries the pipeline executes.

For air-gapped or locked-down clusters, refer to [Air-gapped and disconnected deployment](deployment-options.md#air-gapped-deployment).

**Connected environments:**

On Debian or Ubuntu hosts:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg
```

For the bundled service container at runtime:

```bash
docker run -e INSTALL_FFMPEG=true nemo-retriever-service
```

For Helm, when package-repo egress and the image security policy allow startup install:

```yaml
service:
  installFfmpeg: true
```

This path fails with `allowPrivilegeEscalation: false` or `readOnlyRootFilesystem: true`.

## Can't start new thread error

In rare cases, when you run a job you might an see an error similar to `can't start new thread`.
This error occurs when the maximum number of processes available to a single user is too low.
To resolve the issue, set or raise the maximum number of processes (`-u`) by using the [ulimit](https://ss64.com/bash/ulimit.html) command.
Before you change the `-u` setting, consider the following:

- Apply the `-u` setting directly to the user (or the environment of the pod or process) that runs your ingest service.
- For `-u` we recommend 10,000 as a baseline, but you might need to raise or lower it based on your actual usage and system configuration.

```bash
ulimit -u 10000
```



## Out-of-Memory (OOM) Error when Processing Large Datasets

When you process a very large dataset with thousands of documents, you might encounter an Out-of-Memory (OOM) error.
This happens because NeMo Retriever Library materializes extraction results in system memory (RAM) while the job runs.
If the total size of the results exceeds the available memory, the process fails.

To reduce memory pressure, try one or more of the following:

- Process documents in smaller batches instead of submitting the entire corpus in one job.
- Route outputs to a sink (for example, `.vdb_upload(...)`, `.webhook(...)`, or `.store(...)`) so results are written out instead of held in memory until the job finishes.
- In `run_mode="service"`, pass `return_results=False` to `.ingest(...)` when you do not need the full result payload returned to the client. For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).
- Increase available host or pod memory for the ingest workload.



## Embedding service fails to start with an unsupported batch size error

On certain hardware, for example RTX 6000,
the embedding service might fail to start and you might see an error similar to the following.

```bash
ValueError: Configured max_batch_size (30) is larger than the model''s supported max_batch_size (3).
```

If you are using hardware where the embedding NIM uses the ONNX model profile,
you must set `EMBEDDER_BATCH_SIZE=3` in your environment.
You can set the variable in your .env file or directly in your environment.



## ModuleNotFoundError: No module named open_clip when using nemotron_parse { #modulenotfounderror-no-module-named-open-clip-when-using-nemotron-parse }

When you run PDF extraction with `extract_method="nemotron_parse"`, you might see an error similar to the following:

```text
ModuleNotFoundError: No module named 'open_clip'
```

The Nemotron Parse NIM client requires the `open_clip` Python module, provided by `open-clip-torch`. That package is not part of the default `nemo-retriever` install or the `[local]` extra.

Install the dedicated PyPI extra before running Nemotron Parse extraction:

```bash
pip install "nemo-retriever[nemotron-parse]"
```

For local GPU inference with Nemotron Parse, combine extras:

```bash
pip install "nemo-retriever[local,nemotron-parse]"
```

Also refer to [What is NeMo Retriever Library?](overview.md) and [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md#software-requirements).

## Extract method nemotron-parse doesn't support image files

Currently, extraction with Nemotron parse doesn't support image files, only scanned PDFs.
To work around this issue, convert image files to PDFs before you use `extract_method="nemotron_parse"`.

## Hosted Page Elements NIM image size limits { #hosted-page-elements-nim-image-size-limits }

[NVIDIA-hosted Page Elements NIM](https://build.nvidia.com/nvidia/nemotron-page-elements-v3) endpoints on `ai.api.nvidia.com` (and the matching build.nvidia.com model experience) enforce a strict limit on **inline** image payloads. The same limit applies to hosted **Table Structure** and **Graphic Elements** object-detection NIMs because they share the same `/v1/infer` request shape.

The following table summarizes inline payload limits by deployment:

| Deployment | Inline base64 limit | Oversized images |
|------------|---------------------|------------------|
| Hosted (`build.nvidia.com`, `ai.api.nvidia.com`) | About **180,000 characters** on the base64 portion of the data URL (roughly 180 KB; build.nvidia.com validates `len(image_b64) < 180_000`) | Upload with the [NVCF Asset API](https://docs.api.nvidia.com/cloud-functions/reference/createasset), then reference `data:image/<format>;asset_id,<asset_id>` in the `url` field |
| Self-hosted NIM container | Higher; the NeMo Retriever client downscales HTTP payloads above **512,000 characters** before calling the NIM | Resize or re-encode the source image, or rely on the client downscaling |

The [Object Detection NIM API reference](https://docs.nvidia.com/nim/ingestion/object-detection/latest/api-reference.html) states only that “very large images may cause processing issues.” For hosted integrations, treat **180,000 characters** as the inline cap unless NVIDIA publishes a different limit for your endpoint.

### NeMo Retriever Library pipeline users

When you route extraction to hosted Page Elements NIM URLs (for example `page_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"`), the library:

- Renders PDF pages with default `render_mode="fit_to_model"` (targets about 1024 px on the long edge instead of full raster DPI).
- Downscales base64 page images before remote object-detection NIM HTTP calls when payloads exceed the client limit (512,000 characters for Page Elements and Table Structure).

!!! important

    The library downscales payloads to **512,000** characters before HTTP calls to object-detection NIMs. Hosted endpoints still reject inline base64 above **180,000** characters. Treat the lower hosted cap as the effective limit when `page_elements_invoke_url` points at `ai.api.nvidia.com`.

If you still receive **422** responses mentioning invalid image URLs on hosted endpoints, lower `dpi` in `ExtractParams`, keep `render_mode="fit_to_model"`, or preprocess very large standalone image inputs before ingest. For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

### Direct Page Elements NIM API calls (build.nvidia.com or custom clients)

When you call Page Elements NIM **directly** (build playground, curl, or a custom integration—not through the NeMo Retriever pipeline), use inline base64 only when `len(base64_image) < 180_000`. For larger PNG or JPEG inputs, upload once with the NVCF Asset API and pass an asset reference in the inference payload.

1. **Create an asset** — `POST https://api.nvcf.nvidia.com/v2/nvcf/assets` with `Authorization: Bearer $NVIDIA_API_KEY`, plus JSON `contentType` (for example `image/png`) and `description`.
2. **Upload the file** — `PUT` the image bytes to the `uploadUrl` from step 1. Set `Content-Type` to match `contentType`, and set `x-amz-meta-nvcf-asset-description` to the same description string.
3. **Infer** — `POST` to your Page Elements invoke URL with `"url": "data:image/png;asset_id,<assetId>"` inside each `input[]` item (same `type: image_url` schema as inline base64).

For the full asset workflow (including reuse across requests), refer to [NVCF assets](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/assets.html) in the Cloud Functions user guide and the [Create Asset](https://docs.api.nvidia.com/cloud-functions/reference/createasset) API reference. Hosted calls require the same [`NVIDIA_API_KEY`](api-keys.md#nvidia-api-key) you use for other build.nvidia.com NIM endpoints.

For the request schema, refer to the [Object Detection NIM API reference](https://docs.nvidia.com/nim/ingestion/object-detection/latest/api-reference.html).

??? example "Create an NVCF asset, upload a PNG, and call Page Elements"

    ```python
    import os
    import requests

    API_KEY = os.environ["NVIDIA_API_KEY"]
    PAGE_ELEMENTS_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"
    IMAGE_PATH = "large_page.png"

    create = requests.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={"contentType": "image/png", "description": "page-elements-large-input"},
        timeout=60,
    )
    create.raise_for_status()
    asset = create.json()

    with open(IMAGE_PATH, "rb") as image_file:
        upload = requests.put(
            asset["uploadUrl"],
            headers={
                "Content-Type": "image/png",
                "x-amz-meta-nvcf-asset-description": "page-elements-large-input",
            },
            data=image_file,
            timeout=120,
        )
    upload.raise_for_status()

    payload = {
        "input": [{
            "type": "image_url",
            "url": f"data:image/png;asset_id,{asset['assetId']}",
        }]
    }
    response = requests.post(
        PAGE_ELEMENTS_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    ```

Supported inline formats remain **PNG** and **JPEG**, encoded as `data:image/<format>;base64,<data>` or `data:image/<format>;asset_id,<uuid>`. OpenAPI specs for Page Elements v2 and v3 are linked from the [Object Detection NIM API reference](https://docs.nvidia.com/nim/ingestion/object-detection/latest/api-reference.html#openapi-reference-for-page-elements).

## Too many open files error

In rare cases, when you run a job you might an see an error similar to `too many open files` or `max open file descriptor`.
This error occurs when the open file descriptor limit for your service user account is too low.
To resolve the issue, set or raise the maximum number of open file descriptors (`-n`) by using the [ulimit](https://ss64.com/bash/ulimit.html) command.
Before you change the `-n` setting, consider the following:

- Apply the `-n` setting directly to the user (or the environment of the pod or process) that runs your ingest service.
- For `-n` we recommend 10,000 as a baseline, but you might need to raise or lower it based on your actual usage and system configuration.

```bash
ulimit -n 10000
```



## Triton server INFO messages incorrectly logged as errors

Sometimes messages are incorrectly logged as errors, when they are information.
When this happens, you can ignore the errors, and treat the messages as information.
For example, you might see log messages that look similar to the following.

```bash
ERROR 2025-04-24 22:49:44.266 nimutils.py:68] tritonserver: /usr/local/lib/libcurl.so.4: ...
ERROR 2025-04-24 22:49:44.268 nimutils.py:68] I0424 22:49:44.265292 98 cache_manager.cc:480] "Create CacheManager with cache_dir: '/opt/tritonserver/caches'"
ERROR 2025-04-24 22:49:44.431 nimutils.py:68] I0424 22:49:44.431796 98 pinned_memory_manager.cc:277] "Pinned memory pool is created at '0x7f8e4a000000' with size 268435456"
ERROR 2025-04-24 22:49:44.432 nimutils.py:68] I0424 22:49:44.432036 98 cuda_memory_manager.cc:107] "CUDA memory pool is created on device 0 with size 67108864"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] I0424 22:49:44.433448 98 model_config_utils.cc:753] "Server side auto-completed config: "
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] name: "yolox"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] platform: "tensorrt_plan"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] max_batch_size: 32
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] input {
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] name: "input"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] data_type: TYPE_FP32
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] dims: 3
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] dims: 1024
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] dims: 1024
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] }
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] output {
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] name: "output"
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] data_type: TYPE_FP32
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] dims: 21504
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] dims: 9
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] }
...
```



## Related Topics

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Deployment options](deployment-options.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
- [About getting started](getting-started-about.md) (prerequisites and deployment)
