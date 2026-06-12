# Troubleshoot NeMo Retriever Library

Use this documentation to troubleshoot issues that arise when you use [NeMo Retriever Library](overview.md).

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
- [NeMo Retriever Library — prerequisites / deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) (supported **Helm** charts)
