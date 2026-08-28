# Configure Ray Logging

[NeMo Retriever Library](overview.md) uses [Ray](https://docs.ray.io/en/latest/index.html) for batch ingest.
You can set Ray environment variables for fine-grained control over [Ray's logging behavior](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html).
To control whether worker logs appear in the driver process, use `--ray-log-to-driver` or `ray_log_to_driver`.

!!! important

    You must set Ray environment variables before you initialize the pipeline.
    Restart the pipeline if you change variable values.



## Quick Start - Control Worker Logs on the Driver { #quick-start-control-worker-logs-on-the-driver }

`retriever ingest batch` exposes `--ray-log-to-driver` and `--no-ray-log-to-driver`.
The default is to forward Ray worker logs to the driver.
This option is batch-only.
`retriever ingest` and `retriever ingest local` reject it.

The following command keeps worker logs in worker files instead of the driver process.
Replace `/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs --no-ray-log-to-driver
```

The following Python example sets the same behavior on `create_ingestor`.
The default value of `ray_log_to_driver` is `True`.
Replace `/path/to/your/pdfs/*.pdf` with a glob of PDF files that you supply.

```python
from nemo_retriever import create_ingestor

results = (
    create_ingestor(run_mode="batch", ray_log_to_driver=False)
    .files("/path/to/your/pdfs/*.pdf")
    .extract()
    .ingest()
)
```

Batch ingest passes this value to `ray.init(log_to_driver=...)`.
Use the CLI flag or the Python argument, not `RAY_LOG_TO_DRIVER`, for that path.



## Configuration Reference { #configuration-reference }

The following library controls apply to batch ingest:

| Control | Interface | Description | Default |
|---------|-----------|-------------|----------|
| `--ray-log-to-driver` / `--no-ray-log-to-driver` | `retriever ingest batch` | Forward Ray worker logs to the driver, or keep them in worker log files. | Enabled (`True`) |
| `ray_log_to_driver` | `create_ingestor` / `GraphIngestor` | Same control in Python. | `True` |

The following Ray environment variables control Ray's own logging:

| Variable                          | Type                             | Description | Valid Values | Default |
|-----------------------------------|----------------------------------|-------------|--------------|---------|
| `RAY_DEDUP_LOGS`                  | Log flow control                 | Specify whether to log multiple instances of repeated events or to combine into a single entry. `1` to combine repeated messages (for example, `[repeated 5x]`). | `0`, `1` | `1` |
| `RAY_DISABLE_IMPORT_WARNING`      | Ray internal logging             | `1` to suppress the `Ray X.Y.Z started` message and other warnings during initialization. | `0`, `1` | `0` |
| `RAY_LOG_TO_DRIVER`               | Log flow control                 | Ray environment variable for worker-to-driver log forwarding. For NeMo Retriever Library batch ingest, use `--ray-log-to-driver` instead. The library passes `log_to_driver` to `ray.init()`, so this variable does not control that path. | `true`, `false` | `true` |
| `RAY_LOGGING_ADDITIONAL_ATTRS`    | Core logging control             | Add Python logger fields like function names and line numbers to each log entry. | Comma-separated list | (empty) |
| `RAY_LOGGING_ENCODING`            | Core logging control             | Specify the format for log messages. | `TEXT`, `JSON` | `TEXT` |
| `RAY_LOGGING_LEVEL`               | Core logging control             | Specify what events to log. `DEBUG` to log all Ray internals. `WARNING` to log only significant events. | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `RAY_LOGGING_ROTATE_BACKUP_COUNT` | File rotation                    | Specify the number of old log files retained. Total storage = (count + 1) × file size. | Integer | `19` |
| `RAY_LOGGING_ROTATE_BYTES`        | File rotation                    | Specify the log file size before Ray creates a new log file. Use this to prevent unbounded disk usage. | Bytes | `1073741824` (1 GB) |
| `RAY_USAGE_STATS_ENABLED`         | Ray internal logging             | `1` to enable telemetry collection and related log messages. `0` to disable. | `0`, `1` | `1` |

Set these variables before you initialize the pipeline. NeMo Retriever Library does not expand or validate them.



## Configuration Examples { #configuration-examples }

### Reduce Log Volume { #reduce-log-volume }

By default, Ray generates significant logging output.
The following example configures Ray to reduce log volume.

```bash
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_LOGGING_LEVEL=WARNING
```

To keep worker logs out of the driver process, pass `--no-ray-log-to-driver` or `ray_log_to_driver=False`.
Refer to [Quick Start - Control Worker Logs on the Driver](#quick-start-control-worker-logs-on-the-driver).


### Minimal Logging { #minimal-logging }

The following example logs only errors and suppresses Ray initialization warnings.

```bash
export RAY_LOGGING_LEVEL=ERROR
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=1
```

Use this together with `--no-ray-log-to-driver` on `retriever ingest batch` when you want worker logs isolated from the driver.


### Structured Logging for Analysis { #structured-logging-for-analysis }

The following example results in machine-parseable JSON with metadata for log aggregation systems.

```bash
export RAY_LOGGING_ENCODING=JSON
export RAY_LOGGING_ADDITIONAL_ATTRS=name,funcName,lineno,thread,process
```


### Set Custom Storage Limits { #set-custom-storage-limits }

The following example automatically cleans up files when logs exceed 5 GB.
The oldest files are removed first.

```bash
# 5 GB total log storage (500 MB × 10 files)
export RAY_LOGGING_ROTATE_BYTES=524288000
export RAY_LOGGING_ROTATE_BACKUP_COUNT=9
```



## Log Output Examples { #log-output-examples }

### Default INFO Level

```
2024-01-15 10:30:15,123 INFO worker.py:1234 -- Task task_id=abc123 started
2024-01-15 10:30:15,124 INFO worker.py:1235 -- Processing batch size=100
2024-01-15 10:30:15,125 INFO worker.py:1236 -- Task task_id=abc123 completed
```

### WARNING Level

```
2024-01-15 10:30:20,456 WARNING worker.py:1240 -- Task retry attempt 2/3
2024-01-15 10:30:25,789 ERROR worker.py:1245 -- Task failed: Connection timeout
```

### JSON Encoding

```json
{
    "asctime": "2024-01-15 10:30:15,123",
    "levelname": "INFO",
    "filename": "worker.py",
    "lineno": 1234,
    "message": "Task started",
    "name": "ray.worker",
    "funcName": "execute_task",
    "job_id": "01000000",
    "worker_id": "abc123",
    "task_id": "def456"
}
```


## Related Topics { #related-topics }

- [Environment Variables](environment-config.md)
- [Python API guide](nemo-retriever-api-reference.md)
- [CLI ingest options](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md)
