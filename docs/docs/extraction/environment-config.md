# Environment Variables for NeMo Retriever Library

The following are the environment variables that you can use to configure [NeMo Retriever Library](overview.md).
Set them in the process environment before you run the public SDK or the `retriever` CLI. The SDK and CLI do not load a working-directory `.env` file automatically.

If you keep values in a `.env` file, source that file into the current shell first:

```bash
set -a
source .env
set +a
```

On Windows PowerShell, set the variables in the session, for example `$env:NVIDIA_API_KEY = "nvapi-..."`. Refer to [Authentication and API keys](api-keys.md).


## General Environment Variables { #general-environment-variables }

| Name                             | Example                        | Description                                                           |
|----------------------------------|--------------------------------|-----------------------------------------------------------------------|
| `HF_ACCESS_TOKEN`                | -                                                         | A token for Hugging Face Hub downloads when your runtime needs one. The default chunking tokenizer is public; refer to [Token-based splitting](concepts.md#token-based-splitting) for container caching and offline behavior. |
| `NVIDIA_API_KEY`                    | `nvapi-*************` <br/>                              | An authorized build.nvidia.com API key, used to interact with NVIDIA-hosted NIMs. Create through build.nvidia.com or through [NGC](https://org.ngc.nvidia.com/setup/api-keys). |
| `NGC_API_KEY`                | —                                                          | The key that NIM microservices in the cluster use to access NGC resources. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`    | `http://otel-collector:4317` <br/>                       | The endpoint for the OpenTelemetry exporter, used for sending telemetry data. |
| `OTEL_METRICS_EXPORTER` | `otlp` | The retriever service metrics exporter. Set to `none` to disable OpenTelemetry metric export while retaining other supported telemetry. |
| `OTEL_METRIC_EXPORT_INTERVAL` | `5000` | The metric export interval in milliseconds for Helm deployments. Set a larger value to reduce export frequency. |


## Retriever Service Log Level { #retriever-service-log-level }

The current Retriever service reads log verbosity from `logging.level`.
The default is `INFO`.
`INGEST_LOG_LEVEL` does not control `retriever service start`, the current service container, or Helm.

The following table lists the supported controls.

| Interface | Control | Notes |
|-----------|---------|-------|
| CLI | `retriever service start --log-level LEVEL` | Overrides YAML `logging.level`. |
| YAML | `logging.level` in `retriever-service.yaml` | Bundled default is `INFO`. |
| Helm | `serviceConfig.logging.level` | Rendered into the service ConfigMap. |

Typical values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

The following CLI example starts the service at `CRITICAL`:

```bash
retriever service start --log-level CRITICAL
```

The following YAML sets the same level in `retriever-service.yaml`:

```yaml
logging:
  level: CRITICAL
```

The following Helm values set the same level for a chart deployment:

```yaml
serviceConfig:
  logging:
    level: CRITICAL
```

Setting `INGEST_LOG_LEVEL` in the process environment or in Helm `service.env` does not change `logging.level`.
That variable remains in the `examples/launch_libmode_*.py` scripts, the legacy `docker/scripts/entrypoint.sh` service entrypoint, and `.devcontainer/devcontainer.json`.
The current service image and Helm deployment do not use those paths.

For Ray worker logging, refer to [Configure Ray Logging](ray-logging.md).


## Related Topics { #related-topics }

- [Configure Ray Logging](ray-logging.md)
- [Authentication and API keys](api-keys.md)
- [Python API guide](nemo-retriever-api-reference.md)
