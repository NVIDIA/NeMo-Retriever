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
| `INGEST_LOG_LEVEL`               | - `DEBUG` <br/> - `INFO` <br/> - `WARNING` <br/> - `ERROR` <br/> - `CRITICAL` <br/> | The log level for the ingest service, which controls the verbosity of the logging output. |
| `NVIDIA_API_KEY`                    | `nvapi-*************` <br/>                              | An authorized build.nvidia.com API key, used to interact with NVIDIA-hosted NIMs. Create through build.nvidia.com or through [NGC](https://org.ngc.nvidia.com/setup/api-keys). |
| `NGC_API_KEY`                | —                                                          | The key that NIM microservices in the cluster use to access NGC resources. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`    | `http://otel-collector:4317` <br/>                       | The endpoint for the OpenTelemetry exporter, used for sending telemetry data. |


## Related Topics { #related-topics }

- [Configure Ray Logging](ray-logging.md)
- [Authentication and API keys](api-keys.md)
- [Python API guide](nemo-retriever-api-reference.md)
