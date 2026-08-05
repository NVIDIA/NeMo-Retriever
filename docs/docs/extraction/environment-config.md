# Environment Variables for NeMo Retriever Library

The following are the environment variables that you can use to configure [NeMo Retriever Library](overview.md).
You can specify these in a .env file in your working directory or directly as shell environment variables.


## General Environment Variables

| Name                             | Example                        | Description                                                           |
|----------------------------------|--------------------------------|-----------------------------------------------------------------------|
| `HF_ACCESS_TOKEN`                | -                                                         | A token for Hugging Face Hub downloads when your runtime needs one. The default chunking tokenizer is public; refer to [Token-based splitting](concepts.md#token-based-splitting) for container caching and offline behavior. |
| `INGEST_LOG_LEVEL`               | - `DEBUG` <br/> - `INFO` <br/> - `WARNING` <br/> - `ERROR` <br/> - `CRITICAL` <br/> | The log level for the ingest service, which controls the verbosity of the logging output. |
| `NVIDIA_API_KEY`                    | `nvapi-*************` <br/>                              | An authorized build.nvidia.com API key, used to interact with NVIDIA-hosted NIMs. Create through build.nvidia.com or through [NGC](https://org.ngc.nvidia.com/setup/api-keys). |
| `NGC_API_KEY`                | —                                                          | The key that NIM microservices in the cluster use to access NGC resources. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`    | `http://otel-collector:4317` <br/>                       | The endpoint for the OpenTelemetry exporter, used for sending telemetry data. |


## Related Topics

- [Configure Ray Logging](https://docs.nvidia.com/nemo/retriever/latest/extraction/ray-logging/)
