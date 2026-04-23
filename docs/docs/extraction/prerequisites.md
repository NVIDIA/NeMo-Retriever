# Prerequisites for NeMo Retriever Library

Before you begin using [NeMo Retriever Library](overview.md), ensure the following software and hardware prerequisites are met.

## Software Requirements

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= `535`, CUDA >= `12.2`)
- [Python](https://www.python.org/downloads/) `3.12` — required to install and run the NeMo Retriever Library Python API, CLI, and related packages from PyPI (for example `pip` or `uv`). Older Python versions will fail dependency resolution without a clear error.
- [UV Python package and environment manager](https://docs.astral.sh/uv/getting-started/installation/) (optional; recommended for creating isolated environments)


!!! note

    When you use UV, create the environment with Python 3.12 — for example, `uv venv --python 3.12`. This matches the `requires-python` metadata in the library packages.


## Hardware Requirements

The full ingestion pipeline is designed to consume significant CPU and memory resources to achieve maximal parallelism. 
Resource usage scales up to the limits of your deployed system.

For additional hardware details, refer to [Support Matrix](support-matrix.md).


### Recommended Production Deployment Specifications

- **System Memory**: At least 256 GB RAM
- **CPU Cores**: At least 32 CPU cores
- **GPU**: NVIDIA GPU with at least 24 GB VRAM (e.g., A100, H100, L40S, or equivalent)

!!! note

    Using less powerful systems or lower resource limits is still viable, but performance will suffer.


### Resource Consumption Notes

- The pipeline performs runtime allocation of parallel resources based on system configuration
- Memory usage can reach up to the full system capacity for large document processing
- CPU utilization scales with the number of concurrent processing tasks
- GPU is required for inference using HuggingFace models or NIMs
- GPU is NOT required for build.nvidia.com hosted inference


### Scaling Considerations

For production deployments processing large volumes of documents, consider:
- Higher memory configurations for processing large PDF files or image collections
- Additional CPU cores for improved parallel processing
- Multiple GPUs for distributed processing workloads


### Environment Requirements

Ensure your deployment environment meets these specifications before running the full pipeline. Resource-constrained environments may experience performance degradation.



## Related Topics

- [Support Matrix](support-matrix.md)
- [Troubleshooting](troubleshoot.md)
- [Deploy As a Python Library](quickstart-library-mode.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)
