# Deploy Without Containers (Library Mode) for NeMo Retriever Library

Use the [Quick Start for NeMo Retriever Library](https://github.com/NVIDIA/NeMo-Retriever/blob/26.03/nemo_retriever/README.md) to set up and run the NeMo Retriever Library locally, so you can build a GPU‑accelerated, multimodal RAG ingestion pipeline that parses PDFs, HTML, text, audio, and video into LanceDB vector embeddings, integrates with Nemotron RAG models (locally or via NIM endpoints), which includes Ray‑based scaling with built‑in recall evaluation. Python 3.12 or later is required (refer to [Prerequisites](prerequisites.md)).

## `run_pipeline`

The primary Python entry point for launching the Ray-based ingestion pipeline in library mode is `run_pipeline` in `nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners`.

```python
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
```

### Parameters

The following table matches the function signature in source (defaults and optionality). **None of these parameters are required** in the sense of having no default; omit them to use the defaults shown.

| Parameter | Required | Type (default) | Description |
|-----------|----------|----------------|-------------|
| `pipeline_config` | No | `Optional[PipelineConfigSchema]` (`None`) | Validated pipeline configuration. If `None` and `libmode=True`, the default library-mode pipeline is loaded automatically. If `None` and `libmode=False`, a `ValueError` is raised—you must pass a configuration. |
| `block` | No | `bool` (`True`) | If `True`, the call blocks until the pipeline finishes. If `False`, returns immediately with a handle object (refer to [Return type](#return-type)). |
| `disable_dynamic_scaling` | No | `Optional[bool]` (`None`) | If set, overrides the same field from the pipeline configuration. |
| `dynamic_memory_threshold` | No | `Optional[float]` (`None`) | If set, overrides the same field from the pipeline configuration. |
| `run_in_subprocess` | No | `bool` (`False`) | If `True`, runs the pipeline in a separate Python subprocess (`multiprocessing.Process`). If `False`, runs in the current process. |
| `stdout` | No | `Optional[TextIO]` (`None`) | When using a subprocess, optional stream for child stdout; if `None`, stdout is discarded. |
| `stderr` | No | `Optional[TextIO]` (`None`) | When using a subprocess, optional stream for child stderr; if `None`, stderr is discarded. |
| `libmode` | No | `bool` (`True`) | If `True` and `pipeline_config` is `None`, loads the default library-mode pipeline. If `False`, `pipeline_config` must be provided. |
| `quiet` | No | `Optional[bool]` (`None`) | If `True`, reduces logging noise for library use. If `None`, defaults to `True` when `libmode=True`. |

### Return type

`run_pipeline` returns a **union** of three possible types, depending on `block` and `run_in_subprocess`:

| Mode | Return type | Notes |
|------|-------------|--------|
| In-process, `block=True` | `float` | Elapsed time in seconds. |
| In-process, `block=False` | `RayPipelineInterface` | Handle to control the in-process pipeline (defined in `nv_ingest.framework.orchestration.ray.primitives.ray_pipeline`). |
| Subprocess, `block=False` | `RayPipelineSubprocessInterface` | Handle to control the subprocess-based pipeline (same module). **This is not** `RayPipelineInterface`; the two classes are separate implementations of `PipelineInterface`. Use `isinstance(..., RayPipelineSubprocessInterface)` when you launch with `run_in_subprocess=True` and `block=False`. |
| Subprocess, `block=True` | `float` | Returns `0.0` when blocking in subprocess mode. |

For the authoritative contract (including raised exceptions), refer to the docstring on `run_pipeline` in `src/nv_ingest/framework/orchestration/ray/util/pipeline/pipeline_runners.py`.
