# NeMo Retriever Library Mode Review Standards

These rules apply only to `nemo_retriever/`. For universal rules, see
the root `.greptile/rules.md`.

---

## Model Lifecycle Management

Library mode loads models directly in-process (PyTorch, vLLM, Nemotron
packages). This is fundamentally different from the legacy service mode
where models run in external Triton containers.

### Loading and Initialization

- Models must be loaded lazily, not at import time
- Support both automatic download and pre-cached local paths
- Model initialization should report progress for large downloads
- Failed model loads must produce actionable errors with resolution steps

### Resource Release

- Every model holder must have an explicit unload/cleanup method
- GPU tensors must be deleted and `torch.cuda.empty_cache()` called
- Context managers (`with model:`) are preferred for scoped usage
- Never rely on garbage collection for GPU memory release

### Device Placement

- Never hardcode `cuda:0` -- accept device configuration
- Check `torch.cuda.is_available()` before attempting GPU placement
- Support multi-GPU configurations where applicable
- Provide clear errors when requested device is unavailable

---

## Public API Surface

### Contract Stability

Every public function, class, and method is a user-facing contract.
Treat changes with the same care as a REST API:

- New optional parameters are safe to add
- Removing or renaming parameters is a breaking change
- Changing return types is a breaking change
- Adding new public methods to existing classes is safe

### Error Messages

Users of this library are developers integrating it into their pipelines.
Error messages must be actionable:

```python
# Bad
raise RuntimeError("Model failed to load")

# Good
raise RuntimeError(
    f"Failed to load model '{model_name}': {e}. "
    f"Ensure the model is downloaded with `retriever download {model_name}` "
    f"or set NEMO_RETRIEVER_MODEL_DIR={expected_path}"
)
```

### Type Safety

- All public interfaces must have complete type annotations
- Use `Union` and `Optional` explicitly, never `Any` unless unavoidable
- Pydantic models for all structured configuration
- Return concrete types, not internal implementation types

---

## Concurrency and Ray Integration

- Ray actors wrapping models must handle concurrent requests safely
- Batch processing should be preferred over per-item processing
- Ray object store should be used for passing large data between actors
- Actor initialization failures must be surfaced clearly, not silently
  retried forever
