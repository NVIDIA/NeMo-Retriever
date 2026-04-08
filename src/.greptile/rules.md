# NV-Ingest Service Mode Review Standards

These rules apply only to `src/` (the legacy service-mode pipeline).
For universal rules, see the root `.greptile/rules.md`.

---

## Ray Pipeline Patterns

### Required Decorator Stack

Every function that processes a `ControlMessage` as part of the pipeline
must use the standard decorator stack:

```python
@filter_by_task(["task_name"])
@nv_ingest_node_failure_context_manager(annotation_id="stage_name")
@traceable(trace_name="StageName")
def process_fn(control_message: ControlMessage, **kwargs) -> ControlMessage:
    ...
```

The order matters:
1. `@filter_by_task` -- outermost, skips non-relevant messages
2. `@nv_ingest_node_failure_context_manager` -- catches failures, annotates the message
3. `@traceable` -- innermost, records entry/exit timestamps

A stage missing any of these decorators will silently break tracing,
error recovery, or task routing.

### ControlMessage Failure Flow

When a pipeline stage encounters an error, the `ControlMessage` must be
annotated with the failure (not silently dropped). The
`@nv_ingest_node_failure_context_manager` decorator handles this
automatically. **Never** catch and swallow exceptions inside a pipeline
stage without re-raising or annotating the `ControlMessage`.

Bad:
```python
def process(msg):
    try:
        result = do_work(msg)
    except Exception:
        pass  # silently lost
    return msg
```

Good:
```python
@nv_ingest_node_failure_context_manager(annotation_id="my_stage")
def process(msg):
    result = do_work(msg)
    return msg
```

### Single Responsibility in Pipeline Stages

Each pipeline stage should do exactly one thing: extract text, split chunks,
embed content, etc. If a stage function is handling multiple concerns (e.g.,
extraction AND validation AND storage), it should be decomposed.

### Separation of Configuration and Logic

Pipeline stage behavior should be driven by configuration passed through
`ControlMessage` task specs, not by hardcoded conditionals. If you see a
stage with `if config_value == "mode_a": ... elif config_value == "mode_b": ...`
growing beyond two branches, suggest extracting a strategy pattern.

---

## Ray Actor Lifecycle

Ray actors that hold GPU resources, database connections, or large caches
must implement proper cleanup:

- Implement a `shutdown()` or `cleanup()` method that releases resources
- Use `ray.actor.exit_actor()` for controlled shutdown
- Never rely on `__del__` alone -- Ray does not guarantee its execution
- Explicitly `del` GPU tensors and call `torch.cuda.empty_cache()` when
  releasing models

### Avoiding Ray Anti-Patterns

- Do not pass large objects (DataFrames, tensors, images) directly as Ray
  task arguments. Use the Ray object store (`ray.put()` / `ray.get()`)
  or shared memory references.
- Do not block the event loop inside async Ray actors. Use
  `await asyncio.to_thread()` for CPU-bound work.
- Do not create unbounded numbers of Ray tasks in a loop. Use
  `ray.wait()` with a concurrency limit or batch submissions.
