# NeMo Retriever Code Review Standards

These rules supplement the structured rules in `config.json` with detailed
guidelines, rationale, and code examples. When reviewing, apply the judgment
of a distinguished engineer: prioritize correctness, security, and
maintainability over stylistic preferences.

---

## Architecture and Design

### Package Dependency Direction

This monorepo has a strict layering:

```
api/  (lowest level -- shared types and schemas)
  ^
  |
client/  and  src/  (mid-level -- depend on api/)
                ^
                |
         nemo_retriever/  (highest level -- depends on everything)
```

**Never** import upward. If `api/` needs something from `src/`, the design
is wrong -- extract the shared abstraction into `api/` instead.

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

### Ray Actor Lifecycle

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

---

## Error Handling

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

### Exception Specificity

Catch the most specific exception possible:

```python
# Bad
try:
    response = client.query(params)
except Exception as e:
    logger.error(f"Query failed: {e}")

# Good
try:
    response = client.query(params)
except ConnectionError as e:
    logger.error(f"Connection to service lost: {e}")
    raise
except TimeoutError as e:
    logger.warning(f"Query timed out, retrying: {e}")
    response = client.query(params, timeout=extended_timeout)
```

### Logging Context

Always include actionable context in log messages. A log message should
answer: what happened, where, and what identifiers can be used to trace it.

```python
# Bad
logger.error("Processing failed")

# Good
logger.error(
    "Failed to extract text from document",
    extra={"source_id": doc.source_id, "doc_type": doc.document_type, "stage": "pdf_extraction"},
    exc_info=True,
)
```

---

## Security

### Document Processing Security

This pipeline ingests enterprise documents that may contain sensitive
information (PII, financial data, trade secrets). Every stage must:

- Never log document content at INFO level or below
- Never write document content to temporary files without cleanup
- Sanitize any content before including it in error messages
- Validate file paths to prevent path traversal attacks

### Secrets and Credentials

Credentials must come from environment variables or a secrets manager.
Review for:

- Hardcoded strings that look like tokens, keys, or passwords
- Default parameter values that contain credentials
- Test fixtures that contain real credentials
- Configuration files committed with actual secrets

### Input Validation at Boundaries

Every entry point (API endpoint, CLI command, client method) must validate:

- File sizes before processing (prevent OOM)
- File types against an allowlist (prevent malicious file processing)
- String lengths and content (prevent injection)
- Numeric ranges (prevent resource exhaustion)

---

## Testing Standards

### Test Quality Over Quantity

A test that only verifies the happy path with no assertions on error
behavior is incomplete. Each test function should:

- Test one specific behavior (not multiple scenarios in one function)
- Include assertions on both return values AND side effects
- Use descriptive names: `test_pdf_extraction_raises_on_corrupted_file`
  not `test_pdf_3`

### Mocking Discipline

- Mock at the boundary, not deep inside the call stack
- Use `spec=True` or `spec_set=True` when creating mocks to catch
  API drift
- Verify mock call arguments when the interaction contract matters
- Never mock the unit under test itself

### Integration Test Markers

Tests that require external services (Triton, Redis, Milvus, GPUs) must
be marked with `@pytest.mark.integration` so they are excluded from the
default unit test run.

---

## API Design

### Pydantic Model Discipline

All API request/response models must use Pydantic with:

- Explicit field types (no `Any` unless truly necessary)
- Field validators for business rules
- `model_config = ConfigDict(strict=True)` where type coercion is dangerous
- Descriptive `Field(description="...")` for OpenAPI documentation

### Backward Compatibility

When evolving APIs:

- Add new optional fields with defaults; never remove or rename existing
  fields without a deprecation cycle
- New endpoints can be added freely
- Changes to response shapes must be additive (new fields, not restructured)
- Breaking changes require a new API version path (e.g., `/v2/`)

---

## Performance Considerations

### Memory Awareness

The pipeline processes large documents (multi-GB PDFs, high-resolution
images). Be vigilant about:

- Holding entire documents in memory when streaming is possible
- Creating intermediate copies of large byte arrays or DataFrames
- Accumulating results in a list when a generator would work
- Not releasing GPU tensors after inference completes

### Concurrency and Throughput

- Pipeline stages should be stateless where possible to allow horizontal
  scaling via Ray
- Shared state between stages must go through Ray's object store or an
  external service (Redis), never through module-level globals
- Batch operations (batch inference, batch database writes) are preferred
  over item-by-item processing

---

## Infrastructure and Deployment

### Docker Best Practices

- Multi-stage builds to minimize image size
- Pin base image tags (never use `latest`)
- Run as non-root user
- Do not copy secrets or credentials into the image
- Use `.dockerignore` to exclude test data, docs, and dev files

### Helm Chart Standards

- All configuration must be exposed through `values.yaml`
- Use `{{ .Values.x | default "y" }}` patterns for sensible defaults
- Include resource requests AND limits for every container
- Define liveness and readiness probes
- Support configurable image repositories and tags for air-gapped deployments
