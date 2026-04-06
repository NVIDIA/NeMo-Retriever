# NeMo Retriever Code Review Standards

Universal guidelines that apply across all packages. Package-specific
rules live in their own `.greptile/` directories (`nemo_retriever/.greptile/`
and `src/.greptile/`).

---

## Architecture and Design

### Package Dependency Direction

```
         api/  (base -- shared types and schemas, no upstream imports)
        /    \
   client/   src/  (parallel mid-level -- each depends ONLY on api/)
        \    /
    nemo_retriever/  (top-level -- may depend on all three)
```

**Never** import upward or sideways between peers. `client/` must never
import from `src/` (or vice versa). If `api/` needs something from `src/`,
the design is wrong -- extract the shared abstraction into `api/` instead.

### Single Responsibility

Each module, class, or function should have one reason to change. If a
function exceeds 50 lines or a class has more than 10 public methods,
it likely does too much.

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

## Error Handling

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

Always include actionable context in log messages:

```python
# Bad
logger.error("Processing failed")

# Good
logger.error(
    "Failed to extract text from document",
    extra={"source_id": doc.source_id, "doc_type": doc.document_type},
    exc_info=True,
)
```

---

## Testing Standards

### Test Quality Over Quantity

Each test function should:

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

### Backward Compatibility

When evolving APIs:

- Add new optional fields with defaults; never remove or rename existing
  fields without a deprecation cycle
- New endpoints can be added freely
- Changes to response shapes must be additive (new fields, not restructured)
- Breaking changes require a new API version path (e.g., `/v2/`)

---

## Performance

### Memory Awareness

The pipeline processes large documents (multi-GB PDFs, high-resolution
images). Be vigilant about:

- Holding entire documents in memory when streaming is possible
- Creating intermediate copies of large byte arrays or DataFrames
- Accumulating results in a list when a generator would work
- Not releasing GPU tensors after inference completes

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
