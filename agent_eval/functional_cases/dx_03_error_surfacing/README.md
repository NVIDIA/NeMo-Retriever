# Functional test suite — DX: error surfacing (clear, machine-parseable failures)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s)/flags **and the exact error that comes
back**.

This suite covers the **DEVELOPER EXPERIENCE (DX)** job: when something goes wrong
(unsupported file, missing index, malformed query, embedding rate-limit), the user must get
a **clear, machine-parseable error** — not a raw Python stack trace — with **retry/backoff
guidance where relevant**, and with **permanent failures labeled distinctly from transient
ones**.

---

## The user task under test

> **JTBD: DEVELOPER EXPERIENCE — P0.** "Failures (extraction error, embedding rate limit,
> vDB unavailable, malformed query) surface clear, machine-parseable errors with
> retry/backoff guidance where relevant."

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: each induced failure returns a **clear, machine-parseable** error, **not** a bubbled traceback. CLI layer = `Error: <message>` on stderr + non-zero exit (1 for SDK `_ROOT_CLI_ERRORS`; 2 for Typer option-validation). Service layer = a **structured** object — `error_metadata {task, status=ERROR, source_id, error_msg}` from `create_exception_tag()` + failure annotation via `cm_set_failure`. **Transient** (429/5xx/timeout) carries **retry/backoff** guidance; **permanent** (unsupported format, missing file, missing table, bad flag) is labeled distinctly and **not** retried. |
| Time | **fast — ≤ 30s** per case (each case induces a failure offline; no successful network ingest/embed needed) |
| Trigger rate | ≥ 95% — a "trigger this failure / try to load X / what happens if the index is gone" prompt must fire the skill and drive the CLI to reproduce the error |
| Subcommand accuracy | ≥ 90% — drive the right subcommand to induce the named failure: `ingest <unsupported>`; `query … --table-name <missing>`; `query … --top-k 0` / `--candidate-k < --top-k`. **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"What does a rate-limit error from the embedding service look like? Trigger one."*
- *"Try to load sample.eml."* (unsupported format)
- *"What happens if the index is unavailable mid-query?"*

---

## How errors surface in this CLI (grounded in source + induced live)

Verified by reading `src/nemo_retriever/cli/main.py`, `cli/sdk_workflow.py`,
`src/nemo_retriever/api/util/exception_handlers/` (a shim →
`common/api/util/exception_handlers/`), and `models/nim/primitives/nim_client.py`; and by
**inducing the CLI errors live & offline** against the venv binary.

**Two error layers — the contract:**

1. **CLI layer.** `cli/main.py` wraps both the `ingest` and `query` SDK calls in:
   ```python
   except _ROOT_CLI_ERRORS as exc:
       typer.echo(f"Error: {exc}", err=True)
       raise typer.Exit(1) from exc
   ```
   `_ROOT_CLI_ERRORS = (OSError, RuntimeError, ValueError, ValidationError)`. So an SDK-raised
   `ValueError`/`FileNotFoundError`/etc. is reduced to **one** `Error: <message>` line on
   **stderr** with **exit code 1** — never a bubbled traceback. Out-of-range option values
   (e.g. `--top-k 0`, declared `min=1`) are caught **earlier** by Typer/Click and print a
   `Usage:` header + a boxed `Error: Invalid value for '--top-k': …` with **exit code 2**.

2. **Service / extract layer (structured JSON-shaped errors).** From
   `api/util/exception_handlers/` (`pdf.py`, `decorators.py`):
   - `create_exception_tag(error_message, source_id)` builds
     `error_metadata = {task: EXTRACT, status: ERROR, source_id, error_msg}`, validates it
     via `validate_metadata`, and returns it as a metadata row — the **machine-parseable
     `{status/code, message, source}` shape** the extract layer emits instead of crashing.
   - The `nv_ingest_node_failure_*` decorators / `CMNVIngestFailureContextManager` call
     `cm_set_failure(control_message, error_message)` and
     `annotate_task_result(…, TaskResultStatus.FAILURE, message=…)`. `raise_on_failure` is
     **False** by default, so a single bad doc is **annotated** (parseable) rather than
     crashing the pipeline.

**Transient vs permanent (retry/backoff).** `models/nim/primitives/nim_client.py`:
- **HTTP 429 (Too Many Requests)** → retried up to `max_429_retries = 5` with **exponential
  backoff** (`backoff_time = base_delay * 2**retries_429`), logging
  `Received HTTP 429 (Too Many Requests) from <model>. Attempt N of 5.`
- **HTTP 503 / 5xx** and **`requests.Timeout`** → retried up to `max_retries` with the same
  backoff; only after retries are exhausted is the `HTTPError`/`TimeoutError` raised.
- A non-retryable **4xx** (or a 5xx after the final attempt) is **raised immediately**.

So **transient = rate-limit/5xx/timeout** (retried, backoff guidance inherent);
**permanent = unsupported format / missing file / missing table / bad flag** (fail fast, no
retry).

### Real messages captured (induced live, offline, against the venv binary)

| Induced failure | Command | Captured output | Exit |
|---|---|---|---|
| Unsupported format | `retriever ingest sample.eml` | `Error: Unsupported input file type(s) for retriever ingest: …/sample.eml` | 1 |
| vDB unavailable | `retriever query "…" --table-name does_not_exist` | `Error: Table 'does_not_exist' was not found` | 1 |
| Malformed (range) | `retriever query "x" --top-k 0` | `Invalid value for '--top-k': 0 is not in the range x>=1.` (+ `Usage:` header) | 2 |
| Malformed (semantic) | `retriever query "x" --top-k 5 --candidate-k 2` | `Error: candidate_k (2) must be greater than or equal to top_k (5).` | 1 |

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `dx-err-001` | **Baseline.** One unsupported-format ingest (`.eml`) → clear PERMANENT error naming the type; no traceback. The DX floor. | `ingest` |
| 2 | `dx-err-002` | **Query/vDB layer.** Failure now from the QUERY path against a missing table → `Error: Table '…' was not found`. | `query --table-name` |
| 3 | `dx-err-003` | **Caller-input validation, two layers.** Bad flag values: Typer range error (`--top-k 0`, exit 2) vs SDK semantic error (`--candidate-k < --top-k`, exit 1). | `query --top-k`, `query --candidate-k` |
| 4 | `dx-err-004` | **Transient vs permanent.** Contrast a 429/5xx/timeout (retried w/ exponential backoff, retry guidance) against the permanent `.eml` error (no retry). | `ingest` (+ grounded 429 path) |
| 5 | `dx-err-005` | **Acceptance gate.** Induce failures across ingest + query + validation; prove each is clean & parseable, label permanent/transient, cite the service-layer structured-error contract. | `ingest`, `query --table-name`, `query --top-k` |

The ladder: T1 proves an induced failure surfaces cleanly at all (ingest/format); T2 moves
the failure to the query/vDB layer; T3 adds caller-input validation and splits the
option-parser layer from the SDK layer; T4 adds the transient-vs-permanent taxonomy and the
retry/backoff semantics; T5 composes everything — failures at every stage, each parseable,
labeled, with retry guidance only where it belongs.

---

### T1 — `dx-err-001` · unsupported format (.eml)  *(complexity 1)*
- **Satisfies:** DX error-surfacing core, simplest form (induce one failure → clear error).
- **Data:** `cases/dx-err-001/data/sample.eml` (a 17-line plain-text email — unsupported ext).
- **Expected (induced live):** `RETRIEVER ingest data/sample.eml` →
  `Error: Unsupported input file type(s) for retriever ingest: data/sample.eml` on stderr,
  **exit 1**. PERMANENT; no retry guidance; no stack trace.

### T2 — `dx-err-002` · vDB unavailable / missing table  *(complexity 2)*
- **Satisfies:** the "index unavailable mid-query" seed.
- **Data:** `cases/dx-err-002/data/test.pdf` (a valid 1-page PDF, mounted so a real index
  could optionally be built first to show the contrast).
- **Adds:** the failure now comes from the **query** path against the vector DB.
- **Expected (induced live):** `RETRIEVER query "…" --table-name does_not_exist` →
  `Error: Table 'does_not_exist' was not found`, **exit 1**. The LanceDB lookup error is
  caught by `_ROOT_CLI_ERRORS` and reduced to one line naming the table; no traceback.

### T3 — `dx-err-003` · malformed query / invalid flag value  *(complexity 3)*
- **Satisfies:** the "malformed query" seed.
- **Data:** `cases/dx-err-003/data/test.pdf`.
- **Adds:** caller-input validation surfacing from **two layers**:
  - `--top-k 0` → **Typer/Click** option validation, *before* the SDK runs:
    `Invalid value for '--top-k': 0 is not in the range x>=1.` + `Usage:` header, **exit 2**.
  - `--candidate-k 2 --top-k 5` → **SDK** `ValueError` caught by `_ROOT_CLI_ERRORS`:
    `Error: candidate_k (2) must be greater than or equal to top_k (5).`, **exit 1**.
- Each message names the offending flag and the violated constraint (usable/actionable);
  neither crashes with a traceback. Both are PERMANENT caller-input errors (no retry).

### T4 — `dx-err-004` · transient vs permanent (rate-limit)  *(complexity 4)*
- **Satisfies:** the rate-limit seed AND the transient-vs-permanent clause.
- **Data:** `cases/dx-err-004/data/sample.eml` (permanent side) + `test.pdf` (valid doc).
- **Adds:** the retry/backoff taxonomy.
  - **TRANSIENT** — an embed/extract NIM returns **HTTP 429** → `NimClient` retries up to
    `max_429_retries = 5` with exponential backoff (`base_delay * 2**n`), logging
    `Received HTTP 429 (Too Many Requests) from <model>. Attempt N of 5.`; 503/5xx and
    timeouts behave the same; the `HTTPError` is raised only after retries are exhausted.
    **Retry/backoff guidance is inherent.** (Grounded in `nim_client.py`; see caveat.)
  - **PERMANENT** — `RETRIEVER ingest data/sample.eml` →
    `Error: Unsupported input file type(s) …` (exit 1), **not** retried.
- **Adds (the contrast):** retry guidance present for the transient class, absent for the
  permanent class.

### T5 — `dx-err-005` · acceptance gate, failures across the pipeline  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the DX error-surfacing row.
- **Data:** `cases/dx-err-005/data/sample.eml` + `test.pdf`.
- **Expected:** induce a failure at each stage and prove each is clean & parseable:
  - **ingest/format** — `Error: Unsupported input file type(s) … sample.eml` (exit 1, PERMANENT);
  - **query/vDB** — `Error: Table 'does_not_exist' was not found` (exit 1, PERMANENT);
  - **validation** — `Invalid value for '--top-k': 0 is not in the range x>=1.` (exit 2, PERMANENT caller-input);
  - then cite the **transient** class (429/5xx/timeout → retried w/ backoff) as the one that
    carries retry guidance, and the **service-layer structured contract**
    (`error_metadata {task,status,source_id,error_msg}` from `create_exception_tag` +
    `cm_set_failure` failure annotation) as the parseable shape distinct from the CLI
    `Error: …`+exit layer.
- **Adds (the gate):** every induced failure is a single `Error: …` line (or Typer
  Usage+boxed Invalid-value) with a non-zero exit — **no traceback** — permanent/transient
  is explicitly labeled, and retry/backoff guidance is attributed only to the transient class.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** the named failure is actually induced via the right subcommand/flag;
**(b)** the output is the clean CLI `Error: <message>` (or Typer Usage+boxed Invalid-value),
naming the offending file/table/flag, with the correct **non-zero exit code** (1 for SDK
errors, 2 for Typer option errors);
**(c)** **no raw Python stack trace** is shown to the user;
**(d)** **permanent vs transient** is labeled correctly, and **retry/backoff guidance** is
present for the transient (429/5xx/timeout) class and **absent** for permanent ones;
**(e)** the agent can cite the **service-layer structured-error contract**
(`error_metadata {task,status,source_id,error_msg}` + `cm_set_failure`) as the
machine-parseable shape at the API/service layer, distinct from the CLI layer;
**(f)** no `--input-type` flag is used (it does not exist).

**Note on live runs.** The four CLI errors in rungs 1–3 (and the permanent side of rungs 4–5)
were **induced live, offline** against the venv binary
(`$RETRIEVER_VENV/bin/retriever`, version `2026.06.10.devXXXX`) — see the captured-messages
table above; those are the **real** messages and exit codes. The **transient / rate-limit
(HTTP 429)** path in rungs 4–5 is **grounded in source** (`nim_client.py`: 429 → up to 5
retries with exponential backoff) and was **not run live**, because inducing a real 429
requires a live, throttled hosted embed/extract endpoint
(`integrate.api.nvidia.com` / `ai.api.nvidia.com`) plus an API key and may make billable
calls. A live transient run would capture the actual 429 retry log lines, the realized
backoff intervals, and the final raised `HTTPError` after the 5th attempt. Likewise, the
service-layer `error_metadata` structured tag is grounded by reading
`api/util/exception_handlers/` and is asserted as a contract, not exercised end-to-end here.
