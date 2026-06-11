# Functional test suite — INGEST: run a long ingestion job (>1 min) in the background

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` / `retriever query`,
plus the `retriever service` sub-app).

Each test is a self-contained triple: **a prompt** the agent receives, **a folder of data
files** mounted alongside it, and **an expected output** — including the *correct
`retriever` subcommand(s)* and the *backgrounding mechanism* appropriate for this task. The
five tests climb a single complexity ladder.

---

## The user task under test

> **JTBD: INGEST.** "Run a long ingestion job (>1 min) in the background. Async ingestion
> for jobs that take > 1 min." — **P0**

**Operational pass** = a long-running NRL ingest job is moved to the **background** so the
agent can perform other tasks while it runs. The acknowledgment is **fast** (a started/queued
signal); the full job is **slow** (the whole point — it takes > 1 min).

**Success criteria for the row** (from the JTBD tab + the seed-queries tab):

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) skill does **not** block on full completion — it backgrounds the ingest and returns a quick ack; (2) agent stays responsive (can do another action mid-flight); (3) progress observable; (4) terminal status (succeeded/failed/partial) detectable + index queryable after |
| Time | **two-phase.** Ack is **fast: ≤ 5 s** after submission (the gated SLA). The full job is **slow** and is *not* wall-clock-gated here — the test shape is under test, not literal duration |
| Trigger rate | ≥ 95% — a "start a big ingest in the background, I'll check on it later" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — must reach `retriever ingest` for the work and **background it** (shell-level `nohup … &` / background task, tailing the `--no-quiet` log), or drive **service mode** (`retriever service start` then `retriever service ingest --sse`/`--no-sse`). Must **not** invent a `--async` flag or a native `task_id` REST poll on the plain CLI |
| Token usage | tracked, not gated |

The seed queries this suite is derived from:
- *"Start a big ingest job on these N PDFs — I'll come back to check on it."*
- *"Kick this off in the background; don't make me wait."*
- *"I have a lot of docs — start ingesting and let me poll for status."*

---

## How the CLI behaves for this task (verified, not assumed)

Grounded by reading `src/nemo_retriever/cli/main.py`, `src/nemo_retriever/service/cli.py`,
`src/nemo_retriever/service/client.py`, and
`src/nemo_retriever/service/service_ingestor.py`, plus `retriever ingest … --dry-run`
(offline) and `--help` on `retriever ingest` / `retriever service {start,ingest}`.

**The plain `retriever ingest` is SYNCHRONOUS.** It blocks until the LanceDB write completes
and prints one summary line:
`Ingested N file(s) → M row(s) in LanceDB lancedb/<table>.`
There is **no** `--async` / `--background` / `--detach` flag, and **no** `--run-mode service`
on the plain ingest command — `--run-mode` accepts only `inprocess|batch` (a `--dry-run`
shows `create_ingestor.run_mode = inprocess`). `--run-mode batch` is for Ray-Data scale-out,
not backgrounding.

So there are exactly **two grounded ways** to satisfy "run it in the background":

### Path A — shell-level backgrounding (no server needed)
Wrap the synchronous `retriever ingest` in a background process at the orchestration layer:

```
nohup $RETRIEVER ingest data/ --no-quiet > ingest.log 2>&1 &   # background it
echo "started pid $!"                                          # the immediate ACK (PID)
tail -n 20 ingest.log                                          # observe PROGRESS (--no-quiet)
wait $!; echo "exit=$?"                                        # TERMINAL status (exit 0 + summary line)
$RETRIEVER query "…" --top-k 5                                 # POST-COMPLETION query
```
- **Ack** = the captured PID (returned in well under 5 s).
- **Progress** = tail the `--no-quiet` log (per-stage / per-file lines).
- **Terminal status** = process exit code 0 **and** the final `Ingested N file(s) → M row(s)`
  line in the log (failure = non-zero exit / no summary line).

### Path B — service mode (the closest thing to native async)
The `retriever service` sub-app exposes a streaming surface built on
**`ServiceIngestor.aingest_stream()`**:

```
$RETRIEVER service start &                                  # boot the FastAPI server (port 7670)
$RETRIEVER service ingest data/*.pdf --sse                  # open a job, stream events
#   or:  $RETRIEVER service ingest data/*.pdf --no-sse --poll-interval 2.0   # bulk-poll mode
```
`service ingest` opens a server-side job aggregate via `POST /v1/ingest/job` (returns an
opaque **`job_id`**, sized to `len(files)`), uploads each doc, then subscribes to the
per-job SSE stream `GET /v1/ingest/job/{job_id}/events`. Event lifecycle:

`job_created` (carries `job_id` + `expected_documents`) → `job_started` →
`upload_complete` / `document_complete` (per file) → `job_progress` (completed/failed
counts) → terminal **`job_finalized` | `job_partial` | `job_failed`**.

- **Ack** = the first `job_created` event (carries the `job_id`).
- **Progress** = `job_progress` / `document_complete` events.
- **Terminal status** = `job_finalized` (success) vs `job_partial` / `job_failed`.
- `--no-sse` mode bulk-polls `POST /v1/ingest/status/batch` every `--poll-interval` s
  (default 2.0) and the aggregate is fetchable at `GET /v1/ingest/job/{job_id}`.

Either path is an acceptable "pass." Path A needs no server; Path B is closer to a real
async API but still requires a running server and is **not** a plain-CLI `task_id` call.

Convention in every command below: `RETRIEVER=/raid/nemo_retriever/.venv/bin/retriever`.

---

## Fixtures

Each case mounts the **same small 5-PDF folder** in its `data/`:
`multimodal_test.pdf`, `woods_frost.pdf`, `table_test.pdf`, `test-page-form.pdf`,
`embedded_table.pdf` (copied from `/raid/nemo_retriever/data/`). On this corpus the real
wall-clock is **well under a minute** — a genuine > 1 min job needs a larger corpus (e.g. the
financebench 10-K set under `/raid/retriever-sdg-v3/test-data/financebench/`). The **shape**
— background submission + ≤ 5 s ack + observable progress/terminal status + post-completion
query — is what's under test, **not** literal wall-clock here.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `ingest-async-001` | **Baseline.** Background the folder ingest and return an immediate ack (PID / job-started). Control returns to the agent. | `ingest` (backgrounded) |
| 2 | `ingest-async-002` | **Non-blocking.** Agent performs another action (count/list the submitted PDFs) *while the job runs* — proves the launch isn't just a fast synchronous run. | `ingest` (backgrounded) |
| 3 | `ingest-async-003` | **Progress observable.** Expose a way to *poll* the in-flight job: tail the `--no-quiet` log, or service-mode `job_progress`/`document_complete` SSE events. | `ingest` (backgrounded) |
| 4 | `ingest-async-004` | **Terminal status + queryable.** Detect succeeded/failed/partial from the same channel; then a post-completion `query` returns a grounded hit. | `ingest`, `query` |
| 5 | `ingest-async-005` | **Acceptance gate.** All of the above composed: ≤ 5 s ack, responsive mid-flight, progress observed, terminal status surfaced, post-completion table-cell query — with the explicit skill-layer/no-native-async note. | `ingest --use-table-structure`, `query --content-types text,table` |

The ladder adds exactly one dimension per rung: launch-and-ack → stay responsive → observe
progress → detect terminal status + query → the full composed operational pass.

---

### T1 — `ingest-async-001` · background launch + immediate ack  *(complexity 1)*
- **Satisfies:** the INGEST async core — background the job, return control with a quick ack.
- **Prompt:** "Start a big ingest job on the PDFs in `data/` for me — I'll come back to
  check on it later. Kick it off in the background and just confirm it started; don't make
  me wait for it to finish."
- **Data:** `data/` = the 5-PDF folder.
- **Expected:** `nohup $RETRIEVER ingest data/ --no-quiet > ingest.log 2>&1 & echo "started
  pid $!"` → a PID ack within ~5 s; the agent's turn is **not** blocked on the full ingest.
  `ingest.log` will *eventually* hold `Ingested 5 file(s) → M row(s) in LanceDB
  lancedb/nemo-retriever.` (not awaited at rung 1).
- **Why it's the floor:** one folder, default flags, no progress/terminal requirement — just
  "is the job backgrounded and is an ack returned fast?"

### T2 — `ingest-async-002` · non-blocking (concurrent agent action)  *(complexity 2)*
- **Satisfies:** the "agent stays responsive" clause.
- **Prompt:** "I have a lot of docs in `data/` — start ingesting them in the background, and
  while that's running, tell me how many PDFs you actually submitted and which files they
  are. Don't wait for the ingest to finish before answering."
- **Data:** `data/` = the 5-PDF folder.
- **Expected:** background launch + PID ack (as T1), then `ls data/*.pdf | wc -l` → **5**,
  naming the five files — produced **before** the ingest summary line appears.
- **Adds:** a concurrent action overlapping the live job, proving the launch is genuinely
  non-blocking.

### T3 — `ingest-async-003` · progress observable  *(complexity 3)*
- **Satisfies:** the "progress is observable" clause.
- **Prompt:** "Kick off the ingest of `data/` in the background, then let me poll its status
  — show me progress as files get processed so I can see it's moving, not just that it
  started."
- **Data:** `data/` = the 5-PDF folder.
- **Expected:** background launch + ack, then a progress read of the **in-flight** job —
  `tail -n 20 ingest.log` showing per-stage/per-file lines (Path A), or service-mode
  `job_progress` (completed/failed counts) / `document_complete` SSE events (Path B). The
  observed progress shows files **moving** (a count climbing toward 5), not just the start.
- **Adds:** a polling/progress read of a running job. **Note:** polling is a *skill-layer*
  construct over the stream (log tail or `aingest_stream()` SSE), **not** a native CLI status
  endpoint.

### T4 — `ingest-async-004` · terminal status + post-completion query  *(complexity 4)*
- **Satisfies:** the "terminal status observable + queryable after" clause.
- **Prompt:** "Run the ingest of `data/` in the background and let me know when it's done and
  whether it succeeded. Once it's finished, prove the index works: who owns the woods in the
  Robert Frost poem?"
- **Data:** `data/` = the 5-PDF folder.
- **Expected:** background launch + ack; **terminal SUCCEEDED** detected = process exits 0
  **and** `ingest.log` ends with `Ingested 5 file(s) → M row(s) …` (service-mode equivalent:
  a `job_finalized` event vs `job_partial`/`job_failed`); then `$RETRIEVER query "Who owns
  the woods?" --top-k 5` → the owner's **"house is in the village"** (he will not see the
  speaker stopping), citing `woods_frost.pdf` p1.
- **Adds:** terminal-status detection (the succeeded/failed/partial distinction must be
  *observed*, not assumed) plus a post-completion retrieval against the freshly built index.

### T5 — `ingest-async-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row — all four criteria composed.
- **Prompt:** "I have a big batch of PDFs in `data/` that'll take a while to ingest. Start
  the job in the background and confirm within a few seconds that it's running. While it
  runs, tell me how many docs are in the batch. Then poll it so I can see progress, let me
  know when it finishes and whether it fully succeeded, and finally prove the index is live:
  what value did James have in 2019 (pull it from the table)?"
- **Data:** `data/` = the 5-PDF folder.
- **Expected:**
  - `nohup $RETRIEVER ingest data/ --use-table-structure --no-quiet > ingest.log 2>&1 & echo
    "started pid $!"` → PID ack **within ~5 s** (the gated fast-ack SLA);
  - `ls data/*.pdf | wc -l` → **5** (reported while the job runs);
  - `tail -n 20 ingest.log` → progress on the in-flight job;
  - `wait $!; echo "exit=$?"` → terminal **SUCCEEDED** (exit 0 + final summary line);
  - `$RETRIEVER query "James value in 2019" --top-k 5 --content-types text,table` → **978**
    from a `content_type=table` row, citing `table_test.pdf` p1.
- **Adds (the gate):** every prior dimension at once — ≤ 5 s ack, responsiveness, observed
  progress, surfaced terminal status, and a table-cell post-completion query — plus the
  explicit note below that polling/`task_id` is a skill-layer construct, not a native API.

---

## Running / grading

Drop into the existing `agent_eval` harness: mount each test's `data/` folder into the agent
workdir, give it the `prompt`, and grade against `pass_when` in `cases.json`. T1–T2 grade the
background launch + quick ack (and, for T2, the concurrent action). T3 adds the progress-read
mechanism. T4 adds terminal-status detection + a grounded post-completion query. T5 is the
composed acceptance gate (ack ≤ 5 s + responsive + progress + terminal status + table-cell
query). In all cases, using a fictional `--async`/`--background` flag or a native plain-CLI
`task_id` REST poll is an automatic fail.

---

## Note on live runs (not run live)

Expected outputs are grounded in the CLI source (`cli/main.py`, `service/cli.py`,
`service/client.py`, `service/service_ingestor.py`), `--help`, and the offline `--dry-run`
plan — they have **not** been executed live. A live run would call the bundled/HF embedder
(and hosted/local extraction backends), and would capture real per-file row counts, the
actual ack latency (PID return / first `job_created` event), the progress-tail cadence, the
terminal exit code / terminal SSE event, and the post-completion query latency and hit
contents.

## ⚠️ GAP CAVEAT — no native async / task_id API in NRL 26.05

**NRL 26.05 has NO public `task_id` + REST-poll API on the plain `retriever ingest` CLI, and
no `--async` / `--background` / `--detach` / `--run-mode service` flag on it** (plain ingest
`--run-mode` is `inprocess|batch` only; `retriever ingest` is synchronous and blocks until
the LanceDB write finishes). The "submit → get a task_id → poll `/status` until done"
experience must be **built at the SKILL LAYER** on top of one of two real surfaces:

1. **Shell-level backgrounding** of `retriever ingest` (`nohup … &` / a background task) —
   the **ack is the captured PID**, progress is the `--no-quiet` log tail, terminal status is
   the exit code + final `Ingested N file(s) → M row(s)` summary line. No server required.
2. **Service mode** — `retriever service start` then `retriever service ingest --sse`
   (or `--no-sse --poll-interval`), which exposes a server-side `job_id` and an SSE event
   stream (`job_created` → `job_progress` → `job_finalized`/`job_partial`/`job_failed`) that
   is the surface of **`ServiceIngestor.aingest_stream()`**. This is the closest thing to a
   native async API, but still requires a running server and is **not** a plain-CLI
   `task_id` call.

A "pass" therefore means the skill **backgrounds the job and surfaces a quick ack + a way to
observe progress/terminal status via the stream** — *not* a fictional `--async`/`task_id`
REST call. Any expected output naming such a flag/endpoint on the plain CLI is wrong by
construction and must fail the case.
