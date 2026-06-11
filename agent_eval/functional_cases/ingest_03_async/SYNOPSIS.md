# Synopsis — INGEST: run a long ingestion job (>1 min) in the background

**What user task this covers.** A user has a big batch of documents that will take a while to
ingest, and doesn't want to sit and wait for it. They want NeMo Retriever to **start the job
in the background**, hand control straight back ("confirm it started; don't make me wait"),
let them **check on progress**, tell them **when it's done and whether it succeeded**, and
have the **index ready to query** afterward. Success is *operational*: the long job is moved
to the background so the agent stays free to do other things, with a fast acknowledgment and
an observable terminal status.

**The key reality (and the caveat).** NeMo Retriever 26.05 has **no native async API** — the
plain `retriever ingest` command is *synchronous* and has no `--async`/`--background` flag,
and there is **no `task_id` + REST-poll** endpoint on it. So the "background + poll" experience
is built **at the skill layer** on top of one of two real surfaces: (A) **shell-level
backgrounding** of `retriever ingest` (`nohup … &`) — the acknowledgment is the captured
process PID, progress is a tail of the `--no-quiet` log, and terminal status is the exit code
plus the final `Ingested N file(s) → M row(s)` summary line; or (B) **service mode**
(`retriever service start` then `retriever service ingest --sse`/`--no-sse`), which exposes a
server-side `job_id` and an SSE event stream (`job_created` → `job_progress` →
`job_finalized`/`job_partial`/`job_failed`) — the surface of `ServiceIngestor.aingest_stream()`.
A "pass" is the skill choosing one of these real paths, **not** inventing an async flag or a
task_id poll.

**How we test it.** Five agent prompts, each mounting the same small 5-PDF folder. We check
the agent backgrounds `retriever ingest` (or drives service mode), returns a quick ack, and
exposes progress / terminal status without blocking — then queries the finished index.

**The five tests, simplest to hardest:**

1. **Background launch + ack** — kick off the folder ingest in the background and confirm it
   started (PID / job-started) within ~5 s; control returns to the agent. The floor.
2. **Non-blocking** — while the job runs, the agent answers a second question (how many PDFs
   were submitted, and which) *before* the ingest finishes — proving the launch isn't just a
   fast synchronous run.
3. **Progress observable** — expose a way to poll the in-flight job: tail the `--no-quiet`
   log, or read service-mode `job_progress`/`document_complete` events. Progress shows files
   moving, not just "started."
4. **Terminal status + query** — detect succeeded/failed/partial from the same channel (exit
   0 + summary line, or a `job_finalized` event), then prove the index works: *who owns the
   woods?* → the owner whose **"house is in the village"** (`woods_frost.pdf` p1).
5. **Acceptance gate** — all of it composed: ≤ 5 s ack, responsive mid-flight, progress
   observed, terminal status surfaced, and a post-completion **table-cell** query
   (*James, 2019 → 978* from `table_test.pdf`) — with the explicit no-native-async note.

**Why this order.** Each rung adds exactly one new thing: first "is the job backgrounded with
a fast ack," then "can the agent keep working," then "can we watch progress," then "can we
tell it finished and use the result," then everything together as the operational-pass gate.

**Fixture note.** The 5-PDF corpus runs in well under a minute, so the literal ">1 min" isn't
hit here — a genuine long job needs a larger corpus (e.g. the financebench 10-Ks). The
**shape** (background + quick ack + observable terminal status + post-completion query) is
what's under test, not wall-clock.

**Status.** Tests are authored and grounded in the real CLI source (`cli/main.py`,
`service/cli.py`, `service/client.py`, `service/service_ingestor.py`), `--help`, and the
offline `--dry-run` plan; **not yet run live**. A live run would capture the real ack latency
(PID / first `job_created`), progress cadence, terminal exit code / event, and the
post-completion query results. See `README.md` for the full spec (and the prominent
no-native-async/task_id gap caveat) and `cases.json` for the machine-gradable definitions.
