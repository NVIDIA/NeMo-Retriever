# Synopsis — INGEST: extracted content is immediately queryable (no manual glue)

**What user task this covers.** A user just got some documents and wants to ask about them
**right away** — drop the files in, ask a question, get a grounded answer, all in one go.
The platform promise under test: **extracted content is immediately queryable, with no
manual glue between ingestion stages.** There is no "first ingest, then separately build an
index, then come back and ask" — the agent runs `retriever ingest` and then immediately
`retriever query` against the same just-built index, in the **same turn**.

**How we test it.** Five agent prompts, each handing the agent one small real document (or a
small folder) and a question phrased like a real user ("I just got this — load it and tell
me X; don't make me ask twice"). We check the agent drives the `retriever` CLI correctly:
`retriever ingest <path>` (format auto-detected — there is **no** `--input-type` flag) then,
**without any handoff back to the user**, `retriever query "<q>"` against the **same default
table** (`lancedb/nemo-retriever`). Because ingest and query default to the same target, the
query reads exactly the rows the ingest just wrote — and we verify that by checking the
returned hit's `source` is one of the files just ingested.

**The five tests, simplest to hardest:**

1. **Baseline no-glue loop** — ingest ONE doc, then immediately query it in the same turn
   (no "ask twice"). Proves the loop closes in one turn. (Ground truth: the woods' owner
   lives in the village.)
2. **Provenance** — pull a precise table cell (James, 2019 → **978**) and show it came from
   the just-ingested file via the hit's `source`/`page_number` — proving the answer is from
   the freshly-built index, not model memory.
3. **Folder + cross-doc** — drop a small **folder** (2 PDFs) in one call, then ask a
   cross-doc question answered in the same turn (which Frost collection was published in
   **1923** → New Hampshire). The query must pick the right source.
4. **Timing** — once it's loaded, the answer comes back **fast**: the query leg returns in
   **≤ 30s** after ingest completes, with no rebuild in between (Giraffe → Driving a car).
5. **Acceptance gate** — a **folder of 3** docs dropped in and a "tell me about each"
   question answered in ONE turn, every claim cited to a distinct freshly-ingested source,
   no second "now ask again" step. This is the test the others build up to.

**Why this order.** Each rung adds exactly one new thing: first "does the ingest→query loop
close in one turn at all," then provenance (the answer is provably from the just-built
index), then a multi-doc folder + cross-doc query, then the ≤30s post-ingest query SLA, then
everything composed — a dropped folder and a per-doc "tell me about each" answered in one
turn with per-source citations.

**What makes this suite distinct from the other INGEST suites.** The variable here is the
**single-turn, no-manual-glue, fast-after-ingest** behavior — that the freshly-extracted
content is immediately queryable with no glue step the user has to perform — not extraction
depth or host routing.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run`, offline) and the
skill's ingest/query references; **not yet run live**. Live runs need a reachable embedding
backend for the ingest+query loop (the bundled HuggingFace embedder by default, or a hosted
endpoint with an API key). See `README.md` for the full spec and `cases.json` for the
machine-gradable definitions.
