# Functional test suite — INGEST: extracted content is immediately queryable (no manual glue)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **INGEST** job: content the user just dropped in must be
**immediately queryable through the platform — with no manual glue between ingestion
stages**. The distinct variable under test (vs other INGEST suites) is the **single-turn,
no-manual-glue, fast-after-ingest** behavior: the agent runs `retriever ingest <folder>/`
then **immediately** `retriever query "<q>"` against the **same default table**
(`lancedb/nemo-retriever`) **in the same turn** — there is no "ingest first, then ask
again" handoff, and the query reads the just-built index.

---

## The user task under test

> **JTBD: INGEST — P0.** "Extracted content is immediately queryable through the platform —
> no manual glue between ingestion stages."

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: a **single skill turn** handles **both** the ingest and query phases (no "ingest first, then ask again" handoff); the query phase operates on the **freshly-built index** and returns grounded hits; provenance is verified because each returned hit's `source` is one of the just-ingested files |
| Time | **fast** — the **query phase** responds **≤ 30s** after ingest completes (the no-glue query leg is the gated leg; ingest itself is not) |
| Trigger rate | ≥ 95% — a "drop these docs in and tell me X" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <path>` then **immediately** `retriever query "<text>"` in the **same turn**, against the **same default table**; no manual index/embed glue between them; **no `--input-type` flag** (it does not exist) |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"Drop these 10-Ks in and tell me what each company says about AI."*
- *"I just got these docs — pull out X's stance on supply chain."*
- *"Process my docs folder and rank the items by some metric."*

This row is about the **loop closing in one turn with no glue**, not about extraction depth
(that is the EXTRACT suites) or host routing (the SETUP suites). The fixtures are therefore
small, deterministic text/table PDFs whose answers are exact and grounded in real content.

---

## How the CLI behaves for this task (grounded in `--dry-run` + skill references)

Verified with `retriever ingest <path> --dry-run` (offline, no network) on the venv binary
(`2026.06.10.devXXXXXXXX`):

- **No glue — shared default target.** `retriever ingest` writes rows into LanceDB and
  `retriever query` reads from the **same** default target: `--lancedb-uri lancedb`,
  `--table-name nemo-retriever`. Because the defaults match, a query issued **right after**
  an ingest **in the same turn** reads exactly the rows the ingest just wrote. No separate
  "now build/load the index" step exists or is required.
- **`overwrite: true` by default.** The resolved `vdb_upload` plan shows
  `{uri: lancedb, table_name: nemo-retriever, overwrite: true}` — each turn's ingest builds
  a fresh index, so the immediately-following query reads exactly that turn's content
  (clean provenance).
- **One file → `branch_summary: pdf:1`; a folder of 3 → `branch_summary: pdf:3`** (confirmed
  via `--dry-run`). A single `retriever ingest <folder>/` call ingests every file in the
  folder in one invocation (no per-file loop).
- **Format auto-detection.** `.pdf` is resolved from the extension. There is **no
  `--input-type` flag**.
- **Default extraction is local text/pdfium.** For these text/table fixtures the resolved
  plan uses `method: pdfium`, `extract_text: true` — so the loop is offline-capable on the
  bundled embedder and the query leg is a fast dense-vector lookup. (Visual NIM routing is
  out of scope here; that's the EXTRACT/SETUP suites.)
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/nemo-retriever.`
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. The `source` field is what proves provenance — it is the
  path/name of the just-ingested file.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `ingest-iq-001` | **Baseline no-glue loop.** Ingest ONE doc then immediately query it in the SAME turn (no "ask twice"). The floor: the loop closes in one turn. | `ingest`, `query` |
| 2 | `ingest-iq-002` | **Provenance.** The answer (a precise table cell) must be traceable to the just-ingested file via the hit's `source`/`page_number` — the query reads the freshly-built index, not memory. | `ingest`, `query --content-types text,table` |
| 3 | `ingest-iq-003` | **Folder + cross-doc.** A small FOLDER (2 PDFs, `pdf:2`) ingested in one call, then a cross-doc question answered same turn; the query must pick the right source. | `ingest`, `query` |
| 4 | `ingest-iq-004` | **Timing.** Explicit latency assertion: the query leg returns **≤ 30s** after ingest completes; no re-ingest/rebuild before answering. | `ingest`, `query --content-types text,table` |
| 5 | `ingest-iq-005` | **Acceptance gate.** Folder of 3 (`pdf:3`) + a "tell me about each" question answered in ONE turn, every claim cited to a distinct freshly-ingested source, no second "now ask again" step, fast. | `ingest`, 3× `query` |

The ladder: T1 proves the ingest→query loop closes in one turn; T2 adds the provenance
assertion (answer traceable to the just-built index via `source`); T3 changes scope to a
multi-doc folder and a cross-doc query (right source picked); T4 adds the ≤30s post-ingest
query SLA; T5 composes everything — a dropped folder and a per-doc "tell me about each"
answered in one turn with per-source citations and fast queries.

---

### T1 — `ingest-iq-001` · baseline no-glue loop  *(complexity 1)*
- **Satisfies:** the INGEST "immediately queryable / no manual glue" core, simplest form.
- **Data:** `cases/ingest-iq-001/data/woods_frost.pdf` (2pg; p1 Robert Frost poem).
- **Expected:** `RETRIEVER ingest data/woods_frost.pdf` → `Ingested 1 file(s) -> N row(s)`
  (`branch_summary pdf:1`); then **in the same turn** `RETRIEVER query "Who owns the woods?"
  --top-k 5` → owner's "house is in the village" (the owner lives in the village), citing
  `woods_frost.pdf` p1. **Adds:** nothing — it is the floor (the loop closes in one turn).

### T2 — `ingest-iq-002` · provenance from the just-built index  *(complexity 2)*
- **Satisfies:** the "query operates on the freshly-built index, verified by `source`
  provenance" clause.
- **Data:** `cases/ingest-iq-002/data/table_test.pdf` (Year × {Bill,Amy,James,Ted,Susan}).
- **Adds:** an explicit provenance assertion using a precise cell so a generic answer can't
  pass. **Ground truth: James 2019 = 978**, and the supporting hit must carry
  `source=table_test.pdf`, `page_number=1` — proving the value came from the index built this
  turn, not from model memory or a stale prior index.

### T3 — `ingest-iq-003` · folder + immediate cross-doc question  *(complexity 3)*
- **Satisfies:** the "process my docs folder" seed query.
- **Data:** `cases/ingest-iq-003/data/{woods_frost.pdf, table_test.pdf}` (2 PDFs).
- **Adds:** folder-level ingest (`branch_summary pdf:2`) in one call + a cross-doc query
  answered the same turn. **Ground truth: New Hampshire (1923)**, citing `woods_frost.pdf`
  p2 (collection table) — the query must pick the right source out of the multi-doc index.

### T4 — `ingest-iq-004` · fast query after ingest  *(complexity 4)*
- **Satisfies:** the Time criterion (fast: query leg **≤ 30s** after ingest completes).
- **Data:** `cases/ingest-iq-004/data/multimodal_test.pdf` (3pg; p1 Table 1).
- **Adds:** an explicit latency assertion on the query leg. The agent must **not**
  re-ingest/rebuild before answering — the post-ingest query is a fast dense lookup over the
  just-written table. **Ground truth: Giraffe → Driving a car (At the beach), Table 1**,
  citing `multimodal_test.pdf` p1.

### T5 — `ingest-iq-005` · acceptance gate: folder + "tell me about each"  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the INGEST "immediately queryable, no
  glue" row.
- **Data:** `cases/ingest-iq-005/data/{woods_frost.pdf, table_test.pdf, multimodal_test.pdf}`.
- **Expected:** one `RETRIEVER ingest data/` (`branch_summary pdf:3`), then — **same turn,
  no second "now ask again" step** — three queries answering a "tell me about each" question:
  - **poem** — owner's house is in the village (`woods_frost.pdf` p1);
  - **numbers** — James 2019 = 978 (`table_test.pdf` p1);
  - **testing** — Giraffe → Driving a car, Table 1 (`multimodal_test.pdf` p1).
- **Adds (the gate):** every claim cited to a **distinct, freshly-ingested** source (per-doc
  provenance), every hit carrying `source` + `page_number`, the query legs fast (≤30s each),
  and **no** "ingest first, then ask again" handoff.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** BOTH `retriever ingest` and `retriever query` run in **one turn** — no "ingest first,
then ask again" handoff and no manual index/embed glue step in between;
**(b)** the query reads the **same default table** (`lancedb/nemo-retriever`) the ingest just
wrote — i.e. the **freshly-built index**;
**(c)** provenance: each returned hit's `source` is one of the **just-ingested** files (and,
for the multi-doc rungs, the correct file per claim), every hit carrying `source` +
`page_number`;
**(d)** answers match the grounded values (owner lives in the village; James 2019 = 978; New
Hampshire 1923; Giraffe → Driving a car) with the right page citation;
**(e)** timing: the **query leg** responds **≤ 30s** after ingest completes, with no
re-ingest/rebuild before answering;
**(f)** no `--input-type` flag is used (it does not exist); a folder is ingested in a single
call (`branch_summary pdf:2` / `pdf:3`), not a per-file loop.

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`**
(offline) and the skill's ingest/query references; the suite has **not** been run live yet.
A live run requires a reachable embedding backend for the ingest+query loop — the bundled
HuggingFace embedder by default (offline-capable for these text/table fixtures), or a hosted
endpoint (`integrate.api.nvidia.com`, needs `NVIDIA_API_KEY`, makes small billable calls).
Running live would capture the real per-file/per-folder row counts, the actual **query-leg
latency after ingest** (the ≤30s SLA under test), the exact returned `source`/`page_number`
provenance, and token baselines.
