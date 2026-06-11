# Functional test suite — INGEST: resumable & idempotent ingest  ⚠️ KNOWN NRL 26.05 GAP

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the real
CLI (`retriever ingest` / `retriever query`). Each test is a self-contained triple — a prompt, a
per-case `cases/<id>/data/` folder of real PDFs, and an expected output naming the correct
`retriever` subcommand(s) and flags.

> ## ⚠️ THIS IS A GAP-DOCUMENTING SUITE — IT IS EXPECTED TO SURFACE A GAP, NOT TO PASS A FEATURE
>
> The user task below (**resume a killed ingest without re-processing finished files; re-ingest a
> folder idempotently; add one new file cleanly**) is **NOT natively supported in NRL 26.05.**
> These tests exist to **demonstrate that gap honestly** and **track it until it is closed.**
>
> - **A "pass" of a case = the gap is correctly surfaced** (e.g. re-ingest rebuilds the whole
>   table under `--overwrite`; `--append` duplicates rows; no `--resume` flag exists; the agent
>   reports this truthfully).
> - **A "fail" of the suite = a false claim** that resume/idempotency works — e.g. an agent that
>   invents a `--resume` flag, or claims a re-ingest "skipped already-indexed files" or "didn't
>   duplicate anything" when it did.
>
> This row should appear in the operational dashboard as a **documented gap with tracking tests**,
> not as a failing feature.

---

## The user task under test

> **JTBD: INGEST — P1 (escalating to P0).** "Partial ingestion can resume without full restart.
> Re-ingesting the same folder is idempotent — doesn't duplicate content or corrupt the index."

Seed queries this suite is derived from (paraphrased into the prompts):
- *"I had to kill the ingest halfway through — restart it without re-processing what's already done."*
- *"Re-ingest this folder — I think I dropped some new files in."*
- *"Add `<NEW_DOC>.pdf` to whatever index I built earlier."*

**Desired (aspirational) validation path — the behaviors these tests measure against:**

| Desired behavior | What a passing FEATURE would do |
|---|---|
| **Resume** | Kill mid-ingest, restart same command → already-indexed files **not** re-extracted (verify via row count + timing). |
| **Idempotency** | Re-ingest the same folder twice → **no** duplicate rows for unchanged files (verify via `(pdf_basename, page_number)` uniqueness). |
| **Incremental add** | Add one new file to an existing index → **only** that file processed; existing rows untouched. |
| **No corruption** | Index remains queryable throughout, no corruption mid-write. |

**Success criteria for this row (operational — reframed for a gap suite):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass = **the gap is correctly surfaced**: (a) default ingest resolves with `overwrite=true` (full rebuild, no skip); (b) `--append` duplicates `(pdf_basename, page_number)` rows on re-run; (c) no `--resume`/`--checkpoint`/`--skip-existing`/source-hash flag exists; (d) the agent does **not** fabricate a flag or claim native resume/idempotency |
| Time | **medium — ≤ 2 min** per case (small 3–4 PDF folder, one or two ingests) |
| Trigger rate | ≥ 95% — a "resume / re-ingest / add this file to my index" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — correct real flags (`--overwrite` default vs `--append`); must **not** invent `--resume`/`--checkpoint`/`--skip-existing`; query (when used) hits the same `--lancedb-uri`/`--table-name` |
| Token usage | tracked, not gated |

---

## How the CLI actually behaves for this task (grounded in `--help` + `--dry-run`)

Verified directly against `$RETRIEVER ingest --help` (full long-flag list scanned) and
`$RETRIEVER ingest <paths> --dry-run` (offline, no network):

- **Default write mode is `--overwrite`.** The resolved dry-run plan's `vdb` block shows
  `"overwrite": true`. Every ingest **replaces** the target LanceDB table — the entire folder is
  re-extracted and re-embedded. There is **no per-document skip** of already-indexed files.
- **`--append` flips `overwrite` to `false` and adds rows WITHOUT duplicate checks.** Dry-run with
  `--append` shows `"overwrite": false`. The CLI help text is explicit:
  > *"Overwrite the target LanceDB table by default. Use `--append` to add rows to an existing
  > table **without duplicate checks**; **rerunning the same inputs in append mode creates
  > duplicates.**"*
- **There is NO resume/checkpoint/incremental flag.** The complete `ingest` long-flag list
  contains **no** `--resume`, `--checkpoint`, `--skip-existing`, `--incremental`, `--filter`, or
  source-hash option. A killed ingest can only be restarted by re-running the command.
- **`--dedup` is NOT document dedup.** `--dedup` / `--dedup-iou-threshold` (default IoU 0.45) is an
  **image bounding-box** dedup stage that removes overlapping detected image regions before
  captioning/embedding (DedupParams). It does **not** skip already-processed documents and gives
  **no** cross-run idempotency. (Confirmed: dry-run `dedup` is `null` by default; the help text
  describes a bbox IoU threshold.)
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
  (`--dry-run` shows `branch_summary: pdf:N` for a folder of N PDFs.)
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. The CLI exposes only the **summary** row count — row-level
  `(pdf_basename, page_number)` uniqueness is inspected directly against the LanceDB table in a
  live run (the idempotency probe).

**The gap, stated plainly.** NRL 26.05 has **no native resume/checkpoint API and no source-hash
idempotency** on the ingest path. Achieving "resume without re-processing" and "idempotent
re-ingest" requires **skill-layer state tracking** (remember indexed basenames) or a custom
**`.filter()` pre-stage** that consults LanceDB for known `(pdf_basename, page_number)` before
extraction. Neither ships today.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity — each documents one facet of the gap)

| # | id | Facet of the user task | What it adds / which gap it exposes | Subcommands |
|---|---|---|---|---|
| 1 | `ingest-ri-001` | establish the index | **Baseline.** Ingest the 3-PDF folder once; record clean row count **N0**; confirm queryable. *No gap asserted yet.* | `ingest`, `query` |
| 2 | `ingest-ri-002` | "restart without re-processing" | **Re-ingest, default `--overwrite`.** Adds a 2nd identical run → whole table rebuilt, all 3 files reprocessed. **Resume / re-work gap.** | `ingest` (×2) |
| 3 | `ingest-ri-003` | "doesn't duplicate content" | **Re-ingest with `--append`.** Same inputs → rows ~double; `(basename,page)` keys appear twice. **Idempotency gap.** | `ingest`, `ingest --append` |
| 4 | `ingest-ri-004` | "add a new file to my index" | **Incremental add.** 3 indexed + new `test.pdf` → no clean add path (overwrite redoes all; append-folder duplicates; append-single is a manual workaround). **No clean-add gap.** | `ingest --append`, `ingest --overwrite`, `query` |
| 5 | `ingest-ri-005` | "resume after a kill" | **Acceptance gate.** Kill mid-ingest, restart → no `--resume`; restart rebuilds (overwrite) or duplicates (append). Composes idempotency + not-corrupted checks. **Headline resume gap.** | `ingest` |

The ladder: T1 pins the baseline; T2 runs the same folder again (resume/re-work gap); T3 switches
to append (idempotency gap); T4 introduces a genuinely new file (no clean-add gap); T5 adds an
interruption (the composed resume gap and acceptance gate).

---

### T1 — `ingest-ri-001` · baseline ingest establishes the index  *(complexity 1)*
- **Facet:** establish the "index I built earlier."
- **Data:** `cases/ingest-ri-001/data/{woods_frost,table_test,multimodal_test}.pdf`.
- **Expected:** `RETRIEVER ingest data/` → `Ingested 3 file(s) -> N0 row(s)` (`branch_summary
  pdf:3`, default `--overwrite`); `RETRIEVER query "How far does the speaker still have to go
  before sleeping?" --top-k 5` → **"miles to go before I sleep"**, cite `woods_frost.pdf` p1.
- **Gap:** none — this rung only fixes the baseline **N0** the others measure against.

### T2 — `ingest-ri-002` · re-ingest same folder, default `--overwrite`  *(complexity 2)*
- **Facet:** "re-run the ingest… pick up where it left off."
- **Data:** same 3 PDFs.
- **Adds:** a second identical `RETRIEVER ingest data/`. **Gap exposed:** default `--overwrite`
  **rebuilds the whole table**; all 3 files are re-extracted/re-embedded; row count returns to
  **N0** and second-run wall-clock ≈ first-run (no work saved). There is no skip and no resume.
- **Pass = gap surfaced:** agent does **not** invent a resume/skip flag and **honestly reports**
  the full re-work. **Fail:** agent claims it "picked up where it left off."

### T3 — `ingest-ri-003` · re-ingest with `--append` → duplicates  *(complexity 3)*
- **Facet:** "re-ingest… make sure nothing gets duplicated."
- **Data:** same 3 PDFs.
- **Adds:** `RETRIEVER ingest data/ --append` over the **same** inputs. **Gap exposed:** `--append`
  adds rows with **no dedup**; total rows grow **N0 → ~2·N0**; every `(pdf_basename, page_number)`
  that was count=1 is now count=2. Re-ingest is **not idempotent**.
- **Pass = gap surfaced:** duplication is verifiable and the agent reports NRL does not dedupe on
  re-ingest. **Fail:** agent claims the re-ingest was idempotent / "nothing duplicated."

### T4 — `ingest-ri-004` · incremental add of one new file  *(complexity 4)*
- **Facet:** "add just the new one — don't reprocess or duplicate the others."
- **Data:** the 3 originals **plus** held-back **`test.pdf`** (`cases/ingest-ri-004/data/`).
- **Adds:** a genuinely new file in the folder. **Gap exposed — no clean add path:**
  - **Option A** `RETRIEVER ingest data/test.pdf --append` → adds only the new rows, originals
    untouched — **but only because the agent manually isolates `test.pdf`** (the CLI can't
    auto-detect which files are already indexed).
  - **Option B** `RETRIEVER ingest data/` (`--overwrite`) → **reprocesses all 4** (re-work gap).
  - **Option C** `RETRIEVER ingest data/ --append` → adds the new file **but duplicates** the 3
    originals (idempotency gap).
- **Pass = gap surfaced:** agent shows there is no native auto-incremental-add; the safe path
  (Option A) is a **manual** workaround that depends on the human knowing which file is new.
  **Fail:** agent claims NRL automatically detected and added only the new file.

### T5 — `ingest-ri-005` · acceptance gate: resume after a kill  *(complexity 5)*
- **Facet:** "killed halfway through — restart, pick up where it left off, don't corrupt it."
- **Data:** same 3 PDFs.
- **Adds (the gate):** a **mid-ingest interruption**. **Headline gap exposed:** there is **no
  `--resume`/`--checkpoint`** flag — the only restart is re-running `RETRIEVER ingest data/`,
  which under default `--overwrite` **rebuilds all 3 from scratch** (no work picked up), and under
  `--append` would **duplicate** whatever the killed run had committed. Composes:
  - **idempotency probe** — after a clean `--overwrite` restart, every `(pdf_basename,page_number)`
    is count=1 (correctness bought by **full re-work**, not resume);
  - **not-corrupted** — the table is queryable end-to-end after the overwrite restart, but there is
    **no partial-resume** that preserves prior work.
- **Pass = gap surfaced:** agent confirms no `--resume` exists, that restart re-does everything (or
  duplicates under append), and points at the skill-layer/`.filter()` workaround. **Fail:** agent
  claims a native resume picked up where the kill left off.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this **gap suite**:
**(a)** the agent uses the **real** flags (`--overwrite` default vs `--append`) and **never
fabricates** `--resume`/`--checkpoint`/`--skip-existing`/a dedup-on-reingest flag;
**(b)** re-ingest under `--overwrite` is shown to **rebuild the whole table** (resume/re-work gap);
**(c)** re-ingest under `--append` is shown to **duplicate** `(pdf_basename, page_number)` rows
(idempotency gap); **(d)** the incremental-add has **no native clean path** (manual single-file
`--append` is the only safe workaround); **(e)** the agent **honestly characterizes** the gap and
points at the skill-layer state / `.filter()` pre-stage as the real fix. The suite **fails** on any
**false claim** that resume/idempotency works natively.

---

## Note on live runs (not run live)

⚠️ **Not run live.** Expected outputs are grounded in the CLI **source + `--help` flag scan +
`--dry-run`** (the resolved plan shows `vdb.overwrite=true` by default, flipping to `false` under
`--append`; no `--resume`/`--checkpoint`/source-hash flag exists), **not** yet executed live. A
live run would capture: the concrete baseline row count **N0**, the second-run wall-clock vs the
first (confirming **no** time saved under `--overwrite`), the exact post-`--append` total
(≈ **2·N0**) and the duplicated `(pdf_basename, page_number)` keys, the incremental-add row deltas
under Options A/B/C, and the post-kill restart behavior (full rebuild under `--overwrite`, table
still queryable). Live ingest may hit billable hosted extraction/embedding endpoints or need a GPU
per the SETUP suites. **This suite is expected to demonstrate the ABSENCE of native
resume/idempotency — it tracks a documented NRL 26.05 gap and is not expected to "pass" a feature
that does not yet ship.**
