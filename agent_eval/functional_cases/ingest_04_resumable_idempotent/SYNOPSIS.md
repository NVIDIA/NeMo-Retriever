# Synopsis — INGEST: resumable & idempotent ingest  ⚠️ DOCUMENTED NRL 26.05 GAP

> **READ THIS FIRST (stakeholder note).** This suite is a **known-gap tracking suite**, not a
> normal feature test. The behavior the user wants — *resume a killed ingest without
> re-processing finished files, and re-ingest a folder without duplicating or corrupting the
> index* — is **NOT natively supported in NeMo Retriever Library 26.05**. These five tests
> exist to **surface that gap cleanly and honestly** and to **track it until it is closed**.
> A "pass" here means *the gap was correctly demonstrated*. A "fail" means an agent (or a
> future build) **falsely claimed resume/idempotency works** when it does not.

**What user task this covers.** A user has already built an index and now wants to (a) **resume**
an ingest that was killed halfway, (b) **re-ingest** the same folder safely (no duplicates, no
corruption), and (c) **add one new file** to the existing index without redoing the rest.
JTBD priority: **P1, escalating to P0.**

**The reality in 26.05 (verified directly against the CLI).**
- `retriever ingest` defaults to **`--overwrite`** — every run **rebuilds the LanceDB table from
  scratch**. Re-ingesting a folder re-extracts and re-embeds *every* file. There is **no
  incremental skip**.
- `--append` adds rows **without any duplicate check**. The CLI help says it verbatim:
  *"rerunning the same inputs in append mode creates duplicates."* So re-ingest is **not
  idempotent**.
- There is **no `--resume`, `--checkpoint`, `--skip-existing`, `--incremental`, or source-hash
  flag** anywhere in `retriever ingest --help` (the full flag list was scanned). A killed ingest
  can only be restarted by re-running the command — which redoes everything.
- `--dedup` does **not** help: it is an *image bounding-box* IoU dedup stage (for overlapping
  detected picture regions before captioning), **not** document-level dedup or a "skip already
  processed" mechanism.
- Closing the gap would require **skill-layer state tracking** (remember indexed basenames) or a
  custom **`.filter()` pre-stage** that consults LanceDB for known `(pdf_basename, page_number)`
  before extraction. Neither ships today.

**How we test it.** Five agent prompts paraphrasing the real seed queries ("I killed the ingest
halfway — restart without re-processing"; "re-ingest this folder, I dropped new files in"; "add
test.pdf to the index I built earlier"). Each hands the agent a small folder of 3–4 real PDFs and
checks that the agent (1) uses the **real** flags, (2) does **not** invent a `--resume`/dedup
flag, and (3) **honestly reports** the resulting re-work or duplication as the gap.

**The five tests, simplest → hardest (each documents one facet of the gap):**

1. **Baseline** — ingest the 3-PDF folder once; record the clean row count **N0** and confirm the
   index is queryable. *(No gap asserted — this just pins the baseline the others measure against.)*
2. **Re-ingest, default `--overwrite`** — run the same folder again. **Gap:** the whole table is
   rebuilt; all 3 files are reprocessed; no skip, no resume, no time saved.
3. **Re-ingest, `--append`** — the agent tries to "add without rebuilding." **Gap:** rows are
   added with no dedup; total rows ~**double**; every `(pdf_basename, page_number)` now appears
   twice. Re-ingest is **not idempotent**.
4. **Incremental add of one new file** — 3 indexed PDFs + a new `test.pdf`. **Gap:** there is **no
   clean add path** — `--overwrite` redoes all 4, `--append` on the folder duplicates the 3
   originals, and the only safe option (`--append` just the new file) works **only because a human
   manually identifies the new file** (the CLI can't auto-detect it).
5. **Acceptance gate — resume after a kill** — kill the ingest mid-run, then restart. **Headline
   gap:** no `--resume` exists; the only restart re-runs the command, which rebuilds everything
   (overwrite) or duplicates already-written rows (append). Correctness is bought by full re-work,
   never by resuming. Composes the idempotency probe and the "index queryable / not corrupted
   mid-write" check.

**Why this order.** Each rung adds exactly one dimension: a clean baseline → a second identical
run (resume/re-work gap) → switching to append (idempotency gap) → introducing a genuinely new
file (no clean-add gap) → an interruption (the composed resume gap, the acceptance gate).

**Status.** This row is an **acknowledged NRL 26.05 gap**. The tests are authored and grounded in
the real CLI (`--help` flag scan + `--dry-run` resolved plan showing `vdb.overwrite=true` by
default, flipping to `false` under `--append`). They are **not yet run live**. They are expected
to **demonstrate the absence** of resume/idempotency, and to keep tracking it until a future
build (or a skill-layer state/`.filter()` workaround) closes it. See `README.md` for the full spec
and `cases.json` for the machine-gradable definitions.
