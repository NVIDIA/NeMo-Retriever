# Functional test suite — INGEST: mixed-format, multi-modal folder in one invocation

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **INGEST** job: point the skill at a **folder** of mixed-format,
multi-modal documents and have it ingest **everything in a single invocation** — each file
auto-routed to its modality-specific extractor by extension, unsupported files skipped (not
errored), into one combined, queryable LanceDB index.

---

## The user task under test

> **JTBD: INGEST — P0.** "Ingest a folder of mixed-format, multi-modal documents in a
> single skill invocation."

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: a **single** `retriever ingest <folder>/` auto-detects each file's format and routes it to its modality-specific extractor by extension; **unsupported extensions are skipped** while supported files still index (the folder ingest does **not** error out); the run returns an **end-of-run summary** (total files, by_type / `branch_summary`, total rows, index path); the LanceDB index lands at the expected uri/table with the **canonical schema** and is **loadable in a fresh process** |
| Time | **slow — ≤ 10 min** on a small (3–6 file) mixed folder (the spec's ≤ 10 min on a 50-doc folder, scaled down to a functional fixture) |
| Trigger rate | ≥ 95% — a "load / ingest / process this whole folder and let me query it" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — a **single** `retriever ingest <folder>/` on the directory (no per-file loop, **no `--input-type` flag** — it does not exist), then `retriever query` to prove the combined index |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"Load this whole folder and let me query it later — it has PDFs, scans, slide decks, the works."*
- *"Make this mixed folder of docs searchable."*
- *"Process everything in here and tell me when it's ready."*

---

## How the CLI behaves for this task (grounded in `--dry-run` + source)

Verified with `retriever ingest <folder>/ --dry-run` (offline, no network) and the source in
`src/nemo_retriever/cli/sdk_workflow.py`, `src/nemo_retriever/common/input_files.py`, and
`src/nemo_retriever/common/vdb/lancedb_schema.py`:

- **Single invocation over a directory.** `retriever ingest` takes `DOCUMENTS...` =
  "One or more files, directories, or globs. Supported file types are detected
  automatically." A folder arg is expanded by `resolve_input_files` (recursive glob over the
  supported-extension patterns) — **one call ingests the whole folder; no per-file loop**.
- **No `--input-type` flag.** Format is auto-detected from the extension. Auto-detected
  extensions: `.pdf .docx .pptx .html .txt .png .jpg .jpeg .bmp .tif .tiff .svg .wav .mp3
  .m4a .mp4 .mkv .mov`.
- **Routing by extension into modality branches.** The resolved plan reports a
  `branch_summary` and a `branches` list, one branch per family:
  - **pdf** — `.pdf`, plus `.docx`/`.pptx` (converted to PDF by libreoffice, then the pdf branch);
  - **image** — `.png/.jpg/.jpeg/.bmp/.tif/.tiff/.svg`;
  - **html** — `.html`; **text** — `.txt`;
  - **audio** — `.wav/.mp3/.m4a`; **video** — `.mp4/.mkv/.mov`.

  Grounded examples (real `--dry-run` on this suite's fixtures):
  - 2 PDFs → `branch_summary: "pdf:2"`
  - PDF + DOCX + HTML → `branch_summary: "pdf:2, html:1"` (docx joins the pdf branch)
  - PDF + DOCX + PNG + HTML → `branch_summary: "pdf:2, image:1, html:1"`
- **Unsupported files are skipped, not errored — for a folder.** A **directory** ingest
  globs only supported extensions, so an unsupported member (e.g. `.xlsx`) is **excluded
  from the resolved plan** before validation and the folder ingest **completes normally**.
  Verified: a folder of `fy27_bookings.xlsx` + `multimodal_test.pdf` + `test.html` resolves
  to `branch_summary: "pdf:1, html:1"` — the xlsx is dropped, no error.
  **Caveat:** the shipped CLI skips by *exclusion* (the file never enters the plan); it does
  **not** print an explicit per-file `skipped: <reason>` log line in `--dry-run`. By
  contrast, pointing ingest **directly at a lone `.xlsx`** raises
  `Unsupported input file type(s) for retriever ingest: …`. So the validation-path phrase
  "logged + skipped with a per-file reason" is *aspirational*; the shipped-CLI assertion is
  "supported files indexed, unsupported skipped, **no error on the folder**".
- **Canonical LanceDB schema** (`lancedb_schema()`): `vector` (fixed-size list<float32>,
  dim 2048), `pdf_page`, `filename`, `pdf_basename`, `page_number` (int32), `source`
  (JSON `{source_id}`), `source_id`, `path`, `text`, `metadata` (JSON), `stored_image_uri`,
  `content_type`, `bbox_xyxy_norm` (JSON). All twelve fields named in the validation path
  are present (the table additionally carries `pdf_basename`). The table is created with
  `db.create_table(name, data=rows, schema=schema, …)`, so it is **loadable in a fresh
  process** via `lancedb.connect(uri).open_table(name)`.
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
  (`branch_summary` appears in `--dry-run`.)
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. (`content_type`/`bbox_xyxy_norm`/`stored_image_uri`
  live in the table, not in the CLI hit shape.)

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `ingest-mix-001` | **Baseline.** One ingest call over a folder of 2 same-format PDFs → combined index. The INGEST floor: a directory is ingested in one call (no per-file loop). | `ingest`, `query` |
| 2 | `ingest-mix-002` | **Mixed formats.** PDF + DOCX + HTML routed by extension in the same call (`pdf:2, html:1`); libreoffice prereq for the docx. | `ingest --use-table-structure`, `query` |
| 3 | `ingest-mix-003` | **Multi-MODAL.** Add a raster image/scan (`.png`) → 3 branch families at once (`pdf:2, image:1, html:1`); the png goes through the image branch, not pdf. | `ingest --dry-run`, `ingest --use-table-structure`, `query` |
| 4 | `ingest-mix-004` | **Unsupported skip.** A folder with a `.xlsx` → it's skipped (not errored), supported `.pdf`/`.html` still index (`pdf:1, html:1`). | `ingest --dry-run`, `ingest`, `query` |
| 5 | `ingest-mix-005` | **Acceptance gate.** One-shot mixed multi-modal ingest into a named index → end-of-run summary + canonical schema + fresh-load + a query spanning TWO source files returns a hit from each. | `ingest --use-table-structure --table-name`, `query --table-name` |

The ladder: T1 proves a folder ingests in one call; T2 makes the folder multi-**format**
(routing by extension); T3 makes it multi-**modal** (an image branch alongside the docs);
T4 adds an **unsupported** member to prove graceful skip; T5 composes everything — one-shot
ingest, the summary, the canonical schema, a fresh-process load, and a cross-file query.

---

### T1 — `ingest-mix-001` · baseline folder ingest  *(complexity 1)*
- **Satisfies:** the INGEST core — a single `retriever ingest <folder>/` over a directory
  produces one combined queryable index.
- **Data:** `cases/ingest-mix-001/data/` — `multimodal_test.pdf`, `table_test.pdf`.
- **Expected:** `RETRIEVER ingest data/` → `Ingested 2 file(s) -> N row(s)` (`--dry-run`
  `branch_summary pdf:2`); `RETRIEVER query "James value in 2019" --top-k 5 --content-types
  text,table` → **James 2019 = 978**, cite `table_test.pdf` p1.

### T2 — `ingest-mix-002` · mixed formats routed by extension  *(complexity 2)*
- **Satisfies:** "mixed-format folder, each file routed to its modality-specific extractor
  by extension".
- **Data:** `cases/ingest-mix-002/data/` — `multimodal_test.pdf`, `multimodal_test.docx`,
  `test.html`.
- **Adds:** a non-same-format folder. `--dry-run` → `branch_summary pdf:2, html:1` (docx →
  libreoffice → pdf branch; html → html branch). **Ground truth:** HTML starts with
  **"My First Heading"** (`test.html`); Table 1 **Giraffe → Driving a car → At the beach**.

### T3 — `ingest-mix-003` · true multi-modal folder  *(complexity 3)*
- **Satisfies:** the **multi-modal** part of the task (not just multi-format).
- **Data:** `cases/ingest-mix-003/data/` — `multimodal_test.pdf`, `multimodal_test.docx`,
  `multimodal_test.png`, `test.html`.
- **Adds:** a raster image/scan member. `--dry-run` → `branch_summary pdf:2, image:1,
  html:1`; the `.png` lands in a `(family=image, extraction_mode=image)` branch, **not** the
  pdf branch. One call fans out across 3 branch families. **Ground truth:** Table 1 Giraffe
  → Driving a car → At the beach.

### T4 — `ingest-mix-004` · unsupported file skipped, not errored  *(complexity 4)*
- **Satisfies:** "unsupported extensions are skipped (not errored out), supported files still index".
- **Data:** `cases/ingest-mix-004/data/` — `multimodal_test.pdf`, `test.html`,
  `fy27_bookings.xlsx` (**unsupported**).
- **Adds:** an unsupported member. `--dry-run` → `branch_summary pdf:1, html:1` and a
  resolved `documents` list **without** the xlsx; the folder ingest **does not** raise
  `Unsupported input file type(s)`. **Ground truth:** HTML **"My First Heading"**
  (`test.html`). See the case `known_caveat` for the exclusion-vs-explicit-log nuance.

### T5 — `ingest-mix-005` · acceptance gate  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the INGEST row.
- **Data:** `cases/ingest-mix-005/data/` — `woods_frost.pdf`, `table_test.pdf`,
  `multimodal_test.png`, `test.html` (`branch_summary pdf:2, image:1, html:1`).
- **Expected:** one `RETRIEVER ingest data/ --lancedb-uri lancedb --table-name mixed_smoke
  --use-table-structure`, then queries on the **same** `--table-name mixed_smoke`.
- **Adds (the gate):** (a) an **end-of-run summary** — total_files=4, by_type
  `pdf:2, image:1, html:1`, total_rows non-zero, index_path `lancedb/mixed_smoke`;
  (b) the index lands with the **canonical schema** (all 12 named fields);
  (c) it is **loadable in a fresh process** (`lancedb.connect('lancedb').open_table('mixed_smoke')`);
  (d) a query spanning **two** source files returns a hit from each — **New Hampshire (1923)**
  citing `woods_frost.pdf` p2 **and** **James 2019 = 978** citing `table_test.pdf` p1;
  (e) completion within the **≤ 10 min** slow SLA.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** a **single** `retriever ingest <folder>/` call ingests the whole directory (no
per-file loop, no `--input-type`); **(b)** each file is routed to its modality branch by
extension — the `--dry-run` `branch_summary` matches (`pdf:2` / `pdf:2, html:1` / `pdf:2,
image:1, html:1` / `pdf:1, html:1`); **(c)** an unsupported member (`.xlsx`) is **skipped**
and the folder ingest does **not** error; **(d)** the combined index is queryable and answers
the grounded values (978; "My First Heading"; Giraffe / Driving a car / At the beach; New
Hampshire 1923) with the right citation; **(e)** the acceptance gate returns the end-of-run
summary, lands the canonical schema, loads in a fresh process, and answers across two files.

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`**
(offline), **not** yet run live. A live run requires the visual/multi-modal extraction
prerequisites: **libreoffice** (host pkg, `sudo apt-get install -y libreoffice`) for the
DOCX/PPTX members (rungs 2–3); the `[multimedia]` extra + **ffmpeg** if a "the works" folder
ever includes audio/video members (this suite's fixtures do not, so it is not needed for
these five cases — noted for completeness); and a reachable extraction backend for
page-elements/OCR/table-structure and image extraction — either the **hosted**
`ai.api.nvidia.com` endpoints (needs `NVIDIA_API_KEY`, makes small billable calls) or a
**local GPU** configured per the SETUP-GPU suite. Running live would capture the real
per-branch row counts (by content type), the materialized canonical schema, fresh-load row
totals, and the end-to-end wall-clock latency against the ≤ 10 min SLA.
