# Functional test suite — EXTRACT: standalone visual-document extraction (PDF / DOCX / PPTX)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **EXTRACT** job: given a visual document (PDF, DOCX, or PPTX), pull
out **text, tables, charts, infographics, and pictures** — and prove the extraction
actually produced chunks of each modality (not a silent text-only fallback).

---

## The user task under test

> **JTBD: EXTRACT — P0.** "Standalone visual-document extraction (PDF / DOCX / PPTX) —
> text, tables, charts, infographics, pictures."

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: ingest **1 file** and get back a **non-zero** number of chunks; the visual pipeline fires; tables surface as discrete `content_type=table` rows with row/col structure; picture/chart regions are detected and captioned or stored as image content; **no** silent text-only fallback when tables/charts are present |
| Time | **medium — ≤ 2 min** per case (single small file ingest + one query) |
| Trigger rate | ≥ 95% — an "extract / load this document and answer from it" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <file>` (format auto-detected; tables/charts on by default; `--use-table-structure` to materialize row/col tables) then `retriever query … --content-types text,table` to confirm table rows. **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"What was Apple's net revenue in fiscal year 2022?"* (from a 10-K PDF)
- *"What does <doc>.pdf say about climate change risk? I want exact page references."*
- *"From aurora_qbr_q1.pptx, what's the FY27 Q1 revenue projection and which region has the largest pipeline?"*

This row is **EXTRACT, not full INGEST** — the user only wants extraction. But the shipped
CLI's `retriever ingest` *is* the extraction+index entrypoint, so each test runs an
`ingest` (extraction producing non-zero chunks) and then a `query` purely to **prove the
extracted rows exist** — including the table/chart/picture modalities.

---

## How the CLI behaves for this task (grounded in `--dry-run` + skill references)

Verified with `retriever ingest <file> --dry-run` (offline, no network) and the skill's
`references/install.md` + `references/cli/ingest.md`:

- **Format auto-detection.** `retriever ingest <file>` resolves `.pdf` / `.docx` / `.pptx`
  from the extension. There is **no `--input-type` flag**.
- **Visual extraction is ON by default.** The resolved `extract` plan shows
  `extract_text`, `extract_tables`, `extract_charts`, `extract_images`,
  `extract_infographics` all **true**, with `use_page_elements: true`, `ocr_version: v2`,
  `method: pdfium`, `dpi: 200`. The stack is **nemotron-page-elements-v3 → nemotron-ocr-v2**
  by default.
- **Table row/col structure is OFF by default.** `use_table_structure` defaults to
  **false** (`table_output_format: pseudo_markdown`). Passing **`--use-table-structure`**
  flips it true and switches `table_output_format` to `markdown`, materializing
  **nemotron-table-structure-v1** so table cells are queryable as discrete
  `content_type=table` rows. The table-bearing rungs (2–5) use this flag.
- **DOCX / PPTX route through the PDF branch via libreoffice.** A single `.docx` or
  `.pptx` `--dry-run` reports `branch_summary: pdf:1` with one branch
  `(family=pdf, extraction_mode=pdf)` — i.e. libreoffice converts the office file to PDF
  and the **same** pdf extraction pipeline runs. **Prereq:** `references/install.md` lists
  `.docx`/`.pptx` → **libreoffice (host pkg)** → `sudo apt-get install -y libreoffice`.
  Without it the conversion step fails and no chunks are produced.
- **Where the visual NIMs run is host-dependent (and out of scope here).** Page-elements /
  OCR / table-structure default to hosted `ai.api.nvidia.com` (key from `NVIDIA_API_KEY`)
  on a no-GPU box, or run on a local GPU when configured (see the two SETUP suites). This
  EXTRACT suite is **host-agnostic**: the assertion is that the visual stack **fired and
  produced table/picture rows**, not where it ran.
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. (`--content-types text,table,chart,image,infographic`
  filters to typed hits.)

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `extract-vd-001` | **Baseline.** One multi-page visual PDF → non-zero chunks + grounded body-text answer. The EXTRACT floor: file ingests, rows produced. | `ingest`, `query` |
| 2 | `extract-vd-002` | **Table cell.** Answer must come from a TABLE cell → `--use-table-structure` (table-structure-v1) + `--content-types text,table`. | `ingest --use-table-structure`, `query --content-types text,table` |
| 3 | `extract-vd-003` | **DOCX format routing.** Same content as a `.docx` → libreoffice→pdf branch (`pdf:1`); libreoffice prereq. | `ingest --use-table-structure`, `query --content-types text,table` |
| 4 | `extract-vd-004` | **PPTX + chart.** A slide deck whose key figures live in a CHART (slide 3) plus a text KPI slide → chart-region extraction. | `ingest --use-table-structure`, `query --content-types text,table,chart` |
| 5 | `extract-vd-005` | **Acceptance gate.** One visual-rich PDF where text + table + chart + picture ALL land and are proven by per-modality queries with page/bbox citations. | `ingest --use-table-structure`, 4× `query` (per modality) |

The ladder: T1 proves a visual doc ingests to non-zero chunks; T2 adds the table modality
(a specific cell as a typed row); T3 changes only the input format (DOCX→pdf branch); T4
changes the format again (PPTX) and adds the chart modality; T5 composes everything —
text + table + chart + picture from one document, each provable with a citation.

---

### T1 — `extract-vd-001` · baseline visual-PDF extraction  *(complexity 1)*
- **Satisfies:** EXTRACT operational-pass core (ingest 1 file → non-zero chunks), simplest form.
- **Data:** `cases/extract-vd-001/data/multimodal_test.pdf` (3pg "TestingDocument"; 2 tables,
  2 charts, a picture, 3 bullet points).
- **Expected:** `RETRIEVER ingest data/multimodal_test.pdf` → non-zero rows (`branch_summary
  pdf:1`); `RETRIEVER query "How many tables, charts, and bullet points should be extracted?"
  --top-k 5` → the **conclusion (p3)** says "2 tables, 2 charts, and 3 bullet points".

### T2 — `extract-vd-002` · table cell as a typed row  *(complexity 2)*
- **Satisfies:** "tables surface as discrete `content_type=table` rows with row/col structure".
- **Data:** `cases/extract-vd-002/data/table_test.pdf` (Year × {Bill,Amy,James,Ted,Susan}).
- **Adds:** `--use-table-structure` (nemotron-table-structure-v1) + a `--content-types
  text,table` query. **Ground truth: James 2019 = 978**, cite `table_test.pdf` p1. Fails if
  the table is flattened to plain text and 978 cannot be recovered as a table cell.

### T3 — `extract-vd-003` · DOCX format routing  *(complexity 3)*
- **Satisfies:** the **DOCX** modality of the EXTRACT task.
- **Data:** `cases/extract-vd-003/data/multimodal_test.docx` (same content as the multimodal PDF).
- **Adds:** a non-PDF source. `.docx` is auto-detected, libreoffice converts it to PDF, and
  the same pdf branch runs (`--dry-run` → `branch_summary pdf:1`). **Ground truth: Table 1
  Giraffe → Driving a car → At the beach.** Prereq: libreoffice host package.

### T4 — `extract-vd-004` · PPTX slide deck + chart  *(complexity 4)*
- **Satisfies:** the **PPTX** modality and the **charts/infographics** clause.
- **Data:** `cases/extract-vd-004/data/aurora_qbr_q1.pptx` (3-slide QBR deck).
- **Adds:** a slide-deck source whose key numbers live in a **chart** (slide 3 "Pipeline by
  Region") alongside a text KPI slide (slide 2 "Revenue Trajectory"). **Ground truth: FY27
  Q1 revenue projection = $487M (+18% YoY)** (slide 2 text); **largest pipeline = Americas
  at $1,200M** (slide 3 chart: Americas 1200 vs EMEA 480, APAC 420). Fails if the chart is
  dropped and the Americas=1200 figure is unrecoverable.

### T5 — `extract-vd-005` · acceptance gate, all four modalities  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the EXTRACT row.
- **Data:** `cases/extract-vd-005/data/multimodal_test.pdf` (text + 2 tables + 2 charts + picture).
- **Expected:** one `ingest … --use-table-structure`, then four typed queries proving each
  modality landed:
  - **text** — Introduction body text (p1);
  - **table** — Giraffe row, `content_type=table` (Driving a car / At the beach, p1);
  - **chart** — Chart 1 "gadgets and (fictitious) costs" (p1);
  - **picture** — the "high-quality picture of some shapes" region (referenced p2).
- **Adds (the gate):** every hit must carry non-null `source` + `page_number` (and
  `bbox_xyxy_norm` where applicable), and there must be **no silent text-only fallback** that
  loses the table/chart/picture rows.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** ingest produces a **non-zero** chunk count for the file; **(b)** the visual stack
fired (page-elements → OCR → table-structure) — table cells are recoverable as
`content_type=table` rows, charts/pictures are detected; **(c)** for DOCX/PPTX, the file is
auto-detected and routed through the libreoffice→pdf branch (`branch_summary pdf:1`);
**(d)** answers match the grounded values (978; Giraffe/Driving a car/At the beach; $487M /
Americas; 2 tables/2 charts/3 bullets) with the right page citation; **(e)** no
`--input-type` flag is used (it does not exist).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`** and
the skill's install/ingest references; the suite has **not** been run live yet. A live run
requires the visual-extraction prerequisites: **libreoffice** (host pkg) for the DOCX/PPTX
rungs, and a reachable extraction backend for page-elements/OCR/table-structure — either the
**hosted** `ai.api.nvidia.com` endpoints (needs `NVIDIA_API_KEY`, makes small billable
calls) or a **local GPU** configured per the SETUP-GPU suite. Running live would capture the
real per-file row counts (by content type), table-structure cell fidelity, chart/picture
detection rates, latencies, and token baselines.
