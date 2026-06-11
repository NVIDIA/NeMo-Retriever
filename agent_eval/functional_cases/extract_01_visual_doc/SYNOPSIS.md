# Synopsis — EXTRACT: standalone visual-document extraction (PDF / DOCX / PPTX)

**What user task this covers.** A user has a visual document — a PDF, a Word file, or a
PowerPoint deck — and wants NeMo Retriever to **pull everything out of it**: the body text,
the tables (as real rows and columns), the charts, infographics, and pictures. Success
means: feed in **one file**, get back a **non-zero** set of chunks, and be able to prove
that the tables and charts actually came through — not a silent text-only fallback that
quietly drops them.

**How we test it.** Five agent prompts, each handing the agent one small real document and a
question that can only be answered if the right modality was extracted. We check the agent
drives the `retriever` CLI correctly: `retriever ingest <file>` (the format is auto-detected
— there is **no** `--input-type` flag; tables and charts are on by default; we add
`--use-table-structure` to materialize true row/col tables), then a `retriever query` with
`--content-types text,table` (and `chart`/`image` at the top rungs) to confirm the
extracted rows exist and answer correctly. This row is EXTRACT, not full ingest, but the
CLI's `ingest` *is* the extraction entrypoint, so the query simply proves the chunks landed.

**The five tests, simplest to hardest:**

1. **Baseline PDF extraction** — extract a multi-page visual PDF and get a grounded answer
   from its body text. Proves the file ingests to a non-zero chunk count. (Ground truth: the
   doc's own conclusion says it should yield 2 tables, 2 charts, 3 bullet points.)
2. **Table cell** — pull a specific value out of a table (James, 2019 → **978**), forcing the
   table-structure model and a table-typed query. Catches a silent text-only fallback.
3. **DOCX, same content** — hand over the Word version; it is auto-detected, converted to PDF
   by libreoffice, and run through the same pipeline (Giraffe → Driving a car → At the beach).
   The only new variable is the input format (and the libreoffice prereq).
4. **PPTX slide deck + chart** — a QBR deck whose key numbers live in a chart: FY27 Q1 revenue
   projection (**$487M**, text slide) and the region with the largest pipeline (**Americas at
   $1.2B**, chart slide). Adds chart-region extraction.
5. **Acceptance gate** — one visual-rich PDF where text, a table cell, a chart, and a picture
   must *all* come through, each proven by its own query with page references. This is the
   test the others build up to.

**Why this order.** Each rung adds exactly one new thing: first "does a visual doc extract to
chunks at all," then the table modality, then a new input format (DOCX), then another format
plus the chart modality (PPTX), then everything composed — all four modalities from one
document with citations.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run`) and the skill's
install/ingest references; **not yet run live**. Live runs need libreoffice (for DOCX/PPTX)
and a reachable visual-extraction backend (hosted `ai.api.nvidia.com` with an API key, or a
local GPU). See `README.md` for the full spec and `cases.json` for the machine-gradable
definitions.
