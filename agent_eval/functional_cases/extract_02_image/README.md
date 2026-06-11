# Functional test suite — EXTRACT: standalone image extraction (PNG / JPEG / TIFF / BMP and SVG)

An agent-driven functional test suite for the **NeMo Retriever Library skill**, built
against the real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` /
`retriever query`). Each test is a self-contained triple — a prompt, a per-case `data/`
folder, and an expected output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **EXTRACT** task *"Standalone image extraction (PNG / JPEG / TIFF /
BMP and SVG"* — i.e. handing the CLI an image file (or a folder of images) and getting back
non-zero searchable chunks, with text recovered via OCR and per-row lineage.

---

## The user task under test

> **JTBD: EXTRACT.** "Standalone image extraction (PNG / JPEG / TIFF / BMP and SVG)." —
> **P0**

**Validation (operational pass):** ingest 1 image file and return chunks (non-zero). The
image is routed to the **OCR path** (`nemotron-ocr-v2`, `--ocr-lang` default = `multi`);
returned rows have **non-empty text** for legible images; each row carries **`source_id` +
`bbox_xyxy_norm`** (and `page_number`, which is **1** for a standalone image).

**Success criteria for the row:**

| Dimension | Target |
|---|---|
| Quality (operational) | binary pass: ingest 1 image (or a folder) → non-zero rows; legible images yield non-empty OCR text; every row carries `source_id` + `bbox_xyxy_norm`; `page_number == 1` |
| Time | **fast** — single-image ingest+query ≤ 30 s (excluding model cold start) |
| Trigger rate | ≥ 95% — an "ingest this image / make these scans searchable" prompt fires the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <image\|folder>` then `retriever query`; OCR via `nemotron-ocr-v2`; **no `--input-type`** (format auto-detected) |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased):
- *"What date is on this scanned form?"*
- *"What's the main heading on this document image?"*
- *"Make this folder of scanned forms searchable."*

---

## How the CLI handles standalone images (verified against the shipped source)

Grounded by `common/modality/image/load.py`, `common/input_files.py`,
`service/utils/file_type.py`, and `retriever ingest … --dry-run` (offline):

- **Auto-detection.** Image formats are detected from the file extension; there is **no
  `--input-type` flag**. Supported extensions:
  `.png .jpg .jpeg .tiff .tif .bmp .svg` (`SUPPORTED_IMAGE_EXTENSIONS`).
- **Routing.** Each image is routed to the **image branch**. `--dry-run` reports
  `branch_summary: "image:N"`, `extraction_mode: "image"`, `family: "image"`.
- **One image == one page.** `image_bytes_to_pages_df()` builds a **one-row** page
  DataFrame matching the PDF extraction schema: `page_number == 1`, a `page_image`, and one
  full-frame `images` entry with `bbox_xyxy_norm == [0,0,1,1]`, `metadata.needs_ocr_for_text
  == True`. The standard extraction stack (page-elements → **OCR** → table/chart/
  infographic) then runs on it exactly as for a PDF page.
- **OCR.** Default `--dry-run` shows `extract.ocr_version == "v2"` (**nemotron-ocr-v2**) and
  `extract.ocr_lang == None`, which selects the multilingual (`multi`) model. Selectable
  with `--ocr-version v1|v2` and `--ocr-lang <lang>`.
- **Lineage.** Rows carry `source_id` + `bbox_xyxy_norm`; at the CLI query layer each hit is
  `{source, page_number, text}` with `page_number` 1-indexed (== 1 for a standalone image).

### SVG — important correction to the suite brief

The suite brief hypothesised that **SVG** would route to a *vector-graphics path that reads
embedded XML `<text>` directly (not rasterized + OCR'd)*. **The shipped CLI does not do
this.** Verified in `common/modality/image/load.py`:

- `.svg` is in `_SVG_EXTENSIONS` and is handled by `_svg_to_pil()`, which calls
  **`cairosvg.svg2png()`** to **RASTERIZE** the SVG to a PNG bitmap, then runs the **same
  OCR/image extraction path** as raster formats. `--dry-run` on the `.svg` resolves to the
  identical plan as a PNG: `branch_summary "image:1"`, `extraction_mode "image"`,
  `ocr_version "v2"`, default `ocr_lang`. There is **no** `<text>`/XML-text reader in the
  image path.
- The catalog fixture `multimodal_test.svg` is itself just a `<svg>` wrapper around a
  single base64-embedded JPEG `<image>` element with **zero `<text>`/`<tspan>` elements** —
  so even a hypothetical XML-text reader would find nothing and fall back to rasterizing.
- The **genuine distinct dimension** for SVG is therefore the **`cairosvg` rasterization
  dependency** (optional `cairosvg` / `[multimedia]` extra; `cairosvg` install needs
  network, so SVG is unavailable in air-gapped installs), **not** a vector-text path.

Case `img-004` is authored to this **verified shipped behavior**, and tests the
rasterize-then-OCR distinction explicitly.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## Fixtures & ground truth

- `multimodal_test.{png,jpeg,tiff,bmp,svg}` — the **same** source document image (the
  "TestingDocument" multimodal page). PNG/JPEG/TIFF and the SVG source are 849×1090; the
  **BMP is the large one** (~8 MB, 1275×1650 RGBA) and is used **once** (case `img-003`).
- `scanned_form.png` — a real scanned **FUNSD** form (`0060308251`), an *"A. T. Co. Tar &
  Nicotine Change Form"*. Its annotated text includes the heading and a **Date** field with
  value **`7/ 24/ 90`** — the ground-truth answer for the acceptance query.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `img-001` | **Baseline.** One **PNG** → non-zero OCR rows; image branch (`image:1`), `nemotron-ocr-v2`, `page_number==1`, `source_id`+`bbox_xyxy_norm`. | `ingest`, `query` |
| 2 | `img-002` | **Format invariance.** Same page as **JPEG** routes identically — no `--input-type`, same image branch + OCR. | `ingest`, `query` |
| 3 | `img-003` | **Third raster format + explicit lineage.** **BMP** (the large one, used once); assert each row carries `source_id`+`bbox_xyxy_norm`, `page_number==1`. | `ingest`, `query` |
| 4 | `img-004` | **SVG — the distinct format.** Rasterized via **cairosvg** then OCR'd (NOT a vector-XML path); requires the optional `cairosvg`/`[multimedia]` extra. | install `cairosvg`, `ingest`, `query` |
| 5 | `img-005` | **Acceptance gate.** Folder of **mixed formats** (PNG+JPEG+TIFF+scanned form) in one ingest; per-row lineage; query surfaces a **specific** image's text (the form's date). | `ingest`, `query` |

The ladder: T1 proves a single image yields non-zero OCR rows; T2 proves a different raster
format routes the same; T3 adds a third raster format and pins the lineage assertion
(`source_id`+`bbox_xyxy_norm`, `page_number==1`); T4 takes on SVG — the only format that
behaves distinctly (cairosvg rasterization), and corrects the brief's vector-text
hypothesis; T5 composes folder ingest + format mix + grounded multi-source retrieval into
the row's real operational-pass gate.

---

### T1 — `img-001` · single PNG → non-zero OCR rows  *(complexity 1)*
- **Satisfies:** the core task at its simplest (ingest 1 image → non-zero chunks).
- **Data:** `data/multimodal_test.png`.
- **Expected:** `RETRIEVER ingest data/multimodal_test.png` (→ `branch_summary image:1`,
  OCR via `nemotron-ocr-v2`) → `RETRIEVER query "What does this document image say?"
  --top-k 5`. ≥ 1 hit, non-empty text, `page_number == 1`, citing `multimodal_test.png`.

### T2 — `img-002` · JPEG routes the same  *(complexity 2)*
- **Satisfies:** the task for a second raster format; adds **format invariance**.
- **Data:** `data/multimodal_test.jpeg` (same source content as T1).
- **Adds:** the agent must NOT special-case the format (no `--input-type`); image branch +
  OCR behave identically to the PNG.

### T3 — `img-003` · BMP + explicit lineage  *(complexity 3)*
- **Satisfies:** the task for a third raster format; adds the **lineage assertion**.
- **Data:** `data/multimodal_test.bmp` (the ~8 MB / 1275×1650 RGBA copy — used **once**).
- **Adds:** every returned row must carry `source_id` + `bbox_xyxy_norm` and
  `page_number == 1` (standalone image == one page; full-frame bbox `[0,0,1,1]`).

### T4 — `img-004` · SVG rasterized via cairosvg → OCR  *(complexity 4)*
- **Satisfies:** the task for the SVG format; adds the **cairosvg rasterization dependency**.
- **Data:** `data/multimodal_test.svg`.
- **Expected:** ensure `cairosvg` (optional `[multimedia]` extra) is installed →
  `RETRIEVER ingest data/multimodal_test.svg` (rasterized to a bitmap, then the SAME image
  branch + `nemotron-ocr-v2`; `--dry-run` ≡ the PNG plan) → query → non-empty OCR text,
  `page_number == 1`, citing `multimodal_test.svg`.
- **Correction built into the test:** SVG is **rasterized + OCR'd**, *not* read as vector
  XML. The test fails if a run claims it read SVG text from XML without rasterization (that
  path does not exist), and asserts the routing is the image branch (`image:1`).

### T5 — `img-005` · acceptance: mixed-format folder → specific image's date  *(complexity 5)*
- **Satisfies:** the full operational-pass row + the multi-source / "make this folder
  searchable" seed query.
- **Data:** `data/` = `multimodal_test.png` + `multimodal_test.jpeg` +
  `multimodal_test.tiff` + `scanned_form.png` (the FUNSD form).
- **Expected:** one `RETRIEVER ingest data/` (→ `branch_summary image:4`, non-zero rows,
  no per-file loop) → `RETRIEVER query "What date is on the scanned form?" --top-k 5` →
  **`7/24/90`**, citing `scanned_form.png` p1; every hit carries non-null `source` +
  `page_number == 1`.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite: **(a)** images are
auto-detected (no `--input-type`) and routed to the image branch (`branch_summary image:N`);
**(b)** ingest returns **non-zero** rows; **(c)** OCR ran via `nemotron-ocr-v2`; **(d)** rows
carry `source_id` + `bbox_xyxy_norm` and `page_number == 1`; **(e)** for SVG, the path is
**cairosvg rasterize → OCR**, not a vector-XML reader.

**Prerequisites for live runs:** the extraction stack (hosted `ai.api.nvidia.com` NIMs or
local GPU NIMs) must be reachable for OCR. **SVG (case `img-004`) additionally requires the
optional `cairosvg` dependency** (`pip install 'cairosvg>=2.7.0'` or the `[multimedia]`
extra); without it, ingesting a `.svg` raises `ImportError`. cairosvg install needs network,
so SVG is unavailable in air-gapped installs.

**Note on live runs — not run live.** The expected outputs here are grounded in the CLI
source (`common/modality/image/load.py`, `service/utils/file_type.py`,
`common/input_files.py`) and offline `--dry-run` plans (`branch_summary image:N`,
`ocr_version v2`), **not yet executed live**. A live ingest/query may hit billable hosted
OCR endpoints or need a GPU. A live run would capture the real **row counts** per image, the
actual **OCR text** (and confirm it is non-empty for these legible pages and matches across
the PNG/JPEG/TIFF/BMP/SVG copies of the same page), per-row **`source_id`/`bbox_xyxy_norm`**
values, the FUNSD form's recovered **date (`7/24/90`)**, **latencies** (≤ 30 s target), and
**token baselines** — and confirm the cairosvg rasterization path for the SVG case.
