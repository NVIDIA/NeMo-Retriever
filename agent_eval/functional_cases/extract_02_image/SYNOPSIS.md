# Synopsis — EXTRACT: standalone image extraction (PNG / JPEG / TIFF / BMP and SVG)

**What user task this covers.** A user hands NeMo Retriever a **standalone image** — a
PNG, JPEG, TIFF, BMP, or SVG — and expects it to become searchable: the library should
ingest the file and return non-zero chunks, with the text recovered by **OCR** and each
chunk tagged with where it came from (its source file, a bounding box, and a page number).
A standalone image is treated as a single page, so the page number is always 1. Success is
simple and binary: ingest one image, get back chunks with non-empty text for legible pages.

**How we test it.** Five agent prompts, each handing the agent one image (or, at the top, a
folder of images) and checking that the agent drives the `retriever` CLI correctly: it runs
`ingest` then `query`, lets the format be **auto-detected from the file extension** (there
is no `--input-type` flag), routes the image to the OCR path (`nemotron-ocr-v2`, default
multilingual), and returns hits whose text is non-empty and whose lineage (source + bounding
box + page 1) is intact.

**The five tests, simplest to hardest:**

1. **Single PNG** — ingest one PNG and show the OCR text. The baseline: non-zero rows, the
   image branch, page 1, lineage present.
2. **JPEG** — the same page as a JPEG routes identically; the agent must not special-case
   the format.
3. **BMP** — a third raster format (the large ~8 MB copy, used once), with an explicit check
   that every chunk carries its source and bounding box and is page 1.
4. **SVG** — the one format that behaves differently under the hood: the CLI **rasterizes**
   the SVG (via the optional `cairosvg` dependency) and then OCRs the bitmap, just like a
   raster image. The agent must set up that dependency.
5. **Acceptance** — a folder of **mixed** image formats (PNG, JPEG, TIFF, and a real scanned
   form) ingested in one call, each producing chunks with proper lineage, then a query that
   surfaces a **specific** image's text: the date on the scanned form.

**Why this order.** Each rung adds exactly one thing: rung 1 establishes the happy path;
rung 2 proves a different raster format routes the same way; rung 3 adds a third format and
pins the lineage assertion; rung 4 takes on SVG — the only format with distinct internal
handling; rung 5 composes folder ingest, a format mix, and grounded multi-source retrieval
into the real pass/fail gate the others build toward.

**Important correction (SVG).** The original brief assumed SVG would be read as
vector/XML text without rasterizing. The **shipped CLI does the opposite**: it rasterizes
the SVG with `cairosvg` and OCRs the resulting bitmap — exactly like a raster image — and
the catalog SVG fixture is itself just a wrapper around an embedded JPEG with no XML text at
all. This suite is authored to the **verified real behavior**, and the SVG test makes that
rasterize-then-OCR distinction explicit (see `README.md`).

**Status.** Tests are authored and grounded in the real CLI source and offline `--dry-run`
plans; **not yet run live** (live runs may hit billable hosted OCR endpoints or need a GPU,
and the SVG case needs the optional `cairosvg` dependency installed). See `README.md` for
the full spec and `cases.json` for the machine-gradable definitions.
