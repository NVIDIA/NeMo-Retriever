# Functional test suite — EXTRACT: image / picture-region captioning

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` / `retriever query`).

This suite covers the **EXTRACT** task of *captioning* visual content: generating
natural-language descriptions of an image — whether a **standalone image file** or a
**picture region detected inside a visual document** (PDF / DOCX / PPTX) — and indexing
those captions so the visual content becomes **retrievable by a plain-text query**.

Each test is a self-contained triple — a prompt, a per-case `data/` folder, and an
expected output naming the correct `retriever` subcommand(s) and the captioning flags.

---

## The user task under test

> **JTBD: EXTRACT — image / picture-region captioning.** "Generate natural-language
> descriptions of image content, whether the input is a standalone image file OR a picture
> region detected inside a visual document (PDF / DOCX / PPTX). Captions are indexed
> alongside other extracted chunks so visual content is retrievable by plain text query."
> — **P0**

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) `retriever ingest --caption` runs a VLM caption stage via a **caption-capable** ingestor, (2) each captioned image/chart/region row carries a **non-empty NL caption** in `text`, (3) that caption is **retrievable by a plain-text query** |
| Time | **medium** — each ingest+query round-trip **≤ 2 min** |
| Trigger rate | ≥ 95% — a "describe / caption this image or chart in plain English" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <input> --caption` then `retriever query`; **must not drop `--caption`** or fall back to a no-op base ingestor |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased):
- *"Tell me in plain English what this chart is showing."*
- *"Describe this diagram in a sentence or two."*
- *"Caption this infographic — what's the takeaway?"*

---

## How the CLI captions (verified against the source + dry-run)

Grounded by `skills/nemo-retriever/references/cli/ingest.md` (the `--caption` flag table) +
`references/query.md` ("Charts and images" / the captioning note), and the CLI source:

- **`retriever ingest <input> --caption`** adds a VLM caption stage *after* extraction. It
  is **never enabled by any profile** — the agent must request it explicitly. Related flags:
  `--caption-invoke-url` (pin a remote/local OpenAI-compatible VLM endpoint),
  `--caption-model-name` (override the VLM), `--caption-infographics` (also caption
  infographic crops), `--caption-context-text-max-chars` (feed nearby OCR text into the
  caption prompt). Confirmed via `retriever ingest … --caption --dry-run` (offline): the
  resolved plan grows a `caption` block (`model_name`, `prompt "Caption the content of this
  image:"`, `caption_infographics`, `endpoint_url`).
- **VLM model:** `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (settable with
  `--caption-model-name`; a `--dry-run` confirms the override lands in `caption.model_name`).
- **Where the caption lands:** for images/charts the VLM caption is written into the row's
  **`text`** field; for **infographic** crops it is written to a separate **`caption`** field
  (preserving the OCR `text`). A region captioned **inside a document** keeps its `source`/
  `page_number` and carries **`source_id` + `bbox_xyxy_norm`** (the region's normalized box).
- **Routing.** On a **CPU-only** host, `--caption` with no `--caption-invoke-url` uses the
  hosted default VLM endpoint and reads `NVIDIA_API_KEY` (the `CaptionCPUActor` raises if
  neither a key nor a `--caption-invoke-url` is set — it does *not* silently emit empty
  captions). On a **GPU** host, `--caption` can run a **local VLM** (`CaptionGPUActor`) with
  no hosted call, or `--caption-invoke-url` pins a local container.
- **The documented fail mode (the trap this suite guards).** The base abstract
  `ingestor.caption()` (`ingestor/core.py`) is a **`_not_implemented` stub**. Only
  **`ServiceIngestor.caption()`** (`service/service_ingestor.py`) and
  **`GraphIngestor.caption()`** (`ingestor/graph_ingestor.py`) actually run the stage. If the
  skill drops `--caption`, or the caption stage resolves to the bare base in-process
  `Ingestor`, captioning **silently no-ops**: the visual rows keep empty `text` and stay
  **unsearchable by description** — a FAIL. The skill must use a caption-capable ingestor.
- **Retrieval.** After ingest with `--caption`, a plain-text description query surfaces the
  visual hit whose `text` is the generated caption (`--content-types chart,image` — add
  `infographic` when `--caption-infographics` was used). See `query.md` "Charts and images"
  + the "Image / chart captioning" note.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `extract-caption-001` | **Baseline.** Caption one standalone `chart.png`; a chart/image row with a non-empty NL caption in `text` exists. | `ingest --caption` |
| 2 | `extract-caption-002` | **Retrievable.** A plain-text description query surfaces the image via its caption (closes the ingest→query loop). | `ingest --caption`, `query` |
| 3 | `extract-caption-003` | **Region inside a doc.** Caption picture regions inside `multimodal_test.pdf`; the caption row carries `source_id` + `bbox_xyxy_norm`. | `ingest --caption`, `query` |
| 4 | `extract-caption-004` | **Infographics.** `--caption-infographics` also captions infographic crops (caption in the `caption` field). | `ingest --caption --caption-infographics`, `query` |
| 5 | `extract-caption-005` | **Acceptance gate.** Mixed corpus → fresh named index, all visuals captioned + text-queryable, bbox on in-doc regions, **and** caption ran via a caption-capable ingestor (not the base stub). | `ingest --caption`, `query` |

The ladder: T1 proves a single caption is produced; T2 proves it's searchable; T3 moves from
a standalone image to a **region inside a document** (the second half of the task statement,
adding `bbox_xyxy_norm`); T4 adds the infographic-crop dimension via one flag; T5 composes
everything into the row's operational-pass gate and explicitly guards the documented no-op
fail mode.

---

### T1 — `extract-caption-001` · standalone chart caption  *(complexity 1)*
- **Satisfies:** the task at its simplest — caption one standalone image.
- **Data:** `data/chart.png`.
- **Expected:** `RETRIEVER ingest data/chart.png --caption` → a chart/image-type row whose
  `text` is a non-empty NL caption (exact wording not asserted), produced by a
  caption-capable ingestor.

### T2 — `extract-caption-002` · caption is retrievable  *(complexity 2)*
- **Satisfies:** "captions … retrievable by plain text query."
- **Data:** `data/chart.png`.
- **Expected:** ingest `--caption` → `RETRIEVER query "a chart showing the data" --top-k 5
  --content-types chart,image` returns ≥ 1 hit citing `chart.png` whose `text` is the
  generated caption.
- **Adds:** the retrieval leg over rung 1.

### T3 — `extract-caption-003` · picture region inside a PDF  *(complexity 3)*
- **Satisfies:** the "picture region detected inside a visual document" half of the task.
- **Data:** `data/multimodal_test.pdf` (3-page doc with charts).
- **Expected:** ingest `--caption` (PDF branch detects picture regions, then captions them) →
  a captioned region row carries `source_id` + `bbox_xyxy_norm`; a plain-text query surfaces
  it citing `multimodal_test.pdf` with a page number.
- **Adds:** document input + region localization (`bbox_xyxy_norm`).

### T4 — `extract-caption-004` · also caption infographics  *(complexity 4)*
- **Satisfies:** the "caption this infographic — what's the takeaway?" seed query.
- **Data:** `data/multimodal_test.png`.
- **Expected:** `RETRIEVER ingest data/multimodal_test.png --caption --caption-infographics`
  (dry-run shows `caption_infographics == true`) → infographic crops carry a non-empty
  `caption` field (OCR `text` preserved); query with `--content-types chart,image,infographic`.
- **Adds:** the `--caption-infographics` flag dimension.

### T5 — `extract-caption-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row + the caption-capable-ingestor validation
  path.
- **Data:** `data/` (`chart.png`, `table.png`, `multimodal_test.pdf`).
- **Expected:** `RETRIEVER ingest data/ --caption --caption-infographics --table-name
  caption_smoke` → `RETRIEVER query "a bar chart of values" --table-name caption_smoke
  --top-k 5 --content-types chart,image,infographic`; every visual carries a non-empty NL
  caption, in-doc regions carry `bbox_xyxy_norm`, the query surfaces them, and the caption
  stage verifiably ran on a `ServiceIngestor`/`GraphIngestor` (not the base stub).
- **Adds (the trap):** custom `--table-name` aligned across both commands, the
  all-visuals-captioned assertion, and the explicit caption-capable-ingestor check that
  catches a silent no-op.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The checks unique to this suite: **(a)** `--caption` is
actually used (not dropped); **(b)** the produced visual rows carry **non-empty NL captions**
that are retrievable by a plain-text query; **(c)** in-document regions carry
`bbox_xyxy_norm`; **(d)** the caption stage ran on a **caption-capable** ingestor
(`ServiceIngestor`/`GraphIngestor.caption`), **not** the base `Ingestor._not_implemented`
stub.

**Note on live runs — not run live.** Expected outputs are grounded in the CLI source and a
`retriever ingest … --caption --dry-run` (offline, no network), **not** yet executed live.
A live captioning run requires a reachable **VLM endpoint**: on a CPU-only host that means
`NVIDIA_API_KEY` (or `--caption-invoke-url`) for the hosted default VLM
(`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`); on a GPU host it can use a local VLM
container. Live ingest/query may hit billable hosted endpoints. A live run would capture the
actual row counts, the generated caption strings, the `bbox_xyxy_norm` values on the PDF
regions, per-round-trip latencies (vs. the ≤ 2 min bucket), and token baselines — and would
confirm that the caption stage ran on a caption-capable ingestor rather than no-opping
through the base `Ingestor` stub.
