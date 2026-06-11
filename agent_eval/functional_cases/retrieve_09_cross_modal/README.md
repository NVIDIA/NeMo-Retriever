# Functional test suite — RETRIEVE: cross-modal retrieval (text query surfaces non-text content)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` / `retriever query`).

This suite covers the **RETRIEVE** task of *cross-modal retrieval*: a plain **text query**
surfaces **non-text content** — charts, images, infographics, and picture regions — so the
developer never has to pre-extract the visual into text or stand up a separate visual-search
path. Each test is a self-contained triple — a prompt, a per-case `data/` folder, and an
expected output naming the correct `retriever` subcommand(s) and flags.

This is **RETRIEVE**, graded by an **operational pass** (not RAGAS): a text query must return
a hit whose **row `content_type` ∈ {chart, image, infographic, table}**, surfaced by a
cross-modal route, semantically matching the query intent.

---

## The user task under test

> **JTBD: RETRIEVE — cross-modal retrieval (text query surfaces non-text content).** "Text
> queries can retrieve images, charts, infographics, and picture regions without the
> developer pre-extracting them into text or writing a separate visual search." — **P0**

**Success criteria for the row (operational pass — not RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: a **TEXT** query returns ≥ 1 hit whose backing row `content_type` ∈ {chart, image, infographic, table}, surfaced via a **cross-modal route** (VL embedding **or** a VLM caption), **semantically matching** the query; no hand pre-extraction, no separate visual-search path |
| Time | **RETRIEVE ≤ 1 min** per query (ingest is setup, not part of the retrieve SLA) |
| Trigger rate | ≥ 95% — a "find the chart/image in my docs that shows X" / "what does this chart show" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <corpus>` (visual stack on by default; `--caption` for the caption route) then `retriever query "<text>"` with `--content-types chart,image,infographic,table`; no `--input-type` flag (does not exist) |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased):
- *"What does this chart show? Pull out the most striking data point."*
- *"Find me the chart in my docs that shows revenue growth."*
- *"Which images in my corpus discuss `<topic>`?"*

**Why operational, not RAGAS.** Per `CONVENTIONS.md`, functional RETRIEVE tests use the
operational pass (right subcommand + a correct, grounded hit of the right *type*), not a
RAGAS judge score. The graded/RAGAS flavor lives only in the separate performance-eval
suites.

---

## How the CLI does cross-modal retrieval (grounded against source + dry-run)

Grounded by `skills/nemo-retriever/references/cli/ingest.md` (the `--caption` /
`--embed-modality` tables) + `references/query.md` ("Charts and images") and the CLI source.
A text query crosses the modality boundary by **one of two routes** — either satisfies the row:

- **Route A — VL embedding.** Visual rows (chart / image / infographic) are embedded with a
  **vision-language model** so a text query embeds into the *same* space and matches them
  directly, with no caption text. Ingest side: `--embed-modality text_image` (or `image`);
  query side, the **VL reranker** `llama-nemotron-rerank-vl` (`--reranker-model-name`
  `…-vl-1b-v2`, `--rerank`) keeps the visual hit at the top. The companion embedder is the VL
  family `llama-nemotron-embed-vl`.
- **Route B — VLM caption.** `retriever ingest --caption` writes a natural-language VLM
  caption into each visual row's **`text`** (default VLM class
  `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`); a plain-text query then matches that
  caption with the ordinary text embedder. This route ties to **`extract_06_captioning`**.
  Confirmed via `retriever ingest … --caption --dry-run` (offline): the resolved plan grows a
  `caption` block (`model_name`, `prompt "Caption the content of this image:"`,
  `caption_infographics`, `endpoint_url`); `branch_summary` is e.g. `pdf:1` or `image:1`.

- **The `--content-types` discriminator.** `retriever query … --content-types
  chart,image,infographic,table` (comma-separated) **keeps only** hits whose row
  `content_type` is one of those values and **excludes untyped / plain-text rows**. This is
  how the suite *proves* a text query surfaced genuinely non-text content: if the only hits
  were plain text, a `chart,image,infographic,table` constraint returns **zero** — the fail
  signal.
- **Hit shape.** Query stdout is a JSON array; **each hit has exactly `{source,
  page_number, text}`** (`page_number` a 1-indexed int). `content_type` is a *row/index-layer*
  property set at ingest by the visual pipeline / caption stage — asserted at the index layer
  per the success criteria, **not** a 4th key in the CLI hit JSON.
- **Routing.** Visual extraction (`nemotron-page-elements-v3` → `nemotron-ocr-v2` →
  `nemotron-table-structure-v1`), `--caption`, and VL embedding default to hosted
  `ai.api.nvidia.com` / `integrate.api.nvidia.com` endpoints (key from `NVIDIA_API_KEY`) **or**
  run on a local GPU when configured. This suite is **host-agnostic** about *where* the
  VL/caption stack runs; the assertion is that a **text query surfaced a non-text row**.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

### What distinguishes this from the EXTRACT visual suites

`extract_01_visual_doc` proves the visual stack **extracts** chart/table/image rows;
`extract_06_captioning` proves `--caption` **writes** a caption. This suite proves the
**RETRIEVE leg**: a plain **text** query *crosses* the modality boundary and *surfaces* those
non-text rows by semantic intent. The one new dimension is **TEXT → non-text retrieval
(cross-modal)** — not the extraction or the captioning itself.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `retrieve-xm-001` | **Baseline.** A text query returns a `content_type=table` hit from a PDF (Giraffe row of Table 1). Text → table-typed content. | `ingest --use-table-structure`, `query --content-types text,table` |
| 2 | `retrieve-xm-002` | **Chart.** A text query surfaces a `content_type=chart` hit — non-text content with no native text, requiring the cross-modal route. | `ingest --caption`, `query --content-types chart,image` |
| 3 | `retrieve-xm-003` | **Visual-only.** `--content-types chart,image,infographic,table` filters OUT plain-text rows, isolating cross-modal hits across a mixed corpus. | `ingest --caption`, `query --content-types chart,image,infographic,table` |
| 4 | `retrieve-xm-004` | **Caption route.** A plain-English *description* (not OCR labels) retrieves a standalone image via its VLM caption (ties to captioning). | `ingest --caption`, `query --content-types chart,image` |
| 5 | `retrieve-xm-005` | **Acceptance gate.** One plain query → fresh named index → the *semantically-correct* gadget-cost chart as the top visual hit, with content_type + citation, no pre-extraction, ≤ 1 min. | `ingest --caption`, `query --content-types chart,image,infographic,table` |

The ladder: T1 proves text → a **table** (a typed visual row that *does* carry OCR text);
T2 moves to a **chart** (no native text → must use the cross-modal route); T3 adds the
**visual-only `--content-types` constraint** so every hit is non-text by construction; T4
pins the **caption route** for a **standalone image** retrieved by a description that doesn't
match its labels; T5 composes everything into the row's operational-pass gate — the right
visual, by intent, with a citation, in ≤ 1 min, with no hand pre-extraction.

---

### T1 — `retrieve-xm-001` · text query → TABLE-type hit  *(complexity 1)*
- **Satisfies:** cross-modal retrieve at its simplest — text query surfaces a non-text (table) row.
- **Data:** `data/multimodal_test.pdf` (3-page visual doc).
- **Expected:** `RETRIEVER ingest data/multimodal_test.pdf --use-table-structure` →
  `RETRIEVER query "What activity is the Giraffe doing and where?" --top-k 5 --content-types
  text,table` → a `content_type=table` hit (Table 1: **Giraffe / Driving a car / At the
  beach**), citing `multimodal_test.pdf` p1.

### T2 — `retrieve-xm-002` · text query → CHART hit  *(complexity 2)*
- **Satisfies:** the heart of the task — "text query → image/chart content."
- **Data:** `data/multimodal_test.pdf`.
- **Expected:** `RETRIEVER ingest data/multimodal_test.pdf --caption` →
  `RETRIEVER query "a chart showing gadgets and their costs" --top-k 5 --content-types
  chart,image` → a `content_type=chart` hit (Chart 1, "Gadgets and their cost"), p1; answer
  names the most striking point (**Premium desk fan, the tallest bar ≈ $150**).
- **Adds:** a chart (no native text) — must use the VL-embedding / VLM-caption route.

### T3 — `retrieve-xm-003` · constrain to VISUAL types only  *(complexity 3)*
- **Satisfies:** proving `--content-types` filters out plain text.
- **Data:** `data/chart.png`, `data/table.png`, `data/multimodal_test.pdf`.
- **Expected:** `RETRIEVER ingest data/ --caption --use-table-structure` →
  `RETRIEVER query "gadget costs and car colors" --top-k 10 --content-types
  chart,image,infographic,table` → **every** hit is non-text (gadget-cost chart + car-color
  table); zero plain-text passages.
- **Adds:** the visual-only constraint across a mixed corpus (2 images + a PDF).

### T4 — `retrieve-xm-004` · captioned-image path  *(complexity 4)*
- **Satisfies:** the caption route — "which images discuss `<topic>`," retrieved by description.
- **Data:** `data/chart.png` (standalone bar chart).
- **Expected:** `RETRIEVER ingest data/chart.png --caption` →
  `RETRIEVER query "a bar graph comparing how much different tools and appliances cost"
  --top-k 5 --content-types chart,image` → an image/chart hit citing `chart.png` whose `text`
  is the **non-empty VLM caption** (exact wording not asserted).
- **Adds:** a *standalone image* retrieved by a *description* that does not match its OCR
  labels — pins the VLM-caption route (ties to `extract_06_captioning`).

### T5 — `retrieve-xm-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row.
- **Data:** `data/chart.png`, `data/table.png`, `data/multimodal_test.pdf`.
- **Expected:** `RETRIEVER ingest data/ --caption --use-table-structure --table-name
  crossmodal_smoke` → `RETRIEVER query "the chart in my docs that shows how much gadgets cost"
  --table-name crossmodal_smoke --top-k 5 --content-types chart,image,infographic,table`; the
  **top** visual hit is the **semantically-correct** gadget-cost chart (not the car-color
  table), carries a `content_type` (chart/image) **and** a `source` + `page_number` citation,
  was surfaced via the cross-modal route **without** hand pre-extraction or a separate visual
  search, and returns in **≤ 1 min**.
- **Adds (the gate):** a fresh `--table-name` aligned across both commands, the
  *semantic-correctness* check (right visual, not just any visual), the content_type +
  citation contract, and the ≤ 1-min RETRIEVE SLA.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The checks unique to this suite: **(a)** a **text** query
returns a hit whose backing row `content_type` ∈ {chart, image, infographic, table}; **(b)**
the visual was surfaced via a **cross-modal route** (VL embedding **or** `--caption`), **not**
by the developer hand-transcribing it or building a separate visual search; **(c)** the
returned visual **semantically matches** the query intent; **(d)** `--content-types` with the
visual set actually filters out plain-text rows (an empty constrained result = no cross-modal
route = FAIL).

**Note on live runs — not run live.** Expected outputs are grounded in the CLI source and a
`retriever ingest … --caption --dry-run` (offline, no network), **not** yet executed live.
A live cross-modal run requires a working **cross-modal route**: either `--caption` (VLM)
or **VL embedding** must be enabled and reachable. `--caption` is **off by default and never
enabled by any profile** (the base abstract `Ingestor.caption()` is a `_not_implemented`
stub — if it is dropped, captioning silently no-ops and visuals stay unsearchable by text);
on a **CPU/hosted** host `--caption` / VL embedding need `NVIDIA_API_KEY` (or pinned
`--caption-invoke-url` / `--embed-invoke-url`); on a **GPU** host they can run on a local VLM
/ VL container. Live ingest/query may hit billable hosted endpoints. A live run would capture
the actual row counts, the generated caption strings (and the VL-embedding ranks), which
`content_type` each returned hit carried, the `page_number` / citation values, per-query
latencies (vs the ≤ 1-min RETRIEVE bucket), and token baselines — and would confirm the text
query truly surfaced a non-text row rather than returning empty under the visual-only
`--content-types` constraint.
