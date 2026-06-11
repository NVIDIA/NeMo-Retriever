# Functional test suites — authoring conventions (shared)

This file is the single source of truth for authoring agent-driven functional test suites
for the **NeMo Retriever Library skill**. Two complete exemplars already exist — read them
before authoring a new suite:

- `setup_01_cpu_hosted/` — CPU-only via build.nvidia.com (hosted)
- `setup_02_local_gpu/` — local GPU (on-device)

Every suite covers **one user task** from the NeMo Retriever JTBD spec (the "functional
tests" tab). The goal: demonstrate how an agent drives the real `retriever` CLI (`ingest` /
`query`) to perform that user task.

---

## Deliverable per suite (4 artifacts — all required)

A suite lives at `agent_eval/functional_cases/<suite_id>/` and contains:

1. **`SYNOPSIS.md`** — concise, plain-English, stakeholder-readable narrative (≈1 page):
   what the user task is, how we test it, the five tests simplest→hardest, why that order.
   No deep CLI detail. (REQUIREMENT for every suite.)
2. **`README.md`** — full spec: the user task + priority + source, the success-criteria
   table, how the CLI behaves for this task (grounded, not guessed), and each of the 5
   tests with its data, expected commands/output, and *what it adds over the previous rung*.
3. **`cases.json`** — machine-gradable. Top-level: `suite_id, jtbd, user_task,
   user_task_priority, source, success_criteria{}, environment{}, cases[]`. Each case:
   `id, title, complexity (1-5), complexity_axis, prompt, data_files[], expected_skill,
   expected_subcommands[], expected_commands[], expected_output{}, pass_when[]`
   (+ optional `known_caveat`). Match the structure used in the two exemplar `cases.json`.
4. **`cases/<case-id>/data/`** — one **per-case** data subfolder, holding the file(s) that
   case mounts. Copy real fixtures from the catalog below (do **not** symlink). A case that
   ingests a "folder" gets multiple files in its `data/`.

**The 5-test rule.** Exactly five tests per suite, climbing **one** complexity ladder.
Each test's `complexity_axis` must state *what user task it satisfies* and *what it adds
over the previous rung* (one new dimension at a time). Rung 1 = simplest happy path;
rung 5 = the composed operational-pass / acceptance gate for the row.

**Disclaimer convention.** End README with a "not run live" note: expected outputs are
grounded in CLI source + dry-run, not yet executed live (live ingest/query may hit billable
hosted endpoints or need a GPU). State what a live run would capture (row counts,
latencies, token baselines).

---

## Success-criteria framework (applies to every row — from the spec)

- **Quality** — two flavors:
  - *Operational* (SETUP/EXTRACT/INGEST/RETRIEVE/EVALUATE/DX): operation completes with the
    expected end state, binary pass/fail. **This is what FUNCTIONAL tests use.**
  - *Graded* (RETRIEVE only, and only in the separate **performance-eval** suites — NOT
    these functional ones): ≥80% of queries get a RAGAS judge score ≥0.75. For functional
    RETRIEVE tests, use operational pass (correct grounded answer + right subcommand).
- **Time** — wall-clock. Non-retrieval buckets fall in `fast (≤30s) · medium (≤2min) ·
  slow (≤10min)`. SETUP rows carry their own SLAs. RETRIEVE ≤1 min except agentic ≤5 min.
- **Trigger rate ≥95%** — the agent fires the skill when it should.
- **Subcommand-selection accuracy ≥90%** — right `retriever` subcommand + right flags.
- **Token usage** — tracked, not gated.

---

## CLI grounding (verified against `src/nemo_retriever/cli/main.py` + references)

Binary (for local grounding / dry-run): `/raid/nemo_retriever/.venv/bin/retriever`
(already installed, `retriever --version` → e.g. `2026.06.10.devXXXX`). In suite docs use
the placeholder `$RETRIEVER` = `<RETRIEVER_VENV>/bin/retriever`.

**Two subcommands only:** `retriever ingest <paths…>` and `retriever query "<text>"`.
**There is NO `--input-type` flag** — format is auto-detected from extension. (The skill's
older `setup.md` mentions `--input-type`; the shipped CLI does not have it. Trust the code.)

### `retriever ingest <paths…>` — key flags
- `--profile auto|fast-text` · `--run-mode inprocess|batch` · `--dry-run` (prints resolved
  plan as JSON, **offline, no network** — use it to ground expected command/flag output)
- `--lancedb-uri` (default `lancedb`) · `--table-name` (default `nemo-retriever`) ·
  `--overwrite/--append`
- Extraction toggles: `--extract-text/--extract-tables/--extract-charts/--extract-images/
  --extract-infographics/--extract-page-as-image` (and `--no-…`), `--use-page-elements`,
  `--use-table-structure`, `--method`, `--dpi`
- OCR: `--ocr-version v1|v2` · `--ocr-lang <lang>` (multilingual selector)
- Audio/video: `--segment-audio`, `--audio-split-type size|time|frame`,
  `--audio-split-interval`, `--video-extract-audio`, `--video-extract-frames`,
  `--video-frame-fps`, `--video-av-fuse`
- Captioning: `--caption` (adds a VLM caption stage), `--caption-invoke-url`,
  `--caption-model-name`, `--caption-infographics`
- Embedding: `--embed-invoke-url` (hosted endpoint), `--embed-model-name`,
  `--local-ingest-embed-backend`, `--embed-modality text|image|text_image`
- Chunking: `--text-chunk`, `--text-chunk-max-tokens`, `--text-chunk-overlap-tokens`
- `--api-key` (bearer for hosted extract/embed/caption; else read from `NVIDIA_API_KEY`)
- Batch-mode GPU actors: `--page-elements-gpus-per-actor / --ocr-gpus-per-actor /
  --table-structure-gpus-per-actor / --embed-gpus-per-actor` (with `--run-mode batch`)
- NIM endpoint overrides: `--page-elements-invoke-url / --ocr-invoke-url /
  --table-structure-invoke-url / --graphic-elements-invoke-url`
- `--quiet/--no-quiet` (quiet default; one summary line on success)

**Ingest success output (stdout):**
`Ingested N file(s) → M row(s) in LanceDB lancedb/<table>.`
(`branch_summary` like `pdf:3` appears in `--dry-run`.)

### `retriever query "<text>"` — key flags
- `--top-k` (default 10) · `--candidate-k` (pool before dedup/filter; ≥ top-k)
- `--page-dedup/--no-page-dedup` · `--content-types text,table,chart,image,infographic`
  (comma-sep; untyped hits excluded)
- `--lancedb-uri` / `--table-name` (must match the ingest table!)
- `--embed-invoke-url` / `--embed-model-name` / `--query-embed-backend vllm|hf`
- `--reranker-invoke-url` / `--reranker-model-name` / `--reranker-backend vllm|hf`
- `--rerank/--no-rerank` (off by default; implicitly on if any reranker flag is set)

**Query success output (stdout):** a JSON array of hits; **each hit has exactly three keys**
`{source, page_number, text}`. `page_number` is an int, **1-indexed**. No `metadata` /
`pdf_page` / `_distance` keys at the CLI layer.

### Routing (from the two exemplar suites)
- **CPU-only / hosted (build.nvidia.com):** extraction NIM invoke-urls default to hosted
  `ai.api.nvidia.com`; embedding hosted via `--embed-invoke-url
  https://integrate.api.nvidia.com/v1`; reranker hosted; key from `NVIDIA_API_KEY`.
- **Local GPU (on-device):** `[local]` install (cu130 torch); query uses
  `--query-embed-backend hf --reranker-backend hf --rerank --embed-model-name
  nvidia/llama-nemotron-embed-1b-v2`; ingest uses default vLLM; no hosted endpoints / key;
  visual extraction on-GPU needs batch GPU actors or localhost NIMs (else it falls back to
  hosted — a thing to test). Defaults: lancedb uri `lancedb`, table `nemo-retriever`.
- **Default (unspecified host):** embedding runs on the bundled CPU/GPU HF model; text
  extraction is local pdfium (no network); visual extraction defaults to hosted endpoints.

Models referenced in validation paths (internal spec — fine to name in expected outputs):
embedder `nvidia/llama-nemotron-embed-1b-v2`, reranker `llama-nemotron-rerank-1b-v2`
(VL: `…-vl-1b-v2`), extraction `nemotron-page-elements-v3` → `nemotron-ocr-v2` →
`nemotron-table-structure-v1`, ASR `nvidia/parakeet-ctc-1.1b`, VLM caption
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.

---

## Fixture catalog (copy from these — all real, on disk)

Primary small-fixture dir: **`/raid/nemo_retriever/data/`**
- **Visual docs (PDF):** `test.pdf` (1pg: "Here is one line of text…/…another line/…an image."),
  `woods_frost.pdf` (2pg: p1 Robert Frost poem "Stopping by Woods on a Snowy Evening" — owner's
  "house is in the village", "miles to go before I sleep"; p2 table of Frost collections×years,
  incl. *New Hampshire — 1923*), `table_test.pdf` (1pg table Year × {Bill,Amy,James,Ted,Susan};
  **James 2019 = 978**, Susan 2023 = 970), `embedded_table.pdf`, `test-page-form.pdf` (form),
  `multimodal_test.pdf` (3pg: "TestingDocument"; p1 animal/activity/place table — *Giraffe / Driving
  a car / At the beach*; charts; sections), `functional_validation.pdf`, `test-shapes.pdf`.
- **Same multimodal content in other modalities** (matched ground truth):
  `multimodal_test.docx`, `multimodal_test.pptx`, `multimodal_test.png`, `multimodal_test.jpeg`,
  `multimodal_test.tiff`, `multimodal_test.bmp`, `multimodal_test.svg`, `multimodal_test.wav`.
  Also `woods_frost.docx`.
- **Standalone images:** `chart.png` (a chart), `table.png` (a table image).
- **Text formats:** `test.html` ("My First Heading" / "My first paragraph."),
  `test.md` ("# Hello World! / This is a test"), `test.txt` ("The quick brown fox jumped over the
  lazy dog."), `test.json` (`{"a":4,"b":2}`), `test.sh` (`echo "Hello World!"`).

Richer corpora: **`/raid/retriever-sdg-v3/test-data/`**
- `multiformat/docx/{procurement_memo_q4,procurement_memo_iris}.docx`,
  `multiformat/pptx/aurora_qbr_q1.pptx`, `multiformat/html/{index,architecture,runbook}.html`,
  `multiformat/md/aurora_README.md`, `multiformat/txt/api_changelog.txt`,
  `multiformat/xlsx/fy27_bookings.xlsx` (**.xlsx = UNSUPPORTED modality — negative tests only**),
  `multiformat/mp4/aurora_townhall_excerpt.mp4` (the video fixture).
- `multilingual/{nl,ru,ru_en_mixed}/*.jpg` + `multilingual/ground-truth.json` (available langs:
  Dutch, Russian, Russian+English mixed — note the spec's en/fr/zh/ar set is only partly covered;
  state the gap and use what's here, e.g. `ru` (trained lang → extract) vs `nl` (untrained → the
  skill-gate behavior)).
- `funsd/` (scanned form images), `chartqa/` `docvqa/` (chart/doc images),
  `librispeech/*.flac` (**.flac = UNSUPPORTED audio format — negative tests only**;
  supported audio = `.mp3`/`.wav`, use `data/multimodal_test.wav`),
  `financebench/` + `vidorev3-finance/` (many 10-K PDFs — for multi-doc RETRIEVE; keep the
  per-case subset SMALL, 2–4 PDFs, and ground expected answers in their actual text),
  `unsupported/` (assorted unsupported formats for negative tests).

Audio/video need the `[multimedia]` extra + ffmpeg; DOCX/PPTX need libreoffice (host pkg) —
note these as install prerequisites in the relevant EXTRACT suites.

---

## Process for authoring a suite
1. Read this file + both exemplar suites (`setup_01_cpu_hosted`, `setup_02_local_gpu`).
2. Pick 5 fixtures/fixture-sets from the catalog appropriate to the user task; verify the
   expected answer against the real fixture content (use the venv binary or read the file).
3. Create `cases/<id>/data/` per case and copy the fixtures in.
4. Write `cases.json`, `README.md`, `SYNOPSIS.md`. Optionally run `retriever ingest … --dry-run`
   (offline) to ground the expected resolved plan.
5. Keep prompts realistic (paraphrase the spec's seed queries); keep answers exact and
   grounded; one new complexity dimension per rung.
