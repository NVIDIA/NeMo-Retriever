# NeMo Retriever skill — functional test suites (coverage index)

Agent-driven functional test cases for the **NeMo Retriever Library skill**, one suite per
user task from the JTBD "functional tests" tab. Every suite drives the real `retriever` CLI
(`ingest` / `query`) and is graded **operationally** (right subcommand + flags + a correct,
grounded answer / end state) — the RAGAS-graded RETRIEVE variants live in a separate
performance-eval track, not here.

- **Authoring rules & CLI/fixture grounding:** `CONVENTIONS.md` (read first).
- **Per suite:** `SYNOPSIS.md` (stakeholder narrative) · `README.md` (full spec) ·
  `cases.json` (machine-gradable) · `cases/<case-id>/data/` (per-case fixtures).
- **5 tests per suite**, climbing one complexity ladder; rung 1 = simplest happy path,
  rung 5 = the composed operational-pass / acceptance gate.

**Status:** 29 suites · 145 test cases authored. **Not yet run live** — expected outputs
are grounded in CLI source, offline `--dry-run`, and verified fixture content. Live runs
need (per suite) a reachable hosted endpoint + `NVIDIA_API_KEY`, a CUDA-13 GPU host, a GPU
k8s cluster, or the `[multimedia]`/libreoffice extras.

## Coverage map

| JTBD | User task | Suite | Pr |
|---|---|---|---|
| SETUP | CPU-only via build.nvidia.com (hosted) | `setup_01_cpu_hosted` | P0 |
| SETUP | Local GPU (models on the card) | `setup_02_local_gpu` | P0 |
| SETUP | K8s GPU (Helm NIM deploy) | `setup_03_k8s_gpu` | P1 |
| EXTRACT | Visual docs (PDF/DOCX/PPTX) | `extract_01_visual_doc` | P0 |
| EXTRACT | Images (PNG/JPEG/TIFF/BMP/SVG) | `extract_02_image` | P0 |
| EXTRACT | Text formats (HTML/MD/txt/JSON/SH) | `extract_03_text_format` | P0 |
| EXTRACT | Audio (MP3/WAV) | `extract_04_audio` | P0 |
| EXTRACT | Video (MP4/MOV/MKV/AVI) | `extract_05_video` | P0 |
| EXTRACT | Image / picture-region captioning | `extract_06_captioning` | P0 |
| EXTRACT | Multilingual doc extraction | `extract_07_multilingual` | P0 |
| INGEST | Mixed-format multi-modal folder | `ingest_01_mixed_folder` | P0 |
| INGEST | Immediately queryable (no glue) | `ingest_02_immediately_queryable` | P0 |
| INGEST | Async ingestion (>1 min jobs) | `ingest_03_async` | P0 |
| INGEST | Resumable + idempotent (**gap-doc**) | `ingest_04_resumable_idempotent` | P0/P1 |
| RETRIEVE | Embed → rerank | `retrieve_01_embed_rerank` | P0 |
| RETRIEVE | NL question + structured citations | `retrieve_02_nl_citations` | P0 |
| RETRIEVE | Single-source Q&A | `retrieve_03_single_source` | P0 |
| RETRIEVE | Multi-source synthesis | `retrieve_04_multi_source` | P0 |
| RETRIEVE | Aggregation (count/sum/list) | `retrieve_05_aggregation` | P0 |
| RETRIEVE | Comparative | `retrieve_06_comparative` | P0 |
| RETRIEVE | Multi-hop | `retrieve_07_multihop` | P0 |
| RETRIEVE | Agentic multi-turn (rewrite) | `retrieve_08_agentic_multiturn` | P0 |
| RETRIEVE | Cross-modal (text → non-text) | `retrieve_09_cross_modal` | P0 |
| RETRIEVE | Every result has citation metadata | `retrieve_10_citation_metadata` | P0 |
| RETRIEVE | Filter by document attributes | `retrieve_11_filter` | P0 |
| EVALUATE | Retrieval quality (Recall@k/nDCG@k) | `evaluate_01_retrieval_quality` | P1 |
| EVALUATE | Extraction quality · Answer+citation quality | `evaluate_NA_unsupported` (no suite — N/A in 26.05) | P1 |
| DX | End-to-end review (install→…→cite) | `dx_01_e2e_review` | P0 |
| DX | Tinker with config knobs | `dx_02_config_knobs` | P0 |
| DX | Failures surface parseable errors | `dx_03_error_surfacing` | P0 |

## Notable grounded findings (spec vs shipped 26.05 CLI)

Surfaced while authoring; each is documented in the relevant suite's README/`known_caveat`:
- **No `--input-type` flag** — `retriever ingest` auto-detects format.
- **SVG** is rasterized (cairosvg) then OCR'd — *not* parsed as vector/XML text (contra spec).
- **`.md` / `.json` / `.sh`** are not auto-detected by the CLI's input patterns (only
  `.txt` / `.html` for text); the suite ingests the byte-identical content under `.txt`.
- **`--ocr-lang`** accepts only `multi | english` (not freeform `en/fr/zh/...`).
- **No `--where` / source filter** on `query` — source/page filtering uses a LanceDB
  predicate one-liner (or a per-doc table); `--content-types` is the only native filter.
- **Async:** no native `task_id`/`--async`; backgrounding is shell-level or the `service`
  sub-app's SSE stream over `aingest_stream()`.
- **Resumable/idempotent ingest is a real NRL 26.05 gap** — no `--resume`; default
  `--overwrite` rebuilds, `--append` duplicates. `ingest_04` documents this on purpose
  (its tests are designed to surface the gap, not to pass).
- **`recall vdb-recall run`** defaults to table `nv-ingest` while `ingest` writes
  `nemo-retriever` — eval cases must pass `--table-name nemo-retriever` or score zero.
- **Multilingual fixtures** cover only en / ru / ru+en / nl on disk; CJK + Arabic + French
  legs lack fixtures (flagged in `extract_07_multilingual`).
