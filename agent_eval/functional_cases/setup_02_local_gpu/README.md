# Functional test suite — SETUP user task #2 (local GPU, models on the card)

Second suite in the agent-driven functional tests for the **NeMo Retriever Library
skill**, built against the real CLI in `nemo_retriever/nemo_retriever/src`
(`retriever ingest` / `retriever query`).

This is the **GPU mirror** of `setup_01_cpu_hosted`: instead of routing to hosted
build.nvidia.com endpoints with an API key, everything runs **on a local GPU** with **no
cloud calls**. Each test is a self-contained triple — a prompt, a per-case `data/` folder,
and an expected output naming the correct `retriever` subcommand(s) and the **local-GPU**
flags.

---

## The user task under test

> **JTBD: SETUP — row 2.** "NeMo Retriever library setup on local **GPU** machine: GPU
> detected (automatic detection) — **Extraction, Embedding, Reranking models load directly
> on a local GPU**." — **P0**

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) clean `[local]` install with models on GPU, (2) ingest a PDF with models loaded locally, (3) first retrieval served by GPU-resident models |
| Time | end-to-end (clean state → first successful retrieval) **≤ 30 min** |
| Trigger rate | ≥ 95% — an on-device "set me up and load my docs" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest` then `retriever query` with **local-GPU** flags, **not** hosted (`--embed-invoke-url` / `--api-key`) |
| Token usage | tracked, not gated |

Seed queries this suite is derived from:
- *"I have an H100 — set me up to run extraction, embedding, and reranking models locally instead of calling hosted endpoints."*
- *"Get document Q&A running entirely on-device — assume I have a CUDA-capable GPU."*
- *"Install with local GPU support and confirm the models loaded onto the card."*

---

## How the CLI runs on a local GPU (verified against the skill references)

Grounded by `skills/nemo-retriever/references/install.md` + `references/query.md` and the
CLI source (`cli/main.py`):

- **Install:** on a CUDA-13 host the recipe installs GPU torch wheels (`cu130` index) +
  the **`[local]`** extra, so `nvidia/llama-nemotron-embed-1b-v2` runs on the card. The
  detector keys off `nvidia-smi` + `CUDA Version: 13`.
- **Query (local):** the canonical one-off query is
  `retriever query "<q>" --top-k 10 --embed-model-name nvidia/llama-nemotron-embed-1b-v2
  --query-embed-backend hf --reranker-backend hf --rerank`. The `hf` backends load the
  embedder + reranker on the GPU with a fast (~20–30 s) cold start; **ingest** uses the
  default vLLM backend for batch throughput. No `--embed-invoke-url` / `--api-key`.
- **Embedding + reranking** are GPU-resident bundled models — no network.
- **Text extraction** is local **pdfium** (CPU) — also no network. So a **text-only** smoke
  is 100% on-device.
- **Visual extraction** (page-elements / OCR / table-structure) is the catch: its default
  `invoke_url` is the **hosted** `ai.api.nvidia.com`. To run it on the **local GPU** you
  need either local NIM endpoints (`--page-elements-invoke-url http://localhost:…` etc.) or
  **batch run-mode with GPU actors** (`--run-mode batch --page-elements-gpus-per-actor 1
  --ocr-gpus-per-actor 1 --table-structure-gpus-per-actor 1`). This is the genuinely hard
  part of the row — case 004 targets it and is designed to **catch a silent fallback** to
  the hosted endpoint.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `setup-gpu-001` | **Baseline.** `[local]` install, GPU detected, 1 text PDF, local backends. Close the loop on-device with zero network. | install, `nvidia-smi`, `--version`, `ingest`, `query` |
| 2 | `setup-gpu-002` | **Prove it's on the card.** Assert models occupy VRAM (`nvidia-smi`) **and** zero hosted egress during ingest+query. | `ingest`, `query`, VRAM check |
| 3 | `setup-gpu-003` | **Local GPU reranker.** 3-PDF folder; spotlight the on-device reranker (`--reranker-backend hf`), reranked order ≠ dense order. | `ingest`, `query --rerank` |
| 4 | `setup-gpu-004` | **Local GPU extraction.** Table PDF forces page-elements/OCR/table onto the GPU (batch GPU actors or localhost NIM) — no hosted fallback. | `ingest` (batch/GPU), `query` |
| 5 | `setup-gpu-005` | **Acceptance gate.** Fresh **named** index, citation, all 3 model classes in VRAM, zero egress, ≤ 30 min. | install, `ingest`, `query` |

The ladder: T1 proves the on-device loop runs; T2 proves the models are *actually* on the
card and nothing leaked to the cloud; T3 and T4 each move one more model class on-device
(reranker, then the hard one — extraction); T5 composes everything into the row's real
operational-pass gate. (Same fixtures and answers as suite 1 on purpose — the variable
under test here is **where the models run**, not the content.)

---

### T1 — `setup-gpu-001` · on-device smoke loop  *(complexity 1)*
- **Satisfies:** SETUP-GPU operational-pass criteria 1–3, simplest form.
- **Data:** `data/test.pdf`.
- **Expected:** `nvidia-smi` (GPU present) → `RETRIEVER --version` → `RETRIEVER ingest
  data/test.pdf` → `RETRIEVER query "What does the document say?" --top-k 5
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --query-embed-backend hf
  --reranker-backend hf --rerank`. Answer cites `test.pdf` p1; **no network**.

### T2 — `setup-gpu-002` · models resident in VRAM + zero egress  *(complexity 2)*
- **Satisfies:** validation-path bullets *"models loaded into VRAM"* + *"no outbound calls
  to build.nvidia.com."*
- **Data:** `data/woods_frost.pdf` (owner's "house is in the village").
- **Adds:** an explicit `nvidia-smi` VRAM check + a trace assertion that nothing hit a
  hosted endpoint.

### T3 — `setup-gpu-003` · local GPU reranker  *(complexity 3)*
- **Satisfies:** the **reranking-on-GPU** clause + multi-doc ingest.
- **Data:** `data/` (3 PDFs; `woods_frost.pdf` p2 lists *New Hampshire — 1923*).
- **Adds:** folder ingest + the on-device reranker (`--reranker-backend hf`); reranked
  order differs from dense-only, no hosted reranker call.

### T4 — `setup-gpu-004` · local GPU visual extraction  *(complexity 4)*
- **Satisfies:** the **extraction-on-GPU** clause (the hard one).
- **Data:** `data/table_test.pdf` (James 2019 = 978).
- **Expected:** `RETRIEVER ingest data/table_test.pdf --run-mode batch
  --page-elements-gpus-per-actor 1 --ocr-gpus-per-actor 1 --table-structure-gpus-per-actor
  1` (or localhost NIM endpoints) → query → **978**, citing `table_test.pdf` p1.
- **Caveat (built into the test):** a bare `[local]` install gives GPU embed+rerank for
  free, but visual extraction defaults to the **hosted** endpoint unless GPU actors / local
  NIMs are configured. The test **fails** if extraction silently falls back to
  `ai.api.nvidia.com`.

### T5 — `setup-gpu-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row + the no-egress / VRAM / ≤ 30 min
  validation path.
- **Data:** `data/` (3 PDFs).
- **Expected:** ingest into `--table-name gpu_smoke` (GPU actors) → `query … --table-name
  gpu_smoke …` → *"Stopping by Woods on a Snowy Evening"*, citing `woods_frost.pdf` p1.
- **Adds (the trap):** custom `--table-name` aligned across both commands, all-three-model
  VRAM assertion, zero-egress assertion, and the ≤ 30 min gate.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks unique to this suite vs. suite 1:
**(a)** local-GPU flags are used (no hosted endpoints / API key); **(b)** models occupy
VRAM; **(c)** zero outbound calls to `ai.api.nvidia.com` / `integrate.api.nvidia.com` /
`build.nvidia.com`.

**Note on live runs:** these require an actual CUDA-13 GPU host with the `[local]` install.
The expected outputs are grounded in the CLI source and the skill's install/query
references; the suite has **not** been run live yet — run on a GPU box to capture real row
counts, VRAM figures, latencies, and token baselines, and to confirm the case-004
extraction-on-GPU path.
