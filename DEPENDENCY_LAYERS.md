# Dependency Layering Plan

This document describes the restructured optional-extras model for `nemo_retriever/pyproject.toml`.

## Problem

The previous `pyproject.toml` listed ~50 packages as required dependencies, meaning every install
pulled in torch, vLLM, CUDA wheels, nemotron models, GPU monitoring tooling, etc. — regardless of
whether the user intended to run local models or simply call remote NIM endpoints. This made the
package impossible to install on Intel Macs and unnecessarily heavy everywhere.

## Solution: Layered Optional Extras

Dependencies are now split into a slim base plus composable optional extras. Each tier builds on
the previous via self-referencing extras.

### Tier hierarchy

```
nemo_retriever          ← slim base: ray, fastapi, pydantic, HTTP clients, nv-ingest*
 └── [remote]           ← adds: pypdfium2, pillow, nltk, markitdown, langchain-nvidia-ai-endpoints
      └── [local-cpu]   ← adds: torch CPU, transformers, nemotron models (ARM Mac compatible)
           └── [local-gpu]  ← adds: nvidia-ml-py, vLLM  (Linux/CUDA only)
                └── [multimedia]  ← adds: soundfile + scipy (ASR), cairosvg (SVG)
                                    (can also be combined with any tier independently)

[stores]       ← lancedb, duckdb, duckdb-engine, neo4j  (independent, add to any tier)
[benchmarks]   ← datasets, open-clip-torch  (BEIR evaluation only)
[dev]          ← build, pytest
[all]          ← local-gpu + multimedia + stores + benchmarks
```

### Install commands by use case

| Use case | Platform | Command |
|---|---|---|
| All remote (NIM) inference | Intel Mac, any | `uv pip install "nemo_retriever[remote,stores]"` |
| Local PDF ingestion, CPU | ARM Mac | `uv pip install "nemo_retriever[local-cpu,stores]"` |
| Local PDF ingestion, GPU | Linux + CUDA | `uv pip install "nemo_retriever[local-gpu,stores]"` |
| Full multimedia (GPU + audio + SVG) | Linux + CUDA | `uv pip install "nemo_retriever[local-gpu,multimedia,stores]"` |
| Everything | Linux + CUDA | `uv pip install "nemo_retriever[all]"` |

## What Each Extra Contains

### Base (always installed)
Pure framework infrastructure — no ML, no storage.

- `ray[data,serve]` — pipeline orchestration
- `pandas`, `numpy`, `tqdm` — data handling
- `fastapi`, `uvicorn`, `python-multipart` — service API
- `httpx`, `requests`, `urllib3` — HTTP clients
- `pydantic`, `typer`, `pyyaml`, `rich` — config, CLI, output
- `universal-pathlib`, `debugpy` — utilities
- `nv-ingest`, `nv-ingest-api`, `nv-ingest-client` — core ingest packages

### `[remote]`
Everything needed to run the full pipeline via remote NIM endpoints. No GPU, no local models.
Installs cleanly on Intel Macs.

- `pypdfium2` — PDF page splitting and rendering
- `pillow` — image I/O
- `nltk` — text splitting utilities
- `markitdown` — HTML/document-to-markdown conversion
- `langchain-nvidia-ai-endpoints` — LLM/SQL via NVIDIA NIM

### `[local-cpu]`
Adds local HuggingFace model inference. On Linux, torch resolves to a CUDA wheel from the
PyTorch index; on Mac it falls through to the PyPI CPU wheel.

- `transformers`, `tokenizers`, `accelerate==1.12.0` — HuggingFace model loading
- `torch~=2.9.1`, `torchvision` — PyTorch (CPU on Mac, CUDA on Linux)
- `einops`, `easydict`, `addict`, `timm`, `albumentations`, `scikit-learn` — model utilities
- `nemotron-page-elements-v3`, `nemotron-graphic-elements-v1`, `nemotron-table-structure-v1` — layout/table/chart detection
- `nemotron-ocr` — end-to-end OCR (Linux only)

### `[local-gpu]`
Adds GPU monitoring and fast LLM inference on top of `[local-cpu]`.

- `nvidia-ml-py` — GPU memory and utilization monitoring
- `vllm==0.16.0` — fast GPU-accelerated LLM inference (Linux only)

### `[multimedia]`
Specialized media format support. Can be combined with any inference tier.

- `soundfile`, `scipy` — audio file I/O and resampling for local Parakeet ASR
- `cairosvg` — SVG-to-image rendering (requires `libcairo` system library)

### `[stores]`
Vector, SQL, and graph storage backends. Independent of inference tier.

- `lancedb` — vector database for embedding storage and hybrid search
- `duckdb`, `duckdb-engine` — SQL execution on structured/tabular data
- `neo4j` — graph database for knowledge graph ingestion

### `[benchmarks]`
BEIR evaluation tools. Not needed for production use.

- `datasets` — HuggingFace datasets (used in `recall/beir.py`)
- `open-clip-torch` — OpenAI CLIP implementation

## Torch Index Configuration

`[tool.uv.sources]` uses a platform marker so the right torch wheel is resolved automatically:

```toml
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux'" },
  # Mac: falls through to PyPI CPU wheel
]
```

No manual intervention needed — `uv` picks the right wheel per platform.

## Cleanups Applied

The following bugs in the original flat deps list were fixed:

- `accelerate` was listed twice (`>=1.1.0` and `==1.12.0`) — kept `==1.12.0` only
- `tqdm` was listed twice — deduplicated
- `typer` was listed twice — deduplicated
- `[svg]` extra merged into `[multimedia]` (cairosvg is a media format conversion tool)
