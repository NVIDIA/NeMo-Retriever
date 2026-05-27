# Ingest Reference

## Contents

- [Command Selection](#command-selection)
- [Inputs](#inputs)
- [Remote Inference](#remote-inference)
- [Python SDK Notes](#python-sdk-notes)
- [Validation](#validation)

## Command Selection

Use `retriever ingest` for the compact installed-user path:

```bash
retriever ingest ./data/corpus --lancedb-uri ./lancedb --table-name nv-ingest
```

When working from this source checkout and the installed command is absent, use
the project environment instead of stopping:

```bash
uv run --project nemo_retriever retriever ingest ./data/corpus \
  --lancedb-uri ./lancedb \
  --table-name nv-ingest
```

Observed from the root CLI tests:

- Default `--lancedb-uri` is `lancedb`.
- Default `--table-name` is `nv-ingest`.
- Default `--run-mode` is `inprocess`.
- The command overwrites the target table by default. Use `--append` only when
  duplicate rows are acceptable or the caller explicitly wants append behavior.
- Directories are expanded to supported files. Empty directories and unsupported
  extensions are user-facing errors.

Use `retriever pipeline run` when the task needs lower-level controls:

```bash
retriever pipeline run ./data/corpus \
  --input-type pdf \
  --method pdfium \
  --save-intermediate ./processed_docs
```

Important differences:

- `pipeline run` exposes more extraction, chunking, storage, Ray, service, and
  evaluation flags.
- `--save-intermediate` writes extraction results as Parquet, which is needed
  for full-page markdown QA evaluation.
- `--no-vdb` skips vector DB upload.
- `--run-mode service` submits work to a running Retriever service.
- `pipeline run` defaults to `--run-mode batch`; pass `--run-mode inprocess`
  for a small local smoke test.

## Inputs

`retriever ingest` supports `auto`, `pdf`, `doc`, `txt`, `html`, `image`,
`audio`, and `video` input types. `doc` covers Office documents such as DOCX and
PPTX but routes through the PDF/document extraction path.

Media workflows need extra system dependencies:

- TXT chunking uses HuggingFace tokenizers. If `txt_file_to_chunks_df` or a
  text ingest fails with `ModuleNotFoundError: No module named 'transformers'`,
  install or transiently add the missing dependency, then rerun. In an installed
  environment use `uv pip install transformers`; in a source checkout use
  `uv run --project nemo_retriever --with transformers retriever ingest ...`.
- Audio/video: `ffmpeg` / `ffprobe`.
- SVG rendering: `cairosvg` and its system dependencies.
- Local GPU inference: install the `[local]` extra and CUDA-compatible PyTorch.

## Remote Inference

For hosted or self-hosted NIMs, pass the stage endpoints explicitly:

```bash
export NVIDIA_API_KEY=nvapi-...
retriever ingest ./data/corpus \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1 \
  --ocr-version v1 \
  --graphic-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

For remote embedding, query with the same embedding endpoint and model. Do not
mix vectors created by one model with queries embedded by another model.

## Python SDK Notes

When the user wants SDK code, start from:

```python
from nemo_retriever import create_ingestor

ingestor = create_ingestor(run_mode="batch")
dataset = ingestor.files(["./data/file.pdf"]).extract().embed().ingest()
```

Use the CLI for the shortest path to LanceDB. Some docs discuss graph ingestion
and storage separately; the root CLI adapter has the tested one-shot
`extract -> embed -> vdb_upload -> ingest` path.

## Validation

After ingest, validate with the concrete LanceDB table and one smoke query. A
successful process exit alone is not enough.

```python
import lancedb

db = lancedb.connect("./lancedb")
print(db.table_names())
table = db.open_table("nv-ingest")
print(table.count_rows())
```

Then use a query that should match the corpus:

```bash
retriever query "smoke test term from the corpus" --lancedb-uri ./lancedb --table-name nv-ingest --top-k 3
```

For text ingestion, also validate that extraction produced rows before tuning
embedding or LanceDB settings:

Installed environment:

```bash
python -c "from nemo_retriever.txt.split import txt_file_to_chunks_df; print(txt_file_to_chunks_df('file.txt').shape)"
```

Source checkout:

```bash
uv run --project nemo_retriever --with transformers python -c \
  "from nemo_retriever.txt.split import txt_file_to_chunks_df; print(txt_file_to_chunks_df('file.txt').shape)"
```

If validation fails, read `PITFALLS.md` before changing models, table names, or
paths.
