# Quick Start for NeMo Retriever Library

NeMo Retriever Library is a retrieval-augmented generation (RAG) ingestion pipeline for documents that can parse text, tables, charts, and infographics. NeMo Retriever Library parses documents, creates embeddings, optionally stores embeddings in LanceDB, and performs recall evaluation.

This quick start guide shows how to run NeMo Retriever Library as a library all within local Python processes without containers. NeMo Retriever Library supports two inference options:
- Pull and run [Nemotron RAG models from Hugging Face](https://huggingface.co/collections/nvidia/nemotron-rag) on your local GPU(s).
- Make over the network inference calls to build.nvidia.com hosted or locally deployed NeMo Retriever NIM endpoints.

You’ll set up a CUDA 13–compatible environment, install the library and its dependencies, and run GPU‑accelerated ingestion pipelines that convert PDFs, HTML, plain text, audio, or video into vector embeddings stored in LanceDB (on local disk), with Ray‑based scaling and built‑in recall benchmarking.

## Prerequisites

Before starting, make sure your system meets the following requirements:

- The host is running CUDA 13.x so that `libcudart.so.13` is available.
- Your GPUs are visible to the system and compatible with CUDA 13.x.
​
If optical character recognition (OCR) fails with a `libcudart.so.13` error, install the CUDA 13 runtime for your platform and update `LD_LIBRARY_PATH` to include the CUDA lib64 directory, then rerun the pipeline. 

For example, the following command can be used to update the `LD_LIBRARY_PATH` value.

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

## Setup your environment

Complete the following steps to setup your environment. You will create and activate isolated Python and project virtual environments, install the NeMo Retriever Library and its dependencies, and then run the provided ingestion snippets to validate your setup.

1. Create and activate the NeMo Retriever Library environment

Before installing NeMo Retriever Library, create an isolated Python environment so its dependencies do not conflict with other projects on your system. In this step, you set up a new virtual environment and activate it so that all subsequent installs are scoped to NeMo Retriever Library.

In your terminal, run the following commands from any location.

For **local GPU inference** (Nemotron models running on your GPU), install with the `[local]` extra, which includes the model packages, transformers, and GPU tooling:

```bash
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install "nemo-retriever[local]==26.3.0" nv-ingest-client==26.3.0 nv-ingest==26.3.0 nv-ingest-api==26.3.0
```

For **remote NIM inference only** (no local GPU required), the base package is sufficient:

```bash
uv python install 3.12
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install nemo-retriever==26.3.0 nv-ingest-client==26.3.0 nv-ingest==26.3.0 nv-ingest-api==26.3.0
```

This creates a dedicated Python environment and installs the `nemo-retriever` PyPI package, the canonical distribution for the NeMo Retriever Library.

> **Note:** `uv python install 3.12` installs a uv-managed Python that includes development headers (`Python.h`). These headers are required by vLLM, which compiles CUDA kernels at runtime using torch inductor. If you skip this step and use a system Python without headers, vLLM actor initialization will fail with `InductorError: fatal error: Python.h: No such file or directory`.

2. Override Torch and Torchvision with CUDA 13 builds (local GPU only)

The `[local]` extra pulls PyTorch from PyPI, which defaults to a CPU build on Linux. Reinstall from the CUDA 13.0 wheel index to match the CUDA runtime required by the Nemotron model packages:

```bash
uv pip uninstall torch torchvision
uv pip install torch==2.10.0 torchvision -i https://download.pytorch.org/whl/cu130
```

Skip this step if you are using remote NIM inference only.

## Run the pipeline

The [test PDF](../data/multimodal_test.pdf) contains text, tables, charts, and images. Additional test data resides [here](../data/).

> **Note:** `batch` is the primary intended run_mode of operation for this library. Other modes are experimental and subject to change or removal.

The examples below use default local GPU inference (no `invoke_url` specified) and require the `[local]` extra and the CUDA 13 torch override from the setup steps above. For remote NIM inference without a local GPU, see [Run with remote inference](#run-with-remote-inference-no-local-gpu-required).

### Ingest a test pdf
```python
from nemo_retriever import create_ingestor
from nemo_retriever.io import to_markdown, to_markdown_by_page
from pathlib import Path

documents = [str(Path("../data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")

# ingestion tasks are chainable and defined lazily
ingestor = (
  ingestor.files(documents)
  .extract(
    # below are the default values, but content types can be controlled
    extract_text=True,
    extract_charts=True,
    extract_tables=True,
    extract_infographics=True
  )
  .embed()
  .vdb_upload()
)
```

### Optional extras

- **`asr`** — Local ASR (Parakeet). Has a different `transformers` requirement than the core package; install only if you need local ASR:
  ```bash
  uv pip install -e './nemo_retriever[asr]'
  ```

Run the batch pipeline script and point it at the directory that contains your PDFs using the following command.

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/pdfs
```

```python
# ingestor.ingest() actually executes the pipeline
# results are returned as a ray dataset and inspectable as chunks
ray_dataset = ingestor.ingest()
chunks = ray_dataset.get_dataset().take_all()
```

### Ingest a test corpus (CLI)

`graph_pipeline` is the canonical ingestion script used throughout the
[QA evaluation guide](./src/nemo_retriever/evaluation/README.md#step-1-ingest-and-embed-pdfs-nemo-retriever).
Point it at a **directory** of PDFs to produce a ready-to-query LanceDB table.

> **Corpus size matters.** LanceDB's default IVF index needs at least 16
> chunks to train its 16 k-means partitions. Single-PDF ingestion will fail
> at the indexing step; point `graph_pipeline` at a directory with enough
> documents to clear that threshold. Replace `/your-example-dir` below with
> the path to your own corpus.

```bash
python -m nemo_retriever.examples.graph_pipeline \
  /your-example-dir \
  --lancedb-uri lancedb
```

Chunks land at `./lancedb/nv-ingest`, which matches the default `Retriever()`
constructor used in [Run a recall query](#run-a-recall-query) below. With the
`[local]` extra installed (see setup), defaults point at local-GPU extraction
and embedding. For a realistic retrieval corpus, see
[QA evaluation -- Step 1](./src/nemo_retriever/evaluation/README.md#step-1-ingest-and-embed-pdfs-nemo-retriever).

**No local GPU?** Set [`NVIDIA_API_KEY`](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/#nvidia-api-key) (see [Authentication and API keys](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/)) and route extraction and embedding
through [build.nvidia.com](https://build.nvidia.com/) NIMs instead:

```bash
export NVIDIA_API_KEY=nvapi-...

python -m nemo_retriever.examples.graph_pipeline \
  /your-example-dir \
  --lancedb-uri lancedb \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --graphic-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1 \
  --ocr-version v1 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

> **OCR engine default:** As of this release, the default OCR engine is
> **Nemotron OCR v2** (multilingual: EN/ZH/JA/KO/RU; higher throughput per the
> OmniDocBench benchmark). The `--ocr-version` flag accepts `v2` (default) or
> `v1` (legacy English-only). Local v2 model loading requires the
> [`nemotron-ocr`](https://huggingface.co/nvidia/nemotron-ocr-v2) package.
> A hosted v2 endpoint is not yet available on `ai.api.nvidia.com`, so the
> remote-inference example above pins `--ocr-version v1` and points
> `--ocr-invoke-url` at the v1 NIM until v2 ships there.

When you use the remote embedder, pair the `Retriever` with the matching
`embedder=` + `embedding_endpoint=` overrides shown in
[Run a recall query](#run-a-recall-query).

### Inspect extracts
You can inspect how recall accuracy optimized text chunks for various content types were extracted into text representations:
```python
# page 1 raw text:
>>> chunks[0]["text"]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose...'

# markdown formatted table from the first page
'| Table | 1 |\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |'

# a chart from the first page
>>> chunks[2]["text"]
'Chart 1\nThis chart shows some gadgets, and some very fictitious costs.\nGadgets and their cost\n$160.00\n$140.00\n$120.00\n$100.00\nDollars\n$80.00\n$60.00\n$40.00\n$20.00\n$-\nPowerdrill\nBluetooth speaker\nMinifridge\nPremium desk fan\nHammer\nCost'

# markdown formatting for full pages or documents:
# document results are keyed by source filename
>>> to_markdown_by_page(chunks).keys()
dict_keys(['multimodal_test.pdf'])

# results per document are keyed by page number
>>> to_markdown_by_page(chunks)["multimodal_test.pdf"].keys()
dict_keys([1, 2, 3])

>>> to_markdown_by_page(chunks)["multimodal_test.pdf"][1]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose. It contains some \r\nheadings and some placeholder text to fill the space. The text is not important and contains \r\nno real value, but it is useful for testing. Below, we will have some simple tables and charts \r\nthat we can use to confirm Ingest is working as expected.\r\nTable 1\r\nThis table describes some animals, and some activities they might be doing in specific \r\nlocations.\r\nAnimal Activity Place\r\nGira@e Driving a car At the beach\r\nLion Putting on sunscreen At the park\r\nCat Jumping onto a laptop In a home o@ice\r\nDog Chasing a squirrel In the front yard\r\nChart 1\r\nThis chart shows some gadgets, and some very fictitious costs.\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 1\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 1\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 2\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 2\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 3\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 3\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost'

# full document markdown also keyed by source filename
>>> to_markdown(chunks).keys()
dict_keys(['multimodal_test.pdf'])
```

Since the ingestion job automatically populated a lancedb table with all these chunks, you can use queries to retrieve semantically relevant chunks for feeding directly into an LLM:

### Run a recall query

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(
  # default values
  lancedb_uri="lancedb",
  lancedb_table="nv-ingest",
  embedder="nvidia/llama-3.2-nv-embedqa-1b-v2",
  top_k=5,
  reranker=False
)

query = "Given their activities, which animal is responsible for the typos in my documents?"

# you can also submit a list with retriever.queries[...]
hits = retriever.query(query)
```

If you ingested with the remote-NIM recipe above (no local GPU), point the
`Retriever` at the same embedding endpoint so query vectors are produced by the
same model that produced the stored chunk vectors:

```python
retriever = Retriever(
    lancedb_uri="lancedb",
    lancedb_table="nv-ingest",
    embedder="nvidia/llama-nemotron-embed-1b-v2",
    embedding_endpoint="https://integrate.api.nvidia.com/v1/embeddings",
    top_k=5,
    reranker=False,
)
hits = retriever.query(query)
```

```python
# retrieved text from the first page
>>> hits[0]
{'text': 'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose. It contains some \r\nheadings and some placeholder text to fill the space. The text is not important and contains \r\nno real value, but it is useful for testing. Below, we will have some simple tables and charts \r\nthat we can use to confirm Ingest is working as expected.\r\nTable 1\r\nThis table describes some animals, and some activities they might be doing in specific \r\nlocations.\r\nAnimal Activity Place\r\nGira@e Driving a car At the beach\r\nLion Putting on sunscreen At the park\r\nCat Jumping onto a laptop In a home o@ice\r\nDog Chasing a squirrel In the front yard\r\nChart 1\r\nThis chart shows some gadgets, and some very fictitious costs.', 'metadata': '{"page_number": 1, "pdf_page": "multimodal_test_1", "page_elements_v3_num_detections": 9, "page_elements_v3_counts_by_label": {"table": 1, "chart": 1, "title": 3, "text": 4}, "ocr_table_detections": 1, "ocr_chart_detections": 1, "ocr_infographic_detections": 0}', 'source': '{"source_id": "/home/dev/projects/NeMo-Retriever/data/multimodal_test.pdf"}', 'page_number': 1, '_distance': 1.5822279453277588}

# retrieved text of the table from the first page
>>> hits[1]
{'text': '| Table | 1 |\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |', 'metadata': '{"page_number": 1, "pdf_page": "multimodal_test_1", "page_elements_v3_num_detections": 9, "page_elements_v3_counts_by_label": {"table": 1, "chart": 1, "title": 3, "text": 4}, "ocr_table_detections": 1, "ocr_chart_detections": 1, "ocr_infographic_detections": 0}', 'source': '{"source_id": "/home/dev/projects/NeMo-Retriever/data/multimodal_test.pdf"}', 'page_number': 1, '_distance': 1.614684820175171}
```

###  Generate a query answer using an LLM
The above retrieval results are often feedable directly to an LLM for answer generation.

To do so, first install the openai client and set your [build.nvidia.com](https://build.nvidia.com/) API key:
```bash
uv pip install openai
export NVIDIA_API_KEY=nvapi-...
```

```python
from openai import OpenAI
import os

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ.get("NVIDIA_API_KEY")
)

hit_texts = [hit["text"] for hit in hits]
prompt = f"""
Given the following retrieved documents, answer the question: {query}

Documents:
{hit_texts}
"""

completion = client.chat.completions.create(
  model="nvidia/nemotron-3-super-120b-a12b",
  messages=[{"role":"user","content":prompt}],
  stream=False
)

answer = completion.choices[0].message.content
print(answer)
```

Answer:
```
Cat is the animal whose activity (jumping onto a laptop) matches the location of the typos, so the cat is responsible for the typos in the documents.
```

### Live RAG SDK (retrieve + answer in one call)

The pattern above -- retrieve hits, build a prompt, call an LLM -- is baked into the SDK as `Retriever.answer()` so live applications can skip the boilerplate. The same `Retriever` instance powers three entry points:

| Method | Input | Output | Use case |
| --- | --- | --- | --- |
| `Retriever.retrieve(query, top_k=...)` | one query | `RetrievalResult` (`chunks`, `metadata`) | Structured retrieval without an LLM. |
| `Retriever.answer(query, llm=..., judge=None, reference=None, ...)` | one query | `AnswerResult` (answer + chunks + optional scores) | One-shot RAG -- production/live. |
| `Retriever.pipeline().generate(...).score().judge(...).run(queries)` | many queries | `pandas.DataFrame` | Batch RAG over the operator graph, each step optional. |

Install the LLM client extra:
```bash
uv pip install "nemo-retriever[llm]"
export NVIDIA_API_KEY=nvapi-...
```

Single-query live RAG. Point `lancedb_uri` at any table built above; the
`embedder` must match the one used during ingestion so query vectors land in
the same embedding space as the stored chunks.

```python
from nemo_retriever.retriever import Retriever
from nemo_retriever.llm import LiteLLMClient

retriever = Retriever(
    lancedb_uri="lancedb",
    lancedb_table="nv-ingest",
    embedder="nvidia/llama-nemotron-embed-1b-v2",
    embedding_endpoint="https://integrate.api.nvidia.com/v1/embeddings",
    top_k=5,
)
llm = LiteLLMClient.from_kwargs(
    model="nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
    temperature=0.0,
    max_tokens=512,
)

result = retriever.answer("What is RAG?", llm=llm)
print(result.answer)
# 'Retrieval-augmented generation combines external context with an LLM...'
print(len(result.chunks), "chunks from", {m.get("source") for m in result.metadata})
print(f"{result.latency_s:.2f}s on {result.model}")
```

Local-GPU shortcut: if you ingested with default `graph_pipeline` flags
(`--embed` omitted, `[local]` extra installed), drop `embedder=` and
`embedding_endpoint=` to reuse the bundled `VL_EMBED_MODEL`.

Live RAG with scoring and an LLM judge (requires a ground-truth `reference`):
```python
from nemo_retriever.llm import LLMJudge

judge = LLMJudge.from_kwargs(model="nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")
result = retriever.answer(
    "What is RAG?",
    llm=llm,
    judge=judge,
    reference="RAG combines retrieved context with LLM generation.",
)
print(result.token_f1, result.judge_score, result.failure_mode)
# 0.62 4 'correct'
```

Batch RAG over the operator graph -- each builder step is optional:
```python
df = (
    retriever.pipeline()
    .generate(llm)
    .score()
    .judge(judge)
    .run(
        queries=["What is RAG?", "What is reranking?"],
        reference=["RAG combines retrieval with generation.", "Reranking re-scores retrieved passages."],
    )
)
print(df[["query", "answer", "token_f1", "judge_score", "failure_mode"]])
```

Scoring tiers on `AnswerResult`:

- **Tier 1** (`answer_in_context`) -- whether retrieval surfaced the evidence; requires `reference`.
- **Tier 2** (`token_f1`, `exact_match`) -- token-level overlap; requires `reference`.
- **Tier 3** (`judge_score`, `judge_reasoning`) -- LLM-as-judge 1-5 score; requires `reference` and `judge`.
- `failure_mode` -- derived classification (`correct`, `partial`, `retrieval_miss`, `generation_miss`, `refused_*`, `thinking_truncated`).

If only `reference` is supplied, Tier 1 + 2 run. If only `judge` is supplied (without `reference`), a `ValueError` is raised. On generation error, scoring and judge are skipped and `AnswerResult.error` is populated.

### Ingest other types of content:

For PowerPoint and Docx files, ensure libeoffice is installed by your system's package manager. This is required to make their pages renderable as images for our [page-elements content classifier](https://huggingface.co/nvidia/nemotron-page-elements-v3).

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y libreoffice
```

For SVG files, install the optional `cairosvg` dependency. SVG support is available in the NeMo Retriever Library, but not in the container deployment. `cairosvg` requires network access to install, so it will not work in air-gapped environments.
```bash
uv pip install "nemo-retriever[multimedia]"
# or to install only the SVG dependency:
uv pip install "cairosvg>=2.7.0"
```

Example usage:
```python
# docx and pptx files
documents = [str(Path(f"../data/*{ext}")) for ext in [".pptx", ".docx"]]
# mixed types of images
images = [str(Path(f"../data/*{ext}")) for ext in [".png", ".jpeg", ".bmp"]]
ingestor = (
  # above file types can be combined into a single job
  ingestor.files(documents + images)
  .extract()
)
```

*Note:* the `split()` task uses a tokenizer to split texts by a max_token length
### Render results as markdown

If you want a readable markdown view of extracted results, pass the full in-process result list
to `nemo_retriever.io.to_markdown`. The helper now returns a `dict[str, str]` keyed by input
filename, where each value is the document collapsed into one markdown string without per-page
headers, so both single-document and multi-document runs follow the same contract.

PDF text is split at the page level.

HTML and .txt files have no natural page delimiters, so they almost always need to be paired with the `.split()` task.

```python
# html and text files - include a split task to prevent texts from exceeding the embedder's max sequence length
documents = [str(Path(f"../data/*{ext}")) for ext in [".txt", ".html"]]
ingestor = (
  ingestor.files(documents)
  .extract()
  .split(max_tokens=5) #1024 by default, set low here to demonstrate chunking
)
results = ingestor.ingest()
markdown_docs = to_markdown(results)
print(markdown_docs["multimodal_test.pdf"])
```

Use `to_markdown_by_page(results)` when you want a nested
`dict[str, dict[int, str]]` instead, where each filename maps to its per-page markdown strings.
For audio and video files, ensure ffmpeg is installed by your system's package manager.

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y ffmpeg
```

```python
ingestor = create_ingestor(run_mode="batch")
ingestor = ingestor.files([str(INPUT_AUDIO)]).extract_audio()
```

### Store extracted images and text

Use `.store()` to persist extracted images, tables, charts, and text to local disk or object storage (S3, MinIO, GCS via fsspec). Stored URIs are written back to the DataFrame so downstream stages (embed, VDB upload) can reference them. By default, base64 payloads are stripped after writing to reduce memory pressure.

```python
ingestor = (
  ingestor.files(documents)
  .extract()
  .store(
    storage_uri="s3://my-bucket/citation-assets",  # or a local path
    storage_options={"key": "...", "secret": "..."},  # fsspec auth for S3/MinIO
    store_text=True,       # also write .txt files for page text and structured content
    strip_base64=True,     # free image payloads after writing (default)
  )
  .embed()
  .vdb_upload()
)
```

### Explore Different Pipeline Options:

You can use the [Nemotron RAG VL Embedder](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)

```python
ingestor = (
  ingestor.files(documents)
  .extract()
  .embed(
    model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
    #works with plain "text"s, "image"s, and "text_image" pairs
    embed_modality="text_image"  
  )
)
```

You can use a different ingestion pipeline based on [Nemotron-Parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) combined with the default embedder:
```python
ingestor = ingestor.files(documents).extract(method="nemotron_parse")
```

## Run with remote inference, no local GPU required:

For build.nvidia.com hosted inference, set [`NVIDIA_API_KEY`](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/#nvidia-api-key) as an environment variable (see [Authentication and API keys](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/)). 

```python
ingestor = (
  ingestor.files(documents)
  .extract(
    # for self hosted NIMs, your URLs will depend on your NIM container DNS settings
    page_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3",
    graphic_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1",
    ocr_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1",
    table_structure_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"
  )
  .embed(
    embed_invoke_url="https://integrate.api.nvidia.com/v1/embeddings",
    model_name="nvidia/llama-nemotron-embed-1b-v2",
    embed_modality="text",
  )
  .vdb_upload()
)
```

## Ray cluster setup

NeMo Retriever Library uses Ray Data for distributed ingestion and benchmarking. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

### Local Ray cluster with dashboard

To start a Ray cluster with the dashboard on a single machine use the following command.

```bash
ray start --head
```

Open `http://127.0.0.1:8265` in your browser for the Ray Dashboard, and run your NeMo Retriever Library pipeline on the same machine with `--ray-address auto` to attach to this cluster. [Connecting to a remote Ray cluster on Kubernetes](https://discuss.ray.io/t/connecting-to-remote-ray-cluster-on-k8s/7460)

### Single‑GPU cluster on multi‑GPU nodes

To restrict Ray to a single GPU on a multi‑GPU node use the following command.

```bash
CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1
```
Then run your pipeline as before with `--ray-address auto` so it connects to this single‑GPU Ray cluster. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

## Running multiple NIM instances on multi‑GPU hosts

### Resource heuristics (batch mode)

By default, batch mode computes resources using this order:

1. Auto-detected resources (Ray cluster if connected, otherwise local machine)
2. Environment variables
3. Explicit function arguments (highest precedence)

This means defaults are deterministic but easy to override when you need fixed behavior.

### Default behavior

- `cpu_count` / `gpu_count` are detected from Ray (`cluster_resources`) or local host.
- Worker heuristics:
  - `page_elements_workers = gpu_count * page_elements_per_gpu`
  - `detect_workers = gpu_count * ocr_per_gpu`
  - `embed_workers = gpu_count * embed_per_gpu`
  - minimum of `1` per stage
- Stage GPU defaults:
  - If `gpu_count >= 2` and `concurrent_gpu_stage_count == 3`, uses high-overlap values for page-elements/OCR/embed.
  - Otherwise uses `min(max_gpu_per_stage, gpu_count / concurrent_gpu_stage_count)`.

### Override variables

| Variable | Where to set | Meaning |
|---|---|---|
| `override_cpu_count`, `override_gpu_count` | function args | Highest-priority CPU/GPU override |

### Running multiple NIM service instances on multi-GPU hosts

### Start two stacks on separate GPUs

```bash
# GPU 0 stack
GPU_ID=0 \
PAGE_ELEMENTS_HTTP_PORT=8000 PAGE_ELEMENTS_GRPC_PORT=8001 PAGE_ELEMENTS_METRICS_PORT=8002 \
OCR_HTTP_PORT=8019 OCR_GRPC_PORT=8010 OCR_METRICS_PORT=8011 \
docker compose -p retriever-gpu0 up -d page-elements ocr

# GPU 1 stack
GPU_ID=1 \
PAGE_ELEMENTS_HTTP_PORT=8100 PAGE_ELEMENTS_GRPC_PORT=8101 PAGE_ELEMENTS_METRICS_PORT=8102 \
OCR_HTTP_PORT=8119 OCR_GRPC_PORT=8110 OCR_METRICS_PORT=8111 \
docker compose -p retriever-gpu1 up -d page-elements ocr
```

The `-p` project names create isolated stacks, `GPU_ID` pins each stack to a specific physical GPU, and distinct host ports avoid collisions between the services.  

### Check and tear down stacks

To verify that both stacks are running use the following command.

```bash
docker compose -p retriever-gpu0 ps
docker compose -p retriever-gpu1 ps
```

To stop and remove both stacks use the following command.

```bash
docker compose -p retriever-gpu0 down
docker compose -p retriever-gpu1 down
```

## Troubleshooting

### vLLM engine fails to start during CUDA graph capture

When using the vLLM-based VL reranker, the engine may fail to start with errors similar to the following:

```
fatal error: Python.h: No such file or directory
...
torch._inductor.exc.InductorError: CalledProcessError: Command '['/usr/bin/gcc', '...cuda_utils.c', ...]' returned non-zero exit status 1.
...
RuntimeError: Engine core initialization failed.
```

This occurs because Triton compiles a small C extension at runtime during CUDA graph capture and requires the Python development headers. If `Python.h` is not installed, the compilation fails and the vLLM engine cannot start.

To resolve this, install the Python development headers for your Python version:

```bash
# For Python 3.12 on Ubuntu/Debian
sudo apt install python3.12-dev
```

After installing the headers, restart the pipeline.

## ViDoRe Harness Sweep

The harness includes BEIR-style ViDoRe dataset presets in `nemo_retriever/harness/test_configs.yaml` and a ready-made sweep definition in `nemo_retriever/harness/vidore_sweep.yaml`.

The ViDoRe harness datasets are configured to:

- read PDFs from `/datasets/nv-ingest/vidore_v3_corpus_pdf/...`
- ingest with `embed_modality: text_image`
- embed at `embed_granularity: page`
- enable `extract_page_as_image: true` and `extract_infographics: true`
- evaluate with BEIR-style `ndcg` and `recall` metrics

To run the full ViDoRe sweep:

```bash
cd ~/nv-ingest/nemo_retriever
retriever-harness sweep --runs-config harness/vidore_sweep.yaml
```

The same commands also work under the main CLI as `retriever harness ...` if you prefer a single top-level command namespace.

### Harness with image/text storage

The harness can persist extracted images and text alongside other run artifacts. Set `store_images_uri` in `test_configs.yaml` (per-dataset or in `active:`) or via `--override`:

```bash
retriever harness run --dataset bo20 --preset single_gpu \
  --override store_images_uri=stored_images --override store_text=true
```

When `store_images_uri` is a relative path (like `stored_images`), it resolves to `artifact_dir/stored_images/` so each run is isolated. Absolute paths and fsspec URIs (e.g. `s3://bucket/prefix`) are passed through as-is.
