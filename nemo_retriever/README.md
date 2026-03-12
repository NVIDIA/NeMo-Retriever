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

```bash
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nemo-retriever==26.3.0rc2 nv-ingest-client==26.3.0rc2 nv-ingest==26.3.0rc2 nv-ingest-api==26.3.0rc2
```
This creates a dedicated Python environment and installs the `nemo-retriever` PyPI package, the canonical distribution for the NeMo Retriever Library.

2. Install CUDA 13 builds of Torch and Torchvision

To ensure NeMo Retriever Library’s OCR and GPU‑accelerated components run correctly on your system, you need PyTorch and TorchVision builds that are compiled for CUDA 13. In this step, you uninstall any existing Torch/TorchVision packages and reinstall them from a dedicated CUDA 13.0 wheel index so they link against the same CUDA runtime as the rest of your pipeline.

Use the CUDA 13.0 wheels from the dedicated index by running the following command.

```bash
uv pip uninstall torch torchvision
uv pip install torch==2.9.1 torchvision -i https://download.pytorch.org/whl/cu130
```
This ensures the OCR and GPU‑accelerated components in NeMo Retriever Library run against the right CUDA runtime.

3. Run the pipeline on the [multimodal_test PDF](../data/multimodal_test.pdf)

This test PDF contains text, tables, charts, and images. Additional test data resides [here](../data/).

> **Note:** `batch` is the primary intended run_mode of operation for this library. Other modes are experimental and subject to change or removal.

In this procedure, you run the end‑to‑end NeMo Retriever Library pipeline to ingest a collection of test PDFs:
```python
from nemo_retriever import create_ingestor
from pathlib import Path

documents = [str(Path("../data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")

# ingestion tasks are chainable
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

# results are returned as a ray dataset and inspectable as chunks
ray_dataset = ingestor.ingest()
chunks = ray_dataset.get_dataset().take_all()
```

You can inspect how recall accuracy optimized text chunks for various content types were extracted into text representations:
```python
# page 1 raw text:
>>> chunks[0]["text"]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose. It contains some \r\nheadings and some placeholder text to fill the space. The text is not important and contains \r\nno real value, but it is useful for testing. Below, we will have some simple tables and charts \r\nthat we can use to confirm Ingest is working as expected.\r\nTable 1\r\nThis table describes some animals, and some activities they might be doing in specific \r\nlocations.\r\nAnimal Activity Place\r\nGira@e Driving a car At the beach\r\nLion Putting on sunscreen At the park\r\nCat Jumping onto a laptop In a home o@ice\r\nDog Chasing a squirrel In the front yard\r\nChart 1\r\nThis chart shows some gadgets, and some very fictitious costs.'

# a table from the first page
>>> chunks[1]["text"]
'| Table | 1 |\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |'

# a chart from the first page
>>> chunks[2]["text"]
'Chart 1\nThis chart shows some gadgets, and some very fictitious costs.\nGadgets and their cost\n$160.00\n$140.00\n$120.00\n$100.00\nDollars\n$80.00\n$60.00\n$40.00\n$20.00\n$-\nPowerdrill\nBluetooth speaker\nMinifridge\nPremium desk fan\nHammer\nCost'
```

Since the ingestion job automatically populated a lancedb table with all these chunks, you can use queries to retrieve semantically relevant chunks for feeding directly into an LLM:

4. Run a recall query and generate an answer using an LLM

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(
  # default values
  lancedb_uri="lancedb",
  lancedb_table="nv-ingest",
  embedder="nvidia/llama-3.2-nv-embedqa-1b-v2",
  top_k=5
)

query = "Given their activities, which animal is responsible for the typos in my documents?"

# you can also submit a list with retriever.queries[...]
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

The above retrieval results are often feedable directly to an LLM for answer generation.

To do so, first install the openai client and set your [build.nvidia.com](https://build.nvidia.com/) API key:
```bash
uv pip install -y openai
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

5. Ingest other types of content:

For PowerPoint and Docx files, ensure libeoffice is installed by your system's package manager.

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y libreoffice
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

# html and text files - include a split task to prevent texts from exceeding the embedder's max sequence length
documents = [str(Path(f"../data/*{ext}")) for ext in [".txt", ".html"]]
ingestor = (
  ingestor.files(documents)
  .extract()
  .split(max_tokens=5) #1024 by default, set low here to demonstrate chunking
)
```
For audio and video files, ensure ffmpeg is installed by your system's package manager.

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y ffmpeg
```

```python
ingestor = create_ingestor(run_mode="inprocess")
ingestor = ingestor.files([str(INPUT_AUDIO)]).extract_audio()

chunks = ingestor.ingest()
```

7. Explore Different Pipeline Options:

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
ingestor = create_ingestor(run_mode="inprocess")
ingestor = ingestor.files(documents).extract(
  method="pdfium",
  batch_tuning={
    "nemotron_parse_workers": float(1),
    "gpu_nemotron_parse": float(1),
    "nemotron_parse_batch_size": float(1)
  }
)

chunks = ingestor.ingest()
```



All `ExtractParams` options (`extract_text`, `extract_tables`, `extract_charts`, `extract_infographics`) apply to image ingestion.

### Render one document as markdown

If you want a readable page-by-page markdown view of a single in-process result, pass the
single-document result from `results[0]` to `nemo_retriever.io.to_markdown`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.io import to_markdown

ingestor = (
    create_ingestor(run_mode="inprocess")
    .files("data/multimodal_test.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
)
results = ingestor.ingest()
print(to_markdown(results[0]))
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
