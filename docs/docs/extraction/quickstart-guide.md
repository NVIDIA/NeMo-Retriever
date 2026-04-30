# Self-hosted NeMo Retriever Library (Helm)

!!! note

    This documentation describes NeMo Retriever Library.


This guide helps you get started using [NeMo Retriever Library](overview.md) in self-hosted mode on Kubernetes.


## Step 1: Deploy on Kubernetes

Install and upgrade NeMo Retriever extraction with the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md). The chart pulls NIM and microservice images from NGC; first startup can take significant time while models load.

!!! warning

    NIM containers on their first startup can take 10-15 minutes to pull and fully load models.


For a comparison of deployment modes, see [Deployment options](deployment-options.md). For tunable service settings, see [Environment variables](environment-config.md).

a. Git clone the repo:

    `git clone https://github.com/NVIDIA/NeMo-Retriever`

b. Change the directory to the cloned repo by running the following code.
   
    `cd NeMo-Retriever`.

c. [Generate API keys](api-keys.md) for your deployment: create an NGC personal key with the scopes listed there, then supply it to the cluster through the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) (`helm upgrade` flags such as `ngcImagePullSecret` / `ngcApiSecret`, Helm repo login with username `$oauthtoken`, or your organization’s equivalent secrets).

d. Create a .env file that contains your NVIDIA Build API key.

    !!! note

        If you use an NGC personal key, then you should provide the same value for all keys, but you must specify each environment variable individually. In the past, you could create an API key. If you have an API key, you can still use that. For more information, refer to [Authentication and API keys](api-keys.md) and [Environment Configuration Variables](environment-config.md).

    ```
    # Container images must access resources from NGC.

    NGC_API_KEY=<key to download containers from NGC>
    NIM_NGC_API_KEY=<key to download model files after containers start>
    ```
   
e. Ensure your GPU nodes meet the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html) and driver requirements from the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md).

f. Install or upgrade the release. The documented path uses LanceDB as the vector database (embedded, in-process). For more information about optional components (audio, nemotron-parse, VLM captioning), refer to [Profile Information](#profile-information) and [Data Upload](vdbs.md).

    Follow the `helm upgrade --install` flow in the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md), including NGC authentication and any `--set` values your cluster requires.

    !!! tip "LanceDB"

        LanceDB runs in-process alongside your client; you do not deploy separate vector-database containers for this path. For details, refer to [Data Upload](vdbs.md).

    !!! tip

        If logs are too noisy during bring-up, set `NIM_TRITON_LOG_VERBOSE=0` (or equivalent) on the relevant NIM workloads in your Helm values, then roll out the change.

    !!! tip

        The default chart values might not fit on a single GPU for some hardware targets. Use chart overrides (for example files under [`helm/overrides/`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/helm/overrides)) or custom `values.yaml` fragments to lower per-service memory, batch sizes, or concurrency. See [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md).

g. When core services have fully started, `nvidia-smi` on a GPU node should show NIM / Triton processes similar to the following:

    ```
    # If it's taking > 1m for `nvidia-smi` to return, the bus will likely be busy setting up the models.
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |    0   N/A  N/A     83791      C   tritonserver                               2492MiB |
    |    0   N/A  N/A     85605      C   tritonserver                               1896MiB |
    |    0   N/A  N/A     85889      C   tritonserver                               2824MiB |
    |    0   N/A  N/A     88253      C   tritonserver                               2824MiB |
    |    0   N/A  N/A     91194      C   tritonserver                               4546MiB |
    +---------------------------------------------------------------------------------------+
    ```

h. Run `kubectl get pods -n <your-namespace>`. You should see pods reach `Running` / `Ready` similar to the following (names vary by release):

    ```text
    NAME                                          READY   STATUS    RESTARTS   AGE
    nemo-retriever-ms-runtime-xxxxxxxxxx-xxxxx    1/1     Running   0          7m
    nemoretriever-embedding-xxxxxxxxxx-xxxxx      1/1     Running   0          7m
    nemoretriever-page-elements-xxxxxxxxxx-xxxxx  1/1     Running   0          7m
    redis-master-0                                1/1     Running   0          7m
    ```

## Step 2: Ingest Documents

You can submit jobs programmatically in Python or using the [CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli).

!!! important "Python version"

    Install the client and CLI into an environment that uses Python 3.12 or later. The published packages require Python `>= 3.12`; using Python 3.10 or 3.11 typically fails with dependency resolution errors. Refer to [Prerequisites](prerequisites.md) and [Support Matrix](support-matrix.md).

The following examples demonstrate how to extract text, charts, tables, and images:

- **extract_text** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to find and extract text from pages.
- **extract_images** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to extract images.
- **extract_tables** — Uses [object detection family of NIMs](https://docs.nvidia.com/nim/ingestion/object-detection/latest/overview.html) to find tables and charts, and NemoRetriever OCR for table extraction.
- **extract_charts** — Enables or disables chart extraction, also based on the object detection NIM family.


### In Python

!!! tip

    For more Python examples, refer to the [Python Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).

<a id="ingest_python_example"></a>
```python
import logging, os, time
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
client = NvIngestClient(                                                                         
    message_client_port=7670,                                                               
    message_client_hostname="localhost"        
)                                                                 
# do content extraction from files                               
ingestor = (
    Ingestor(client=client)
    .files("data/multimodal_test.pdf")
    .extract(             
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        table_output_format="markdown",
        extract_infographics=True,
        # extract_method="nemotron_parse", # Slower, but maximally accurate, especially for PDFs with pages that are scanned images
        text_depth="page"
    ).embed()
    .vdb_upload(
        vdb_op="lancedb",
        uri="./lancedb_data",
        table_name="test",
        hybrid=False,
        # for llama-3.2 embedder, use 1024 for e5-v5
        dense_dim=2048,
    )
)

print("Starting ingestion..")
t0 = time.time()

# Return both successes and failures
# Use for large batches where you want successful chunks/pages to be committed, while collecting detailed diagnostics for failures.
results, failures = ingestor.ingest(show_progress=True, return_failures=True)

# Return only successes
# results = ingestor.ingest(show_progress=True)

t1 = time.time()
print(f"Total time: {t1-t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))

if failures:
    print(f"There were {len(failures)} failures. Sample: {failures[0]}")
```

!!! note

    For advanced visual parsing in self-hosted mode, uncomment `extract_method="nemotron_parse"` in the previous code. For more information, refer to [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse).


The output looks similar to the following.

```
Starting ingestion..
Indexed records into LanceDB
logged 8 records
Total time: 5.479151725769043 seconds
This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost   Chart 1 - Hammer - Powerdrill - Bluetooth speaker - Minifridge - Premium desk fan Dollars $- - $20.00 - $40.00 - $60.00 - $80.00 - $100.00 - $120.00 - $140.00 - $160.00 Cost
Table 1
| This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. |
| Animal | Activity | Place |
| Giraffe | Driving a car | At the beach |
| Lion | Putting on sunscreen | At the park |
| Cat | Jumping onto a laptop | In a home office |
| Dog | Chasing a squirrel | In the front yard |
TestingDocument
A sample document with headings and placeholder text
Introduction
This is a placeholder document that can be used for any purpose. It contains some 
headings and some placeholder text to fill the space. The text is not important and contains 
no real value, but it is useful for testing. Below, we will have some simple tables and charts 
that we can use to confirm Ingest is working as expected.
Table 1
This table describes some animals, and some activities they might be doing in specific 
locations.
Animal Activity Place
Gira@e Driving a car At the beach
Lion Putting on sunscreen At the park
Cat Jumping onto a laptop In a home o@ice
Dog Chasing a squirrel In the front yard
Chart 1
This chart shows some gadgets, and some very fictitious costs.
image_caption:[]
image_caption:[]
Below,is a high-quality picture of some shapes          Picture
Table 2
| This table shows some popular colors that cars might come in | This table shows some popular colors that cars might come in | This table shows some popular colors that cars might come in | This table shows some popular colors that cars might come in |
| Car | Color1 | Color2 | Color3 |
| Coupe | White | Silver | Flat Gray |
| Sedan | White | Metallic Gray | Matte Gray |
| Minivan | Gray | Beige | Black |
| Truck | Dark Gray | Titanium Gray | Charcoal |
| Convertible | Light Gray | Graphite | Slate Gray |
Section One
This is the first section of the document. It has some more placeholder text to show how 
the document looks like. The text is not meant to be meaningful or informative, but rather to 
demonstrate the layout and formatting of the document.
• This is the first bullet point
• This is the second bullet point
• This is the third bullet point
Section Two
This is the second section of the document. It is more of the same as we’ve seen in the rest 
of the document. The content is meaningless, but the intent is to create a very simple 
smoke test to ensure extraction is working as intended. This will be used in CI as time goes 
on to ensure that changes we make to the library do not negatively impact our accuracy.
Table 2
This table shows some popular colors that cars might come in.
Car Color1 Color2 Color3
Coupe White Silver Flat Gray
Sedan White Metallic Gray Matte Gray
Minivan Gray Beige Black
Truck Dark Gray Titanium Gray Charcoal
Convertible Light Gray Graphite Slate Gray
Picture
Below, is a high-quality picture of some shapes.
image_caption:[]
image_caption:[]
This chart shows some average frequency ranges for speaker drivers. Frequency Ranges ofSpeaker Drivers   Tweeter - Midrange - Midwoofer - Subwoofer Chart2 Hertz (log scale) 1 - 10 - 100 - 1000 - 10000 - 100000 FrequencyRange Start (Hz) - Frequency Range End (Hz)
Chart 2
This chart shows some average frequency ranges for speaker drivers.
Conclusion
This is the conclusion of the document. It has some more placeholder text, but the most 
important thing is that this is the conclusion. As we end this document, we should have 
been able to extract 2 tables, 2 charts, and some text including 3 bullet points.
image_caption:[]

```

### Using the CLI

!!! tip

    There is a Jupyter notebook available to help you get started with the CLI. For more information, refer to the [CLI Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/cli_client_usage.ipynb).

<a id="ingest_cli_example"></a>
```shell
nemo-retriever \
  --doc ./data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true", "extract_charts": "true"}' \
  --client_host=localhost \
  --client_port=7670
```

You should see output that indicates the document processing status followed by a breakdown of time spent during job execution.

```
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[nltk_data] Downloading package punkt_tab to
[nltk_data]     /path/to/your/venv/lib/python3.12/site-
[nltk_data]     packages/llama_index/core/_static/nltk_cache...
[nltk_data]   Package punkt_tab is already up-to-date!
INFO:retriever_client.cli:Processing 1 documents.
INFO:retriever_client.cli:Output will be written to: ./processed_docs
Processing files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.34s/file, pages_per_sec=1.28]
INFO:retriever_client.processing:message_broker_task_source: Avg: 2.39 ms, Median: 2.39 ms, Total Time: 2.39 ms, Total % of Trace Computation: 0.06%
INFO:retriever_client.processing:broker_source_network_in: Avg: 9.51 ms, Median: 9.51 ms, Total Time: 9.51 ms, Total % of Trace Computation: 0.25%
INFO:retriever_client.processing:job_counter: Avg: 1.47 ms, Median: 1.47 ms, Total Time: 1.47 ms, Total % of Trace Computation: 0.04%
INFO:retriever_client.processing:job_counter_channel_in: Avg: 0.46 ms, Median: 0.46 ms, Total Time: 0.46 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:metadata_injection: Avg: 3.52 ms, Median: 3.52 ms, Total Time: 3.52 ms, Total % of Trace Computation: 0.09%
INFO:retriever_client.processing:metadata_injection_channel_in: Avg: 0.16 ms, Median: 0.16 ms, Total Time: 0.16 ms, Total % of Trace Computation: 0.00%
INFO:retriever_client.processing:pdf_content_extractor: Avg: 475.64 ms, Median: 163.77 ms, Total Time: 2378.21 ms, Total % of Trace Computation: 62.73%
INFO:retriever_client.processing:pdf_content_extractor_channel_in: Avg: 0.31 ms, Median: 0.31 ms, Total Time: 0.31 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:image_content_extractor: Avg: 0.67 ms, Median: 0.67 ms, Total Time: 0.67 ms, Total % of Trace Computation: 0.02%
INFO:retriever_client.processing:image_content_extractor_channel_in: Avg: 0.21 ms, Median: 0.21 ms, Total Time: 0.21 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:docx_content_extractor: Avg: 0.46 ms, Median: 0.46 ms, Total Time: 0.46 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:docx_content_extractor_channel_in: Avg: 0.20 ms, Median: 0.20 ms, Total Time: 0.20 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:pptx_content_extractor: Avg: 0.68 ms, Median: 0.68 ms, Total Time: 0.68 ms, Total % of Trace Computation: 0.02%
INFO:retriever_client.processing:pptx_content_extractor_channel_in: Avg: 0.46 ms, Median: 0.46 ms, Total Time: 0.46 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:audio_data_extraction: Avg: 1.08 ms, Median: 1.08 ms, Total Time: 1.08 ms, Total % of Trace Computation: 0.03%
INFO:retriever_client.processing:audio_data_extraction_channel_in: Avg: 0.20 ms, Median: 0.20 ms, Total Time: 0.20 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:dedup_images: Avg: 0.42 ms, Median: 0.42 ms, Total Time: 0.42 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:dedup_images_channel_in: Avg: 0.42 ms, Median: 0.42 ms, Total Time: 0.42 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:filter_images: Avg: 0.59 ms, Median: 0.59 ms, Total Time: 0.59 ms, Total % of Trace Computation: 0.02%
INFO:retriever_client.processing:filter_images_channel_in: Avg: 0.57 ms, Median: 0.57 ms, Total Time: 0.57 ms, Total % of Trace Computation: 0.02%
INFO:retriever_client.processing:table_data_extraction: Avg: 240.75 ms, Median: 240.75 ms, Total Time: 481.49 ms, Total % of Trace Computation: 12.70%
INFO:retriever_client.processing:table_data_extraction_channel_in: Avg: 0.38 ms, Median: 0.38 ms, Total Time: 0.38 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:chart_data_extraction: Avg: 300.54 ms, Median: 299.94 ms, Total Time: 901.62 ms, Total % of Trace Computation: 23.78%
INFO:retriever_client.processing:chart_data_extraction_channel_in: Avg: 0.23 ms, Median: 0.23 ms, Total Time: 0.23 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:infographic_data_extraction: Avg: 0.77 ms, Median: 0.77 ms, Total Time: 0.77 ms, Total % of Trace Computation: 0.02%
INFO:retriever_client.processing:infographic_data_extraction_channel_in: Avg: 0.25 ms, Median: 0.25 ms, Total Time: 0.25 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:caption_ext: Avg: 0.55 ms, Median: 0.55 ms, Total Time: 0.55 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:caption_ext_channel_in: Avg: 0.51 ms, Median: 0.51 ms, Total Time: 0.51 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:embed_text: Avg: 1.21 ms, Median: 1.21 ms, Total Time: 1.21 ms, Total % of Trace Computation: 0.03%
INFO:retriever_client.processing:embed_text_channel_in: Avg: 0.21 ms, Median: 0.21 ms, Total Time: 0.21 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:message_broker_task_sink_channel_in: Avg: 0.42 ms, Median: 0.42 ms, Total Time: 0.42 ms, Total % of Trace Computation: 0.01%
INFO:retriever_client.processing:No unresolved time detected. Trace times account for the entire elapsed duration.
INFO:retriever_client.processing:Processed 1 files in 2.34 seconds.
INFO:retriever_client.processing:Total pages processed: 3
INFO:retriever_client.processing:Throughput (Pages/sec): 1.28
INFO:retriever_client.processing:Throughput (Files/sec): 0.43
```

## Step 3: Inspecting and Consuming Results

After the ingestion steps above have been completed, you should be able to find the `text` and `image` subfolders inside your processed docs folder. Each will contain JSON-formatted extracted content and metadata.

When processing has completed, you'll have separate result files for text and image data:
```shell
ls -R processed_docs/
```
```shell
processed_docs/:
image  structured  text

processed_docs/image:
multimodal_test.pdf.metadata.json

processed_docs/structured:
multimodal_test.pdf.metadata.json

processed_docs/text:
multimodal_test.pdf.metadata.json
```

For the full metadata definitions, refer to [Content Metadata](content-metadata.md). 

We also provide a script for inspecting [extracted images](https://github.com/NVIDIA/NeMo-Retriever/blob/main/src/util/image_viewer.py).

First, install `tkinter` by running the following code. Choose the code for your OS.

- For Ubuntu/Debian Linux:

    ```shell
    sudo apt-get update
    sudo apt-get install python3-tk
    ```

- For Fedora/RHEL Linux:

    ```shell
    sudo dnf install python3-tkinter
    ```

- For macOS using Homebrew:

    ```shell
    brew install python-tk
    ```

Then, run the following command to execute the script for inspecting the extracted image:

```shell
python src/util/image_viewer.py --file_path ./processed_docs/image/multimodal_test.pdf.metadata.json
```

!!! tip

    Beyond inspecting the results, you can read them into things like [llama-index](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/llama_index_multimodal_rag.ipynb) or [langchain](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/langchain_multimodal_rag.ipynb) retrieval pipelines. Also, checkout our [Enterprise RAG Blueprint on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) to query over document content pre-extracted with the retriever pipeline.



## Profile Information

The following table maps **logical capability bundles** (optional NIM and storage stacks) to what you enable in **Helm values** when deploying NeMo Retriever extraction. Enable the rows you need for your workload; exact value paths depend on chart version—use the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) as the source of truth.

| Capability            | Type     | Description                                                       | 
|-----------------------|----------|-------------------------------------------------------------------| 
| `retrieval`           | Core     | Embedding NIM for passage and query embeddings; pair with LanceDB in your client configuration for vector storage. | 
| `audio`               | Advanced | [parakeet-1-1b-ctc-en-us](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html) ASR NIM for audio files. See [Audio Processing](audio-video.md). | 
| `nemotron-parse`      | Advanced | [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) for higher-accuracy text and table extraction. |
| `vlm`                 | Advanced | [Nemotron Nano 12B v2 VL](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl/modelcard) for image captioning. Enables the `caption` API. See [Use Multimodal Embedding](embedding.md) and [Extract Captions from Images](nemo-retriever-api-reference.md). | 

### Example: VLM captioning for infographics

Infographics often combine text, charts, and diagrams into complex visuals. Vision-language model (VLM) captioning generates natural language descriptions that capture this complexity, making the content searchable and more accessible for downstream applications.

Enable the VLM / captioning NIM and the retrieval (embedding) stack your values require in your Helm values, then upgrade the release. Refer to the chart README and [embedding](embedding.md) for the environment variables that select the multimodal embedding and caption stack.

## Air-gapped deployment (Kubernetes)

When deploying without internet or NGC registry access from the cluster, follow [Air-Gapped Deployment (Kubernetes)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) in the Helm chart documentation and the [NVIDIA NIM Operator air-gap guide](https://docs.nvidia.com/nim-operator/latest/air-gap.html): mirror images to your registry, load secrets, then install or upgrade the chart with those private references. Keep the same image tags and values between staging and production so configuration stays consistent.

## Helm values and GPU-specific overrides

Default chart settings might exceed VRAM on a single GPU for some hardware targets. Use **values fragments** or the checked-in examples under [`helm/overrides/`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/helm/overrides) (for example A10G, A100 40GB, L40S) to lower per-service memory, batch sizes, or concurrency. Merge the override file with `-f` when you run `helm upgrade`, or maintain a single consolidated `values.yaml` per environment.

| Example override asset (repo) | GPU target |
|--------------------------------|------------|
| `helm/overrides/values-a10g.yaml` | NVIDIA A10G |
| `helm/overrides/values-a100-40gb.yaml` | NVIDIA A100-SXM4-40GB |
| `helm/overrides/values-l40s.yaml` | NVIDIA L40S |

For RTX Pro 6000 Server Edition and other GPUs with limited VRAM, start from the override that best matches your GPU memory, then tune further.

### Example: merge a GPU override with `helm upgrade`

```shell
helm upgrade --install nemo-retriever <chart-or-tgz> -n "${NAMESPACE}" \
  -f my-base-values.yaml \
  -f helm/overrides/values-a100-40gb.yaml
```

Use the same pattern for other override files, substituting the path that matches your GPU.


## Specify MIG slices for NIM models

When you deploy the pipeline with NIM models on MIG‑enabled GPUs, MIG device slices are requested and scheduled through the `values.yaml` file for the corresponding NIM microservice. For IBM Content-Aware Storage (CAS) deployments, this allows NIM pods to land only on nodes that expose the desired MIG profiles (see the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)).

To target a specific MIG profile—for example, a 3g.20gb slice on an A100, which is a hardware-partitioned virtual GPU instance that gives your workload a fixed mid-sized share of the A100’s compute plus 20 GB of dedicated GPU memory and behaves like a smaller independent GPU—for a given NIM, configure the `resources` and `nodeSelector` under that NIM’s values path in `values.yaml`.

The following example shows the pattern. Paths vary by NIM, such as `nvingest.nvidiaNim.nemoretrieverPageElements` instead of the generic `nvingest.nim` placeholder. For details refer to the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) and the chart listing on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo-microservices/containers).
Set `resources.requests` and `resources.limits` to the name of the MIG resource that you want (for example, `nvidia.com/mig-3g.20gb`).
```shell
nvingest:
  nvidiaNim:
    nemoretrieverPageElements:
      modelName: "nvidia/nemotron-page-elements-v3"  # Page-elements NIM (not a text LLM)
      resources:
        limits:
          nvidia.com/mig-3g.20gb: 1               # MIG profile resource
        requests:
          nvidia.com/mig-3g.20gb: 1
      nodeSelector:
        nvidia.com/gpu.product: A100-SXM4-40GB-MIG-3g.20gb
```
Key points:

 * Use the appropriate NIM‑specific values path (for example, `nvingest.nvidiaNim.nemoretrieverPageElements.resources`) rather than the generic `nvingest.nim` placeholder.
 * Set `resources.requests` and `resources.limits` to the desired MIG resource name (for example, `nvidia.com/mig-3g.20gb`).
 * Use `nodeSelector` (or tolerations/affinity, if you prefer) to target nodes labeled with the corresponding MIG‑enabled GPU product (for example, `nvidia.com/gpu.product: A100-SXM4-40GB-MIG-3g.20gb`).
This syntax and structure can be repeated for each NIM model used by CAS, ensuring that each NeMo Retriever Library NIM pod is mapped to the correct MIG slice type and scheduled onto compatible nodes.

!!! important

    Advanced features require additional GPU support and disk space. For more information, refer to [Support Matrix](support-matrix.md).

## Related Topics

- [Troubleshoot](troubleshoot.md)
- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)
- [Notebooks](notebooks.md)
- [Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
