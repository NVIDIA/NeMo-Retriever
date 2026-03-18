<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

**Important: The default branch is main, which tracks active development and may be ahead of the latest supported release.**

For the latest stable release:

Use the latest release/* branch (for example, release/26.03) from the branch dropdown.

See the corresponding NeMo Retriever Library documentation at https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/

# NeMo Retriever Library

NeMo Retriever Library is a scalable, performance-oriented document content and metadata extraction microservice. It uses specialized NVIDIA NIM microservices 
to find, contextualize, and extract text, tables, charts and infographics that you can use in downstream generative applications.

> [!Note]
> NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.

NeMo Retriever Library enables parallelization of splitting documents into pages where artifacts are classified (such as text, tables, charts, and infographics), extracted, and further contextualized through optical character recognition (OCR) into a well defined JSON schema. From there, NeMo Retriever Library manages computaiton of embeddings for the extracted content as well as storing them in a vector database [Milvus](https://milvus.io/).

The following diagram shows the NeMo Retriever Library pipeline.

![Pipeline Overview](https://docs.nvidia.com/nemo/retriever/extraction/images/overview-extraction.png)

## Table of Contents
1. [NeMo Retriever Library](#nemo-retriever-library)
2. [Prerequisites](#prerequisites)
3. [Quickstart](#library-mode-quickstart)
4. [Benchmarking](#benchmarking)
5. [GitHub Repository Structure](#github-repository-structure)
6. [Notices](#notices)


## What is NeMo Retriever Library?

The NeMo Retriever Library is a library and microservice framework designed to perform the following functions::

- Accept a job specification that contains a document payload and a set of ingestion tasks to perform on that payload.
- Store the result of each job to retrieve later. The result is a dictionary that contains a list of metadata that describes the objects extracted from the base document, and processing annotations and timing/trace data.
- Support multiple methods of extraction for each document type to balance trade-offs between throughput and accuracy. For example, for .pdf documents, extraction can be performeded using pdfium, [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse), Unstructured.io, and Adobe Content Extraction Services.
- Support various types of before and after processing operations, including text splitting and chunking, transform and filtering, embedding generation, and image offloading to storage.


NeMo Retriever Extraction supports PDFs, docx/pptx, HTML, text files, as well as audio and video files. You can find a complete list [here](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/).

### What NeMo Retriever Library Isn't

NeMo Retriever Library does not do the following:

- Run a static pipeline or fixed set of operations on every submitted document.
- Act as a wrapper for any specific document parsing library.


For more information, refer to the [NeMo Retriever Library documentation](https://docs.nvidia.com/nemo/retriever/extraction/overview/).

## Documentation Resources

- **[Official Documentation](https://docs.nvidia.com/nemo/retriever/extraction/)** - Complete user guides, API references, and deployment instructions
- **[Getting Started Guide](https://docs.nvidia.com/nemo/retriever/extraction/overview/)** - Overview and prerequisites for production deployments
- **[Benchmarking Guide](https://docs.nvidia.com/nemo/retriever/extraction/benchmarking/)** - Performance testing and recall evaluation framework
- **[MIG Deployment](https://docs.nvidia.com/nemo/retriever/extraction/mig-benchmarking/)** - Multi-Instance GPU configurations for Kubernetes
- **[API Documentation](https://docs.nvidia.com/nemo/retriever/extraction/api/)** - Python client and API reference


## Prerequisites

For production-level performance and scalability, we recommend that you deploy the pipeline and supporting NIMs by using Kubernetes ([helm charts](helm)). For more information, refer to [prerequisites](https://docs.nvidia.com/nv-ingest/user-guide/getting-started/prerequisites).


## Library Quickstart

For small-scale workloads, such as workloads of fewer than 100 PDFs, you can use our library setup which works with HuggingFace models on local GPUs or with NIMs hosted on build.nvidia.com.

After [following the installation steps](nemo_retriever), you can start ingesting content like with the following snippet:
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

# ingestor.ingest() actually executes the pipeline
# results are returned as a ray dataset and inspectable as chunks
ray_dataset = ingestor.ingest()
chunks = ray_dataset.get_dataset().take_all()
```

You can see the extracted text that represents the content of the ingested test document.

```python
# page 1 raw text:
>>> chunks[0]["text"]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose...'

# markdown formatted table from the first page
>>> chunks[1]["text"]
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

### Step 3: Query Ingested Content

To query for relevant snippets of the ingested content, and use them with an LLM to generate answers, use the following code.

```python
from nemo_retriever.retriever import Retriever
from openai import OpenAI
import os

retriever = Retriever()

query = "Given their activities, which animal is responsible for the typos in my documents?"

# you can also submit a list with retriever.queries[...]
hits = retriever.query(query)

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
```shell
Cat is the animal whose activity (jumping onto a laptop) matches the location of the typos, so the cat is responsible for the typos in the documents.
```

> [!TIP]
> Beyond inspecting the results, you can read them into things like [llama-index](examples/llama_index_multimodal_rag.ipynb) or [langchain](examples/langchain_multimodal_rag.ipynb) retrieval pipelines.
>
> Please also checkout our [demo using a retrieval pipeline on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) to query over document content pre-extracted w/ NVIDIA Ingest.


## Benchmarking

nv-ingest includes a comprehensive testing framework for benchmarking performance and evaluating retrieval accuracy.

### Quick Start

```bash
cd tools/harness

uv sync

# Run end-to-end benchmark
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Evaluate retrieval accuracy
uv run nv-ingest-harness-run --case=e2e_recall --dataset=bo767
```

### Available Benchmarks

- **End-to-End Performance** - Measure ingestion throughput, latency, and resource utilization
- **Retrieval Accuracy** - Evaluate recall@k metrics against ground truth datasets
- **MIG Benchmarking** - Test performance with NVIDIA Multi-Instance GPU (MIG) configurations

### Documentation

- **[Testing Framework Guide](https://docs.nvidia.com/nemo/retriever/extraction/benchmarking/)** - Complete guide to benchmarking and testing nv-ingest (same as `tools/harness/README.md`)
- **[MIG Benchmarking](https://docs.nvidia.com/nemo/retriever/extraction/mig-benchmarking/)** - GPU partitioning for multi-tenant deployments on Kubernetes/Helm

### Benchmark Datasets

- **bo767** - 767 PDF documents with ground truth for recall evaluation
- **bo20** - 20 PDF documents for quick validation
- **single** - singular multimodal pdf for quick validation
- **earnings** - earnings reports ppt and pdf dataset
-- **financebench** - financial data
- **Custom datasets** - Use your own datasets with the testing framework

For more information, see the [benchmarking documentation](https://docs.nvidia.com/nemo/retriever/extraction/benchmarking/).


## GitHub Repository Structure

The following is a description of the folders in the GitHub repository.

- [.devcontainer](https://github.com/NVIDIA/nv-ingest/tree/main/.devcontainer) — VSCode containers for local development
- [.github](https://github.com/NVIDIA/nv-ingest/tree/main/.github) — GitHub repo configuration files
- [api](https://github.com/NVIDIA/nv-ingest/tree/main/api) — Core API logic shared across python modules
- [ci](https://github.com/NVIDIA/nv-ingest/tree/main/ci) — Scripts used to build the nv-ingest container and other packages
- [client](https://github.com/NVIDIA/nv-ingest/tree/main/client) — Readme, examples, and source code for the nv-ingest-cli utility
- [config](https://github.com/NVIDIA/nv-ingest/tree/main/config) — Various .yaml files defining configuration for OTEL, Prometheus
- [data](https://github.com/NVIDIA/nv-ingest/tree/main/data) — Sample PDFs for testing
- [deploy](https://github.com/NVIDIA/nv-ingest/tree/main/deploy) — Brev.dev-hosted launchable
- [docker](https://github.com/NVIDIA/nv-ingest/tree/main/docker) — Scripts used by the nv-ingest docker container
- [docs](https://github.com/NVIDIA/nv-ingest/tree/main/docs/docs) — Documentation for NV Ingest
- [evaluation](https://github.com/NVIDIA/nv-ingest/tree/main/evaluation) — Notebooks that demonstrate how to test recall accuracy
- [examples](https://github.com/NVIDIA/nv-ingest/tree/main/examples) — Notebooks, scripts, and tutorial content
- [helm](https://github.com/NVIDIA/nv-ingest/tree/main/helm) — Documentation for deploying nv-ingest to a Kubernetes cluster via Helm chart
- [skaffold](https://github.com/NVIDIA/nv-ingest/tree/main/skaffold) — Skaffold configuration
- [src](https://github.com/NVIDIA/nv-ingest/tree/main/src) — Source code for the nv-ingest pipelines and service
- [tests](https://github.com/NVIDIA/nv-ingest/tree/main/tests) — Unit tests for nv-ingest


## Notices

### Third Party License Notice:

If configured to do so, this project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use:

https://pypi.org/project/pdfservices-sdk/

- **`INSTALL_ADOBE_SDK`**:
  - **Description**: If set to `true`, the Adobe SDK will be installed in the container at launch time. This is
    required if you want to use the Adobe extraction service for PDF decomposition. Please review the
    [license agreement](https://github.com/adobe/pdfservices-python-sdk?tab=License-1-ov-file) for the
    pdfservices-sdk before enabling this option.
- **Built With Llama**:
  - **Description**: The NV-Ingest container comes with the `meta-llama/Llama-3.2-1B` tokenizer pre-downloaded so 
    that the split task can use it for token-based splitting without making a network request. The [Llama 3.2 Community License Agreement](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt) governs your use of these Llama materials.
    
    If you're building the container yourself and want to pre-download this model, you'll first need to set 
    `DOWNLOAD_LLAMA_TOKENIZER` to `True`. Because this is a gated model, you'll also need to 
    [request access](https://huggingface.co/meta-llama/Llama-3.2-1B) and set `HF_ACCESS_TOKEN` to your HuggingFace 
    access token in order to use it.


### Contributing

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original
work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not signed off are not accepted.

To sign off on a commit, use the --signoff (or -s) option when you commit your changes as shown following.

```
$ git commit --signoff --message "Add cool feature."
```

This appends the following text to your commit message.

```
Signed-off-by: Your Name <your@email.com>
```

#### Developer Certificate of Origin (DCO)

The following is the full text of the Developer Certificate of Origin (DCO)

```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```
  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```


## Security Considerations

- NeMo Retriever Extraction doesn't generate any code that may require sandboxing.
- NeMo Retriever Extraction is shared as a reference and is provided "as is". The security in the production environment is the responsibility of the end users deploying it. When deploying in a production environment, please have security experts review any potential risks and threats; define the trust boundaries, implement logging and monitoring capabilities, secure the communication channels, integrate AuthN & AuthZ with appropriate access controls, keep the deployment up to date, ensure the containers/source code are secure and free of known vulnerabilities.
- A frontend that handles AuthN & AuthZ should be in place as missing AuthN & AuthZ could provide ungated access to customer models if directly exposed to e.g. the internet, resulting in either cost to the customer, resource exhaustion, or denial of service.
- NeMo Retriever Extraction doesn't require any privileged access to the system.
- The end users are responsible for ensuring the availability of their deployment.
- The end users are responsible for building the container images and keeping them up to date.
- The end users are responsible for ensuring that OSS packages used by the developer blueprint are current.
- The logs from nginx proxy, backend, and demo app are printed to standard out. They can include input prompts and output completions for development purposes. The end users are advised to handle logging securely and avoid information leakage for production use cases.
