# Support Matrix for NeMo Retriever Library

Before you begin using [NeMo Retriever Library](overview.md), ensure that you have the hardware for your use case.


## Core and Advanced Pipeline Features

The Nemo Retriever Library extraction core pipeline features run on a single A10G or better GPU. 
The core pipeline models (for document type inputs) include the following:

- [llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) — Embedding model for converting text chunks into vectors.
- [nemotron-page-elements-v3](https://huggingface.co/nvidia/nemotron-page-elements-v3) — Detects and classifies images on a page as a table, chart or infographic.
- [nemotron-table-structure-v1](https://huggingface.co/nvidia/nemotron-table-structure-v1) — Detects rows, columns, and cells within a table to preserve table structure and convert to Markdown format. 
- [nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2) — Image OCR model to detect and extract text from images.

Advanced features (for example, for audio/video) require additional GPU support and disk space. 
This includes the following:

- [parakeet-1-1b-ctc-en-us](https://huggingface.co/nvidia/parakeet-ctc-1.1b) for transcript extraction from [audio and video](audio.md).
- [nemotron-parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) - for [maximally accurate table extraction](nemoretriever-parse.md).
- [nemotron-nano-12b-v2-vl](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2) for image captioning of unstructured (not charts, tables, infographics) images. 
    
    !!! note
    
        While nemotron-nano-12b-v2-vl is the default VLM, you can configure and use other vision language models for image captioning based on your specific use case requirements. For more information, refer to [Extract Captions from Images](python-api-reference.md#extract-captions-from-images).

- [llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) for improved retrieval accuracy.

## Hardware Requirements

NeMo Retriever Library supports the following GPU hardware given system constraints in the table:

|Feature         | GPU Option                | [RTX Pro 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)  | [B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)          | [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/)      | [H100](https://www.nvidia.com/en-us/data-center/h100/)        | [A100 80GB](https://www.nvidia.com/en-us/data-center/a100/)   | A100 40GB     | [A10G](https://aws.amazon.com/ec2/instance-types/g5/)          | L40S   | [RTX PRO 4500 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) |
|----------------|---------------------------|---------------|---------------|---------------|-------------|-------------|---------------|---------------|--------|------------------------|
| GPU            | Memory                    | 96GB          | 180GB         | 141GB         | 80GB        | 80GB        | 40GB          | 24GB          | 48GB   | 32GB GDDR7 (GB203)     |
| Core Features  | Total GPUs                | 1             | 1             | 1             | 1           | 1           | 1             | 1             | 1      | 1                      |
| Core Features  | Total Disk Space          | ~150GB        | ~150GB        | ~150GB        | ~150GB      | ~150GB      | ~150GB        | ~150GB        | ~150GB | ~150GB                 |
| Audio (parakeet-1-1b-ctc-en-us) | Additional Dedicated GPUs | 1             | 1             | 1             | 1           | 1           | 1             | 1             | 1      | 1¹                     |
| Audio (parakeet-1-1b-ctc-en-us) | Additional Disk Space     | ~37GB         | ~37GB         | ~37GB         | ~37GB       | ~37GB       | ~37GB         | ~37GB         | ~37GB  | ~37GB¹                 |
| nemotron-parse | Additional Dedicated GPUs | Not supported | Not supported | Not supported | 1           | 1           | 1             | 1             | 1      | Not supported²         |
| nemotron-parse | Additional Disk Space     | Not supported | Not supported | Not supported | ~16GB       | ~16GB       | ~16GB         | ~16GB         | ~16GB  | Not supported²         |
| VLM            | Additional Dedicated GPUs | 1             | 1             | 1             | 1           | 1           | Not supported | Not supported | 1      | Not supported³         |
| VLM            | Additional Disk Space     | ~16GB         | ~16GB         | ~16GB         | ~16GB       | ~16GB       | Not supported | Not supported | ~16GB  | Not supported³         |
| Reranker       | With Core Pipeline        | Yes           | Yes           | Yes           | Yes         | Yes         | No*           | No*           | No*    | No*                    |
| Reranker       | Standalone (recall only)  | Yes           | Yes           | Yes           | Yes         | Yes         | Yes           | Yes           | Yes    | Yes                    |

¹ Audio runs but requires runtime engine build — no pre-defined model profile.

² Nemotron Parse fails to start on 32GB.

³ VLM fails to load on 32GB, 32GB is below the minimum threshold.

\* GPUs with less than 80GB VRAM cannot run the reranker concurrently with the core pipeline. 
To perform recall testing with the reranker on these GPUs, shut down the core pipeline NIM microservices 
and run only the embedder, reranker, and your vector database.

## Related Topics

- [Prerequisites](prerequisites.md)
- [Release Notes](releasenotes.md)
- [NVIDIA NIM for Vision Language Models Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NVIDIA Speech NIM Microservices](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html)
