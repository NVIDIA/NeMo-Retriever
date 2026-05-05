# What is NeMo Retriever Library?

NVIDIA NeMo Retriever Library (NRL) is a high retrieval accuracy, performant, and scalable framework for content and metadata extraction from various media types (PDFs, HTML, Word docs, Powerpoint, audio, video, and image files). It supports both NVIDIA NIM microservices and a range of models to find, contextualize, and extract text, tables, charts, infographics, and transcripts for use in downstream generative and retrieval-augmented applications.

NeMo Retriever Library enables parallelization of splitting documents into pages where sub-page content is classified (such as text paragraphs, tables, charts, and infographics), extracted, and further contextualized through optical character recognition (OCR) into a standard schema. From there, NeMo Retriever Library manages computation of embeddings for the extracted content,
and can store vectors in [LanceDB](https://lancedb.com/) for the recommended embedded path when you pass `vdb_op="lancedb"` to upload (see [Data Upload](vdbs.md)).

## What NeMo Retriever Library Is ✔️

The following diagram shows the retriever pipeline.

![Overview diagram](images/overview-extraction.png)

NeMo Retriever Library does the following:

- Accept directories of input files and a series of configurable ingestion tasks to perform on that input
- Allow the extracted content be retrieved from a VDB containing discrete metadata element
- Support multiple methods of extraction for each document type to balance trade-offs between throughput and accuracy. For example, for .pdf documents, extraction is performed by using pdfium and [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse).
- Support various types of pre- and post- processing operations, including text splitting and chunking, transform and filtering, embedding generation, and image offloading to storage.

NeMo Retriever Library supports the following file types:

- `avi`
- `bmp`
- `docx`
- `html` (converted to markdown format)
- `jpeg`
- `json` (treated as text)
- `md` (treated as text)
- `mkv` 
- `mov` 
- `mp3`
- `mp4` 
- `pdf`
- `png`
- `pptx`
- `sh` (treated as text)
- `svg` (NeMo Retriever Library only, requires `cairosvg`)
- `tiff`
- `txt`
- `wav`

## Related Topics

- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Deploy As a Library](quickstart-library-mode.md)
- [Deploy on Kubernetes with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)
- [Deployment options](deployment-options.md) — library, Helm, hosted vs self-hosted NIMs in one place
- [Notebooks](notebooks.md)
- [NVIDIA AI Blueprints catalog](https://build.nvidia.com/explore/discover) — solution cards, enterprise RAG blueprints, and end-to-end patterns (including [Enterprise RAG — multimodal PDF data extraction](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)); for integration pathways, also see [Integrations](integrations-langchain-llamaindex-haystack.md).
