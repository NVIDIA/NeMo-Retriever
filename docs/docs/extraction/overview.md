# What is NeMo Retriever Library?

NVIDIA NeMo Retriever Library (NRL) is a high retrieval accuracy, performant, and scalable framework for content and metadata extraction from various media types (PDFs, HTML, Word docs, Powerpoint, audio, video, and image files). It supports both NVIDIA NIM microservices and a range of models to find, contextualize, and extract text, tables, charts, infographics, and transcripts for use in downstream generative and retrieval-augmented applications.

!!! note

    This documentation describes NeMo Retriever Library.

NeMo Retriever Library enables parallelization of splitting documents into pages where sub-page content is classified (such as text paragraphs, tables, charts, and infographics), extracted, and further contextualized through optical character recognition (OCR) into a standard schema. From there, NeMo Retriever Library manages computation of embeddings for the extracted content, 
and stores into a vector database ([LanceDB by default](https://lancedb.com/).

## What NeMo Retriever Library Is ✔️

The following diagram shows the retriever pipeline.

![Overview diagram](images/overview-extraction.png)

NeMo Retriever Library is a microservice service that does the following:

- Accept a JSON job description, containing a document payload, and a set of ingestion tasks to perform on that payload.
- Allow the results of a job to be retrieved. The result is a JSON dictionary that contains a list of metadata describing objects extracted from the base document, and processing annotations and timing/trace data.
- Support multiple methods of extraction for each document type to balance trade-offs between throughput and accuracy. For example, for .pdf documents, extraction is performed by using pdfium and [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse).
- Support various types of pre- and post- processing operations, including text splitting and chunking, transform and filtering, embedding generation, and image offloading to storage.

NeMo Retriever Library supports the following file types:

- `avi` (early access)
- `bmp`
- `docx`
- `html` (converted to markdown format)
- `jpeg`
- `json` (treated as text)
- `md` (treated as text)
- `mkv` (early access)
- `mov` (early access)
- `mp3`
- `mp4` (early access)
- `pdf`
- `png`
- `pptx`
- `sh` (treated as text)
- `svg` (NeMo Retriever Library only, requires `cairosvg`)
- `tiff`
- `txt`
- `wav`


## What NeMo Retriever Library Isn't ✖️

NeMo Retriever Library does not do the following:

- Run a static pipeline or fixed set of operations on every submitted document.
- Act as a wrapper for any specific document parsing library.


## Related Topics

- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)
- [Notebooks](notebooks.md)
- [Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
