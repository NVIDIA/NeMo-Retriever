# Concepts

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed NeMo Retriever Library.


These terms appear throughout NeMo Retriever Library documentation.

## Job

A **job** is a unit of work you submit with a JSON description: a document payload (or reference) and a list of **ingestion tasks** to run on that payload. Results are retrieved as structured metadata and annotations.

## Pipeline and tasks

NeMo Retriever Library does **not** run one static pipeline on every document. You configure **tasks** such as parsing, chunking, embedding, storage, and filtering per job. Related topics: [Customize your pipeline](user-defined-functions.md), [user-defined stages](user-defined-stages.md).

## Extraction metadata

Output is typically a **JSON dictionary** listing extracted objects (text regions, tables, images, and so on), processing notes, and timing or trace data. Field-level detail is in the [metadata reference](content-metadata.md).

## Embeddings and retrieval

Optionally, the library can compute **embeddings** for extracted content and store vectors in a database such as [LanceDB](https://lancedb.com/) or [Milvus](https://milvus.io/) for downstream **semantic or hybrid search** in your application.

## Deployment modes

- **Library mode** — Run without the full container stack where appropriate ([quickstart](quickstart-library-mode.md)).
- **Helm / Kubernetes** — [Helm-based deployment](helm.md) for cluster operations.
- **Notebooks** — [Jupyter examples](notebooks.md) for experimentation and RAG demos.

For a concise comparison, see [Choose your path](choose-your-path.md).
