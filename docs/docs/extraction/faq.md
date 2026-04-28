# Frequently Asked Questions for NeMo Retriever Library

This documentation contains the Frequently Asked Questions (FAQ) for [NeMo Retriever Library](overview.md).

## What if I already have a retrieval pipeline? Can I just use NeMo Retriever Library? 

You can use the CLI or Python APIs to perform extraction only, and then consume the results.
Using the Python API, `results` is a list object with one entry.
For code examples, refer to the Jupyter notebooks [Multimodal RAG with LlamaIndex](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/llama_index_multimodal_rag.ipynb) 
and [Multimodal RAG with LangChain](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/langchain_multimodal_rag.ipynb).



## Where does NeMo Retriever Library ingest to?

NeMo Retriever Library supports extracting text representations of various forms of content,
and ingesting to a vector database. **[LanceDB](https://lancedb.com/) is the default**; [Milvus](https://milvus.io/) is also fully supported.
NeMo Retriever Library does not store data on disk except through the vector database (LanceDB uses local Lance files; Milvus uses its server and MinIO).
You can ingest to other data stores; however, you must configure other data stores yourself.
For more information, refer to [Data Upload](vdbs.md).



## How would I process unstructured images?

For images that `nemoretriever-page-elements-v3` does not classify as tables, charts, or infographics,
you can use our VLM caption task to create a dense caption of the detected image. 
That caption is then be embedded along with the rest of your content. 
For more information, refer to [Extract Captions from Images](python-api-reference.md#extract-captions-from-images).



## When should I consider advanced visual parsing?

For scanned documents, or documents with complex layouts, 
we recommend that you use [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse). 
Nemotron parse provides higher-accuracy text extraction. 
For more information, refer to [Advanced Visual Parsing](nemotron-parse.md).



## Why are the environment variables different between library mode and self-hosted mode?

### Self-Hosted Deployments

For [self-hosted deployments](deployment-options.md#when-to-self-host-nims), you should set the environment variables `NGC_API_KEY` and `NIM_NGC_API_KEY`.
For more information, refer to [Generate Your NGC Keys](api-keys.md).

For advanced scenarios, you might want to set environment variables for NIM container paths, tags, and batch sizes on the ingestion runtime. Configure them in your Helm values, Kubernetes `Secret`/`ConfigMap`, or follow [Environment variables](environment-config.md).

### Library Mode

For production environments, you should use the provided Helm charts. For [library mode](quickstart-library-mode.md), you should set the environment variable `NVIDIA_API_KEY`. This is because the NeMo Retriever containers and the NeMo Retriever services running inside them do not have access to arbitrary variables on your laptop or jump host unless you inject them into the workload (for example via Helm, `Secret`, or the client environment as documented for library mode).

For advanced scenarios, you might want to use library mode with self-hosted NIM instances. 
You can set custom endpoints for each NIM. 
For examples of `*_ENDPOINT` variables, refer to [Environment variables](environment-config.md) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md).







## What parameters or settings can I adjust to optimize extraction from my documents or data? 

Refer to the [Performance guide](performance_guide.md) for extraction tuning and optimization guidance.

You can configure the `extract`, `caption`, and other tasks by using the [Ingestor API](python-api-reference.md).

To choose what types of content to extract, use code similar to the following. 
For more information, refer to [Extract Specific Elements from PDFs](python-api-reference.md#extract-specific-elements-from-pdfs).

```python
Ingestor(client=client)
    .files("data/multimodal_test.pdf")
    .extract(              
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        extract_infographics=True,
        text_depth="page"
    )
```

To generate captions for images, use code similar to the following.
For more information, refer to [Extract Captions from Images](python-api-reference.md#extract-captions-from-images).

```python
Ingestor(client=client)
    .files("data/multimodal_test.pdf")
    .extract()
    .embed()
    .caption()

```
