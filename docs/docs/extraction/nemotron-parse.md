# Advanced Visual Parsing with Nemotron Parse

For scanned documents, or documents with complex layouts, 
we recommend that you use [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse). 
Nemotron parse provides higher-accuracy text extraction. 

This documentation describes the following three methods 
to run [NeMo Retriever Library](overview.md) with nemotron-parse.

- Run the NIM on your infrastructure (for example [Helm / Kubernetes](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) or the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html))
- Use NVIDIA Cloud Functions (NVCF) endpoints for cloud-based inference
- Run the Ray batch pipeline with nemotron-parse ([library mode](quickstart-library-mode.md))

## Limitations

Currently, the limitations to using `nemotron-parse` with NeMo Retriever Library are the following:

- Extraction with `nemotron-parse` only supports PDFs, not image files. For more information, refer to [Troubleshoot Nemo Retriever Extraction](troubleshoot.md).
- `nemotron-parse` is not supported on RTX Pro 6000, B200, or H200 NVL. For more information, refer to the [Nemotron Parse Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-parse).


## Run the Nemotron Parse NIM on your infrastructure

Use the following procedure to deploy nemotron-parse with NeMo Retriever Library on infrastructure you control.

!!! important

    Due to limitations in available VRAM controls in the current release of nemotron-parse, it must run on a [dedicated additional GPU](support-matrix.md). In Kubernetes, pin the workload to a dedicated GPU using resource limits, node selectors, or the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html).


1. Deploy the NeMo Retriever Library stack with **nemotron-parse** enabled. Follow [Deployment options](deployment-options.md) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) to enable the service and wire `NEMOTRON_PARSE_HTTP_ENDPOINT` (or equivalent) into the ingestion runtime.

2. After the services are running, you can interact with the pipeline by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells the pipeline to use `nemotron-parse` for extracting text and metadata from images.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.pdf")
        .extract(
            document_type="pdf",  # Ingestor should detect type automatically in most cases
            extract_method="nemotron_parse"
        )
    )
    ```

    !!! tip

        For more Python examples, refer to [Python Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).


## Using NVCF Endpoints for Cloud-Based Inference

Instead of running the pipeline locally, you can use NVCF to perform inference by using remote endpoints.

1. Set the authentication token in the `.env` file.

    ```
    NVIDIA_API_KEY=nvapi-...
    ```

2. Point the ingestion runtime at the hosted `nemotron-parse` endpoint (for example set `NEMOTRON_PARSE_HTTP_ENDPOINT` in Helm values or your process environment):

    ```yaml
    # build.nvidia.com hosted nemotron-parse
    NEMOTRON_PARSE_HTTP_ENDPOINT: "https://integrate.api.nvidia.com/v1/chat/completions"
    # NEMOTRON_PARSE_HTTP_ENDPOINT: "http://nemotron-parse:8000/v1/chat/completions"  # in-cluster NIM
    ```

3. Run inference by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells the pipeline to use `nemotron-parse` for extracting text and metadata from images.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.pdf")
        .extract(
            document_type="pdf",  # Ingestor should detect type automatically in most cases
            extract_method="nemotron_parse"
        )
    )
    ```

    !!! tip

        For more Python examples, refer to [Python Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).



## Related Topics

- [Support Matrix](support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use the Python API](python-api-reference.md)
