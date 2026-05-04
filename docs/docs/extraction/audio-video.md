# Extract Speech with NeMo Retriever Library

This documentation describes two methods to run [NeMo Retriever Library](overview.md) 
with the [parakeet-1-1b-ctc-en-us ASR NIM microservice](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html) 
(`nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us`) to extract speech from audio files.

- Run the NIM locally on your cluster with the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)
- Use NVIDIA Cloud Functions (NVCF) endpoints for cloud-based inference

Currently, you can extract speech from the following file types:

- `mp3`
- `wav`


## Overview

[NeMo Retriever Library](overview.md) supports extracting speech from audio files for Retrieval Augmented Generation (RAG) applications. 
Similar to how the multimodal document extraction pipeline leverages object detection and image OCR microservices, 
NeMo Retriever Library uses the [parakeet-1-1b-ctc-en-us ASR NIM microservice](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html) 
to transcribe speech to text, which is then embedded by using the NeMo Retriever embedding NIM. 

!!! important

    Due to limitations in available VRAM controls in the current release, the parakeet-1-1b-ctc-en-us ASR NIM microservice must run on a [dedicated additional GPU](prerequisites-support-matrix.md#model-hardware-requirements). For the full list of requirements, refer to the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md#model-hardware-requirements).

This pipeline enables users to retrieve speech files at the segment level.



![Overview diagram](images/audio.png)



## Run the NIM locally (Helm)

Use the following procedure to run the NIM on your own infrastructure. Self-hosted Parakeet runs on Kubernetes via the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md).

!!! important

    The parakeet-1-1b-ctc-en-us ASR NIM microservice must run on a [dedicated additional GPU](prerequisites-support-matrix.md#model-hardware-requirements). Pin the workload to that GPU with your Helm values or the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html) (for example, node selectors, resource limits, or device requests appropriate to your cluster).

1. Create [your NGC personal key](api-keys.md) with the scopes required for `nvcr.io` image pulls and NGC API access. Configure it for the cluster using the [NeMo Retriever Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) (for example `ngcImagePullSecret` / `ngcApiSecret` on `helm upgrade --install`, or pre-created secrets with username `$oauthtoken`).

2. **Optional — local tooling only.** Helm-managed pods get NGC credentials from chart secrets (step 1), not from a file on your laptop. If you also run **Python clients, CLIs, or other scripts on your machine** (outside the cluster) that load `NGC_API_KEY` from disk, create a `.env` in that working directory:

    ```ini
    NGC_API_KEY=<your-ngc-key>
    ```

    Skip this step if you only operate the cluster through Helm and do not run such local processes.

3. Deploy or upgrade NeMo Retriever Library with the Helm chart and enable the ASR / audio components your release requires (Parakeet and related services). Follow [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md) and [Deployment options](deployment-options.md). Ensure the chart values for your cluster request the ASR NIM.

4. After the services are running, you can interact with the pipeline by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells the pipeline to extract information from WAV audio files.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.wav")
        .extract(
            document_type="wav",  # Ingestor should detect type automatically in most cases
            extract_method="audio",
            extract_audio_params={
                "segment_audio": True,
            },
        )
    )
    ```

    To generate one extracted element for each sentence-like ASR segment, include `extract_audio_params={"segment_audio": True}` when calling `.extract(...)`. This option applies when audio extraction runs with a Parakeet NIM (either self-hosted on your cluster or remotely via NVCF) but has no effect when using the local Hugging Face Parakeet model.

    !!! tip

        For more Python examples, refer to [Python Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).


## Use NVCF Endpoints for Cloud-Based Inference

Instead of running the pipeline locally, you can use NVCF to perform inference by using remote endpoints.

1. NVCF requires an authentication token and a function ID for access. Ensure you have these credentials ready before making API calls.

2. Run inference by using Python. Provide an NVCF endpoint along with authentication details.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells the pipeline to extract information from WAV audio files.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.mp3")
        .extract(
            document_type="mp3",
            extract_method="audio",
            extract_audio_params={
                "grpc_endpoint": "grpc.nvcf.nvidia.com:443",
                "auth_token": "<API key>",
                "function_id": "<function ID>",
                "use_ssl": True,
                "segment_audio": True,
            },
        )
    )
    ```

    !!! tip

        For more Python examples, refer to [Python Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).



## Related Topics

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use the Python API](nemo-retriever-api-reference.md)
