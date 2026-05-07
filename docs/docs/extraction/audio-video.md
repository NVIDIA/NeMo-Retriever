# Audio and video ingestion

Use this page for speech and audio extraction with Parakeet ASR and for video workflows that combine audio with OCR on frames or derived images.

**Sections:** [Speech and audio (Parakeet)](#speech-and-audio-extraction) · [Run Parakeet on the cluster (Helm)](#run-parakeet-on-the-cluster-helm) · [Parakeet with hosted inference (build.nvidia.com)](#parakeet-hosted-inference-build-nvidia) · [Video and frame OCR](#video-and-frame-ocr)

## Speech and audio extraction {#speech-and-audio-extraction}

This documentation describes two ways to run [NeMo Retriever Library](overview.md) with the [parakeet-1-1b-ctc-en-us ASR NIM microservice](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html) (`nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us`) to extract speech from audio files:

- Run the NIM locally on your cluster with the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
- Use NVIDIA Cloud Functions (NVCF) endpoints for cloud-based inference

Supported file types for speech extraction today:

- `mp3`, `wav`
- `mp4`, `mov`, `mkv`, `avi` — common video containers; the audio track is transcribed (same extensions as in [What is NeMo Retriever Library?](overview.md))

[NeMo Retriever Library](overview.md) supports extracting speech from audio for Retrieval Augmented Generation (RAG). Similar to how the multimodal document pipeline uses detection and OCR microservices, NeMo Retriever Library uses the [parakeet-1-1b-ctc-en-us ASR NIM](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html) to transcribe speech to text, then embeddings via the NeMo Retriever embedding path.

!!! important

    Due to limitations in available VRAM controls in the current release, the parakeet-1-1b-ctc-en-us ASR NIM must run on a [dedicated additional GPU](prerequisites-support-matrix.md#model-hardware-requirements). For the full list of requirements, refer to the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md#model-hardware-requirements).

This pipeline enables retrieval at the speech segment level when you enable segmenting (see examples below).

![Overview diagram](images/audio.png)

## Run Parakeet on the cluster (Helm) {#run-parakeet-on-the-cluster-helm}

Use the following procedure to run the NIM on your own infrastructure. Self-hosted Parakeet runs on Kubernetes via the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

!!! important

    Pin the Parakeet workload to the dedicated GPU with your Helm values or the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html) (for example, node selectors, resource limits, or device requests appropriate to your cluster).

1. Deploy or upgrade NeMo Retriever Library with the Helm chart and enable the ASR / audio components your release requires (Parakeet and related services). Follow [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) and [Deployment options](deployment-options.md). Ensure the chart values for your cluster request the ASR NIM.

2. After the services are running, interact with the pipeline from Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method runs audio extraction.

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


    To generate one extracted element for each sentence-like ASR segment, include `extract_audio_params={"segment_audio": True}` when calling `.extract(...)`. This option applies when audio extraction runs with a self-hosted Parakeet NIM or using build.nvidia.com hosted inference, but has no effect when using the local Hugging Face Parakeet model.


    !!! tip

        For more Python examples, refer to [Python Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).

## Parakeet with hosted inference (build.nvidia.com) {#parakeet-hosted-inference-build-nvidia}

Instead of running the pipeline locally, you can call Parakeet through [build.nvidia.com](https://build.nvidia.com/) hosted inference.

1. On the Parakeet model page on [build.nvidia.com](https://build.nvidia.com/), create or copy an API key and note the function ID for hosted access. You need both before making API calls.

2. Run inference from Python with the hosted gRPC endpoint and credentials from that page (the example below uses the default hosted gRPC hostname; confirm values in the **Get API Key** flow for your deployment).

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method runs audio extraction.
    - The `document_type` parameter is optional because `Ingestor` should detect the file type automatically in most cases.

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

## Video and frame OCR {#video-and-frame-ocr}

For video assets, NeMo Retriever Library can combine audio or speech processing (see [Speech and audio extraction](#speech-and-audio-extraction) above) with visual text extraction when OCR applies to frames or derived images.

For OCR-oriented extract methods on scanned or image-heavy content, see [OCR and scanned documents](multimodal-extraction.md#ocr-and-scanned-documents), [text and layout extraction](multimodal-extraction.md#text-and-layout-extraction), and [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse) for advanced visual parsing.

Container formats and early-access video types are listed under [supported file types and formats](multimodal-extraction.md#supported-file-types-and-formats) (see [What is NeMo Retriever Library?](overview.md) for the full list).

For end-to-end RAG stacks that include multimodal ingestion, see the [NVIDIA AI Blueprints catalog](https://build.nvidia.com/explore/discover) and related solution pages on [NVIDIA Build](https://build.nvidia.com/).

## Related topics {#related-topics}

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Troubleshoot NeMo Retriever extraction](troubleshoot.md)
- [Use the Python API](nemo-retriever-api-reference.md)
- [Chunking](concepts.md#chunking) (includes audio and video segmenting defaults)
