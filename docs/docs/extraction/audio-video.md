# Audio and video ingestion

Use this page for speech and audio extraction with Parakeet ASR and for video workflows that combine audio with OCR on frames or derived images.

For air-gapped or disconnected deployments, refer to [Air-gapped and disconnected deployment](deployment-options.md#air-gapped-deployment).

**Sections:** [Speech and audio (Parakeet)](#speech-and-audio-extraction) · [Run Parakeet on the cluster (Helm)](#run-parakeet-on-the-cluster-helm) · [Parakeet with hosted inference (build.nvidia.com)](#parakeet-hosted-inference-build-nvidia) · [Video and frame OCR](#video-and-frame-ocr)

## Speech and audio extraction { #speech-and-audio-extraction }

This documentation describes two ways to run [NeMo Retriever Library](overview.md) with the [parakeet-1-1b-ctc-en-us ASR NIM microservice](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html) (`nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us`) to extract speech from audio files:

- Run the NIM locally on your cluster with the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
- Use NVIDIA Cloud Functions (NVCF) endpoints for cloud-based inference

Supported file types for speech extraction today:

- `mp3`, `wav`
- `mp4`, `mov`, `mkv`, `avi` — common video containers; the audio track is transcribed (same extensions as in [What is NeMo Retriever Library?](overview.md))

[NeMo Retriever Library](overview.md) supports extracting speech from audio for Retrieval Augmented Generation (RAG). Similar to how the multimodal document pipeline uses detection and OCR microservices, NeMo Retriever Library uses the [parakeet-1-1b-ctc-en-us ASR NIM](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html) to transcribe speech to text, then embeddings through the NeMo Retriever embedding path.

Before running audio extraction from Python with either self-hosted or hosted Parakeet, install the multimedia extra so the Parakeet ASR client can decode and resample audio:

```bash
pip install "nemo-retriever[multimedia]"
# For local GPU inference, include both extras:
pip install "nemo-retriever[local,multimedia]"
```

The Python package includes the `ffmpeg-python` wrapper, and the multimedia
extra adds Python libraries for audio decoding and resampling. These Python
dependencies do not install the `ffmpeg` or `ffprobe` command-line binaries.
For audio and video workflows, install system FFmpeg so both binaries are on
`PATH`:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg
```

Containers use the FFmpeg package from the base Ubuntu image, rather than a
source-built FFmpeg release. If your workflow depends on exact FFmpeg version
or codec behavior, verify the package inside the image against those
requirements.

For Kubernetes deployments with network access to package repositories, set
`service.installFfmpeg=true` in the
[Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#1-service-image)
to install ffmpeg/ffprobe at service startup. This runtime path requires
package-repository network egress, a writable root filesystem, and a security
policy that allows the image's scoped sudo use. For air-gapped clusters, refer to
[Air-gapped and disconnected deployment](deployment-options.md#air-gapped-deployment).

!!! important

    Due to limitations in available VRAM controls in the current release, the parakeet-1-1b-ctc-en-us ASR NIM must run on a [dedicated additional GPU](prerequisites-support-matrix.md#model-hardware-requirements). For the full list of requirements, refer to the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md#model-hardware-requirements).

This pipeline enables retrieval at the speech segment level when you enable segmenting (refer to the examples below).

![Overview diagram](images/audio.png)

## Run Parakeet on the cluster (Helm) { #run-parakeet-on-the-cluster-helm }

Use the following procedure to run the NIM on your own infrastructure. Self-hosted Parakeet runs on Kubernetes through the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md). Enable the ASR NIM per [Optional Helm NIMs](prerequisites-support-matrix.md#optional-helm-nims-not-auto-wired-by-default). GPU pinning and endpoint wiring are documented in [Parakeet ASR](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#audio-video-parakeet).

1. Deploy or upgrade with the [NeMo Retriever Helm chart](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) and enable Parakeet for your release (refer to [Optional Helm NIMs](prerequisites-support-matrix.md#optional-helm-nims-not-auto-wired-by-default)). Follow [Deployment options](deployment-options.md).

2. If the service will process audio or video files, set `service.installFfmpeg=true` in the Helm chart when your cluster allows runtime package installation; for air-gapped clusters, refer to [Air-gapped and disconnected deployment](deployment-options.md#air-gapped-deployment) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#1-service-image) for `service.image` overrides.

3. After the services are running, interact with the pipeline from Python (refer to the [Python API guide](nemo-retriever-api-reference.md) for parameter details).

    - In `batch` mode, pass the in-cluster Parakeet gRPC endpoint through `ASRParams.audio_endpoints` (for example `audio:50051` from your Helm release). The retriever service auto-wires this endpoint; graph ingest does not.

    ```python
    from nemo_retriever import create_ingestor
    from nemo_retriever.common.params.models import ASRParams

    ingestor = (
        create_ingestor(run_mode="batch")
        .files("./data/*.wav")
        .extract_audio(
            asr_params=ASRParams(
                audio_endpoints=("audio:50051", None),  # (grpc_endpoint, http_endpoint)
                segment_audio=True,
            ),
        )
    )
    results = ingestor.ingest()
    ```

    To generate one extracted element for each sentence-like ASR segment, pass `asr_params=ASRParams(segment_audio=True)` to `.extract_audio(...)`. This option applies when audio extraction runs with a self-hosted Parakeet NIM or using build.nvidia.com hosted inference, but has no effect when using the local Hugging Face Parakeet model.

    For more runnable examples, refer to [Workflow: Ingest documents](workflow-document-ingestion.md).

## Parakeet with hosted inference (build.nvidia.com) { #parakeet-hosted-inference-build-nvidia }

Instead of running the pipeline locally, you can call Parakeet through [build.nvidia.com](https://build.nvidia.com/) hosted inference.

1. On the Parakeet model page on [build.nvidia.com](https://build.nvidia.com/), create or copy an API key and note the function ID for hosted access. You need both before making API calls.

2. Run inference from Python with the hosted gRPC endpoint and credentials from that page (the example below uses the default hosted gRPC hostname; confirm values in the **Get API Key** flow for your deployment). Pass hosted endpoint, function ID, and API key through `ASRParams` (`audio_endpoints`, `function_id`, `auth_token`).

    ```python
    from nemo_retriever import create_ingestor
    from nemo_retriever.common.params.models import ASRParams

    ingestor = (
        create_ingestor(run_mode="batch")
        .files("./data/*.mp3")
        .extract_audio(
            asr_params=ASRParams(
                audio_endpoints=("grpc.nvcf.nvidia.com:443", None),  # (grpc_endpoint, http_endpoint)
                function_id="<function ID>",
                auth_token="<API key>",
                segment_audio=True,
            ),
        )
    )
    results = ingestor.ingest()
    ```

    !!! tip

        For more runnable examples, refer to [Workflow: Ingest documents](workflow-document-ingestion.md).

## Video and frame OCR { #video-and-frame-ocr }

For video assets, NeMo Retriever Library can combine audio or speech processing (refer to [Speech and audio extraction](#speech-and-audio-extraction) above) with visual text extraction when OCR applies to frames or derived images.

For OCR-oriented extract methods on scanned or image-heavy content, refer to [OCR and scanned documents](multimodal-extraction.md#ocr-and-scanned-documents), [text and layout extraction](multimodal-extraction.md#text-and-layout-extraction), and [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse) for advanced visual parsing.

Container formats and early-access video types are listed under [supported file types and formats](multimodal-extraction.md#supported-file-types-and-formats) (refer to [What is NeMo Retriever Library?](overview.md) for the full list).

For end-to-end RAG stacks that include multimodal ingestion, refer to the [NVIDIA AI Blueprints catalog](https://build.nvidia.com/explore/discover) and related solution pages on [NVIDIA Build](https://build.nvidia.com/).

## Related topics { #related-topics }

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Troubleshoot NeMo Retriever extraction](troubleshoot.md)
- [Use the Python API](nemo-retriever-api-reference.md)
- [Chunking](concepts.md#chunking) (includes audio and video segmenting defaults)
