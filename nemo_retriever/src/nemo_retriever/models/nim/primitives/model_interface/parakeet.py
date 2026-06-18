# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import base64
import logging
from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import requests

try:
    from scipy.io import wavfile
except ModuleNotFoundError:
    wavfile = None  # type: ignore[assignment]

from nemo_retriever.common.api.internal.primitives.tracing.tagging import traceable_func
from nemo_retriever.common.api.util.string_processing import generate_url

try:
    import librosa
except ImportError:
    librosa = None

logger = logging.getLogger(__name__)

# Parakeet ASR training sample rate. ``convert_to_mono_wav`` resamples to this
# rate before sending audio to the HTTP transcription endpoint.
PARAKEET_SAMPLE_RATE_HZ = 16000

DEFAULT_PARAKEET_HTTP_ENDPOINT = "https://ai.api.nvidia.com/v1/audio/nvidia/parakeet-ctc-1_1b-asr"
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"


def _transcription_url(endpoint: str) -> str:
    """Resolve either a base NIM URL or a full transcription endpoint."""
    url = generate_url(endpoint).rstrip("/")
    if url.endswith("/audio/transcriptions"):
        return url
    if "/v1/audio/" in url:
        return url
    return f"{url}{TRANSCRIPTIONS_PATH}"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalise_segment(segment: Any) -> Optional[dict]:
    if not isinstance(segment, dict):
        return None
    text = _coerce_text(segment.get("text") or segment.get("transcript")).strip()
    if not text:
        return None
    start = segment["start"] if "start" in segment else segment.get("start_time")
    end = segment["end"] if "end" in segment else segment.get("end_time")
    return {
        "start": start,
        "end": end,
        "text": text,
    }


def parse_transcription_response(payload: Any) -> tuple[list[dict], str]:
    """Parse OpenAI-style or simple ASR JSON into ``(segments, transcript)``."""
    if isinstance(payload, str):
        return [], payload
    if not isinstance(payload, dict):
        return [], _coerce_text(payload)

    transcript = _coerce_text(
        payload.get("text") or payload.get("transcript") or payload.get("transcription") or payload.get("result") or ""
    )

    raw_segments = payload.get("segments") or payload.get("chunks") or []
    segments = []
    if isinstance(raw_segments, list):
        for raw_segment in raw_segments:
            segment = _normalise_segment(raw_segment)
            if segment is not None:
                segments.append(segment)
    if not transcript and segments:
        transcript = " ".join(segment["text"] for segment in segments)
    return segments, transcript


class ParakeetClient:
    """
    A simple interface for handling inference with a Parakeet model (e.g., speech, audio-related).
    """

    def __init__(
        self,
        endpoint: str,
        auth_token: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the ParakeetClient.

        Parameters
        ----------
        endpoint : str
            The URL of the Parakeet service endpoint.
        auth_token : Optional[str], default=None
            The authentication token for accessing the service.
        """
        self.endpoint = _transcription_url(endpoint)
        self.auth_token = auth_token
        self.timeout = timeout

    @traceable_func(trace_name="{stage_name}::{model_name}")
    def infer(self, data: dict, model_name: str, **kwargs) -> Any:
        """
        Perform inference using the specified model and input data.

        Parameters
        ----------
        data : dict
            The input data for inference.
        model_name : str
            The model name.
        kwargs : dict
            Additional parameters for inference.

        Returns
        -------
        Any
            The processed inference results, coalesced in the same order as the input images.
        """

        segments, transcript = self.transcribe(data)
        logger.debug("Processing Parakeet inference results (pass-through).")

        return segments, transcript

    def transcribe(
        self,
        audio_content: str,
        language_code: str = "en-US",
        response_format: str = "verbose_json",
    ) -> tuple[list[dict], str]:
        """
        Transcribe an audio file using Parakeet's HTTP transcription endpoint.

        Parameters
        ----------
        audio_content : str
            Base64-encoded audio content to be transcribed.
        language_code : str, default="en-US"
            The language code for transcription.
        response_format : str, default="verbose_json"
            Requested response format. ``verbose_json`` preserves timestamp
            segments when the endpoint supports OpenAI-compatible ASR output.

        Returns
        -------
        tuple[list[dict], str]
            Segment dictionaries and the complete transcript.
        """
        audio_bytes = base64.b64decode(audio_content)
        mono_audio_bytes = convert_to_mono_wav(audio_bytes)
        headers = {"accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        data = {"language": language_code, "response_format": response_format}
        files = {"file": ("audio.wav", mono_audio_bytes, "audio/wav")}
        response = requests.post(self.endpoint, headers=headers, data=data, files=files, timeout=self.timeout)
        response.raise_for_status()
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return parse_transcription_response(payload)


def convert_to_mono_wav(audio_bytes):
    """
    Convert an audio file to mono WAV format using Librosa and SciPy.

    Parameters
    ----------
    audio_bytes : bytes
        The raw audio data in bytes.

    Returns
    -------
    bytes
        The processed audio in mono WAV format.
    """

    if librosa is None:
        raise ImportError("Librosa is required for audio processing. ")

    # Create a BytesIO object from the audio bytes
    byte_io = io.BytesIO(audio_bytes)

    # Load the audio file with librosa.
    # ``sr=PARAKEET_SAMPLE_RATE_HZ`` (16 kHz) matches Parakeet's training rate;
    # ``RecognitionConfig.sample_rate_hertz`` above must stay in sync with it.
    # ``mono=True`` collapses any multichannel input to mono.
    audio_data, sample_rate = librosa.load(byte_io, sr=PARAKEET_SAMPLE_RATE_HZ, mono=True)

    # Ensure audio is properly scaled for 16-bit PCM
    # Librosa normalizes the data between -1 and 1
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9

    # Convert to int16 format for 16-bit PCM WAV
    audio_data_int16 = (audio_data * 32767).astype(np.int16)

    # Create a BytesIO buffer to write the WAV file
    output_io = io.BytesIO()

    # Write the WAV data using scipy
    wavfile.write(output_io, sample_rate, audio_data_int16)

    # Reset the file pointer to the beginning and read all contents
    output_io.seek(0)
    wav_bytes = output_io.read()

    return wav_bytes


def create_audio_inference_client(
    endpoints: Tuple[str, str],
    infer_protocol: Optional[str] = None,
    auth_token: Optional[str] = None,
    timeout: float = 120.0,
):
    """
    Create a ParakeetClient for interfacing with an audio model inference server.

    Parameters
    ----------
    endpoints : tuple
        A tuple containing the gRPC and HTTP endpoints. Only the HTTP endpoint is used.
    infer_protocol : str, optional
        The protocol to use ("grpc" or "http").
        If not specified, defaults to "http" if a valid HTTP endpoint is provided.
    auth_token : str, optional
        Authorization token for authentication (default: None).

    Returns
    -------
    ParakeetClient
        The initialized ParakeetClient configured for audio inference over HTTP.

    Raises
    ------
    ValueError
        If an invalid `infer_protocol` is specified or if no HTTP endpoint is provided.
    """
    _, http_endpoint = endpoints

    if (infer_protocol is None) and (http_endpoint and http_endpoint.strip()):
        infer_protocol = "http"

    # Normalize protocol to lowercase for case-insensitive comparison
    if infer_protocol:
        infer_protocol = infer_protocol.lower()

    if infer_protocol != "http":
        raise ValueError("Audio inference now requires an HTTP endpoint; gRPC is no longer supported.")
    if not http_endpoint:
        raise ValueError("HTTP endpoint must be provided for audio inference.")

    return ParakeetClient(
        http_endpoint,
        auth_token=auth_token,
        timeout=timeout,
    )
