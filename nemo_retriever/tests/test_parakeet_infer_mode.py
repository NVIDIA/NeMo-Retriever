# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from nemo_retriever.models.nim.primitives.model_interface.parakeet import (
    ParakeetClient,
    resolve_audio_infer_mode,
)
from nemo_retriever.common.params import ASRParams


@pytest.mark.parametrize(
    ("mode", "endpoint", "expected"),
    [
        ("auto", "localhost:18019", "offline"),
        ("auto", "parakeet-nim:50051", "offline"),
        ("auto", "audio:50051", "offline"),
        ("auto", "grpc.nvcf.nvidia.com:443", "online"),
        ("online", "localhost:18019", "online"),
        ("offline", "grpc.nvcf.nvidia.com:443", "offline"),
    ],
)
def test_resolve_audio_infer_mode(mode: str, endpoint: str, expected: str) -> None:
    assert resolve_audio_infer_mode(mode, endpoint) == expected


def test_resolve_audio_infer_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="audio_infer_mode"):
        resolve_audio_infer_mode("batch", "localhost:50051")


@patch("nemo_retriever.models.nim.primitives.model_interface.parakeet.riva_client")
def test_parakeet_client_transcribe_uses_offline_for_self_hosted(mock_riva) -> None:
    mock_asr = MagicMock()
    mock_riva.ASRService.return_value = mock_asr
    mock_riva.AudioEncoding.LINEAR_PCM = "LINEAR_PCM"
    mock_riva.RecognitionConfig.return_value = MagicMock()

    client = ParakeetClient("localhost:18019", infer_mode="auto")
    with patch(
        "nemo_retriever.models.nim.primitives.model_interface.parakeet.convert_to_mono_wav",
        return_value=b"RIFFfake",
    ):
        client.transcribe(base64.b64encode(b"audio").decode())

    mock_asr.offline_recognize.assert_called_once()
    mock_asr.streaming_response_generator.assert_not_called()


@patch("nemo_retriever.models.nim.primitives.model_interface.parakeet.riva_client")
def test_parakeet_client_transcribe_uses_offline_when_explicit(mock_riva) -> None:
    mock_asr = MagicMock()
    mock_riva.ASRService.return_value = mock_asr
    mock_riva.AudioEncoding.LINEAR_PCM = "LINEAR_PCM"
    mock_riva.RecognitionConfig.return_value = MagicMock()

    client = ParakeetClient("localhost:18019", infer_mode="offline")
    with patch(
        "nemo_retriever.models.nim.primitives.model_interface.parakeet.convert_to_mono_wav",
        return_value=b"RIFFfake",
    ):
        client.transcribe(base64.b64encode(b"audio").decode())

    mock_asr.offline_recognize.assert_called_once()
    mock_asr.streaming_response_generator.assert_not_called()


@patch("nemo_retriever.models.nim.primitives.model_interface.parakeet.riva_client")
def test_parakeet_client_transcribe_uses_streaming_for_nvcf(mock_riva) -> None:
    mock_asr = MagicMock()
    mock_riva.ASRService.return_value = mock_asr
    mock_riva.AudioEncoding.LINEAR_PCM = "LINEAR_PCM"
    mock_riva.RecognitionConfig.return_value = MagicMock()
    mock_riva.StreamingRecognitionConfig.return_value = MagicMock()
    mock_asr.streaming_response_generator.return_value = []

    client = ParakeetClient(
        "grpc.nvcf.nvidia.com:443",
        function_id="fn-1",
        auth_token="nvapi-test",
        infer_mode="auto",
    )
    with patch(
        "nemo_retriever.models.nim.primitives.model_interface.parakeet.convert_to_mono_wav",
        return_value=b"RIFFfake",
    ), patch.object(client, "_streaming_transcribe", return_value=MagicMock(results=[])) as mock_stream:
        client.transcribe(base64.b64encode(b"audio").decode())

    mock_stream.assert_called_once()
    mock_asr.offline_recognize.assert_not_called()


def test_asr_params_default_infer_mode_is_auto() -> None:
    params = ASRParams(audio_endpoints=("localhost:50051", None))
    assert params.audio_infer_mode == "auto"
