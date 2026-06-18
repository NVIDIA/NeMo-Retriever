# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

from nemo_retriever.models.nim.primitives.model_interface.parakeet import (
    ParakeetClient,
    parse_transcription_response,
)
from nemo_retriever.common.params import ASRParams


def test_parakeet_client_posts_http_transcription_request() -> None:
    response = MagicMock()
    response.json.return_value = {"text": "hello world", "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

    client = ParakeetClient("http://localhost:9000", auth_token="nvapi-test")
    with patch(
        "nemo_retriever.models.nim.primitives.model_interface.parakeet.convert_to_mono_wav",
        return_value=b"RIFFfake",
    ), patch(
        "nemo_retriever.models.nim.primitives.model_interface.parakeet.requests.post", return_value=response
    ) as post:
        segments, transcript = client.transcribe(base64.b64encode(b"audio").decode())

    response.raise_for_status.assert_called_once()
    post.assert_called_once()
    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == "http://localhost:9000/v1/audio/transcriptions"
    assert kwargs["headers"]["Authorization"] == "Bearer nvapi-test"
    assert kwargs["data"] == {"language": "en-US", "response_format": "verbose_json"}
    assert kwargs["files"]["file"] == ("audio.wav", b"RIFFfake", "audio/wav")
    assert transcript == "hello world"
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hello"}]


def test_parse_transcription_response_falls_back_to_segments() -> None:
    segments, transcript = parse_transcription_response(
        {"segments": [{"start": 0.0, "end": 0.5, "text": "hello"}, {"start": 0.5, "end": 1.0, "text": "world"}]}
    )
    assert transcript == "hello world"
    assert segments == [
        {"start": 0.0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "world"},
    ]


def test_asr_params_default_infer_protocol_is_http() -> None:
    params = ASRParams(audio_endpoints=(None, "http://localhost:9000"))
    assert params.audio_infer_protocol == "http"
