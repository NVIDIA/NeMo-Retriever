# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for nemo_retriever.audio: ASRActor (with mocked Parakeet client).

Avoids importing nemo_retriever.models (and thus torch) by not eagerly loading
the model package; local-ASR tests inject a fake nemo_retriever.models.local
into sys.modules so the real module is never loaded.
"""

import base64
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd

from nemo_retriever.operators.extract.audio.asr_actor import ASRActor
from nemo_retriever.operators.extract.audio.asr_actor import DEFAULT_NGC_ASR_FUNCTION_ID
from nemo_retriever.operators.extract.audio.asr_actor import apply_asr_to_df
from nemo_retriever.operators.extract.audio.asr_actor import asr_params_from_env
from nemo_retriever.common.params import ASRParams


NVCF_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"


def test_strip_pad_from_transcript():
    """Transformers backend post-process removes <pad> and normalizes spaces."""
    # Some tests monkeypatch nemo_retriever.models.local with a mock module object.
    # Ensure we import the real package submodule for this test.
    local_mod = sys.modules.get("nemo_retriever.models.local")
    if local_mod is not None and not hasattr(local_mod, "__path__"):
        sys.modules.pop("nemo_retriever.models.local", None)
    from nemo_retriever.models.local.parakeet_ctc_1_1b_asr import _strip_pad_from_transcript

    assert _strip_pad_from_transcript("") == ""
    assert _strip_pad_from_transcript("  ") == ""
    assert _strip_pad_from_transcript("<pad>") == ""
    assert _strip_pad_from_transcript("<pad> hello <pad> world") == "hello world"
    assert _strip_pad_from_transcript("  a  <pad>  b  ") == "a b"
    out = _strip_pad_from_transcript("  <pad> foo <pad> bar <pad>  ")
    assert out == "foo bar"
    assert "<pad>" not in out


def test_asr_actor_empty_batch():
    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None))
        actor = ASRActor(params=params)
        empty = pd.DataFrame(columns=["path", "bytes"])
        out = actor(empty)

        assert isinstance(out, pd.DataFrame)
        assert "text" in out.columns
        assert len(out) == 0
        mock_client.infer.assert_not_called()


def test_asr_actor_mock_transcribe():
    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = ([], "hello world transcript")
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None))
        actor = ASRActor(params=params)
        raw = b"\x00\x00\x00\x00"
        batch = pd.DataFrame(
            [
                {
                    "path": "/tmp/chunk.wav",
                    "bytes": raw,
                    "source_path": "/tmp/source.wav",
                    "duration": 1.0,
                    "chunk_index": 0,
                    "metadata": {"source_path": "/tmp/source.wav", "chunk_index": 0, "duration": 1.0},
                    "page_number": 0,
                }
            ]
        )
        out = actor(batch)

        assert len(out) == 1
        assert out["text"].iloc[0] == "hello world transcript"
        assert out["path"].iloc[0] == "/tmp/chunk.wav"
        assert out["source_path"].iloc[0] == "/tmp/source.wav"
        mock_client.infer.assert_called_once()
        call_arg = mock_client.infer.call_args[0][0]
        assert call_arg == base64.b64encode(raw).decode("ascii")


def test_apply_asr_to_df():
    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = ([], "applied transcript")
        mock_get.return_value = mock_client

        batch = pd.DataFrame(
            [
                {
                    "path": "/p",
                    "bytes": b"x",
                    "source_path": "/s",
                    "duration": 0.5,
                    "chunk_index": 0,
                    "metadata": {},
                    "page_number": 0,
                }
            ]
        )
        out = apply_asr_to_df(batch, asr_params={"audio_endpoints": ("localhost:50051", None)})
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert out["text"].iloc[0] == "applied transcript"


def test_asr_actor_remote_segment_audio():
    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = (
            [
                {"start": 0.0, "end": 1.0, "text": "Hello world."},
                {"start": 1.0, "end": 2.5, "text": "How are you?"},
            ],
            "Hello world. How are you?",
        )
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None), segment_audio=True)
        actor = ASRActor(params=params)
        batch = pd.DataFrame(
            [
                {
                    "path": "/tmp/chunk.wav",
                    "bytes": b"fake_audio",
                    "source_path": "/tmp/source.wav",
                    "duration": 2.5,
                    "chunk_index": 3,
                    "metadata": {"source_path": "/tmp/source.wav", "chunk_index": 3, "duration": 2.5},
                    "page_number": 3,
                }
            ]
        )
        out = actor(batch)

        assert len(out) == 2
        assert out["text"].tolist() == ["Hello world.", "How are you?"]
        assert out["page_number"].tolist() == [3, 3]
        assert out["chunk_index"].tolist() == [3, 3]
        assert out["metadata"].iloc[0]["segment_index"] == 0
        assert out["metadata"].iloc[0]["segment_count"] == 2
        assert out["metadata"].iloc[0]["segment_start_seconds"] == 0.0
        assert out["metadata"].iloc[0]["segment_end_seconds"] == 1.0
        assert out["metadata"].iloc[1]["segment_index"] == 1
        assert out["metadata"].iloc[1]["segment_start_seconds"] == 1.0
        assert out["metadata"].iloc[1]["segment_end_seconds"] == 2.5


def test_apply_asr_to_df_segment_audio():
    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = (
            [
                {"start": 0.0, "end": 0.4, "text": "First sentence."},
                {"start": 0.4, "end": 0.8, "text": "Second sentence!"},
            ],
            "First sentence. Second sentence!",
        )
        mock_get.return_value = mock_client

        batch = pd.DataFrame(
            [
                {
                    "path": "/p",
                    "bytes": b"x",
                    "source_path": "/s",
                    "duration": 0.8,
                    "chunk_index": 0,
                    "metadata": {},
                    "page_number": 0,
                }
            ]
        )
        out = apply_asr_to_df(
            batch,
            asr_params={"audio_endpoints": ("localhost:50051", None), "segment_audio": True},
        )
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 2
        assert out["text"].tolist() == ["First sentence.", "Second sentence!"]
        assert out["metadata"].iloc[0]["segment_count"] == 2


def test_local_asr_does_not_call_get_client():
    """After the CPU/GPU split the local-Parakeet path is :class:`ASRGPUActor`,
    which must never touch the remote ``_get_client`` factory."""
    from nemo_retriever.operators.extract.audio.gpu_actor import ASRGPUActor

    mock_model = MagicMock()
    mock_model.transcribe_with_segments.return_value = [("mocked local transcript", [])]
    mock_class = MagicMock(return_value=mock_model)
    mock_local = MagicMock()
    mock_local.ParakeetCTC1B1ASR = mock_class
    prev_local = sys.modules.get("nemo_retriever.models.local")
    sys.modules["nemo_retriever.models.local"] = mock_local
    try:
        with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
            params = ASRParams(audio_endpoints=(None, None))
            actor = ASRGPUActor(params=params)

            mock_get.assert_not_called()
            assert actor._model is mock_model

            batch = pd.DataFrame(
                [
                    {
                        "path": "/tmp/chunk.wav",
                        "bytes": b"fake_audio_bytes",
                        "source_path": "/tmp/source.wav",
                        "duration": 1.0,
                        "chunk_index": 0,
                        "metadata": {},
                        "page_number": 0,
                    }
                ]
            )
            out = actor(batch)

            assert len(out) == 1
            assert out["text"].iloc[0] == "mocked local transcript"
            mock_model.transcribe_with_segments.assert_called_once()
            # One path passed (temp file or /tmp/chunk.wav)
            call_args = mock_model.transcribe_with_segments.call_args[0][0]
            assert isinstance(call_args, list)
            assert len(call_args) == 1
    finally:
        if prev_local is None:
            sys.modules.pop("nemo_retriever.models.local", None)
        else:
            sys.modules["nemo_retriever.models.local"] = prev_local


def test_asr_params_from_env_default_grpc_endpoint_preserves_nvidia_auth(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("AUDIO_GRPC_ENDPOINT", raising=False)
    monkeypatch.delenv("AUDIO_FUNCTION_ID", raising=False)

    params = asr_params_from_env(default_grpc_endpoint=NVCF_GRPC_ENDPOINT)

    assert params.audio_endpoints[0] == NVCF_GRPC_ENDPOINT
    assert params.auth_token == "nvapi-test"
    assert params.function_id == DEFAULT_NGC_ASR_FUNCTION_ID
    assert params.audio_infer_protocol == "grpc"


def test_asr_params_from_env_without_endpoint_drops_nvidia_auth(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("AUDIO_FUNCTION_ID", "function-test")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("AUDIO_GRPC_ENDPOINT", raising=False)

    params = asr_params_from_env()

    assert params.audio_endpoints == (None, None)
    assert params.auth_token is None
    assert params.function_id is None
    assert params.audio_infer_protocol == "grpc"


def test_asr_cpu_actor_defaults_with_only_nvidia_auth_populate_remote_defaults(monkeypatch):
    from nemo_retriever.operators.extract.audio.cpu_actor import ASRCPUActor

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.delenv("AUDIO_GRPC_ENDPOINT", raising=False)
    monkeypatch.delenv("AUDIO_FUNCTION_ID", raising=False)

    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
        actor = ASRCPUActor(params=asr_params_from_env())

    mock_get.assert_called_once()
    assert actor._params.audio_endpoints[0] == NVCF_GRPC_ENDPOINT
    assert actor._params.auth_token == "nvapi-test"
    assert actor._params.function_id == DEFAULT_NGC_ASR_FUNCTION_ID
    assert actor._params.audio_infer_protocol == "grpc"


def test_local_asr_apply_asr_to_df():
    """apply_asr_to_df with audio_endpoints=(None, None) uses local model when mocked.

    After the ASR CPU/GPU split, the archetype picks the local (GPU) variant
    only when a GPU is detected, so we advertise one via the centralized
    ``gather_local_resources`` source — every dispatch site (executor,
    archetype, resolver, multi-type op) reads through that one attribute.
    """
    from nemo_retriever.common.ray_resource_hueristics import Resources

    mock_model = MagicMock()
    mock_model.transcribe_with_segments.return_value = [("apply local text", [])]
    mock_class = MagicMock(return_value=mock_model)
    mock_local = MagicMock()
    mock_local.ParakeetCTC1B1ASR = mock_class
    prev_local = sys.modules.get("nemo_retriever.models.local")
    sys.modules["nemo_retriever.models.local"] = mock_local
    try:
        with patch(
            "nemo_retriever.common.ray_resource_hueristics.gather_local_resources",
            return_value=Resources(cpu_count=8, gpu_count=1),
        ), patch("nemo_retriever.operators.extract.audio.asr_actor._get_client") as mock_get:
            batch = pd.DataFrame(
                [
                    {
                        "path": "/p",
                        "bytes": b"x",
                        "source_path": "/s",
                        "duration": 0.5,
                        "chunk_index": 0,
                        "metadata": {},
                        "page_number": 0,
                    }
                ]
            )
            out = apply_asr_to_df(batch, asr_params={"audio_endpoints": (None, None)})

            mock_get.assert_not_called()
            assert len(out) == 1
            assert out["text"].iloc[0] == "apply local text"
    finally:
        if prev_local is None:
            sys.modules.pop("nemo_retriever.models.local", None)
        else:
            sys.modules["nemo_retriever.models.local"] = prev_local
