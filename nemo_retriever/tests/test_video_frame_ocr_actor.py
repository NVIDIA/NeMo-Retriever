# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.video.ocr_actor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from nemo_retriever.video.ocr_actor import (
    VideoFrameOCRActor,
    VideoFrameOCRCPUActor,
    VideoFrameOCRGPUActor,
)


def _make_frame_df(image_b64s: list[str]) -> pd.DataFrame:
    rows = []
    for i, b64 in enumerate(image_b64s):
        rows.append(
            {
                "path": f"/tmp/frame_{i}.png",
                "source_path": "/tmp/v.mp4",
                "image_b64": b64,
                "page_number": i,
                "metadata": {
                    "source_path": "/tmp/v.mp4",
                    "frame_index": i,
                    "fps": 1.0,
                    "frame_timestamp_seconds": float(i) + 0.5,
                    "segment_start_seconds": float(i),
                    "segment_end_seconds": float(i) + 1.0,
                    "_content_type": "video_frame",
                    "modality": "video_frame",
                },
                "bytes": b"fake",
            }
        )
    return pd.DataFrame(rows)


def test_archetype_prefers_cpu_when_invoke_url_set() -> None:
    assert VideoFrameOCRActor.prefers_cpu_variant({"ocr_invoke_url": "https://example/ocr"}) is True
    assert VideoFrameOCRActor.prefers_cpu_variant({"invoke_url": "https://example/ocr"}) is True
    # Without invoke_url, prefers GPU when available.
    assert VideoFrameOCRActor.prefers_cpu_variant({}) is False


def test_cpu_actor_calls_remote_batched_with_b64_list() -> None:
    df = _make_frame_df(["AAA", "BBB", ""])  # empty b64 row should pass through with empty text

    fake_response = [
        [{"text_prediction": {"text": "hello world"}}],
        [{"text_prediction": {"text": "frame two"}}],
    ]
    nim_client = MagicMock()
    nim_client.invoke_image_inference_batches = MagicMock(return_value=fake_response)

    actor = VideoFrameOCRCPUActor(ocr_invoke_url="https://example/ocr")
    actor._nim_client = nim_client

    out = actor.run(df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2  # empty-b64 row dropped
    assert out["text"].tolist() == ["hello world", "frame two"]
    nim_client.invoke_image_inference_batches.assert_called_once()
    call_kwargs = nim_client.invoke_image_inference_batches.call_args.kwargs
    assert call_kwargs["image_b64_list"] == ["AAA", "BBB"]
    assert call_kwargs["invoke_url"] == "https://example/ocr"


def test_gpu_actor_invokes_local_model_per_frame() -> None:
    df = _make_frame_df(["b64_one", "b64_two"])

    fake_model = MagicMock()
    fake_model.invoke = MagicMock(
        side_effect=[
            [{"text_prediction": {"text": "alpha"}}],
            [{"text_prediction": {"text": "beta"}}],
        ]
    )

    actor = VideoFrameOCRGPUActor()
    actor._model = fake_model

    out = actor.run(df)
    assert isinstance(out, pd.DataFrame)
    assert out["text"].tolist() == ["alpha", "beta"]
    assert fake_model.invoke.call_count == 2


def test_gpu_actor_drops_empty_text_rows() -> None:
    df = _make_frame_df(["b64_one", "b64_two"])
    fake_model = MagicMock()
    fake_model.invoke = MagicMock(
        side_effect=[
            [{"text_prediction": {"text": "alpha"}}],
            [{"text_prediction": {"text": ""}}],
        ]
    )
    actor = VideoFrameOCRGPUActor()
    actor._model = fake_model

    out = actor.run(df)
    assert len(out) == 1
    assert out["text"].iloc[0] == "alpha"
    assert out["page_number"].iloc[0] == 0


def test_gpu_actor_default_loads_nemotron_ocr_v2(monkeypatch) -> None:
    """No flag → GPU actor must lazy-load NemotronOCRV2, not v1.

    Pins the v2-default switch (commit 8f6ab501) for the video frame OCR
    surface, which doesn't go through ``resolve_ocr_archetype``.
    """
    loaded: list[str] = []

    class _FakeV2:
        def __init__(self, *a: object, **kw: object) -> None:
            loaded.append("v2")

    class _FakeV1:
        def __init__(self, *a: object, **kw: object) -> None:
            loaded.append("v1")

    import nemo_retriever.model.local as local_models

    monkeypatch.setattr(local_models, "NemotronOCRV2", _FakeV2, raising=False)
    monkeypatch.setattr(local_models, "NemotronOCRV1", _FakeV1, raising=False)

    actor = VideoFrameOCRGPUActor()
    actor._ensure_model()
    assert loaded == ["v2"]


def test_cpu_actor_default_endpoint_is_v2() -> None:
    """No flag → CPU actor must point at the v2 NIM endpoint by default."""
    actor = VideoFrameOCRCPUActor()
    assert actor._invoke_url == "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"
