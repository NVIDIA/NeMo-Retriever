# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory audio/video buffers through MultiTypeExtractOperator in auto mode.

``GraphIngestor.buffers()`` rows carry the caller's name in ``path`` and the
payload in ``bytes``. That name must survive as ``source_path`` (the stored
VDB ``source_id``), and buffers that share a basename must not clobber each
other's bytes on the way to ffmpeg.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.common.params import (
    ASRParams,
    AudioVisualFuseParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import _MultiTypeExtractBase

_OPERATOR = "nemo_retriever.operators.graph_ops.multi_type_extract_operator"


class _CopyingMediaInterface:
    """ffmpeg stand-in: each chunk/frame carries the bytes of the file it was read from."""

    def split(self, input_path, output_dir, **_kwargs):
        chunk = Path(output_dir) / "chunk_000.wav"
        chunk.write_bytes(Path(input_path).read_bytes())
        return [str(chunk)]

    def probe_media(self, *_args, **_kwargs):
        return None, None, 1.0

    def extract_frames(self, input_path, output_dir, **_kwargs):
        frame = Path(output_dir) / "frame_000.jpg"
        frame.write_bytes(Path(input_path).read_bytes())
        return [(str(frame), 0.0)]


class _Passthrough:
    def run(self, df):
        return df


@pytest.fixture(autouse=True)
def fake_media(monkeypatch):
    monkeypatch.setattr("nemo_retriever.operators.extract.audio.chunk_actor.is_media_available", lambda: True)
    monkeypatch.setattr("nemo_retriever.operators.extract.audio.chunk_actor.MediaInterface", _CopyingMediaInterface)
    monkeypatch.setattr("nemo_retriever.operators.extract.video.frame_actor.is_ffmpeg_available", lambda: True)
    monkeypatch.setattr("nemo_retriever.operators.extract.video.frame_actor.MediaInterface", _CopyingMediaInterface)


def _run_auto(buffers: list[tuple[str, bytes]]) -> pd.DataFrame:
    op = _MultiTypeExtractBase(
        extraction_mode="auto",
        asr_params=ASRParams(),
        video_frame_params=VideoFrameParams(fps=1.0, dedup=False),
        video_text_dedup_params=VideoFrameTextDedupParams(enabled=False),
        av_fuse_params=AudioVisualFuseParams(enabled=False),
    )
    # Mirrors the DataFrame GraphIngestor builds from ``buffers()``.
    batch = pd.DataFrame([{"bytes": data, "path": name} for name, data in buffers])
    with patch(f"{_OPERATOR}.ASRActor", return_value=_Passthrough()), patch.object(
        op, "_instantiate_resolved", return_value=_Passthrough()
    ):
        out = op.run(batch)
    assert isinstance(out, pd.DataFrame) and not out.empty
    return out


@pytest.mark.parametrize("name", ["call.wav", "demo.mp4"])
def test_auto_mode_media_buffer_keeps_caller_name_as_source_path(name: str) -> None:
    out = _run_auto([(name, b"media bytes")])

    assert set(out["source_path"]) == {name}
    assert {md["source_path"] for md in out["metadata"]} == {name}


@pytest.mark.parametrize("basename", ["call.wav", "demo.mp4"])
def test_auto_mode_media_buffers_sharing_a_basename_do_not_overwrite(basename: str) -> None:
    out = _run_auto([(f"team_a/{basename}", b"team a"), (f"team_b/{basename}", b"team b")])

    seen = {(md["source_path"], bytes(data)) for md, data in zip(out["metadata"], out["bytes"])}
    assert seen == {(f"team_a/{basename}", b"team a"), (f"team_b/{basename}", b"team b")}
