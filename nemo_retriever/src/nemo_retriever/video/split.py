# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoSplitActor: splits one video file into per-stream rows in a single
graph stage so the downstream ASR + OCR + scene fusion stages can run as
separate Ray Data MapBatches operators.

Conceptually parallel to :class:`PDFSplitActor` (one PDF -> per-page rows):
each input video file produces both audio-chunk rows and frame rows.

  - N audio-chunk rows tagged ``_content_type="audio"`` (same shape as
    :class:`MediaChunkActor` output)
  - M frame rows tagged ``_content_type="video_frame"`` (same shape as
    :class:`VideoFrameActor` output)

Both kinds are emitted in the same DataFrame; downstream actors filter on
``_content_type``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.audio.chunk_actor import _chunk_one
from nemo_retriever.audio.media_interface import MediaInterface, is_media_available
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.params import AudioChunkParams, VideoFrameParams
from nemo_retriever.video import _content_types as _CT
from nemo_retriever.video.frame_actor import _extract_one, dedup_video_frames

logger = logging.getLogger(__name__)


@designer_component(
    name="Video Split",
    category="Video",
    compute="cpu",
    description="Splits a video into audio-chunk and frame rows tagged by _content_type.",
    category_color="#ff6b6b",
)
class VideoSplitActor(AbstractOperator, CPUOperator):
    """Per-row: emits audio chunk rows + frame rows from one video file."""

    def __init__(
        self,
        audio_chunk_params: AudioChunkParams | None = None,
        video_frame_params: VideoFrameParams | None = None,
    ) -> None:
        super().__init__(
            audio_chunk_params=audio_chunk_params,
            video_frame_params=video_frame_params,
        )
        if not is_media_available():
            raise RuntimeError("VideoSplitActor requires ffmpeg; install ffmpeg-python and system ffmpeg.")
        self._audio_chunk_params = audio_chunk_params or AudioChunkParams()
        self._video_frame_params = video_frame_params or VideoFrameParams()
        self._interface = MediaInterface()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            path = row.get("path")
            if path is None:
                continue
            path_str = str(path)
            if not path_str.strip():
                continue

            if self._audio_chunk_params.enabled:
                try:
                    chunk_rows = _chunk_one(path_str, self._audio_chunk_params, self._interface)
                except Exception as exc:
                    logger.exception("Audio chunking failed for %s: %s", path_str, exc)
                    chunk_rows = []
                for chunk_row in chunk_rows:
                    chunk_row["_content_type"] = _CT.AUDIO
                    rows.append(chunk_row)

            if self._video_frame_params.enabled:
                try:
                    frame_rows = _extract_one(path_str, self._video_frame_params, self._interface)
                except Exception as exc:
                    logger.exception("Frame extraction failed for %s: %s", path_str, exc)
                    frame_rows = []
                rows.extend(frame_rows)

        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        if self._video_frame_params.dedup and "_content_type" in out.columns:
            frame_mask = out["_content_type"] == _CT.VIDEO_FRAME
            if frame_mask.any():
                deduped_frames = dedup_video_frames(
                    out[frame_mask].reset_index(drop=True),
                    max_hamming_distance=int(self._video_frame_params.dedup_max_hamming_distance),
                    max_dropped_frames=int(self._video_frame_params.dedup_max_dropped_frames),
                )
                non_frames = out[~frame_mask].reset_index(drop=True)
                out = pd.concat([non_frames, deduped_frames], ignore_index=True, sort=False)
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
