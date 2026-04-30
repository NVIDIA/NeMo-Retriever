# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameActor: Ray Data map_batches callable for video frame extraction.

Consumes rows from rd.read_binary_files (path, bytes) and produces one row
per frame with path, source_path, image_b64, bytes, page_number, metadata.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import tempfile
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.audio.media_interface import MediaInterface
from nemo_retriever.audio.media_interface import is_media_available
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.params import VideoFrameParams

logger = logging.getLogger(__name__)

# Output columns for downstream (OCR, embed, VDB).
FRAME_COLUMNS = [
    "path",
    "source_path",
    "image_b64",
    "page_number",
    "metadata",
    "bytes",
    "_content_type",
]


@designer_component(
    name="Video Frame Extractor",
    category="Video",
    compute="cpu",
    description="Extracts video frames at a fixed fps via ffmpeg",
    category_color="#ff6b6b",
)
class VideoFrameActor(AbstractOperator, CPUOperator):
    """
    Ray Data map_batches callable: DataFrame with path -> DataFrame of frame rows.

    Each output row has:
      - ``path``: frame PNG path
      - ``source_path``: original video path
      - ``image_b64``: base64-encoded PNG (the ``VideoFrameOCRActor`` reads this)
      - ``bytes``: raw PNG bytes (kept for compatibility with Ray Data binary readers)
      - ``page_number``: frame index (0, 1, 2, ...)
      - ``metadata``: dict with ``frame_timestamp_seconds``, ``segment_start_seconds``,
        ``segment_end_seconds``, ``fps``, ``source_path``, ``modality="video_frame"``,
        ``_content_type="video_frame"``.

    Frames are streamed to disk to avoid OOM on long videos.
    """

    def __init__(self, params: VideoFrameParams | None = None) -> None:
        super().__init__(params=params)
        if not is_media_available():
            raise RuntimeError(
                "VideoFrameActor requires ffmpeg. Install with: pip install ffmpeg-python and system ffmpeg."
            )
        self._params = params or VideoFrameParams()
        self._interface = MediaInterface()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(columns=FRAME_COLUMNS)

        out_rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            path = row.get("path")
            if path is None:
                continue
            path_str = str(path)
            if not path_str.strip():
                continue
            try:
                frame_rows = _extract_one(path_str, self._params, self._interface)
                out_rows.extend(frame_rows)
            except Exception as e:
                logger.exception("Error extracting frames from %s: %s", path_str, e)
                continue

        if not out_rows:
            return pd.DataFrame(columns=FRAME_COLUMNS)
        return pd.DataFrame(out_rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data


def _extract_one(source_path: str, params: VideoFrameParams, interface: MediaInterface) -> List[Dict[str, Any]]:
    """Extract frames from one video file and return a list of row dicts."""
    fps = float(params.fps)
    half_window = 0.5 / fps
    with tempfile.TemporaryDirectory(prefix="retriever_video_frames_") as tmpdir:
        frames = interface.extract_frames(
            source_path,
            tmpdir,
            fps=fps,
            max_frames=params.max_frames,
        )
        if not frames:
            logger.warning("No frames extracted from %s (ffmpeg returned 0 files)", source_path)
            return []

        rows: List[Dict[str, Any]] = []
        for idx, (frame_path, timestamp) in enumerate(frames):
            try:
                with open(frame_path, "rb") as f:
                    frame_bytes = f.read()
            except Exception as e:
                logger.warning("Could not read frame %s: %s", frame_path, e)
                continue
            image_b64 = base64.b64encode(frame_bytes).decode("ascii")
            metadata = {
                "source_path": source_path,
                "frame_index": idx,
                "fps": fps,
                "frame_timestamp_seconds": float(timestamp),
                "segment_start_seconds": max(0.0, float(timestamp) - half_window),
                "segment_end_seconds": float(timestamp) + half_window,
                "modality": "video_frame",
                "_content_type": "video_frame",
            }
            rows.append(
                {
                    "path": frame_path,
                    "source_path": source_path,
                    "image_b64": image_b64,
                    "page_number": idx,
                    "metadata": metadata,
                    "bytes": frame_bytes,
                    "_content_type": "video_frame",
                }
            )
        return rows


def dedup_video_frames(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Drop frame rows whose ``image_b64`` content matches an earlier row.

    Per ``source_path``, hashes each row's base64 PNG and keeps the first
    occurrence. Preserves the order and metadata of retained rows. Rows
    without an ``image_b64`` are passed through unchanged.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    if "image_b64" not in batch_df.columns:
        return batch_df

    seen: dict[str, set[str]] = {}
    keep: List[bool] = []
    for _, row in batch_df.iterrows():
        b64 = row.get("image_b64")
        source = row.get("source_path") or "<no_source>"
        if not isinstance(b64, str) or not b64:
            keep.append(True)
            continue
        h = hashlib.md5(b64.encode("utf-8")).hexdigest()
        bucket = seen.setdefault(str(source), set())
        if h in bucket:
            keep.append(False)
            continue
        bucket.add(h)
        keep.append(True)
    return batch_df.loc[keep].reset_index(drop=True)


def video_path_to_frames_df(path: str, params: VideoFrameParams | None = None) -> pd.DataFrame:
    """Synchronous loader: one video file path -> DataFrame of frame rows.

    Columns match :data:`FRAME_COLUMNS`. Used by inprocess ingest() when
    ``_pipeline_type == "video"``.
    """
    if not is_media_available():
        raise RuntimeError("video_path_to_frames_df requires ffmpeg.")
    params = params or VideoFrameParams()
    interface = MediaInterface()
    rows = _extract_one(path, params, interface)
    if not rows:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    return pd.DataFrame(rows)
