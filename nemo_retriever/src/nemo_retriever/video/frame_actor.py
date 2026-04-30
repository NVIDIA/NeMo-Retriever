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
import io
import logging
import tempfile
from typing import Any, Dict, List, Optional

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


def _dhash(image_b64: str, hash_size: int = 8) -> Optional[int]:
    """Difference-hash of a base64-encoded PNG, packed into a 64-bit integer.

    dhash is a perceptual hash: resize to ``(hash_size+1) x hash_size``
    grayscale, then compare each pixel to its right neighbour and pack the
    results as bits. Two frames with similar overall brightness layout end
    up with hashes that are close in Hamming distance, even if individual
    pixel values differ from encoder noise. Returns ``None`` if decoding
    fails so the caller can fall back to keeping the row.
    """
    try:
        from PIL import Image  # local import to avoid hard dep at module load
    except Exception:  # pragma: no cover
        return None
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        px = list(img.getdata())
    except Exception:
        return None
    bits = 0
    for r in range(hash_size):
        for c in range(hash_size):
            i = r * (hash_size + 1) + c
            if px[i] > px[i + 1]:
                bits |= 1 << (r * hash_size + c)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _seg_field(md: Any, key: str, default: float = 0.0) -> float:
    if isinstance(md, dict):
        try:
            return float(md.get(key) or default)
        except (TypeError, ValueError):
            return default
    return default


def dedup_video_frames(
    batch_df: pd.DataFrame,
    max_hamming_distance: int = 5,
    max_dropped_frames: int = 2,
) -> pd.DataFrame:
    """Collapse runs of perceptually-similar adjacent frames into one row each.

    Per ``source_path``, frames are sorted by ``segment_start_seconds`` and
    walked in order. A frame joins the current run when:
      - its dhash is within ``max_hamming_distance`` bits of the run's hash, AND
      - the gap to the run's current end is at most
        ``max_dropped_frames / fps`` seconds (so a small number of dropped
        frames in the middle is tolerated, but a long disappearance closes the run).
    The run's first frame is kept; its ``segment_end_seconds`` (and
    ``frame_timestamp_seconds`` midpoint) is extended to span the whole run.
    Other frames in the run are dropped.

    Without the time-window extension the kept row would say a slide was
    visible only for its first frame (e.g. [2.0, 4.0]) even when it stayed
    on screen for minutes — utterances during that span would miss the
    midpoint-overlap recall match. This function fixes that.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    if "image_b64" not in batch_df.columns:
        return batch_df

    threshold = max(0, int(max_hamming_distance))
    max_drop = max(0, int(max_dropped_frames))

    work = batch_df.reset_index(drop=True).copy()
    work["__seg_start"] = work["metadata"].apply(lambda m: _seg_field(m, "segment_start_seconds"))

    output_rows: List[Dict[str, Any]] = []
    for _source, group in work.groupby("source_path", sort=False):
        group = group.sort_values("__seg_start")
        active_hash: Optional[int] = None
        active_idx: Optional[int] = None  # index in output_rows
        active_end: Optional[float] = None

        for _, row in group.iterrows():
            md = row.get("metadata")
            row_start = _seg_field(md, "segment_start_seconds")
            row_end = _seg_field(md, "segment_end_seconds")
            fps = _seg_field(md, "fps", default=1.0) or 1.0
            b64 = row.get("image_b64")

            row_dict = row.to_dict()
            row_dict.pop("__seg_start", None)

            h: Optional[int] = None
            if isinstance(b64, str) and b64:
                h = _dhash(b64)

            max_gap = float(max_drop) / max(fps, 0.001)
            can_merge = (
                active_hash is not None
                and h is not None
                and active_end is not None
                and _hamming(h, active_hash) <= threshold
                and (row_start - active_end) <= max_gap
            )
            if can_merge:
                # Extend the kept run's window through this frame.
                kept = output_rows[active_idx]
                kept_md = dict(kept.get("metadata") or {})
                if row_end > float(kept_md.get("segment_end_seconds") or 0.0):
                    kept_md["segment_end_seconds"] = row_end
                    start = float(kept_md.get("segment_start_seconds") or 0.0)
                    kept_md["frame_timestamp_seconds"] = (start + row_end) / 2.0
                kept_md["dedup_merged_count"] = int(kept_md.get("dedup_merged_count", 1)) + 1
                kept["metadata"] = kept_md
                active_end = row_end
            else:
                # Either no active run, hash too different, gap too large, or unhashable —
                # start a new run rooted at this frame.
                output_rows.append(row_dict)
                active_idx = len(output_rows) - 1
                active_end = row_end
                active_hash = h  # may be None for unhashable frames; that breaks the run

    if not output_rows:
        return pd.DataFrame()
    return pd.DataFrame(output_rows)


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
