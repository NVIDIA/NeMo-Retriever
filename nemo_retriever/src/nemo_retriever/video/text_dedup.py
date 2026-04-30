# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameTextDedup: merge consecutive video_frame rows with identical OCR
text into a single row spanning the union of their time windows.

Image-hash dedup runs before OCR and only catches byte-identical frames.
Slides that re-render every frame (with anti-aliasing or compression
noise) survive that pass but produce identical OCR output. This stage
collapses those streaks per ``(source_path, text)`` so the embedder
indexes one wider-window row instead of dozens of near-duplicates.

Self-join semantics: needs *all* frame rows for a given source in the
same batch. Marked with ``REQUIRES_GLOBAL_BATCH`` so
:class:`RayDataExecutor` forces a single-block / ``batch_size=None``
stage at this point in the graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.params import VideoFrameTextDedupParams

logger = logging.getLogger(__name__)


def _md(row: pd.Series) -> dict:
    md = row.get("metadata")
    return md if isinstance(md, dict) else {}


def _seg_start(row: pd.Series) -> float:
    try:
        return float(_md(row).get("segment_start_seconds") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _seg_end(row: pd.Series) -> float:
    try:
        return float(_md(row).get("segment_end_seconds") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _group_fps(group: pd.DataFrame) -> float:
    """Read fps from any frame in the group; default to 1.0 if missing/zero.

    All frames from the same source share the same fps, but be defensive
    in case a row's metadata is malformed.
    """
    for md in group["metadata"]:
        if isinstance(md, dict):
            try:
                fps = float(md.get("fps") or 0.0)
            except (TypeError, ValueError):
                fps = 0.0
            if fps > 0.0:
                return fps
    return 1.0


def _merge_run(group: pd.DataFrame, indices: List[int], run_start: float, run_end: float) -> Dict[str, Any]:
    """Build a merged row from the first member of the run, widening its time window."""
    first = group.iloc[indices[0]].to_dict()
    md = dict(first.get("metadata") or {})
    md["segment_start_seconds"] = float(run_start)
    md["segment_end_seconds"] = float(run_end)
    md["frame_timestamp_seconds"] = (float(run_start) + float(run_end)) / 2.0
    md["merged_frame_count"] = len(indices)
    first["metadata"] = md
    # Strip helper sort columns if they were added.
    first.pop("__seg_start", None)
    first.pop("__seg_end", None)
    return first


@designer_component(
    name="Video Frame Text Dedup",
    category="Video",
    compute="cpu",
    description="Merges consecutive video_frame rows with identical OCR text.",
)
class VideoFrameTextDedup(AbstractOperator, CPUOperator):
    """Per ``(source_path, text)`` group: merge time-adjacent runs into one row."""

    #: Read by ``RayDataExecutor`` to force one block + ``batch_size=None``.
    REQUIRES_GLOBAL_BATCH: bool = True

    def __init__(self, params: VideoFrameTextDedupParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params or VideoFrameTextDedupParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not self._params.enabled:
            return batch_df
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df
        if "_content_type" not in batch_df.columns:
            return batch_df
        is_frame = batch_df["_content_type"].astype(str) == "video_frame"
        if not is_frame.any():
            return batch_df

        frames = batch_df[is_frame].reset_index(drop=True)
        others = batch_df[~is_frame].reset_index(drop=True)

        # Drop frames that lack a text or source_path — pass them through alongside non-frames.
        has_text = frames["text"].fillna("").astype(str).str.len() > 0
        has_source = frames["source_path"].fillna("").astype(str).str.len() > 0
        mergable = frames[has_text & has_source].reset_index(drop=True)
        leftovers = frames[~(has_text & has_source)].reset_index(drop=True)
        if mergable.empty:
            return batch_df

        # Add helper sort columns; cleaned up in _merge_run.
        mergable["__seg_start"] = mergable.apply(_seg_start, axis=1)
        mergable["__seg_end"] = mergable.apply(_seg_end, axis=1)

        max_dropped = int(self._params.max_dropped_frames)
        merged_rows: List[Dict[str, Any]] = []
        for (_source, _text), group in mergable.groupby(["source_path", "text"], sort=False):
            group = group.sort_values("__seg_start").reset_index(drop=True)
            # Convert "max dropped frames" → seconds using the group's fps:
            # at fps=0.5 (frame interval=2s), 2 dropped frames → 4s gap; at
            # fps=1.0 → 2s; etc. Independent of fps, the rule is the same.
            fps = _group_fps(group)
            max_gap = float(max_dropped) / fps
            run_start = float(group.iloc[0]["__seg_start"])
            run_end = float(group.iloc[0]["__seg_end"])
            run_indices = [0]
            for i in range(1, len(group)):
                row_start = float(group.iloc[i]["__seg_start"])
                row_end = float(group.iloc[i]["__seg_end"])
                gap = row_start - run_end
                if gap <= max_gap:
                    run_end = max(run_end, row_end)
                    run_indices.append(i)
                else:
                    merged_rows.append(_merge_run(group, run_indices, run_start, run_end))
                    run_start = row_start
                    run_end = row_end
                    run_indices = [i]
            merged_rows.append(_merge_run(group, run_indices, run_start, run_end))

        merged_df = pd.DataFrame(merged_rows)
        # Drop helper columns from any branch that may still carry them.
        for col in ("__seg_start", "__seg_end"):
            if col in merged_df.columns:
                merged_df = merged_df.drop(columns=col)

        # Concat: merged frames + frames missing text/source + non-frame passthrough rows.
        # Harmonise columns so concat doesn't introduce NaN columns from misordered keys.
        pieces: List[pd.DataFrame] = [merged_df]
        if not leftovers.empty:
            pieces.append(leftovers)
        if not others.empty:
            pieces.append(others)
        all_cols = sorted({c for p in pieces for c in p.columns})
        for i, p in enumerate(pieces):
            for c in all_cols:
                if c not in p.columns:
                    p[c] = None
            pieces[i] = p[all_cols]
        out = pd.concat(pieces, ignore_index=True, sort=False)

        kept = len(merged_df)
        original = int(is_frame.sum())
        if kept < original:
            logger.info(
                "VideoFrameTextDedup: merged %d frame rows -> %d (max_dropped_frames=%d)",
                original,
                kept,
                max_dropped,
            )
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
