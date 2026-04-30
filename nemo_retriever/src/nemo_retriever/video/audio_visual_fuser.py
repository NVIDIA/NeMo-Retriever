# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AudioVisualFuser: emit per-utterance ``audio_visual`` rows that combine
audio transcript text with concurrent video frame OCR text.

Designed to boost retrieval recall on questions whose answer requires
both audio and visual modalities (``answer_modality="Audio + Visual"``
in the eval ground truth). For each ASR utterance row, we find frame
OCR rows whose timestamps fall within the utterance's wall-clock
window and emit one fused row that the embedder can index together.

The fuser is *additive* — the input audio and frame rows are preserved
in the output so single-modality queries still hit them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.params import AudioVisualFuseParams

logger = logging.getLogger(__name__)


def _row_content_type(row: Any) -> str:
    md = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
    if isinstance(md, dict):
        ct = md.get("_content_type")
        if isinstance(ct, str):
            return ct
    direct = row.get("_content_type") if isinstance(row, dict) else getattr(row, "_content_type", None)
    return str(direct) if isinstance(direct, str) else ""


def _row_segment_window(row: Any) -> tuple[float, float] | None:
    md = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
    if not isinstance(md, dict):
        return None
    try:
        start = float(md["segment_start_seconds"])
        end = float(md["segment_end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    return start, end


@designer_component(
    name="Audio-Visual Fuser",
    category="Video",
    compute="cpu",
    description="Fuses audio utterances with concurrent video frame OCR text",
)
class AudioVisualFuser(AbstractOperator, CPUOperator):
    """Append fused audio+visual rows to a DataFrame of audio + frame rows.

    Self-join semantics: needs *all* rows for a given source (audio
    utterances + frame OCR) to be co-located in a single batch. The
    ``REQUIRES_GLOBAL_BATCH`` marker tells :class:`RayDataExecutor` to
    force a single block + ``batch_size=None`` for this stage, so the
    fuser sees the whole dataset in one ``process()`` call.
    """

    #: Read by ``RayDataExecutor`` to force a global view (one block, one batch).
    REQUIRES_GLOBAL_BATCH: bool = True

    def __init__(self, params: AudioVisualFuseParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params or AudioVisualFuseParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not self._params.enabled:
            return batch_df
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df

        # Bucket frame rows by source_path so we can self-join cheaply.
        # Each entry is (frame_start_seconds, frame_end_seconds, text). Storing
        # the window (rather than the midpoint timestamp) means dedup-merged
        # rows with wide windows still fuse with utterances inside that window.
        frames_by_source: Dict[str, List[tuple[float, float, str]]] = {}
        for row in batch_df.itertuples(index=False):
            if _row_content_type(row) != "video_frame":
                continue
            window = _row_segment_window(row)
            text = getattr(row, "text", None)
            if window is None or not isinstance(text, str) or not text.strip():
                continue
            source = getattr(row, "source_path", None)
            if not isinstance(source, str):
                continue
            f_start, f_end = window
            frames_by_source.setdefault(source, []).append((float(f_start), float(f_end), text.strip()))

        for entries in frames_by_source.values():
            entries.sort(key=lambda t: t[0])

        if not frames_by_source:
            return batch_df

        fused_rows: List[Dict[str, Any]] = []
        sep = self._params.frame_separator
        for row in batch_df.itertuples(index=False):
            if _row_content_type(row) != "audio":
                continue
            window = _row_segment_window(row)
            if window is None:
                continue
            u_start, u_end = window
            source = getattr(row, "source_path", None)
            if not isinstance(source, str):
                continue
            audio_text = getattr(row, "text", None)
            if not isinstance(audio_text, str) or not audio_text.strip():
                continue
            frame_entries = frames_by_source.get(source, [])
            # Window-overlap: a frame fuses when its visibility window
            # intersects the utterance window. Handles narrow per-frame windows
            # (single frame) and wide merged windows (text-dedup output) alike.
            concurrent = [text for f_start, f_end, text in frame_entries if max(u_start, f_start) <= min(u_end, f_end)]
            if not concurrent:
                continue

            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(batch_df.columns, row))
            metadata = dict(row_dict.get("metadata") or {})
            metadata.update(
                {
                    "segment_start_seconds": float(u_start),
                    "segment_end_seconds": float(u_end),
                    "modality": "audio_visual",
                    "_content_type": "audio_visual",
                    "fused_frame_count": len(concurrent),
                }
            )
            fused_text = (
                f"{self._params.audio_label}{audio_text.strip()}" f"\n{self._params.visual_label}{sep.join(concurrent)}"
            )
            fused_row = dict(row_dict)
            fused_row["text"] = fused_text
            fused_row["metadata"] = metadata
            fused_row["_content_type"] = "audio_visual"
            fused_rows.append(fused_row)

        if not fused_rows:
            return batch_df

        fused_df = pd.DataFrame(fused_rows)
        # Make sure both frames carry the union of columns so the LanceDB sink
        # sees ``_content_type`` (top-level) on every fused row, even if the
        # incoming batch lacks that column.
        for col in batch_df.columns:
            if col not in fused_df.columns:
                fused_df[col] = None
        for col in fused_df.columns:
            if col not in batch_df.columns:
                batch_df = batch_df.copy()
                batch_df[col] = None
        fused_df = fused_df[batch_df.columns.tolist()]
        return pd.concat([batch_df, fused_df], ignore_index=True, sort=False)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
