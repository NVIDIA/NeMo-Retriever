# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AudioVisualFuser: emit per-utterance ``audio_visual`` rows that combine
audio transcript text with concurrent video frame OCR text.

Each ASR utterance is paired with the single concurrent frame whose
window is most centred on the utterance (tiebreak: longer OCR), and the
fused text is rendered as ``"<audio>. <visual>"``. The source audio row
is dropped whenever a fused row was produced for its window so retrieval
doesn't see two near-identical embeddings (audio-only and audio+visual)
for the same utterance; audio with no concurrent frames is preserved.

Set :attr:`AudioVisualFuseParams.enabled` to ``False`` to skip fusion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.ocr.shared import concat_with_passthrough
from nemo_retriever.params import AudioVisualFuseParams
from nemo_retriever.video import _content_types as _CT

logger = logging.getLogger(__name__)

#: Hard cap on the visual portion of a fused row — keeps a long slide-OCR
#: blob from dominating the embedding over the (typically much shorter)
#: audio transcript.
FRAME_TEXT_MAX_CHARS: int = 200


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


def _keep_audio(row: Any, fused_window_keys: set[tuple[str, float, float]]) -> bool:
    """Drop audio rows whose window already produced a fused row; keep everything else."""
    if _row_content_type(row) != _CT.AUDIO:
        return True
    window = _row_segment_window(row)
    if window is None:
        return True
    source = getattr(row, "source_path", None)
    if not isinstance(source, str):
        return True
    return (source, float(window[0]), float(window[1])) not in fused_window_keys


@designer_component(
    name="Audio-Visual Fuser",
    category="Video",
    compute="cpu",
    description="Fuses audio utterances with the most-representative concurrent video frame OCR text",
)
class AudioVisualFuser(AbstractOperator, CPUOperator):
    """Replace audio rows with fused audio+visual rows where frames overlap.

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
            if _row_content_type(row) != _CT.VIDEO_FRAME:
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

        if not frames_by_source:
            return batch_df

        fused_rows: List[Dict[str, Any]] = []
        for row in batch_df.itertuples(index=False):
            if _row_content_type(row) != _CT.AUDIO:
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
            concurrent_entries = [
                (f_start, f_end, text)
                for f_start, f_end, text in frame_entries
                if max(u_start, f_start) <= min(u_end, f_end)
            ]
            if not concurrent_entries:
                continue

            # Pick the frame whose midpoint is closest to the utterance
            # midpoint; tiebreak by longer OCR (favours slides that actually
            # carry content vs. near-blank ones).
            u_mid = (u_start + u_end) / 2.0
            best = min(
                concurrent_entries,
                key=lambda fe: (abs((fe[0] + fe[1]) / 2.0 - u_mid), -len(fe[2])),
            )
            visual_text = best[2]
            if len(visual_text) > FRAME_TEXT_MAX_CHARS:
                visual_text = visual_text[:FRAME_TEXT_MAX_CHARS].rstrip()

            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(batch_df.columns, row))
            metadata = dict(row_dict.get("metadata") or {})
            metadata.update(
                {
                    "segment_start_seconds": float(u_start),
                    "segment_end_seconds": float(u_end),
                    "modality": _CT.AUDIO_VISUAL,
                    "_content_type": _CT.AUDIO_VISUAL,
                    "fused_concurrent_total": len(concurrent_entries),
                }
            )
            fused_text = f"{audio_text.strip()}. {visual_text}".strip()
            fused_row = dict(row_dict)
            fused_row["text"] = fused_text
            fused_row["metadata"] = metadata
            fused_row["_content_type"] = _CT.AUDIO_VISUAL
            fused_rows.append(fused_row)

        if not fused_rows:
            return batch_df

        fused_df = pd.DataFrame(fused_rows)

        fused_window_keys = {
            (
                str(row["source_path"] or ""),
                float(row["metadata"]["segment_start_seconds"]),
                float(row["metadata"]["segment_end_seconds"]),
            )
            for row in fused_rows
        }
        keep_mask = [_keep_audio(row, fused_window_keys) for row in batch_df.itertuples(index=False)]
        upstream_df = batch_df.iloc[keep_mask].reset_index(drop=True)

        return concat_with_passthrough(fused_df, upstream_df)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
