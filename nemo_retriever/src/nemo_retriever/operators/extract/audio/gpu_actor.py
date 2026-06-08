# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local ASR variant — loads ``nvidia/parakeet-ctc-1.1b`` via HuggingFace.

This is the GPU-tagged counterpart to the remote-only
:class:`~nemo_retriever.audio.cpu_actor.ASRCPUActor`. Loading the local model
pulls torch + transformers; that's the entire reason this lives in a separate
module from the CPU variant — keeping the import path off the CPU-only
``retriever ingest <mp3>`` flow.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from nemo_retriever.operators.extract.audio.asr_actor import (
    _ASRActorBase,
    _concat_with_passthrough,
    _split_audio_rows,
)
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.common.params import ASRParams

logger = logging.getLogger(__name__)


class ASRGPUActor(_ASRActorBase, AbstractOperator, GPUOperator):
    """Local ``nvidia/parakeet-ctc-1.1b`` via HuggingFace transformers.

    Loads weights eagerly at construction. ``ParakeetCTC1B1ASR`` selects
    ``cuda`` when available and falls back to ``cpu`` otherwise; the
    :class:`~nemo_retriever.audio.asr_actor.ASRActor` archetype prefers this
    variant when no remote ``audio_endpoints`` is specified and a GPU is
    available, but the model itself runs on either device.
    """

    def __init__(self, params: ASRParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params or ASRParams()
        from nemo_retriever.models.local import ParakeetCTC1B1ASR

        self._model = ParakeetCTC1B1ASR()

    def process(self, batch_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return self._empty_output_frame()
        audio_df, passthrough_df = _split_audio_rows(batch_df)
        if audio_df.empty:
            return passthrough_df
        asr_out = self._call_local_batch(audio_df)
        return _concat_with_passthrough(asr_out, passthrough_df)

    def _call_local_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """One batched transcribe call for the whole batch."""
        if self._model is None:
            return self._empty_output_frame()
        temp_paths: List[Optional[str]] = []
        paths_for_model: List[str] = []
        rows_list: List[pd.Series] = []
        for _, row in batch_df.iterrows():
            rows_list.append(row)
            raw = row.get("bytes")
            path = row.get("path")
            path_to_use: Optional[str] = None
            temp_created: Optional[str] = None
            if path and Path(path).exists():
                path_to_use = str(path)
            elif raw is not None:
                try:
                    f = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
                    f.write(raw)
                    f.close()
                    path_to_use = f.name
                    temp_created = f.name
                except Exception as e:
                    logger.warning("Failed to write temp file for ASR: %s", e)
                    path_to_use = ""
            else:
                if path:
                    try:
                        with open(path, "rb") as fp:
                            raw = fp.read()
                    except Exception as e:
                        logger.warning("Could not read %s: %s", path, e)
                        path_to_use = ""
                    else:
                        try:
                            f = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
                            f.write(raw)
                            f.close()
                            path_to_use = f.name
                            temp_created = f.name
                        except Exception as e:
                            logger.warning("Failed to write temp file for ASR: %s", e)
                            path_to_use = ""
                else:
                    path_to_use = ""
            paths_for_model.append(path_to_use or "")
            temp_paths.append(temp_created)

        try:
            decoded = self._model.transcribe_with_segments(paths_for_model) if paths_for_model else []
        finally:
            for p in temp_paths:
                if p:
                    Path(p).unlink(missing_ok=True)

        out_rows: List[Dict[str, Any]] = []
        for row, (transcript, segments) in zip(rows_list, decoded):
            out_rows.extend(self._build_output_rows(row, transcript or "", segments=segments))

        if not out_rows:
            return self._empty_output_frame()
        return pd.DataFrame(out_rows)
