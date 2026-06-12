# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote ASR variant — calls Parakeet/Riva via gRPC, no local model weights.

Mirrors the CPU-actor pattern used by ``page_elements/cpu_actor.py`` and
``ocr/cpu_ocr.py``: a class constant carries the public NIM endpoint and
``__init__`` fills it in when the caller didn't provide one. The default
endpoint is the NVCF Parakeet deployment, so ``ASRCPUActor()`` with no args
"just works" against build.nvidia.com given an exported ``NVIDIA_API_KEY``.

No torch / transformers imports anywhere on this code path.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from nemo_retriever.operators.extract.audio import asr_actor as _asr_actor
from nemo_retriever.operators.extract.audio.asr_actor import (
    DEFAULT_NGC_ASR_FUNCTION_ID,
    DEFAULT_NGC_ASR_GRPC_ENDPOINT,
    _ASRActorBase,
    _concat_with_passthrough,
    _split_audio_rows,
)
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.common.params import ASRParams

logger = logging.getLogger(__name__)


class ASRCPUActor(_ASRActorBase, AbstractOperator, CPUOperator):
    """Remote Parakeet/Riva ASR. Defaults to the public NVCF endpoint.

    When the caller supplies ``ASRParams`` with empty ``audio_endpoints``
    (the default), this actor fills in:
      - ``audio_endpoints = (DEFAULT_GRPC_ENDPOINT, None)``
      - ``audio_infer_protocol = "grpc"``
      - ``function_id = DEFAULT_FUNCTION_ID`` (libmode Parakeet)
      - ``auth_token`` ← ``$NVIDIA_API_KEY`` if unset and the env var is present

    Mirrors the pattern used by ``PageElementDetectionCPUActor`` /
    ``OcrCPUActor`` / ``TableStructureCPUActor`` — CPU variant means remote NIM
    work, not local model inference. The local Parakeet path is the GPU
    variant (:class:`~nemo_retriever.audio.gpu_actor.ASRGPUActor`).
    """

    DEFAULT_GRPC_ENDPOINT = DEFAULT_NGC_ASR_GRPC_ENDPOINT
    DEFAULT_FUNCTION_ID = DEFAULT_NGC_ASR_FUNCTION_ID

    def __init__(self, params: ASRParams | None = None) -> None:
        super().__init__(params=params)
        self._params = self._apply_actor_defaults(params or ASRParams())
        # Dispatch through the source module so tests that ``patch(
        # 'nemo_retriever.audio.asr_actor._get_client')`` still intercept us.
        self._client = _asr_actor._get_client(self._params)

    @classmethod
    def _apply_actor_defaults(cls, params: ASRParams) -> ASRParams:
        """Fill in NVCF defaults when the caller left ``audio_endpoints`` empty.

        Env overrides honoured (in addition to the class-level constants):
          - ``AUDIO_FUNCTION_ID`` — pin a specific NVCF Parakeet function-id
            without code changes. Useful for A/B testing deployments.
          - ``NVIDIA_API_KEY`` — bearer token (also auto-resolved by
            ``_ParamsModel`` for ``api_key``-named fields elsewhere, but ASR
            historically uses ``auth_token`` which isn't matched by that
            mechanism).
        """
        grpc_ep, http_ep = params.audio_endpoints
        if grpc_ep or http_ep:
            return params  # caller supplied an endpoint — respect it
        updates: Dict[str, Any] = {
            "audio_endpoints": (cls.DEFAULT_GRPC_ENDPOINT, None),
            "audio_infer_protocol": "grpc",
        }
        if not params.function_id:
            env_fid = (os.environ.get("AUDIO_FUNCTION_ID") or "").strip() or None
            updates["function_id"] = env_fid or cls.DEFAULT_FUNCTION_ID
        if not params.auth_token:
            env_token = (os.environ.get("NVIDIA_API_KEY") or "").strip() or None
            if env_token:
                updates["auth_token"] = env_token
        return params.model_copy(update=updates)

    def process(self, batch_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return self._empty_output_frame()

        # When ``_content_type`` is set on the batch (mixed audio + video_frame
        # rows from a video pipeline), only ASR the audio rows and pass the
        # rest through unchanged. Audio-only pipelines have no ``_content_type``
        # column, so this branch is a no-op for them.
        audio_df, passthrough_df = _split_audio_rows(batch_df)
        if audio_df.empty:
            return passthrough_df
        asr_out = self._call_remote_batch(audio_df)
        return _concat_with_passthrough(asr_out, passthrough_df)

    def _call_remote_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """One infer call per row; server doesn't batch on its side."""
        out_rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            try:
                out_rows.extend(self._transcribe_one(row))
            except Exception as e:
                logger.exception("ASR failed for row path=%s: %s", row.get("path"), e)
                continue

        if not out_rows:
            return self._empty_output_frame()
        return pd.DataFrame(out_rows)

    def _transcribe_one(self, row: pd.Series) -> List[Dict[str, Any]]:
        raw = row.get("bytes")
        path = row.get("path")
        if raw is None and path:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except Exception as e:
                logger.warning("Could not read %s: %s", path, e)
                return []
        if raw is None:
            return []

        remote_result = self._transcribe_remote(raw, path)
        if remote_result is None:
            return []
        segments, transcript = remote_result
        return self._build_output_rows(row, transcript, segments=segments)

    def _transcribe_remote(self, raw: bytes, path: Optional[str]) -> Optional[tuple[List[Dict[str, Any]], str]]:
        """Send audio bytes to Parakeet and return (segments, transcript)."""
        audio_b64 = base64.b64encode(raw).decode("ascii")
        try:
            segments, transcript = self._client.infer(
                audio_b64,
                model_name="parakeet",
            )
            safe_segments = segments if isinstance(segments, list) else []
            safe_transcript = transcript if isinstance(transcript, str) else ""
            return safe_segments, safe_transcript
        except Exception as e:
            logger.warning("Parakeet infer failed for path=%s: %s", path, e)
            return None
