# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ASRActor: Ray Data map_batches callable for speech-to-text.

Supports remote (Parakeet/Riva gRPC) or local (HuggingFace nvidia/parakeet-ctc-1.1b).
When audio_endpoints are both null/empty, uses local model; otherwise uses remote client.

Core batch function: asr_chunks_to_text(batch_df, model=..., client=..., asr_params=...).
Model injection happens there; inprocess and GPU pool pass a ParakeetCTC1B1ASR instance.
ASRActor is a thin wrapper that holds _model/_client and delegates to asr_chunks_to_text.

Consumes chunk rows (path, bytes, source_path, duration, chunk_index, metadata)
and produces rows with text (transcript) for downstream embed/VDB. For now,
``segment_audio=True`` only fans out rows when using a hosted/remote Parakeet
client, because the local Hugging Face Parakeet model does not emit
punctuation-aware transcripts that can be segmented into sentences.
"""

from __future__ import annotations

import base64
import copy
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from nemo_retriever.params import ASRParams


def _use_remote(params: ASRParams) -> bool:
    """True if at least one of audio_endpoints is set (use remote gRPC client)."""
    grpc = (params.audio_endpoints[0] or "").strip()
    http = (params.audio_endpoints[1] or "").strip()
    return bool(grpc or http)


logger = logging.getLogger(__name__)

# Default NGC/NVCF Parakeet gRPC endpoint when using cloud ASR
DEFAULT_NGC_ASR_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"
# Default NVCF function ID for Parakeet NIM (same as nv-ingest default_libmode_pipeline_impl)
DEFAULT_NGC_ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"


def asr_params_from_env(
    *,
    grpc_endpoint_var: str = "AUDIO_GRPC_ENDPOINT",
    auth_token_var: str = "NGC_API_KEY",
    function_id_var: str = "AUDIO_FUNCTION_ID",
    default_grpc_endpoint: Optional[str] = DEFAULT_NGC_ASR_GRPC_ENDPOINT,
    default_function_id: Optional[str] = DEFAULT_NGC_ASR_FUNCTION_ID,
) -> ASRParams:
    """
    Build ASRParams from environment variables for cloud/NGC ASR.

    - AUDIO_GRPC_ENDPOINT: gRPC endpoint (default: grpc.nvcf.nvidia.com:443 for NGC).
    - NGC_API_KEY: Bearer token for NGC/NVCF (required for cloud).
    - AUDIO_FUNCTION_ID: NVCF function ID for the Parakeet NIM (default: same as nv-ingest libmode).

    Returns ASRParams with auth_token and function_id set from env when present.
    When NGC_API_KEY is set but AUDIO_FUNCTION_ID is not, uses the nv-ingest default Parakeet NIM function ID.
    Local ASR always uses the transformers backend (nvidia/parakeet-ctc-1.1b).
    """
    import os

    auth_token = (os.environ.get(auth_token_var) or "").strip() or None
    function_id = (os.environ.get(function_id_var) or "").strip() or None
    if auth_token and function_id is None and default_function_id:
        function_id = default_function_id

    # Only use remote (NGC) endpoint when credentials are set; otherwise use local Parakeet.
    grpc_from_env = (os.environ.get(grpc_endpoint_var) or "").strip()
    if grpc_from_env:
        grpc_endpoint = grpc_from_env
    elif auth_token or function_id:
        grpc_endpoint = default_grpc_endpoint or ""
    else:
        grpc_endpoint = ""  # Local ASR (nvidia/parakeet-ctc-1.1b via Transformers)

    return ASRParams(
        audio_endpoints=(grpc_endpoint or None, None),
        audio_infer_protocol="grpc",
        function_id=function_id,
        auth_token=auth_token,
    )


try:
    from nv_ingest_api.internal.primitives.nim.model_interface.parakeet import (
        create_audio_inference_client,
    )

    _PARAKEET_AVAILABLE = True
except ImportError:
    create_audio_inference_client = None  # type: ignore[misc, assignment]
    _PARAKEET_AVAILABLE = False


def _get_client(params: ASRParams):  # noqa: ANN201
    if not _PARAKEET_AVAILABLE or create_audio_inference_client is None:
        raise RuntimeError(
            "ASRActor requires nv-ingest-api (Parakeet client). "
            "Install with: pip install nv-ingest-api (or add nv-ingest-api to dependencies)."
        )
    grpc_endpoint = (params.audio_endpoints[0] or "").strip() or None
    http_endpoint = (params.audio_endpoints[1] or "").strip() or None
    if not grpc_endpoint:
        raise ValueError(
            "ASR audio_endpoints[0] (gRPC) must be set for Parakeet (e.g. localhost:50051 or grpc.nvcf.nvidia.com:443)."
        )
    return create_audio_inference_client(
        (grpc_endpoint, http_endpoint or ""),
        infer_protocol=params.audio_infer_protocol or "grpc",
        auth_token=params.auth_token,
        function_id=params.function_id,
        use_ssl=bool("nvcf.nvidia.com" in grpc_endpoint and params.function_id),
        ssl_cert=None,
    )


def _infer_remote(client: Any, raw: bytes, path: Optional[str]) -> Optional[tuple[List[Dict[str, Any]], str]]:
    """Use remote Parakeet client to transcribe audio bytes; return (segments, transcript)."""
    audio_b64 = base64.b64encode(raw).decode("ascii")
    try:
        segments, transcript = client.infer(
            audio_b64,
            model_name="parakeet",
        )
        safe_segments = segments if isinstance(segments, list) else []
        safe_transcript = transcript if isinstance(transcript, str) else ""
        return safe_segments, safe_transcript
    except Exception as e:
        logger.warning("Parakeet infer failed for path=%s: %s", path, e)
        return None


def _asr_out_columns() -> List[str]:
    return [
        "path",
        "source_path",
        "duration",
        "chunk_index",
        "metadata",
        "page_number",
        "text",
    ]


def _build_output_rows(
    row: pd.Series,
    transcript: str,
    *,
    segment_audio: bool,
    segments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build one or more output rows for a chunk, optionally exploding remote punctuation segments."""
    path = row.get("path")
    source_path = row.get("source_path", path)
    duration = row.get("duration")
    chunk_index = row.get("chunk_index", 0)
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {"source_path": source_path, "chunk_index": chunk_index, "duration": duration}
    else:
        metadata = copy.deepcopy(metadata)
    metadata.setdefault("source_path", source_path)
    metadata.setdefault("chunk_index", chunk_index)
    metadata.setdefault("duration", duration)
    page_number = row.get("page_number", chunk_index)

    if segment_audio and segments:
        out_rows: List[Dict[str, Any]] = []
        segment_count = len(segments)
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            segment_text = str(segment.get("text") or "").strip()
            if not segment_text:
                continue
            segment_metadata = copy.deepcopy(metadata)
            segment_metadata["segment_index"] = segment_index
            segment_metadata["segment_count"] = segment_count
            if segment.get("start") is not None:
                segment_metadata["segment_start"] = segment.get("start")
            if segment.get("end") is not None:
                segment_metadata["segment_end"] = segment.get("end")
            out_rows.append(
                {
                    "path": path,
                    "source_path": source_path,
                    "duration": duration,
                    "chunk_index": chunk_index,
                    "metadata": segment_metadata,
                    "page_number": page_number,
                    "text": segment_text,
                }
            )
        if out_rows:
            return out_rows

    return [
        {
            "path": path,
            "source_path": source_path,
            "duration": duration,
            "chunk_index": chunk_index,
            "metadata": metadata,
            "page_number": page_number,
            "text": transcript,
        }
    ]


def asr_chunks_to_text(
    batch_df: pd.DataFrame,
    *,
    model: Any = None,
    client: Any = None,
    asr_params: Optional[dict] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Core batch function: turn chunk rows (path/bytes) into rows with text (transcript).

    model: ParakeetCTC1B1ASR instance for local ASR (or None).
    client: Remote gRPC client for remote ASR (or None).
    asr_params: ASRParams or dict for options / creating backend when none injected.

    If both model and client are None, builds ASRActor(asr_params) and delegates
    with model=actor._model, client=actor._client.

    When ``params.segment_audio`` is enabled for remote Parakeet, punctuation-delimited
    segments are emitted as multiple rows per chunk.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return pd.DataFrame(columns=_asr_out_columns())

    params = ASRParams(**(asr_params or {}))

    # Local path: model has transcribe(paths)
    if model is not None and hasattr(model, "transcribe"):
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
            transcripts = model.transcribe(paths_for_model) if paths_for_model else []
        finally:
            for p in temp_paths:
                if p:
                    Path(p).unlink(missing_ok=True)

        out_rows: List[Dict[str, Any]] = []
        for row, transcript in zip(rows_list, transcripts):
            out_rows.extend(
                _build_output_rows(
                    row,
                    transcript or "",
                    segment_audio=params.segment_audio,
                    segments=None,
                )
            )
        if not out_rows:
            return pd.DataFrame(columns=_asr_out_columns())
        return pd.DataFrame(out_rows)

    # Remote path: client is set
    if client is not None:
        out_rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            raw = row.get("bytes")
            path = row.get("path")
            if raw is None and path:
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                except Exception as e:
                    logger.warning("Could not read %s: %s", path, e)
                    continue
            if raw is None:
                continue
            remote_result = _infer_remote(client, raw, path)
            if remote_result is None:
                continue
            segments, transcript = remote_result
            out_rows.extend(
                _build_output_rows(
                    row,
                    transcript,
                    segment_audio=params.segment_audio,
                    segments=segments,
                )
            )
        if not out_rows:
            return pd.DataFrame(columns=_asr_out_columns())
        return pd.DataFrame(out_rows)

    # Backward compat: no model/client injected, create actor and delegate
    actor = ASRActor(params=params)
    return asr_chunks_to_text(
        batch_df,
        model=actor._model,
        client=actor._client,
        asr_params=params.model_dump(mode="python"),
        **kwargs,
    )


class ASRActor:
    """
    Ray Data map_batches callable: chunk rows (path/bytes) -> rows with text (transcript).

    When audio_endpoints are set, uses Parakeet (Riva ASR) via gRPC. When both are
    null/empty, uses local HuggingFace/NeMo Parakeet (nvidia/parakeet-ctc-1.1b).
    Output rows have path, text, page_number, metadata for downstream embed. When
    ``params.segment_audio`` is enabled for remote Parakeet, punctuation-delimited
    segments are emitted as multiple rows per chunk.
    """

    def __init__(self, params: ASRParams | None = None) -> None:
        self._params = params or ASRParams()
        if _use_remote(self._params):
            self._client = _get_client(self._params)
            self._model = None
        else:
            self._client = None
            from nemo_retriever.model.local import ParakeetCTC1B1ASR

            self._model = ParakeetCTC1B1ASR()

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        return asr_chunks_to_text(
            batch_df,
            model=self._model,
            client=self._client,
            asr_params=self._params.model_dump(mode="python"),
        )
