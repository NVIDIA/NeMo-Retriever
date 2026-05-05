# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ASRActor: Ray Data map_batches callable for speech-to-text.

Supports remote (Parakeet/Riva gRPC) or local (HuggingFace nvidia/parakeet-ctc-1.1b).
When audio_endpoints are both null/empty, uses local model; otherwise uses remote client.

Consumes chunk rows (path, bytes, source_path, duration, chunk_index, metadata)
and produces rows with text (transcript) for downstream embed/VDB. With
``segment_audio=True`` the remote (punctuation-bounded) and local (silence-gap,
from CTC frame timestamps) paths both fan out per-segment rows with start/end
times so ``recall_match_mode: audio_segment`` can match against time-aligned hits.
"""

from __future__ import annotations

import base64
import copy
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.params import ASRParams


def _to_chunk_relative_seconds(value: Any, chunk_duration_secs: float) -> Optional[float]:
    """Coerce a per-utterance timestamp to seconds, divided down from ms when needed.

    Local Parakeet returns seconds; the remote NIM client returns milliseconds.
    A seconds-valued utterance can't exceed the chunk duration — so anything
    past it must be ms. When the chunk duration is unknown (probe_media
    couldn't resolve it for some segmented MP4s and chunk_actor.py substitutes
    0.0), fall back to a value-range check: no legitimate audio segment lasts
    more than an hour, so anything past 3600 must be ms.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if chunk_duration_secs > 0 and v > chunk_duration_secs:
        return v / 1000.0
    if chunk_duration_secs <= 0 and v > 3600:
        return v / 1000.0
    return v


def _use_remote(params: ASRParams) -> bool:
    """True if at least one of audio_endpoints is set (use remote gRPC client)."""
    grpc = (params.audio_endpoints[0] or "").strip()
    http = (params.audio_endpoints[1] or "").strip()
    return bool(grpc or http)


def _split_audio_rows(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a mixed batch into audio rows (to ASR) and passthrough rows.

    Audio-only pipelines emit batches without a ``_content_type`` column;
    in that case the whole batch is treated as audio.
    """
    if "_content_type" not in batch_df.columns:
        return batch_df, pd.DataFrame()
    is_audio = batch_df["_content_type"].astype(str) == "audio"
    return (
        batch_df[is_audio].reset_index(drop=True),
        batch_df[~is_audio].reset_index(drop=True),
    )


def _concat_with_passthrough(processed: pd.DataFrame, passthrough: pd.DataFrame) -> pd.DataFrame:
    """Concat the ASR output with the passthrough rows, harmonising columns."""
    if passthrough is None or passthrough.empty:
        return processed
    if processed is None or processed.empty:
        return passthrough
    for col in processed.columns:
        if col not in passthrough.columns:
            passthrough = passthrough.assign(**{col: None})
    for col in passthrough.columns:
        if col not in processed.columns:
            processed = processed.assign(**{col: None})
    return pd.concat([processed[passthrough.columns.tolist()], passthrough], ignore_index=True, sort=False)


logger = logging.getLogger(__name__)

# Public NVCF Parakeet endpoint and the libmode function ID. Exposed as named
# constants so Python callers can opt into NVCF without hardcoding strings:
#   asr_params_from_env(default_grpc_endpoint=DEFAULT_NGC_ASR_GRPC_ENDPOINT)
DEFAULT_NGC_ASR_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"
DEFAULT_NGC_ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"


def asr_params_from_env(
    *,
    grpc_endpoint_var: str = "AUDIO_GRPC_ENDPOINT",
    auth_token_var: str = "NGC_API_KEY",
    function_id_var: str = "AUDIO_FUNCTION_ID",
    default_grpc_endpoint: Optional[str] = None,
    default_function_id: Optional[str] = DEFAULT_NGC_ASR_FUNCTION_ID,
) -> ASRParams:
    """
    Build ASRParams from environment variables, with optional Python-level defaults.

    Local Parakeet (nvidia/parakeet-ctc-1.1b via Transformers) is the default;
    remote ASR is opted into explicitly. ``NGC_API_KEY`` alone never flips ASR
    to remote — it's set in many environments for unrelated reasons (HF auth,
    other NIMs) and shouldn't silently route a local run to cloud.

    Two opt-in paths to remote, both honoured:

    - **Environment variable**: ``AUDIO_GRPC_ENDPOINT=grpc.nvcf.nvidia.com:443``
      (NVCF) or ``AUDIO_GRPC_ENDPOINT=localhost:50051`` (local NIM).
    - **Python API**: pass ``default_grpc_endpoint=...`` to this function. The
      env var wins when both are present. Use the exported
      :data:`DEFAULT_NGC_ASR_GRPC_ENDPOINT` constant for NVCF.

    - ``NGC_API_KEY`` — Bearer token; only consulted when an endpoint is set.
    - ``AUDIO_FUNCTION_ID`` — NVCF function ID; defaults to ``default_function_id``
      (the nv-ingest libmode Parakeet NIM) when an endpoint is set but the env
      var is unset.
    """
    import os

    grpc_endpoint = (os.environ.get(grpc_endpoint_var) or "").strip()
    if not grpc_endpoint and default_grpc_endpoint:
        grpc_endpoint = default_grpc_endpoint.strip()

    auth_token = (os.environ.get(auth_token_var) or "").strip() or None
    function_id = (os.environ.get(function_id_var) or "").strip() or None

    if not grpc_endpoint:
        # Local path: drop any cloud credentials that happen to be in the env so
        # _use_remote() returns False and the local Parakeet model is loaded.
        auth_token = None
        function_id = None
    elif function_id is None and default_function_id:
        function_id = default_function_id

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


@designer_component(
    name="ASR (Speech-to-Text)",
    category="Audio",
    compute="gpu",
    description="Performs automatic speech recognition on audio chunks",
    category_color="#ff6b6b",
)
class ASRCPUActor(AbstractOperator, CPUOperator):
    """
    Ray Data map_batches callable: chunk rows (path/bytes) -> rows with text (transcript).

    When audio_endpoints are set, uses Parakeet (Riva ASR) via gRPC. When both are
    null/empty, uses local HuggingFace/NeMo Parakeet (nvidia/parakeet-ctc-1.1b).
    Output rows have path, text, page_number, metadata for downstream embed. When
    ``params.segment_audio`` is enabled for remote Parakeet, punctuation-delimited
    segments are emitted as multiple rows per chunk.
    """

    def __init__(self, params: ASRParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params or ASRParams()
        if _use_remote(self._params):
            self._client = _get_client(self._params)
            self._model = None
        else:
            self._client = None
            from nemo_retriever.model.local import ParakeetCTC1B1ASR

            self._model = ParakeetCTC1B1ASR()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(
                columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
            )

        # When ``_content_type`` is set on the batch (mixed audio + video_frame
        # rows from a video pipeline), only ASR the audio rows and pass the
        # rest through unchanged. Audio-only pipelines have no ``_content_type``
        # column, so this branch is a no-op for them.
        audio_df, passthrough_df = _split_audio_rows(batch_df)
        if audio_df.empty:
            return passthrough_df

        if self._client is not None:
            asr_out = self._call_remote_batch(audio_df)
        else:
            asr_out = self._call_local_batch(audio_df)
        return _concat_with_passthrough(asr_out, passthrough_df)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    def _call_remote_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Remote ASR: one infer call per row (no batching on server side)."""
        out_rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            try:
                out_rows.extend(self._transcribe_one(row))
            except Exception as e:
                logger.exception("ASR failed for row path=%s: %s", row.get("path"), e)
                continue

        if not out_rows:
            return pd.DataFrame(
                columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
            )
        return pd.DataFrame(out_rows)

    def _call_local_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Local ASR: one batched transcribe call for the whole batch."""
        if self._model is None:
            return pd.DataFrame(
                columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
            )
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
            return pd.DataFrame(
                columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
            )
        return pd.DataFrame(out_rows)

    def _transcribe_remote(self, raw: bytes, path: Optional[str]) -> Optional[tuple[List[Dict[str, Any]], str]]:
        """Use remote Parakeet client to transcribe audio bytes and return segments + transcript."""
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

    def _transcribe_local(self, raw: bytes, path: Optional[str]) -> Optional[tuple[str, List[Dict[str, Any]]]]:
        """Use local Parakeet model to transcribe; path or temp file with raw bytes."""
        if self._model is None:
            return None
        path_to_use = path
        if not path_to_use or not Path(path_to_use).exists():
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
                f.write(raw)
                path_to_use = f.name
            try:
                results = self._model.transcribe_with_segments([path_to_use])
            finally:
                Path(path_to_use).unlink(missing_ok=True)
        else:
            results = self._model.transcribe_with_segments([path_to_use])
        if not results:
            return ("", [])
        text, segments = results[0]
        return (text, list(segments))

    def _build_output_rows(
        self,
        row: pd.Series,
        transcript: str,
        *,
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

        try:
            chunk_start = float(metadata.get("chunk_start_seconds") or 0.0)
        except (TypeError, ValueError):
            chunk_start = 0.0
        try:
            chunk_dur = float(duration) if duration is not None else 0.0
        except (TypeError, ValueError):
            chunk_dur = 0.0

        if self._params.segment_audio and segments:
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
                seg_s_secs = _to_chunk_relative_seconds(segment.get("start"), chunk_dur)
                seg_e_secs = _to_chunk_relative_seconds(segment.get("end"), chunk_dur)
                # Wall-clock span: chunk start + the chunk-relative times the ASR
                # backend produced. Local Parakeet emits seconds; remote emits
                # milliseconds — normalized above against the chunk duration.
                if seg_s_secs is not None:
                    segment_metadata["segment_start_seconds"] = seg_s_secs + chunk_start
                if seg_e_secs is not None:
                    segment_metadata["segment_end_seconds"] = seg_e_secs + chunk_start
                segment_metadata["_content_type"] = "audio"
                segment_metadata.setdefault("modality", "audio_segment")
                out_rows.append(
                    {
                        "path": path,
                        "source_path": source_path,
                        "duration": duration,
                        "chunk_index": chunk_index,
                        "metadata": segment_metadata,
                        "page_number": page_number,
                        "text": segment_text,
                        "_content_type": "audio",
                    }
                )
            if out_rows:
                return out_rows

        # Per-chunk fallback: anchor the row's span to the chunk's wall-clock
        # window so audio_segment recall still works without per-utterance data.
        metadata.setdefault("segment_start_seconds", chunk_start)
        metadata.setdefault("segment_end_seconds", chunk_start + chunk_dur)
        metadata["_content_type"] = "audio"
        metadata.setdefault("modality", "audio_segment")
        return [
            {
                "path": path,
                "source_path": source_path,
                "duration": duration,
                "chunk_index": chunk_index,
                "metadata": metadata,
                "page_number": page_number,
                "text": transcript,
                "_content_type": "audio",
            }
        ]

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

        if self._client is not None:
            remote_result = self._transcribe_remote(raw, path)
            if remote_result is None:
                return []
            segments, transcript = remote_result
            return self._build_output_rows(row, transcript, segments=segments)
        else:
            local_result = self._transcribe_local(raw, path)
            if local_result is None:
                return []
            transcript, segments = local_result
            return self._build_output_rows(row, transcript, segments=segments)


class ASRGPUActor(ASRCPUActor, GPUOperator):
    """Local Parakeet on GPU.

    Reuses :class:`ASRCPUActor`'s implementation; the only difference is the
    :class:`GPUOperator` mixin so the executor allocates a GPU when scheduling
    and the pipeline registry renders the node as ``[GPU]``. The :class:`ASRActor`
    archetype routes here when no remote ``audio_endpoints`` is configured.
    """

    pass


class ASRActor(ArchetypeOperator):
    """Graph-facing ASR archetype: GPU (local Parakeet) or CPU (remote gRPC)."""

    _cpu_variant_class = ASRCPUActor
    _gpu_variant_class = ASRGPUActor

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        """CPU variant when a remote endpoint is set — no local GPU needed."""
        params = (operator_kwargs or {}).get("params")
        return isinstance(params, ASRParams) and _use_remote(params)

    def __init__(self, params: ASRParams | None = None) -> None:
        resolved_params = params or ASRParams()
        super().__init__(params=resolved_params)
        self._params = resolved_params


def apply_asr_to_df(
    batch_df: pd.DataFrame,
    asr_params: Optional[dict] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Inprocess helper: apply ASR to a DataFrame of chunk rows; returns DataFrame with text column set.

    Used by InProcessIngestor when _pipeline_type == "audio". asr_params can be a dict
    to construct ASRParams (e.g. from model_dump()).
    """
    params = ASRParams(**(asr_params or {}))
    actor = ASRActor(params=params)
    return actor(batch_df)
