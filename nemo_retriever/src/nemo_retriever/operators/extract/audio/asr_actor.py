# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ASRActor: Ray Data map_batches archetype for speech-to-text.

The archetype resolves to one of two hardware-shaped variants:

  - :class:`nemo_retriever.audio.cpu_actor.ASRCPUActor` — remote-only.
    Calls Parakeet/Riva via gRPC. Defaults to the public NVCF endpoint
    (``grpc.nvcf.nvidia.com:443``) when ``audio_endpoints`` is left empty.
    Imports no torch.
  - :class:`nemo_retriever.audio.gpu_actor.ASRGPUActor` — local-only.
    Loads ``nvidia/parakeet-ctc-1.1b`` via HuggingFace transformers.

Consumes chunk rows (path, bytes, source_path, duration, chunk_index, metadata)
and produces rows with text (transcript) for downstream embed/VDB. With
``segment_audio=True`` the remote (punctuation-bounded) and local (silence-gap,
from CTC frame timestamps) paths both fan out per-segment rows with start/end
times so ``recall_match_mode: audio_segment`` can match against time-aligned hits.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.common.params import ASRParams


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
    """True if at least one of audio_endpoints is set (use remote gRPC client).

    Retained for the archetype's ``prefers_cpu_variant`` check; the CPU variant
    constructor doesn't gate on this anymore (it auto-defaults to NVCF when
    both endpoints are empty).
    """
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
# These same constants are the default-fill source for ``ASRCPUActor`` so the
# CPU variant works out of the box without any ``audio_endpoints`` plumbing.
DEFAULT_NGC_ASR_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"
DEFAULT_NGC_ASR_FUNCTION_ID = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"


def asr_params_from_env(
    *,
    grpc_endpoint_var: str = "AUDIO_GRPC_ENDPOINT",
    auth_token_var: str = "NVIDIA_API_KEY",
    function_id_var: str = "AUDIO_FUNCTION_ID",
    default_grpc_endpoint: Optional[str] = None,
    default_function_id: Optional[str] = DEFAULT_NGC_ASR_FUNCTION_ID,
) -> ASRParams:
    """
    Build ASRParams from environment variables, with optional Python-level defaults.

    The CPU variant auto-defaults to NVCF when ``audio_endpoints`` is empty, so
    this helper is now mainly useful for callers who want to populate
    :class:`ASRParams` from env *without* instantiating an actor — e.g. when
    constructing a :class:`~nemo_retriever.graph_ingestor.GraphIngestor`.

    Two opt-in paths to a custom (non-NVCF) endpoint, both honoured:

    - **Environment variable**: ``AUDIO_GRPC_ENDPOINT=grpc.nvcf.nvidia.com:443``
      (NVCF) or ``AUDIO_GRPC_ENDPOINT=localhost:50051`` (local NIM).
    - **Python API**: pass ``default_grpc_endpoint=...`` to this function. The
      env var wins when both are present. Use the exported
      :data:`DEFAULT_NGC_ASR_GRPC_ENDPOINT` constant for NVCF.

    - ``NVIDIA_API_KEY`` — Bearer token; only consulted when an endpoint is set.
    - ``AUDIO_FUNCTION_ID`` — NVCF function ID; defaults to ``default_function_id``
      (the libmode Parakeet NIM) when an endpoint is set but the env var is unset.
    """
    import os

    grpc_endpoint = (os.environ.get(grpc_endpoint_var) or "").strip()
    if not grpc_endpoint and default_grpc_endpoint:
        grpc_endpoint = default_grpc_endpoint.strip()

    auth_token = (os.environ.get(auth_token_var) or "").strip() or None
    function_id = (os.environ.get(function_id_var) or "").strip() or None

    if not grpc_endpoint:
        # Caller did not opt into a custom endpoint — leave audio_endpoints empty
        # and let the actor's default-fill (or the GPU variant) decide. Drop any
        # cloud credentials so they don't leak into a non-NVCF destination.
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
    from nemo_retriever.models.nim.primitives.model_interface.parakeet import (
        create_audio_inference_client,
    )

    _PARAKEET_AVAILABLE = True
except ImportError:
    create_audio_inference_client = None  # type: ignore[misc, assignment]
    _PARAKEET_AVAILABLE = False


def _get_client(params: ASRParams):  # noqa: ANN201
    if not _PARAKEET_AVAILABLE or create_audio_inference_client is None:
        raise RuntimeError(
            "ASRCPUActor requires the Parakeet NIM client (vendored in nemo_retriever.api) "
            "and the nvidia-riva-client gRPC stubs."
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
        infer_mode=params.audio_infer_mode,
    )


class _ASRActorBase:
    """Shared state + presentation helpers for the ASR CPU / GPU variants.

    Carries ``self._params`` and the row-building logic that's identical on
    both sides (the only thing that differs between remote and local is the
    transcription call itself). Subclasses inherit from this **plus** the
    appropriate :class:`AbstractOperator` + :class:`CPUOperator` /
    :class:`GPUOperator` mixins (see ``cpu_actor.py`` / ``gpu_actor.py``).
    """

    _params: ASRParams

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    @staticmethod
    def _empty_output_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
        )

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


@designer_component(
    name="ASR (Speech-to-Text)",
    category="Audio",
    compute="gpu",
    description="Performs automatic speech recognition on audio chunks",
    category_color="#ff6b6b",
)
class ASRActor(ArchetypeOperator):
    """Graph-facing ASR archetype.

    Resolves to:
      - :class:`~nemo_retriever.audio.cpu_actor.ASRCPUActor` when the caller
        passed ``audio_endpoints`` (explicit remote NIM), or when the host has
        no GPU available (auto-defaults to the NVCF Parakeet endpoint).
      - :class:`~nemo_retriever.audio.gpu_actor.ASRGPUActor` otherwise — local
        ``nvidia/parakeet-ctc-1.1b`` via HuggingFace transformers.
    """

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        """CPU variant when a remote endpoint is explicitly set."""
        params = (operator_kwargs or {}).get("params")
        return isinstance(params, ASRParams) and _use_remote(params)

    @classmethod
    def cpu_variant_class(cls) -> type[AbstractOperator]:
        from nemo_retriever.operators.extract.audio.cpu_actor import ASRCPUActor

        return ASRCPUActor

    @classmethod
    def gpu_variant_class(cls) -> type[AbstractOperator]:
        from nemo_retriever.operators.extract.audio.gpu_actor import ASRGPUActor

        return ASRGPUActor

    def __init__(self, params: ASRParams | None = None) -> None:
        resolved_params = params or ASRParams()
        # ``AUDIO_GRPC_ENDPOINT`` lets operators force the remote (CPU) variant
        # from the environment when the caller didn't explicitly set endpoints
        # — mirrors the ``asr_params_from_env`` convention so a single env var
        # works whether you go through the helper or straight through the
        # archetype. Once populated, ``prefers_cpu_variant`` returns True and
        # the archetype resolves to ``ASRCPUActor`` regardless of GPU count.
        if not _use_remote(resolved_params):
            import os

            env_grpc = (os.environ.get("AUDIO_GRPC_ENDPOINT") or "").strip()
            if env_grpc:
                resolved_params = resolved_params.model_copy(
                    update={
                        "audio_endpoints": (env_grpc, resolved_params.audio_endpoints[1]),
                        "audio_infer_protocol": "grpc",
                    }
                )
        super().__init__(params=resolved_params)
        self._params = resolved_params


def apply_asr_to_df(
    batch_df: pd.DataFrame,
    asr_params: Optional[dict] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inprocess helper: apply ASR to a DataFrame of chunk rows; returns DataFrame with text column set.

    Used by InProcessIngestor when ``_pipeline_type == "audio"``. ``asr_params``
    can be a dict to construct :class:`ASRParams` (e.g. from ``model_dump()``).
    """
    params = ASRParams(**(asr_params or {}))
    actor = ASRActor(params=params)
    return actor(batch_df)


def __getattr__(name: str):
    """Lazy re-export so callers can still do
    ``from nemo_retriever.audio.asr_actor import ASRCPUActor`` after the split.

    Defined as PEP 562 module-level ``__getattr__`` to avoid the circular
    import that direct top-level imports would trigger (cpu_actor.py and
    gpu_actor.py both import symbols from this module).
    """
    if name == "ASRCPUActor":
        from nemo_retriever.operators.extract.audio.cpu_actor import ASRCPUActor

        return ASRCPUActor
    if name == "ASRGPUActor":
        from nemo_retriever.operators.extract.audio.gpu_actor import ASRGPUActor

        return ASRGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
