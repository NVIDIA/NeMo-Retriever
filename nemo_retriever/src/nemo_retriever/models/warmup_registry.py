# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local registry for Hugging Face models warmed at worker startup.

Each :class:`~concurrent.futures.ProcessPoolExecutor` child process owns its
own copy of the registry.  The service layer populates it via
:func:`warm_local_models` from a pool worker initializer before ingest jobs
run, so ``nvidia-smi`` shows VRAM allocation at startup rather than on the
first upload.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Any] = {}
_WARMED = False


def is_warmup_active() -> bool:
    """Return True after :func:`warm_local_models` has run in this process."""
    return _WARMED


def get_warmed_model(key: str) -> Any | None:
    """Return a pre-loaded model for *key*, or ``None`` when not warmed."""
    return _REGISTRY.get(key)


def clear_warmed_models() -> None:
    """Reset registry state (for tests)."""
    global _WARMED
    _REGISTRY.clear()
    _WARMED = False


def build_warmup_spec(
    extract_params_dict: dict[str, Any] | None,
    embed_params_dict: dict[str, Any] | None,
    asr_params_dict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Derive which local models a worker should load from pipeline params."""
    extract = extract_params_dict or {}
    stages: list[str] = []

    if not str(extract.get("page_elements_invoke_url") or "").strip():
        stages.append("page_elements")
    if not str(extract.get("ocr_invoke_url") or "").strip():
        stages.append("ocr")
    if extract.get("use_table_structure") and not str(extract.get("table_structure_invoke_url") or "").strip():
        stages.append("table_structure")

    embed_spec: dict[str, Any] | None = None
    if embed_params_dict:
        from nemo_retriever.common.params import EmbedParams
        from nemo_retriever.models.inference.shared import build_embed_kwargs

        kwargs = build_embed_kwargs(EmbedParams(**embed_params_dict))
        endpoint = str(kwargs.get("embedding_endpoint") or kwargs.get("embed_invoke_url") or "").strip()
        if not endpoint and (kwargs.get("model_name") or kwargs.get("embed_model_name")):
            embed_spec = {
                "model_name": kwargs.get("embed_model_name") or kwargs.get("model_name"),
                "backend": kwargs.get("local_ingest_embed_backend") or "hf",
                "device": kwargs.get("local_hf_device") or kwargs.get("device"),
                "hf_cache_dir": kwargs.get("hf_cache_dir"),
                "gpu_memory_utilization": float(kwargs.get("gpu_memory_utilization", 0.45)),
                "enforce_eager": bool(kwargs.get("enforce_eager", False)),
                "dimensions": kwargs.get("dimensions"),
                "normalize": bool(kwargs.get("normalize", True)),
                "max_length": int(kwargs.get("max_length", 8192)),
                "query_max_length": int(kwargs.get("query_max_length", 128)),
            }

    asr_local = False
    if asr_params_dict is not None:
        endpoints = asr_params_dict.get("audio_endpoints")
        if isinstance(endpoints, (list, tuple)):
            asr_local = not any(str(e or "").strip() for e in endpoints)
        else:
            asr_local = True

    if not stages and embed_spec is None and not asr_local:
        return None

    spec: dict[str, Any] = {
        "stages": stages,
        "ocr_version": extract.get("ocr_version", "v2"),
        "ocr_lang": extract.get("ocr_lang"),
    }
    if embed_spec is not None:
        spec["embed"] = embed_spec
    if asr_local:
        spec["asr"] = True
    return spec


def _preload_local_embedder(model: Any, backend: str) -> None:
    backend = (backend or "hf").strip().lower()
    if backend == "hf" and hasattr(model, "_ensure_loaded"):
        model._ensure_loaded()


def warm_local_models(spec: dict[str, Any]) -> None:
    """Load all models described by *spec* into this process's registry."""
    global _WARMED

    clear_warmed_models()
    stages = set(spec.get("stages") or [])

    if "page_elements" in stages:
        from nemo_retriever.models.local import NemotronPageElementsV3

        logger.info("Warming local model: page_elements")
        _REGISTRY["page_elements"] = NemotronPageElementsV3()

    ocr_needed = "ocr" in stages or "table_structure" in stages
    if ocr_needed and "ocr" not in _REGISTRY:
        from nemo_retriever.common.modality.ocr.config import resolve_ocr_v2_lang
        from nemo_retriever.models.local import NemotronOCRV2

        lang = resolve_ocr_v2_lang(
            str(spec.get("ocr_version", "v2")),
            spec.get("ocr_lang"),
        )
        logger.info("Warming local model: ocr (lang=%s)", lang)
        _REGISTRY["ocr"] = NemotronOCRV2(lang=lang)

    if "table_structure" in stages:
        from nemo_retriever.models.local import NemotronTableStructureV1

        logger.info("Warming local model: table_structure")
        _REGISTRY["table_structure"] = NemotronTableStructureV1()

    embed_spec = spec.get("embed")
    if embed_spec:
        from nemo_retriever.models import create_local_embedder

        backend = str(embed_spec.get("backend") or "hf").strip().lower()
        hf_cache = str(embed_spec["hf_cache_dir"]) if embed_spec.get("hf_cache_dir") else None
        hf_device = str(embed_spec["device"]) if embed_spec.get("device") else None
        logger.info("Warming local model: embed (backend=%s)", backend)
        embedder = create_local_embedder(
            embed_spec.get("model_name"),
            backend=backend,
            device=hf_device,
            hf_cache_dir=hf_cache,
            gpu_memory_utilization=float(embed_spec.get("gpu_memory_utilization", 0.45)),
            enforce_eager=bool(embed_spec.get("enforce_eager", False)),
            dimensions=embed_spec.get("dimensions"),
            normalize=bool(embed_spec.get("normalize", True)),
            max_length=int(embed_spec.get("max_length", 8192)),
            query_max_length=int(embed_spec.get("query_max_length", 128)),
        )
        _preload_local_embedder(embedder, backend)
        _REGISTRY["embed"] = embedder

    if spec.get("asr"):
        from nemo_retriever.models.local import ParakeetCTC1B1ASR

        logger.info("Warming local model: asr")
        _REGISTRY["asr"] = ParakeetCTC1B1ASR()

    _WARMED = True
    logger.info(
        "Local model warmup complete in pid=%s — keys=%s",
        __import__("os").getpid(),
        sorted(_REGISTRY.keys()),
    )
