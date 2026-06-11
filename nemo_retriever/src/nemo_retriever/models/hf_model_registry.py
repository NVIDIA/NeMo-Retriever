# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Central registry of pinned HuggingFace model revisions.

Every ``from_pretrained`` call in the codebase should pass
``revision=get_hf_revision(model_id)`` and direct ``hf_hub_download`` calls
should use ``hf_hub_download_with_pinned_revision`` so that we always
download an exact, immutable snapshot rather than tracking the mutable
``main`` branch.

To bump a model version, update the corresponding SHA in
``HF_MODEL_REVISIONS`` and re-test.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

HF_MODEL_REVISIONS: dict[str, str] = {
    "nvidia/llama-3.2-nv-embedqa-1b-v2": "cefc2394cc541737b7867df197984cf23f05367f",
    "nvidia/llama-nemotron-embed-1b-v2": "b4caa8456edd360b3b4e938d94ed4398dd437fad",
    "nvidia/parakeet-ctc-1.1b": "a707e818195cb97c8f7da2fc36b221a29f69a5db",
    "nvidia/NVIDIA-Nemotron-Parse-v1.2": "f42c8040b12ee64370922d108778ab655b722c5d",
    "nvidia/llama-nemotron-embed-vl-1b-v2": "4ef1bfa6da3a909de6bd00611950b7ed99203117",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "intfloat/e5-large-unsupervised": "15af9288f69a6291f37bfb89b47e71abc747b206",
    "nvidia/llama-nemotron-rerank-1b-v2": "8fd3e5d962d44cfe65d4ba0784eebed44cf136b0",
    "nvidia/llama-nemotron-rerank-vl-1b-v2": "edc083f4b3a433d65287cbca916759c9f88fa887",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16": "5d250e2e111dc5e1434131bdf3d590c27a878ade",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8": "7394488badb786e1decc0e00e308de1cab9560e6",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD": "b8d3c170d9ee3a078917ef9bfd508eff988d6de7",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": "24e67ea000b7c2837fc8f9488aa2008524fac8ba",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8": "6647b845a4b786c6e2c7adb1b6a909e1aa71fac2",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4": "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe",
    "nvidia/nemotron-ocr-v1": "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb",
    "nvidia/nemotron-ocr-v2": "86cacb0467fa4f7ce54342fdb250825e0d928ae7",
    "nvidia/nemotron-page-elements-v3": "df62dbb631502575ac4d43b44d700b1674ab1d56",
    "nvidia/nemotron-table-structure-v1": "9350162faa1110320af62699105780b0c87b73ad",
    "nvidia/nemotron-graphic-elements-v1": "4a76546bb1bb4cbab3401361c91cf01706321805",
}


def get_hf_revision(model_id: str, *, strict: bool = True) -> str | None:
    """Return the pinned commit SHA for *model_id*.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier (e.g. ``"nvidia/parakeet-ctc-1.1b"``).
    strict:
        When ``True`` (the default), raise ``ValueError`` if *model_id* has
        no pinned revision.  When ``False``, log a warning and return
        ``None`` so that ``from_pretrained`` falls back to the ``main``
        branch.
    """
    revision = HF_MODEL_REVISIONS.get(model_id)
    if revision is not None:
        return revision

    msg = (
        f"No pinned HuggingFace revision for model '{model_id}'. "
        "Add an entry to HF_MODEL_REVISIONS in hf_model_registry.py to pin it."
    )
    if strict:
        raise ValueError(msg)
    logger.warning(msg + " Falling back to the default (main) branch.")
    return None


def hf_hub_download(*args: Any, **kwargs: Any) -> str:
    """Proxy to Hugging Face's downloader, imported lazily."""
    from huggingface_hub import hf_hub_download as _hf_hub_download

    return _hf_hub_download(*args, **kwargs)


def _is_huggingface_hub_error(exc: Exception) -> bool:
    return exc.__class__.__module__.startswith("huggingface_hub.")


def _hf_download_error_context(repo_id: Any, revision: Any, filename: Any) -> str:
    token_state = "set" if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") else "unset"
    hf_hub_offline = os.environ.get("HF_HUB_OFFLINE", "0")
    return (
        "Failed to resolve Hugging Face asset "
        f"repo_id={repo_id!r}, revision={revision!r}, filename={filename!r}. "
        f"HF_HUB_OFFLINE={hf_hub_offline}; HF token is {token_state}. "
        "For offline startup, pre-cache this exact revision and set HF_HUB_OFFLINE=1. "
        "For online startup, set HF_TOKEN/HUGGING_FACE_HUB_TOKEN if the repo requires auth or the cluster is hitting "
        "anonymous Hub rate limits."
    )


def hf_hub_download_with_pinned_revision(*args: Any, **kwargs: Any) -> str:
    """Call ``hf_hub_download`` with a registry revision when one is known.

    Parameters
    ----------
    *args:
        Positional arguments forwarded to ``huggingface_hub.hf_hub_download``.
        When present, the first positional argument is treated as ``repo_id``.
    **kwargs:
        Keyword arguments forwarded to ``huggingface_hub.hf_hub_download``.
        If ``repo_id`` has a registered pin and ``revision`` is omitted, this
        helper adds the pinned revision before downloading.

    Returns
    -------
    str
        The local path returned by ``huggingface_hub.hf_hub_download``.

    Raises
    ------
    RuntimeError
        If Hugging Face Hub raises while resolving the asset; the original
        exception is chained with startup-focused context.
    """
    repo_id = kwargs.get("repo_id")
    if repo_id is None and args:
        repo_id = args[0]

    if repo_id is not None and kwargs.get("revision") is None:
        revision = get_hf_revision(str(repo_id), strict=False)
        if revision is not None:
            kwargs["revision"] = revision

    try:
        return hf_hub_download(*args, **kwargs)
    except Exception as exc:
        if _is_huggingface_hub_error(exc):
            filename = kwargs.get("filename")
            if filename is None and len(args) > 1:
                filename = args[1]
            raise RuntimeError(_hf_download_error_context(repo_id, kwargs.get("revision"), filename)) from exc
        raise


def install_pinned_hf_hub_download(module: Any) -> None:
    """Patch an upstream module-level ``hf_hub_download`` to use registry pins.

    Parameters
    ----------
    module:
        Imported upstream module object expected to expose a top-level
        ``hf_hub_download`` function.  If the attribute is absent, the helper
        logs a warning and leaves the module unchanged.

    Returns
    -------
    None
        The module is mutated in place when patching succeeds.
    """
    if hasattr(module, "hf_hub_download"):
        # TODO: Move this pinning into the extraction package nightly build/publish
        # once those packages expose revision-aware downloads natively.
        module.hf_hub_download = hf_hub_download_with_pinned_revision
    else:
        logger.warning(
            "install_pinned_hf_hub_download: module %r has no 'hf_hub_download' attribute; "
            "revision pinning was NOT applied.",
            getattr(module, "__name__", module),
        )
