# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Central registry of pinned HuggingFace model revisions.

Every ``from_pretrained`` call in the codebase should pass
``revision=get_hf_revision(model_id)`` so that we always download an
exact, immutable snapshot rather than tracking the mutable ``main``
branch.

To bump a model version, update the corresponding entry in
``HF_MODEL_REGISTRY`` and re-test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HFModelInfo:
    """Metadata for a pinned HuggingFace model."""

    revision: str
    created_at: int
    url: str


HF_MODEL_REGISTRY: dict[str, HFModelInfo] = {
    "nvidia/llama-3.2-nv-embedqa-1b-v2": HFModelInfo(
        revision="cefc2394cc541737b7867df197984cf23f05367f",
        created_at=1729122273,
        url="https://huggingface.co/nvidia/llama-3.2-nv-embedqa-1b-v2",
    ),
    "nvidia/llama-nemotron-embed-1b-v2": HFModelInfo(
        revision="cefc2394cc541737b7867df197984cf23f05367f",
        created_at=1729122273,
        url="https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2",
    ),
    "nvidia/parakeet-ctc-1.1b": HFModelInfo(
        revision="a707e818195cb97c8f7da2fc36b221a29f69a5db",
        created_at=1703778477,
        url="https://huggingface.co/nvidia/parakeet-ctc-1.1b",
    ),
    "nvidia/NVIDIA-Nemotron-Parse-v1.2": HFModelInfo(
        revision="f42c8040b12ee64370922d108778ab655b722c5d",
        created_at=1771533817,
        url="https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2",
    ),
    "nvidia/llama-nemotron-embed-vl-1b-v2": HFModelInfo(
        revision="859e1f2dac29c56c37a5279cf55f53f3e74efc6b",
        created_at=1764702468,
        url="https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2",
    ),
    "meta-llama/Llama-3.2-1B": HFModelInfo(
        revision="4e20de362430cd3b72f300e6b0f18e50e7166e08",
        created_at=1726671794,
        url="https://huggingface.co/meta-llama/Llama-3.2-1B",
    ),
    "intfloat/e5-large-unsupervised": HFModelInfo(
        revision="15af9288f69a6291f37bfb89b47e71abc747b206",
        created_at=1675130616,
        url="https://huggingface.co/intfloat/e5-large-unsupervised",
    ),
    "nvidia/llama-nemotron-rerank-1b-v2": HFModelInfo(
        revision="aee9a1be0bbd89489f8bd0ec5763614c8bb85878",
        created_at=1729905762,
        url="https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2",
    ),
}

# Backward-compatible alias used by callers that import the old dict name.
HF_MODEL_REVISIONS: dict[str, str] = {
    model_id: info.revision for model_id, info in HF_MODEL_REGISTRY.items()
}


def get_hf_revision(model_id: str, *, strict: bool = True) -> str:
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
    info = HF_MODEL_REGISTRY.get(model_id)
    if info is not None:
        return info.revision

    msg = (
        f"No pinned HuggingFace revision for model '{model_id}'. "
        "Add an entry to HF_MODEL_REGISTRY in hf_model_registry.py to pin it."
    )
    if strict:
        raise ValueError(msg)
    logger.warning(msg + " Falling back to the default (main) branch.")
    return None  # type: ignore[return-value]
