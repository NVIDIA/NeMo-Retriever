# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named inference presets for hosted NIM endpoints.

Each preset maps a short name to default ``extract`` and ``embed`` kwargs that
are injected into the ingestor pipeline.  Users can still override individual
fields by passing their own keyword arguments to ``.extract()`` or ``.embed()``.

Currently supported presets
----------------------------
``"build.nvidia.com"``
    Uses the publicly hosted NIMs on `build.nvidia.com
    <https://build.nvidia.com>`_ / ``ai.api.nvidia.com``.  Requires
    ``NVIDIA_API_KEY`` to be set in the environment (or passed explicitly as
    ``api_key``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# build.nvidia.com preset
# ---------------------------------------------------------------------------

_BUILD_NVIDIA_EXTRACT_DEFAULTS: Dict[str, Any] = {
    "page_elements_invoke_url": "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3",
    "graphic_elements_invoke_url": "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1",
    "ocr_invoke_url": "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1",
    "table_structure_invoke_url": "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1",
}

_BUILD_NVIDIA_EMBED_DEFAULTS: Dict[str, Any] = {
    "embed_invoke_url": "https://integrate.api.nvidia.com/v1/embeddings",
    "model_name": "nvidia/llama-nemotron-embed-1b-v2",
    "embed_modality": "text",
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    "build.nvidia.com": (
        _BUILD_NVIDIA_EXTRACT_DEFAULTS,
        _BUILD_NVIDIA_EMBED_DEFAULTS,
    ),
}


def resolve_inference_preset(
    inference: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(extract_defaults, embed_defaults)`` for the given preset name.

    Parameters
    ----------
    inference:
        Preset name (e.g. ``"build.nvidia.com"``), or ``None`` / empty string
        to opt out of any preset (returns empty dicts).

    Raises
    ------
    ValueError
        If *inference* is not ``None`` and is not a recognised preset name.
    """
    if not inference:
        return {}, {}
    key = inference.strip().lower()
    if key not in _PRESETS:
        known = ", ".join(sorted(_PRESETS))
        raise ValueError(
            f"Unknown inference preset {inference!r}.  "
            f"Supported values: {known}"
        )
    extract_defaults, embed_defaults = _PRESETS[key]
    return dict(extract_defaults), dict(embed_defaults)
