# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backwards-compatible re-export.

Evidence shaping moved to :mod:`nemo_retriever.query.evidence` so the service
``/v1/query`` endpoint can reuse it without the service depending on ``cli``.
Import from there directly in new code.
"""

from __future__ import annotations

from nemo_retriever.query.evidence import (
    _evidence_item,
    _normalize_modality,
    build_evidence_result,
)

__all__ = ["build_evidence_result", "_evidence_item", "_normalize_modality"]
