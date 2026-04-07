# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for graph text-embedding operators."""

from __future__ import annotations

__all__ = []


def __getattr__(name: str):
    if name == "_BatchEmbedActor":
        from nemo_retriever.text_embed.gpu_operator import _BatchEmbedActor

        return _BatchEmbedActor
    if name == "_BatchEmbedCPUActor":
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        return _BatchEmbedCPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
