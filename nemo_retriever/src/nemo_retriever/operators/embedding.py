# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports for embedding operators."""

from __future__ import annotations

from nemo_retriever.text_embed.operators import _BatchEmbedActor, _BatchEmbedCPUActor, _BatchEmbedGPUActor

__all__ = ["_BatchEmbedActor", "_BatchEmbedCPUActor", "_BatchEmbedGPUActor"]
