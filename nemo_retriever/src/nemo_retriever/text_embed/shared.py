# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for graph text-embedding operators."""

from __future__ import annotations

from nemo_retriever.params import EmbedParams


def _to_bool(v: object, default: bool = False) -> bool:
    if isinstance(v, str):
        return v.lower() not in ("false", "0", "no", "")
    if v is None:
        return default
    return bool(v)


def build_embed_kwargs(params: EmbedParams) -> dict[str, object]:
    kwargs = {
        **params.model_dump(mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True),
        **params.runtime.model_dump(mode="python", exclude_none=True),
    }
    if "embedding_endpoint" not in kwargs and kwargs.get("embed_invoke_url"):
        kwargs["embedding_endpoint"] = kwargs.get("embed_invoke_url")
    return kwargs
