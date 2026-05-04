# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coalesce legacy per-field chunk params into :class:`TextChunkParamsByType`."""

from __future__ import annotations

from typing import Any

from nemo_retriever.params.models import HtmlChunkParams
from nemo_retriever.params.models import TextChunkParams
from nemo_retriever.params.models import TextChunkParamsByType


def _as_text_params(obj: Any) -> TextChunkParams | None:
    if obj is None:
        return None
    if isinstance(obj, TextChunkParamsByType):
        return None
    return TextChunkParams.model_validate(obj.model_dump())


def coalesce_text_chunk_bundle(
    *,
    text_chunk_by_type: TextChunkParamsByType | None = None,
    text_params: Any | None = None,
    html_params: Any | None = None,
    split_params: Any | None = None,
) -> TextChunkParamsByType | None:
    """Merge explicit bundle + legacy ``text`` / ``html`` / ``split`` params."""

    if text_chunk_by_type is not None:
        return text_chunk_by_type
    if isinstance(split_params, TextChunkParamsByType):
        return split_params

    split_tp = _as_text_params(split_params)
    text_tp = _as_text_params(text_params)
    html_tp = _as_text_params(html_params)

    if split_tp is None and text_tp is None and html_tp is None:
        return None

    if split_tp is not None:
        base = split_tp
    elif text_tp is not None:
        base = text_tp
    else:
        base = TextChunkParams.model_validate(html_tp.model_dump())  # type: ignore[union-attr]

    txt: TextChunkParams | None = None
    html: HtmlChunkParams | None = None
    pdf: TextChunkParams | None = None
    image: TextChunkParams | None = None

    if text_tp is not None and text_tp.model_dump() != base.model_dump():
        txt = text_tp
    if html_tp is not None:
        hp = HtmlChunkParams.model_validate(html_tp.model_dump())
        if hp.model_dump() != base.model_dump():
            html = hp

    if split_tp is not None and split_params is not None and not isinstance(split_params, TextChunkParamsByType):
        if split_tp.model_dump() != base.model_dump():
            pdf = split_tp
            image = split_tp

    return TextChunkParamsByType(base=base, txt=txt, html=html, pdf=pdf, image=image)
