# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import Any


def _has_endpoint(*values: Any) -> bool:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return True
        elif value:
            return True
    return False


@lru_cache(maxsize=1)
def has_local_gpu() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available() and int(torch.cuda.device_count()) > 0)
    except Exception:
        return False


def prefer_cpu_actor(*endpoint_values: Any) -> bool:
    return _has_endpoint(*endpoint_values) or not has_local_gpu()


def page_elements_actor_class(*, extract_params: Any):
    if prefer_cpu_actor(getattr(extract_params, "page_elements_invoke_url", None)):
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        return PageElementDetectionCPUActor
    from nemo_retriever.page_elements.gpu_actor import PageElementDetectionActor

    return PageElementDetectionActor


def ocr_actor_class(*, extract_params: Any):
    if prefer_cpu_actor(getattr(extract_params, "ocr_invoke_url", None)):
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        return OCRCPUActor
    from nemo_retriever.ocr.gpu_ocr import OCRActor

    return OCRActor


def table_structure_actor_class(*, extract_params: Any):
    if prefer_cpu_actor(
        getattr(extract_params, "table_structure_invoke_url", None),
        getattr(extract_params, "ocr_invoke_url", None),
    ):
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        return TableStructureCPUActor
    from nemo_retriever.table.gpu_actor import TableStructureActor

    return TableStructureActor


def graphic_elements_actor_class(*, extract_params: Any):
    if prefer_cpu_actor(
        getattr(extract_params, "graphic_elements_invoke_url", None),
        getattr(extract_params, "ocr_invoke_url", None),
    ):
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        return GraphicElementsCPUActor
    from nemo_retriever.chart.gpu_actor import GraphicElementsActor

    return GraphicElementsActor


def nemotron_parse_actor_class(*, invoke_url: str | None = None):
    if prefer_cpu_actor(invoke_url):
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        return NemotronParseCPUActor
    from nemo_retriever.ocr.gpu_parse import NemotronParseActor

    return NemotronParseActor


def embed_actor_class(*, embed_params: Any):
    if prefer_cpu_actor(getattr(embed_params, "embed_invoke_url", None)):
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        return _BatchEmbedCPUActor
    from nemo_retriever.text_embed.gpu_operator import _BatchEmbedActor

    return _BatchEmbedActor
