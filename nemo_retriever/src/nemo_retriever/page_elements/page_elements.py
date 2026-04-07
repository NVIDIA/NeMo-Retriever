# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.page_elements.shared import detect_page_elements_v3

__all__ = [
    "detect_page_elements_v3",
]


def __getattr__(name: str):
    if name == "PageElementDetectionActor":
        from nemo_retriever.page_elements.gpu_actor import PageElementDetectionActor

        return PageElementDetectionActor
    if name == "PageElementDetectionCPUActor":
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        return PageElementDetectionCPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
