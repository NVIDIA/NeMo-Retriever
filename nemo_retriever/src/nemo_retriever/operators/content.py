# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared content-transform operators."""

from __future__ import annotations

from typing import Any

from nemo_retriever.operators.base import AbstractOperator, CPUOperator
from nemo_retriever.utils.pipeline.content import explode_content_to_rows


class ExplodeContentActor(AbstractOperator, CPUOperator):
    """Expand page-level rows into per-element rows for finer-grained embedding."""

    def __init__(
        self,
        *,
        modality: str = "text",
        text_elements_modality: str | None = None,
        structured_elements_modality: str | None = None,
    ) -> None:
        super().__init__()
        self._modality = modality
        self._text_elements_modality = text_elements_modality
        self._structured_elements_modality = structured_elements_modality

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return explode_content_to_rows(
            data,
            modality=self._modality,
            text_elements_modality=self._text_elements_modality,
            structured_elements_modality=self._structured_elements_modality,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
