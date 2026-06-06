# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operators for transforming extracted page content."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.content_transforms import explode_content_to_rows
from nemo_retriever.graph.designer import designer_component


@designer_component(
    name="Explode Content",
    category="Text & Content",
    compute="cpu",
    description="Explodes page-level content into element-level rows by modality",
    category_color="#42d6a4",
)
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
