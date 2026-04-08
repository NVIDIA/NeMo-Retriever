# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator for persisting extracted content to storage."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator


class StoreOperator(AbstractOperator, CPUOperator):
    """Persist extracted content (images, text) to local or cloud storage.

    This is a side-effecting I/O operator.  It writes files, annotates the
    DataFrame with storage URIs, and optionally strips base64 payloads to
    reduce downstream memory pressure.

    Place *after* OCR/caption (so all content types are populated) and
    *before* embed/content-reshape (so stripped payloads aren't needed
    inline — embed reloads from URIs).  See ``.claude/lessons.md`` for the
    memory reasoning behind this placement.
    """

    def __init__(self, *, params: Any = None) -> None:
        super().__init__()
        self._params = params

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        from nemo_retriever.io.image_store import store_extracted

        store_kwargs = self._params.model_dump(mode="python") if self._params else {}
        return store_extracted(data, **store_kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
