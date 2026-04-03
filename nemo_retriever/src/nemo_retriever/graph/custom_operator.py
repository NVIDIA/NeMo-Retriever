# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Optional

from nemo_retriever.graph.abstract_operator import AbstractOperator


class UDFOperator(AbstractOperator):
    """A small operator wrapper for user-defined Python functions."""

    def __init__(self, fn: Callable[[Any], Any], name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn
        self.name = name or type(self).__name__

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return self.fn(data)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
