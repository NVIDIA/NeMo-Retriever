# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvalOperator -- bridge between graph.AbstractOperator and evaluation operators."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator as GraphAbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator


class EvalOperator(GraphAbstractOperator, CPUOperator):
    """Base class for evaluation operators within the graph framework.

    Bridges ``graph.AbstractOperator`` (which provides ``>>``, ``run()``,
    ``get_constructor_kwargs()``, and executor compatibility) with the
    evaluation domain's DataFrame-in/out contract.

    Subclasses declare ``required_columns`` and ``output_columns`` as
    class variables and implement ``process(data, **kwargs)``.

    Lifecycle (inherited from ``graph.AbstractOperator.run``):
        ``preprocess``  -- validates required columns if *data* is a DataFrame
        ``process``     -- abstract; enrich the DataFrame
        ``postprocess`` -- identity pass-through (override if needed)

    Constructor kwargs are stored by ``GraphAbstractOperator.__init__`` for
    ``get_constructor_kwargs()`` so that executors (including Ray) can
    reconstruct the operator on workers.
    """

    required_columns: ClassVar[tuple[str, ...]] = ()
    output_columns: ClassVar[tuple[str, ...]] = ()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, pd.DataFrame):
            missing = [c for c in self.required_columns if c not in data.columns]
            if missing:
                raise ValueError(f"{type(self).__name__} requires missing columns: {missing}")
        return data

    @abstractmethod
    def process(self, data: Any, **kwargs: Any) -> Any: ...

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
