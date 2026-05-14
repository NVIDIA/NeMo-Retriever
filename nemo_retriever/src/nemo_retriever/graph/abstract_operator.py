# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import os
import time
from typing import Any, TYPE_CHECKING

from nemo_retriever.utils import stage_timing

if TYPE_CHECKING:
    from nemo_retriever.graph.pipeline_graph import Graph, Node


def _safe_len(data: Any) -> int:
    try:
        return len(data)
    except Exception:
        return -1


class AbstractOperator(ABC):
    """Base class for all pipeline operators."""

    def __init__(self, **kwargs: Any) -> None:
        self._graph_init_kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def preprocess(self, data: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def process(self, data: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def postprocess(self, data: Any, **kwargs: Any) -> Any: ...

    def run(self, data: Any, **kwargs: Any) -> Any:
        if not stage_timing.is_enabled():
            data = self.preprocess(data, **kwargs)
            data = self.process(data, **kwargs)
            data = self.postprocess(data, **kwargs)
            return data

        stage = getattr(self, "_nr_stage_name", None) or type(self).__name__
        n_in = _safe_len(data)
        t0 = time.perf_counter()
        data = self.preprocess(data, **kwargs)
        t1 = time.perf_counter()
        data = self.process(data, **kwargs)
        t2 = time.perf_counter()
        data = self.postprocess(data, **kwargs)
        t3 = time.perf_counter()
        stage_timing.record_timing(
            stage=stage,
            n_rows_in=n_in,
            n_rows_out=_safe_len(data),
            preprocess_ms=(t1 - t0) * 1000.0,
            process_ms=(t2 - t1) * 1000.0,
            postprocess_ms=(t3 - t2) * 1000.0,
            total_ms=(t3 - t0) * 1000.0,
            worker_pid=os.getpid(),
            wallclock_start=t0,
        )
        return data

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Make operators directly usable as Ray ``map_batches`` callables."""
        return self.run(data, **kwargs)

    def get_constructor_kwargs(self) -> dict[str, Any]:
        """Best-effort constructor kwargs for executor-side reconstruction."""
        kwargs = dict(getattr(self, "_graph_init_kwargs", {}))
        signature = inspect.signature(type(self).__init__)
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in kwargs:
                continue
            if hasattr(self, name):
                kwargs[name] = getattr(self, name)
                continue
            private_name = f"_{name}"
            if hasattr(self, private_name):
                kwargs[name] = getattr(self, private_name)
        return kwargs

    def __rshift__(self, other: "AbstractOperator | Node") -> "Graph":
        """``operator_a >> operator_b`` — auto-wrap both in Nodes and chain them.

        Returns a :class:`Graph` so the pipeline is immediately usable::

            graph = op_a >> op_b >> op_c
        """
        from nemo_retriever.graph.pipeline_graph import Node

        left = Node(self)
        # Delegate to Node.__rshift__ which returns a Graph
        return left >> other
