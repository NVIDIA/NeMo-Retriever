# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline executors that run a :class:`Graph` against input data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd

from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.utils.ray_resource_hueristics import (
    gather_cluster_resources,
    NEMOTRON_PARSE_BATCH_SIZE,
    NEMOTRON_PARSE_GPUS_PER_ACTOR,
    OCR_GPUS_PER_ACTOR,
)

import logging

logger = logging.getLogger(__name__)

# Heuristic GPU fraction for GPUOperator nodes that load a local model.
# Reuses the same baseline constant as the batch ingest mode.
_DEFAULT_GPU_OPERATOR_NUM_GPUS = OCR_GPUS_PER_ACTOR


class AbstractExecutor(ABC):
    """Base class for pipeline executors.

    An executor takes a :class:`Graph` at init time and provides an
    :meth:`ingest` method that feeds data through the graph.
    """

    def __init__(self, graph: Graph) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"graph must be a Graph, got {type(graph).__name__}")
        self.graph = graph

    @abstractmethod
    def ingest(self, data: Any, **kwargs: Any) -> Any:
        """Execute the graph against *data* and return results."""
        ...


class InprocessExecutor(AbstractExecutor):
    """Executor that runs a :class:`Graph` in-process on pandas DataFrames.

    No Ray dependency — each node's operator is constructed once from
    ``operator_class(**operator_kwargs)`` and called sequentially on the
    accumulated DataFrame.

    Only linear (single-root, no fan-out) graphs are currently supported.
    """

    def __init__(self, graph: Graph, *, show_progress: bool = True) -> None:
        super().__init__(graph)
        self._show_progress = show_progress

    @staticmethod
    def _linearize(graph: Graph) -> List[Node]:
        """Walk a single-root, single-child-per-node graph and return an ordered list."""
        if not graph.roots:
            return []
        if len(graph.roots) > 1:
            raise ValueError("InprocessExecutor currently supports single-root graphs only.")
        ordered: List[Node] = []
        node = graph.roots[0]
        while node is not None:
            ordered.append(node)
            if len(node.children) > 1:
                raise ValueError(
                    f"InprocessExecutor does not support fan-out. "
                    f"Node {node.name!r} has {len(node.children)} children."
                )
            node = node.children[0] if node.children else None
        return ordered

    def ingest(self, data: Any, **kwargs: Any) -> Any:
        """Run the graph in-process on pandas DataFrames.

        Parameters
        ----------
        data
            A ``pandas.DataFrame``, a file path (str), or a list of file
            paths.  When paths are provided, each file is read as raw bytes
            and combined into a single DataFrame with ``bytes`` and ``path``
            columns before being passed through the graph.

        Returns
        -------
        pandas.DataFrame
            The result after all operators have been applied.
        """
        import glob as _glob

        import pandas as pd

        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, str):
            df = self._load_files([data])
        elif isinstance(data, list):
            # Expand globs
            expanded: List[str] = []
            for pattern in data:
                matches = _glob.glob(pattern)
                expanded.extend(sorted(matches) if matches else [pattern])
            df = self._load_files(expanded)
        else:
            raise TypeError(
                f"data must be a pandas.DataFrame, file path, or list of paths, " f"got {type(data).__name__}"
            )

        nodes = self._linearize(self.graph)
        operators = []
        for node in nodes:
            op = node.operator_class(**node.operator_kwargs)
            operators.append((node.name, op))

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if self._show_progress and tqdm is not None:
            pbar = tqdm(operators, desc="Pipeline stages", unit="stage")
            for name, op in pbar:
                pbar.set_postfix_str(name)
                df = op.run(df)
        else:
            for _name, op in operators:
                df = op.run(df)

        return df

    @staticmethod
    def _load_files(paths: List[str]) -> "pd.DataFrame":
        """Read files as raw bytes into a DataFrame with ``bytes`` and ``path`` columns."""
        import pandas as pd
        from pathlib import Path

        rows = []
        for p in paths:
            fp = Path(p)
            if fp.is_file():
                rows.append({"bytes": fp.read_bytes(), "path": str(fp.resolve())})
        if not rows:
            return pd.DataFrame(columns=["bytes", "path"])
        return pd.DataFrame(rows)


class RayDataExecutor(AbstractExecutor):
    """Executor that builds a Ray Data pipeline from a :class:`Graph`.

    For each :class:`Node` in the graph the executor appends a
    ``map_batches`` stage that uses the node's ``operator_class`` with
    ``fn_constructor_kwargs`` for deferred construction on Ray workers.
    This ensures heavy GPU models are loaded on workers, not serialised
    from the driver.

    The operator's ``__call__`` (defined on :class:`AbstractOperator`)
    delegates to ``run()``, so each ``map_batches`` stage executes the
    full preprocess → process → postprocess pipeline.

    Only linear (single-root, no fan-out) graphs are currently supported.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        ray_address: Optional[str] = None,
        batch_size: int = 1,
        batch_format: str = "pandas",
        num_cpus: float = 1,
        num_gpus: float = 0,
        node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(graph)
        self._ray_address = ray_address
        self._default_batch_size = batch_size
        self._default_batch_format = batch_format
        self._default_num_cpus = num_cpus
        self._default_num_gpus = num_gpus
        self._node_overrides = node_overrides or {}

    @staticmethod
    def _linearize(graph: Graph) -> List[Node]:
        """Walk a single-root, single-child-per-node graph and return an ordered list."""
        if not graph.roots:
            return []
        if len(graph.roots) > 1:
            raise ValueError("RayDataExecutor currently supports single-root graphs only.")
        ordered: List[Node] = []
        node = graph.roots[0]
        while node is not None:
            ordered.append(node)
            if len(node.children) > 1:
                raise ValueError(
                    f"RayDataExecutor does not support fan-out. "
                    f"Node {node.name!r} has {len(node.children)} children."
                )
            node = node.children[0] if node.children else None
        return ordered

    def ingest(self, data: Any, **kwargs: Any) -> Any:
        """Build and execute a Ray Data pipeline from the graph.

        Parameters
        ----------
        data
            Input to ``ray.data.read_binary_files`` (a path or list of glob patterns)
            **or** an already-constructed ``ray.data.Dataset``.

        Returns
        -------
        ray.data.Dataset
            The materialized result dataset.
        """
        import glob as _glob

        import ray
        import ray.data as rd

        if self._ray_address or not ray.is_initialized():
            ray.init(address=self._ray_address, ignore_reinit_error=True)

        cluster = gather_cluster_resources(ray)
        available_gpus = cluster.available_gpu_count()

        if isinstance(data, rd.Dataset):
            ds = data
        elif isinstance(data, (str, list)):
            paths = [data] if isinstance(data, str) else list(data)
            expanded: List[str] = []
            for pattern in paths:
                matches = _glob.glob(pattern)
                expanded.extend(sorted(matches) if matches else [pattern])
            ds = rd.read_binary_files(expanded, include_paths=True)
        else:
            raise TypeError(
                f"data must be a path/glob string, list of globs, or ray.data.Dataset, " f"got {type(data).__name__}"
            )

        nodes = self._linearize(self.graph)
        for node in nodes:
            overrides = dict(self._node_overrides.get(node.name, {}))
            target_num_rows_per_block = overrides.pop("target_num_rows_per_block", None)
            batch_size = overrides.pop("batch_size", self._default_batch_size)
            batch_format = overrides.pop("batch_format", self._default_batch_format)
            num_cpus = overrides.pop("num_cpus", self._default_num_cpus)

            # NemotronParseActor uses vLLM which handles its own batching
            # efficiently, so feed it more rows per map_batches call.
            from nemo_retriever.parse.nemotron_parse import NemotronParseActor

            if batch_size == self._default_batch_size and issubclass(node.operator_class, NemotronParseActor):
                batch_size = NEMOTRON_PARSE_BATCH_SIZE

            # When no explicit num_gpus override is given, auto-detect from the
            # GPUOperator mixin using actual cluster GPU availability.
            if "num_gpus" in overrides:
                num_gpus = overrides.pop("num_gpus")
            elif issubclass(node.operator_class, GPUOperator):
                has_remote_endpoint = any("invoke_url" in k and bool(v) for k, v in node.operator_kwargs.items())
                if has_remote_endpoint:
                    # Remote endpoint handles the model — no local GPU needed.
                    num_gpus = self._default_num_gpus
                elif available_gpus > 0:
                    # Local model, GPUs present: assign the heuristic fraction so
                    # Ray can co-schedule multiple actors per GPU.
                    # Exception: NemotronParseActor uses vLLM which manages
                    # its own KV-cache and requires exclusive GPU access.
                    from nemo_retriever.parse.nemotron_parse import NemotronParseActor

                    if issubclass(node.operator_class, NemotronParseActor):
                        num_gpus = max(self._default_num_gpus, NEMOTRON_PARSE_GPUS_PER_ACTOR)
                    else:
                        num_gpus = max(self._default_num_gpus, _DEFAULT_GPU_OPERATOR_NUM_GPUS)
                else:
                    # No GPUs in the cluster — operator will likely fail to load
                    # its CUDA model.  Warn loudly rather than silently requesting
                    # a fraction that would stall the pipeline indefinitely.
                    logger.warning(
                        "Node %r is a GPUOperator with no remote endpoint but "
                        "the Ray cluster reports 0 available GPUs. "
                        "The actor will be scheduled with num_gpus=0 and will "
                        "likely fail to load its model. Pass --ocr-invoke-url / "
                        "--page-elements-invoke-url / --embed-invoke-url to use "
                        "remote endpoints, or ensure GPUs are visible to Ray.",
                        node.name,
                    )
                    num_gpus = self._default_num_gpus
            else:
                num_gpus = self._default_num_gpus

            if target_num_rows_per_block is not None and int(target_num_rows_per_block) > 0:
                ds = ds.repartition(target_num_rows_per_block=int(target_num_rows_per_block))

            # Pass the operator class directly to map_batches with
            # fn_constructor_kwargs for deferred construction on workers.
            # AbstractOperator.__call__ delegates to run(), so each stage
            # executes the full preprocess -> process -> postprocess chain.
            ds = ds.map_batches(
                node.operator_class,
                batch_size=batch_size,
                batch_format=batch_format,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                fn_constructor_kwargs=node.operator_kwargs,
                **overrides,
            )

        return ds.materialize()
