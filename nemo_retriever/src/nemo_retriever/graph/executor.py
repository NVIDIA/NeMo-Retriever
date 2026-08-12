# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline executors that run a :class:`Graph` against input data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.graph.operator_resolution import resolve_graph
from nemo_retriever.common.ray_resource_hueristics import ClusterResources
from nemo_retriever.common.input_files import (
    _is_explicit_glob_path,
    expand_input_file_patterns,
    raise_input_path_not_found,
)
from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime
from nemo_retriever.common import ray_resource_hueristics as _rrh
from nemo_retriever.common.ray_resource_hueristics import (
    gather_cluster_resources,
    NEMOTRON_PARSE_BATCH_SIZE,
    VLLM_GPUS_PER_ACTOR,
    OCR_GPUS_PER_ACTOR,
)

import logging

logger = logging.getLogger(__name__)

# Heuristic GPU fraction for GPUOperator nodes that load a local model.
# Reuses the same baseline constant as the batch ingest mode.
_DEFAULT_GPU_OPERATOR_NUM_GPUS = OCR_GPUS_PER_ACTOR


def _concurrency_target(concurrency: Any) -> int:
    """Return the largest actor-pool size that resource planning can permit."""
    if isinstance(concurrency, tuple):
        return int(concurrency[1] if len(concurrency) == 3 else concurrency[0])
    return int(concurrency)


def _concurrency_initial(concurrency: Any) -> int:
    """Return the number of actors Ray creates when the pool starts."""
    if isinstance(concurrency, tuple) and len(concurrency) == 3:
        return int(concurrency[2])
    return 1


def _planned_concurrency(concurrency: Any, planned: int) -> Any:
    """Preserve Ray's actor-pool tuple while capping its maximum size."""
    if isinstance(concurrency, tuple) and len(concurrency) == 3:
        minimum, _maximum, initial = (int(value) for value in concurrency)
        return (minimum, max(minimum, initial, planned), initial)
    return planned


def preflight_executors(executors: list[Any], cluster_resources: ClusterResources) -> None:
    """Plan all lazy executor pools against one shared Ray resource snapshot."""
    entries = []
    available_cpus = cluster_resources.available_cpu_count()
    available_gpus = cluster_resources.available_gpu_count()
    for executor in executors:
        for node in executor._linearize(resolve_graph(executor.graph, cluster_resources)):
            override = executor._node_overrides.get(node.name, {})
            concurrency = override.get("concurrency", 1)
            entries.append(
                (
                    executor,
                    node.name,
                    concurrency,
                    _concurrency_target(concurrency),
                    _concurrency_initial(concurrency),
                    float(override.get("num_cpus", executor._default_num_cpus)),
                    executor._scheduled_num_gpus(node, override, available_gpus),
                    node.name in executor._auto_concurrency_nodes,
                )
            )
    fixed = [item for item in entries if not item[7]]
    auto = [item for item in entries if item[7]]
    fixed_cpu = sum(item[3] * item[5] for item in fixed)
    fixed_gpu = sum(item[3] * item[6] for item in fixed)
    min_cpu = sum(item[4] * item[5] for item in auto)
    min_gpu = sum(item[4] * item[6] for item in auto)
    if fixed_cpu + min_cpu > available_cpus or fixed_gpu + min_gpu > available_gpus:
        raise ValueError(
            "Infeasible Ray CPU/GPU plan: requested at least "
            f"{fixed_cpu + min_cpu:g} CPUs and {fixed_gpu + min_gpu:g} GPUs, but Ray reports "
            f"{available_cpus} CPUs and {available_gpus} GPUs available. "
            "Reduce explicit *_workers or node_overrides concurrency, or wait for cluster capacity."
        )
    used_cpu, used_gpu = fixed_cpu + min_cpu, fixed_gpu + min_gpu
    planned = {(id(item[0]), item[1]): item[4] for item in auto}
    while True:
        candidates = sorted(
            (item for item in auto if planned[(id(item[0]), item[1])] < item[3]),
            key=lambda item: planned[(id(item[0]), item[1])] / item[3],
        )
        selected = next(
            (
                item
                for item in candidates
                if used_cpu + item[5] <= available_cpus and used_gpu + item[6] <= available_gpus
            ),
            None,
        )
        if selected is None:
            break
        planned[(id(selected[0]), selected[1])] += 1
        used_cpu += selected[5]
        used_gpu += selected[6]
    for executor, name, concurrency, _target, _initial, _cpu, _gpu, _auto in auto:
        executor._node_overrides[name]["concurrency"] = _planned_concurrency(concurrency, planned[(id(executor), name)])
    for executor in executors:
        executor._resources_preflight_complete = True


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
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, (str, list)):
            df = self._load_files(expand_input_file_patterns(data))
        else:
            raise TypeError(
                f"data must be a pandas.DataFrame, file path, or list of paths, " f"got {type(data).__name__}"
            )

        resolved_graph = resolve_graph(self.graph, _rrh.gather_local_resources())
        nodes = self._linearize(resolved_graph)
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
            elif not _is_explicit_glob_path(p):
                raise_input_path_not_found(p)
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
        auto_concurrency_nodes: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(graph)
        self._ray_address = ray_address
        self._default_batch_size = batch_size
        self._default_batch_format = batch_format
        self._default_num_cpus = num_cpus
        self._default_num_gpus = num_gpus
        self._node_overrides = node_overrides or {}
        self._auto_concurrency_nodes = auto_concurrency_nodes or set()
        self._resources_preflight_complete = False

    def _has_remote_endpoint(self, node: Node) -> bool:
        """Return whether a node delegates inference to a remote endpoint."""
        if any("invoke_url" in key and bool(value) for key, value in node.operator_kwargs.items()):
            return True
        return any(
            hasattr(value, "model_dump")
            and any("invoke_url" in key and bool(item) for key, item in value.model_dump(exclude_none=True).items())
            for value in node.operator_kwargs.values()
        )

    def _scheduled_num_gpus(self, node: Node, overrides: Dict[str, Any], available_gpus: int) -> float:
        """Resolve the GPU reservation passed to Ray for a graph node."""
        if "num_gpus" in overrides:
            return float(overrides["num_gpus"])
        if not issubclass(node.operator_class, GPUOperator) or self._has_remote_endpoint(node):
            return float(self._default_num_gpus)
        if available_gpus > 0:
            from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor, NemotronParseGPUActor
            from nemo_retriever.operators.extract.caption.caption import CaptionGPUActor

            if issubclass(node.operator_class, (NemotronParseActor, NemotronParseGPUActor, CaptionGPUActor)):
                return max(float(self._default_num_gpus), VLLM_GPUS_PER_ACTOR)
            return max(float(self._default_num_gpus), _DEFAULT_GPU_OPERATOR_NUM_GPUS)
        logger.warning(
            "Node %r is a GPUOperator with no remote endpoint but the Ray cluster reports 0 available GPUs. "
            "The actor will be scheduled with num_gpus=0 and will likely fail to load its model. "
            "Pass --ocr-invoke-url / --page-elements-invoke-url / --embed-invoke-url to use remote endpoints, "
            "or ensure GPUs are visible to Ray.",
            node.name,
        )
        return float(self._default_num_gpus)

    def _preflight_resources(self, nodes: List[Node], available_cpus: int, available_gpus: int) -> None:
        """Reduce unspecified pools and reject infeasible explicit plans."""
        entries = []
        for node in nodes:
            override = self._node_overrides.get(node.name, {})
            concurrency = override.get("concurrency", 1)
            entries.append(
                (
                    node.name,
                    concurrency,
                    _concurrency_target(concurrency),
                    _concurrency_initial(concurrency),
                    float(override.get("num_cpus", self._default_num_cpus)),
                    self._scheduled_num_gpus(node, override, available_gpus),
                )
            )
        fixed = [item for item in entries if item[0] not in self._auto_concurrency_nodes]
        auto = [item for item in entries if item[0] in self._auto_concurrency_nodes]
        fixed_cpu = sum(count * cpu for _name, _concurrency, count, _initial, cpu, _gpu in fixed)
        fixed_gpu = sum(count * gpu for _name, _concurrency, count, _initial, _cpu, gpu in fixed)
        minimum_cpu = sum(initial * cpu for _name, _concurrency, _count, initial, cpu, _gpu in auto)
        minimum_gpu = sum(initial * gpu for _name, _concurrency, _count, initial, _cpu, gpu in auto)
        if fixed_cpu + minimum_cpu > available_cpus or fixed_gpu + minimum_gpu > available_gpus:
            raise ValueError(
                "Infeasible Ray CPU/GPU plan: requested at least "
                f"{fixed_cpu + minimum_cpu:g} CPUs and {fixed_gpu + minimum_gpu:g} GPUs, but Ray reports "
                f"{available_cpus} CPUs and {available_gpus} GPUs available. "
                "Reduce explicit *_workers or node_overrides concurrency, or wait for cluster capacity."
            )
        used_cpu, used_gpu = fixed_cpu + minimum_cpu, fixed_gpu + minimum_gpu
        planned = {name: initial for name, _concurrency, _count, initial, _cpu, _gpu in auto}
        while True:
            candidates = sorted(
                (item for item in auto if planned[item[0]] < item[2]),
                key=lambda item: planned[item[0]] / item[2],
            )
            selected = next(
                (
                    item
                    for item in candidates
                    if used_cpu + item[4] <= available_cpus and used_gpu + item[5] <= available_gpus
                ),
                None,
            )
            if selected is None:
                break
            name, _concurrency, _count, _initial, cpu, gpu = selected
            planned[name] += 1
            used_cpu += cpu
            used_gpu += gpu
        for name, concurrency, _count, _initial, _cpu, _gpu in auto:
            self._node_overrides[name]["concurrency"] = _planned_concurrency(concurrency, planned[name])

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
        """Build, execute, and materialize a Ray Data pipeline from the graph."""

        return self.build_dataset(data, **kwargs).to_pandas()

    def build_dataset(self, data: Any, **kwargs: Any) -> Any:
        """Build a lazy Ray Data pipeline from the graph.

        Parameters
        ----------
        data
            Input to ``ray.data.read_binary_files`` (a path or list of glob patterns)
            **or** an already-constructed ``ray.data.Dataset``.

        Returns
        -------
        ray.data.Dataset
            The lazy Ray dataset with all graph stages appended.
        """
        ray = ensure_local_ray_runtime(self._ray_address)
        import ray.data as rd

        if not isinstance(data, (rd.Dataset, str, list)):
            raise TypeError(
                f"data must be a path/glob string, list of globs, or ray.data.Dataset, " f"got {type(data).__name__}"
            )

        input_paths: Optional[List[str]] = None
        if isinstance(data, (str, list)):
            input_paths = expand_input_file_patterns(data)

        ctx = rd.DataContext.get_current()
        ctx.enable_rich_progress_bars = True
        ctx.use_ray_tqdm = False

        cluster = gather_cluster_resources(ray)
        available_gpus = cluster.available_gpu_count()
        resolved_graph = resolve_graph(self.graph, cluster)

        if isinstance(data, rd.Dataset):
            ds = data
        else:
            try:
                ds = rd.read_binary_files(input_paths, include_paths=True)
            except FileNotFoundError as exc:
                raise_input_path_not_found(input_paths or [], exc)
        nodes = self._linearize(resolved_graph)
        if nodes and not self._resources_preflight_complete:
            self._preflight_resources(nodes, cluster.available_cpu_count(), available_gpus)
        for node in nodes:
            overrides = dict(self._node_overrides.get(node.name, {}))
            target_num_rows_per_block = overrides.pop("target_num_rows_per_block", None)
            batch_size = overrides.pop("batch_size", self._default_batch_size)
            batch_format = overrides.pop("batch_format", self._default_batch_format)
            num_cpus = overrides.pop("num_cpus", self._default_num_cpus)
            # Ray 2.49+ requires concurrency to be specified for callable classes.
            # Default to 1 when not explicitly set via node_overrides.
            if "concurrency" not in overrides:
                overrides["concurrency"] = 1

            # vLLM-backed actors handle their own batching efficiently
            # (continuous batching), so feed them more rows per map_batches call.
            from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor, NemotronParseGPUActor
            from nemo_retriever.operators.extract.caption.caption import CaptionGPUActor

            if batch_size == self._default_batch_size and issubclass(
                node.operator_class, (NemotronParseActor, NemotronParseGPUActor, CaptionGPUActor)
            ):
                batch_size = NEMOTRON_PARSE_BATCH_SIZE

            # Self-join operators (AudioVisualFuser, VideoFrameTextDedup) need
            # the entire dataset in one batch — see the repartition site below
            # for the actual single-block enforcement.
            requires_global_batch = bool(getattr(node.operator_class, "REQUIRES_GLOBAL_BATCH", False))
            if requires_global_batch:
                batch_size = None
                target_num_rows_per_block = None

            num_gpus = self._scheduled_num_gpus(node, overrides, available_gpus)
            overrides.pop("num_gpus", None)

            if requires_global_batch:
                # ``num_blocks=1`` is exact; ``target_num_rows_per_block`` is a
                # streaming best-effort cap that can leave joins missing rows.
                # When the operator declares ``GLOBAL_BATCH_GROUP_KEYS`` and
                # concurrency > 1, hash-partition by those keys so rows sharing
                # the keys stay co-located while blocks distribute across actors.
                group_keys = list(getattr(node.operator_class, "GLOBAL_BATCH_GROUP_KEYS", None) or ())
                n_blocks = max(1, int(overrides.get("concurrency") or 1)) if group_keys else 1
                if n_blocks > 1:
                    ds = ds.repartition(num_blocks=n_blocks, keys=group_keys, shuffle=True)
                else:
                    ds = ds.repartition(num_blocks=1)
            elif target_num_rows_per_block is not None and int(target_num_rows_per_block) > 0:
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

        return ds
