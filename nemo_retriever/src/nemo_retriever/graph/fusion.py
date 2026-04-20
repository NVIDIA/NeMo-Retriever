# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for compiling eligible linear graph segments into fused operators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.utils.ray_resource_hueristics import OCR_GPUS_PER_ACTOR


class ProcessOnlyFusionSafe:
    """Explicit opt-in contract for stages safe to fuse at ``process()`` level."""

    fusion_stage_id: ClassVar[str]
    fusion_next_stage_ids: ClassVar[tuple[str, ...]] = ()
    fusion_can_start_segment: ClassVar[bool] = False


@dataclass(frozen=True)
class GraphCompilationResult:
    graph: Graph
    node_overrides: dict[str, dict[str, Any]]
    fusion_summary: dict[str, Any]


class FusedOperator(AbstractOperator):
    """Composite operator that runs a known-safe stage chain."""

    def __init__(
        self,
        *,
        stage_specs: list[dict[str, Any]],
        fused_node_names: list[str],
        segment_name: str,
    ) -> None:
        super().__init__(
            stage_specs=deepcopy(stage_specs),
            fused_node_names=list(fused_node_names),
            segment_name=segment_name,
        )
        self.stage_specs = deepcopy(stage_specs)
        self.fused_node_names = tuple(fused_node_names)
        self.segment_name = segment_name
        self._operators: list[AbstractOperator] | None = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        current = data
        for operator in self._ensure_operators():
            current = operator.process(current, **kwargs)
        return current

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Preserve unfused Ray batch semantics by invoking each stage callable."""
        current = self.preprocess(data, **kwargs)
        for operator in self._ensure_operators():
            current = operator(current, **kwargs)
        return self.postprocess(current, **kwargs)

    def _ensure_operators(self) -> list[AbstractOperator]:
        if self._operators is None:
            self._operators = []
            for stage in self.stage_specs:
                operator_class = stage["operator_class"]
                operator_kwargs = deepcopy(stage.get("operator_kwargs", {}))
                self._operators.append(operator_class(**operator_kwargs))
        return self._operators


def compile_graph_for_fusion(
    graph: Graph,
    *,
    enable_fusion: bool = False,
    node_overrides: dict[str, dict[str, Any]] | None = None,
) -> GraphCompilationResult:
    """Compile eligible linear graph segments into fused operators."""

    original_node_names = _node_names(graph)
    original_overrides = deepcopy(node_overrides or {})

    if not enable_fusion:
        return GraphCompilationResult(
            graph=graph,
            node_overrides=original_overrides,
            fusion_summary=_fusion_summary(
                enabled=False,
                applied=False,
                reason="disabled",
                original_nodes=original_node_names,
                compiled_nodes=original_node_names,
                fused_segments=[],
            ),
        )

    linear_nodes = _linearize_if_possible(graph)
    if linear_nodes is None:
        return GraphCompilationResult(
            graph=graph,
            node_overrides=original_overrides,
            fusion_summary=_fusion_summary(
                enabled=True,
                applied=False,
                reason="graph_not_linear",
                original_nodes=original_node_names,
                compiled_nodes=original_node_names,
                fused_segments=[],
            ),
        )

    compiled_nodes: list[Node] = []
    compiled_overrides: dict[str, dict[str, Any]] = {}
    fused_segments: list[dict[str, Any]] = []
    idx = 0
    while idx < len(linear_nodes):
        segment_nodes = _consume_fusable_segment(linear_nodes, idx)
        if len(segment_nodes) > 1:
            fused_node = _make_fused_node(segment_nodes)
            compiled_nodes.append(fused_node)
            aggregated = _aggregate_node_overrides(segment_nodes, original_overrides)
            if aggregated:
                compiled_overrides[fused_node.name] = aggregated
            fused_segments.append(
                {
                    "name": fused_node.name,
                    "stage_names": [node.name for node in segment_nodes],
                    "operator_classes": [node.operator_class.__name__ for node in segment_nodes],
                    "stage_count": len(segment_nodes),
                    "aggregated_overrides": aggregated,
                }
            )
            idx += len(segment_nodes)
            continue

        node = linear_nodes[idx]
        compiled_nodes.append(_clone_node(node))
        if node.name in original_overrides:
            compiled_overrides[node.name] = deepcopy(original_overrides[node.name])
        idx += 1

    if not fused_segments:
        return GraphCompilationResult(
            graph=graph,
            node_overrides=original_overrides,
            fusion_summary=_fusion_summary(
                enabled=True,
                applied=False,
                reason="no_eligible_segments",
                original_nodes=original_node_names,
                compiled_nodes=original_node_names,
                fused_segments=[],
            ),
        )

    compiled_graph = Graph()
    compiled_graph.add_chain(*compiled_nodes)
    compiled_node_names = [node.name for node in compiled_nodes]
    return GraphCompilationResult(
        graph=compiled_graph,
        node_overrides=compiled_overrides,
        fusion_summary=_fusion_summary(
            enabled=True,
            applied=True,
            reason="compiled",
            original_nodes=original_node_names,
            compiled_nodes=compiled_node_names,
            fused_segments=fused_segments,
        ),
    )


def _fusion_summary(
    *,
    enabled: bool,
    applied: bool,
    reason: str,
    original_nodes: list[str],
    compiled_nodes: list[str],
    fused_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "applied": applied,
        "reason": reason,
        "original_stage_count": len(original_nodes),
        "compiled_stage_count": len(compiled_nodes),
        "original_nodes": list(original_nodes),
        "compiled_nodes": list(compiled_nodes),
        "fused_segments": fused_segments,
    }


def _linearize_if_possible(graph: Graph) -> list[Node] | None:
    if not graph.roots:
        return []
    if len(graph.roots) > 1:
        return None

    ordered: list[Node] = []
    node = graph.roots[0]
    while node is not None:
        ordered.append(node)
        if len(node.children) > 1:
            return None
        node = node.children[0] if node.children else None
    return ordered


def _node_names(graph: Graph) -> list[str]:
    linear_nodes = _linearize_if_possible(graph)
    if linear_nodes is not None:
        return [node.name for node in linear_nodes]

    names: list[str] = []
    seen: set[int] = set()

    def _visit(node: Node) -> None:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        names.append(node.name)
        for child in node.children:
            _visit(child)

    for root in graph.roots:
        _visit(root)
    return names


def _consume_fusable_segment(nodes: list[Node], start_idx: int) -> list[Node]:
    start = nodes[start_idx]
    start_class = start.operator_class
    if not _can_start_fused_segment(start_class):
        return [start]

    segment = [start]
    current_class = start_class
    idx = start_idx + 1
    while idx < len(nodes):
        next_node = nodes[idx]
        next_class = next_node.operator_class
        if not _is_process_only_fusion_safe(next_class):
            break
        if _stage_id(next_class) not in _next_stage_ids(current_class):
            break
        segment.append(next_node)
        current_class = next_class
        idx += 1
    return segment if len(segment) > 1 else [start]


def _make_fused_node(segment_nodes: list[Node]) -> Node:
    stage_names = [node.name for node in segment_nodes]
    fused_name = f"Fused[{'+'.join(stage_names)}]"
    stage_specs = [
        {
            "name": node.name,
            "operator_class": node.operator_class,
            "operator_kwargs": deepcopy(node.operator_kwargs),
        }
        for node in segment_nodes
    ]
    operator_kwargs = {
        "stage_specs": stage_specs,
        "fused_node_names": stage_names,
        "segment_name": fused_name,
    }
    operator = FusedOperator(**operator_kwargs)
    return Node(
        operator,
        name=fused_name,
        operator_class=FusedOperator,
        operator_kwargs=operator_kwargs,
    )


def _clone_node(node: Node) -> Node:
    return Node(
        node.operator,
        name=node.name,
        operator_class=node.operator_class,
        operator_kwargs=deepcopy(node.operator_kwargs),
    )


def _aggregate_node_overrides(
    segment_nodes: list[Node],
    original_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_stage_overrides = [deepcopy(original_overrides.get(node.name, {})) for node in segment_nodes]
    aggregated: dict[str, Any] = {}

    for key in ("batch_size", "target_num_rows_per_block", "concurrency"):
        value = _min_positive_override(per_stage_overrides, key)
        if value is not None:
            aggregated[key] = value

    num_cpus = _max_positive_override(per_stage_overrides, "num_cpus")
    if num_cpus is not None:
        aggregated["num_cpus"] = num_cpus

    num_gpus = _sum_num_gpus(segment_nodes, per_stage_overrides)
    if num_gpus is not None:
        aggregated["num_gpus"] = num_gpus

    batch_format = _first_override(per_stage_overrides, "batch_format")
    if batch_format is not None:
        aggregated["batch_format"] = batch_format

    reserved_keys = {
        "batch_size",
        "target_num_rows_per_block",
        "concurrency",
        "num_cpus",
        "num_gpus",
        "batch_format",
    }
    passthrough_keys = {
        key for override in per_stage_overrides for key in override if key not in reserved_keys and key not in aggregated
    }
    for key in sorted(passthrough_keys):
        value = _first_override(per_stage_overrides, key)
        if value is not None:
            aggregated[key] = value

    return aggregated


def _first_override(per_stage_overrides: list[dict[str, Any]], key: str) -> Any:
    for override in per_stage_overrides:
        if key in override:
            return override[key]
    return None


def _min_positive_override(per_stage_overrides: list[dict[str, Any]], key: str) -> int | float | None:
    values = [override.get(key) for override in per_stage_overrides if _is_positive_number(override.get(key))]
    if not values:
        return None
    return min(values)


def _max_positive_override(per_stage_overrides: list[dict[str, Any]], key: str) -> float | None:
    values = [float(override.get(key)) for override in per_stage_overrides if _is_positive_number(override.get(key))]
    if not values:
        return None
    return max(values)


def _sum_num_gpus(segment_nodes: list[Node], per_stage_overrides: list[dict[str, Any]]) -> float | None:
    total = 0.0
    saw_any = False
    for node, override in zip(segment_nodes, per_stage_overrides):
        if "num_gpus" in override:
            saw_any = True
            total += max(float(override.get("num_gpus") or 0.0), 0.0)
            continue
        if issubclass(node.operator_class, GPUOperator):
            saw_any = True
            total += float(OCR_GPUS_PER_ACTOR)
    if not saw_any:
        return None
    return total


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _is_process_only_fusion_safe(operator_class: type) -> bool:
    return issubclass(operator_class, ProcessOnlyFusionSafe) and bool(getattr(operator_class, "fusion_stage_id", ""))


def _can_start_fused_segment(operator_class: type) -> bool:
    return _is_process_only_fusion_safe(operator_class) and bool(
        getattr(operator_class, "fusion_can_start_segment", False)
    )


def _stage_id(operator_class: type) -> str:
    return str(getattr(operator_class, "fusion_stage_id", ""))


def _next_stage_ids(operator_class: type) -> tuple[str, ...]:
    return tuple(getattr(operator_class, "fusion_next_stage_ids", ()))
