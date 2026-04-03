# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.utils.ray_resource_hueristics import ClusterResources, Resources, gather_local_resources


def resolve_operator_class(
    operator_class: type[AbstractOperator],
    resources: ClusterResources | Resources,
    operator_kwargs: dict | None = None,
) -> type[AbstractOperator]:
    if issubclass(operator_class, ArchetypeOperator):
        return operator_class.resolve_operator_class(resources, operator_kwargs=operator_kwargs)
    return operator_class


def resolve_graph(
    graph: Graph,
    resources: ClusterResources | Resources,
) -> Graph:
    resolved = Graph()
    visited: dict[int, Node] = {}

    def _clone(node: Node) -> Node:
        node_id = id(node)
        if node_id in visited:
            return visited[node_id]

        operator = node.operator
        if isinstance(operator, ArchetypeOperator):
            operator = type(operator)(**node.operator_kwargs)

        cloned = Node(
            operator,
            name=node.name,
            operator_class=resolve_operator_class(node.operator_class, resources, operator_kwargs=node.operator_kwargs),
            operator_kwargs=dict(node.operator_kwargs),
        )
        visited[node_id] = cloned
        for child in node.children:
            cloned.children.append(_clone(child))
        return cloned

    for root in graph.roots:
        resolved.roots.append(_clone(root))

    if graph._tail is not None:
        resolved._tail = visited.get(id(graph._tail))
    return resolved


def resolve_graph_for_local_execution(graph: Graph) -> Graph:
    return resolve_graph(graph, gather_local_resources())
