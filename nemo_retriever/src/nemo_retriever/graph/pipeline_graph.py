# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directed pipeline graph composed of Nodes that wrap AbstractOperators."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from nemo_retriever.graph.abstract_operator import AbstractOperator


def _ensure_node(obj: Union["Node", "Graph", AbstractOperator]) -> "Node":
    """Wrap *obj* in a :class:`Node` if it is a bare operator.

    If *obj* is a :class:`Graph`, its first root node is returned.
    """
    if isinstance(obj, Graph):
        if not obj.roots:
            raise ValueError("Cannot extract a Node from an empty Graph.")
        return obj.roots[0]
    if isinstance(obj, Node):
        return obj
    if isinstance(obj, AbstractOperator):
        return Node(obj)
    raise TypeError(f"Expected a Node, Graph, or AbstractOperator, got {type(obj).__name__}")


class Node:
    """A single node in a pipeline graph.

    Each node wraps an :class:`AbstractOperator` and maintains an ordered list
    of child nodes that should execute after it.

    The ``>>`` operator chains two nodes and returns a :class:`Graph`::

        graph = a >> b >> c   # Graph with root=a, a->b->c
        graph.add_root(...)   # add more roots if needed
    """

    def __init__(
        self,
        operator: AbstractOperator,
        name: Optional[str] = None,
        *,
        operator_class: Optional[type] = None,
        operator_kwargs: Optional[dict] = None,
    ) -> None:
        if not isinstance(operator, AbstractOperator):
            raise TypeError(f"operator must be an AbstractOperator, got {type(operator).__name__}")
        self.operator = operator
        self.name = name or type(operator).__name__
        self.children: List[Node] = []
        self.operator_class = operator_class or type(operator)
        self.operator_kwargs = operator_kwargs if operator_kwargs is not None else operator.get_constructor_kwargs()

    def add_child(self, child: Union["Node", AbstractOperator]) -> "Node":
        """Append *child* to this node's children and return the child node."""
        child_node = _ensure_node(child)
        self.children.append(child_node)
        return child_node

    def __rshift__(self, other: Union["Node", AbstractOperator]) -> "Graph":
        """``self >> other`` — add *other* as a child of *self*.

        Returns a :class:`Graph` with ``self`` as root so that further ``>>``
        appends to the leaf nodes::

            graph = a >> b >> c
            # Graph(roots=['A']), a -> b -> c
        """
        child_node = self.add_child(other)
        g = Graph()
        g.roots.append(self)
        g._tail = child_node
        return g

    def __repr__(self) -> str:
        child_names = [c.name for c in self.children]
        return f"Node(name={self.name!r}, children={child_names})"


class Graph:
    """A directed acyclic pipeline graph.

    A graph owns one or more root :class:`Node` instances and can execute them
    in topological (depth-first) order.

    The ``>>`` operator appends a node to the current leaf nodes and returns
    the graph itself, enabling fluent chaining::

        graph = a >> b >> c >> d
        # or
        graph = Graph()
        graph.add_root(a)
        graph >> b >> c >> d

    Bare :class:`AbstractOperator` instances passed to :meth:`add_root`,
    :meth:`add_chain`, or ``>>`` are auto-wrapped in :class:`Node`.
    """

    def __init__(self) -> None:
        self.roots: List[Node] = []
        self._tail: Optional[Node] = None

    def add_root(self, node: Union[Node, "Graph", AbstractOperator]) -> Node:
        """Register *node* as a root (entry point) of the graph."""
        if isinstance(node, Graph):
            # Merge another graph's roots into this one.
            for r in node.roots:
                if r not in self.roots:
                    self.roots.append(r)
            return node.roots[0] if node.roots else None  # type: ignore[return-value]
        root = _ensure_node(node)
        self.roots.append(root)
        return root

    def add_chain(self, *nodes: Union[Node, AbstractOperator]) -> None:
        """Chain *nodes* in order and register the first as a root.

        Each element may be a :class:`Node` or a bare :class:`AbstractOperator`.
        """
        if not nodes:
            return
        wrapped = [_ensure_node(n) for n in nodes]
        self.roots.append(wrapped[0])
        for i in range(len(wrapped) - 1):
            wrapped[i].add_child(wrapped[i + 1])
        self._tail = wrapped[-1]

    def __rshift__(self, other: Union[Node, AbstractOperator]) -> "Graph":
        """``graph >> node`` — add *other* as a child of the current tail.

        If the graph has no roots yet, *other* becomes the first root.
        Returns ``self`` for fluent chaining.
        """
        node = _ensure_node(other)
        if not self.roots:
            self.roots.append(node)
            self._tail = node
        elif self._tail is not None:
            self._tail.add_child(node)
            self._tail = node
        else:
            for leaf in self._leaf_nodes():
                leaf.add_child(node)
            self._tail = node
        return self

    def _leaf_nodes(self) -> List[Node]:
        """Return all leaf nodes (nodes with no children) reachable from roots."""
        leaves: List[Node] = []
        for root in self.roots:
            self._collect_leaves(root, leaves)
        return leaves

    def _collect_leaves(self, node: Node, leaves: List[Node]) -> None:
        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)

    def execute(self, data: Any, **kwargs: Any) -> List[Any]:
        """Execute every root and its descendants depth-first.

        Each root receives *data*; children receive their parent's output.
        Returns a list of leaf outputs (one per leaf node reached).
        """
        results: List[Any] = []
        for root in self.roots:
            self._execute_node(root, data, results, **kwargs)
        return results

    def _execute_node(self, node: Node, data: Any, results: List[Any], **kwargs: Any) -> None:
        output = node.operator.run(data, **kwargs)
        if node.children:
            for child in node.children:
                self._execute_node(child, output, results, **kwargs)
        else:
            results.append(output)

    def __repr__(self) -> str:
        root_names = [r.name for r in self.roots]
        return f"Graph(roots={root_names})"
