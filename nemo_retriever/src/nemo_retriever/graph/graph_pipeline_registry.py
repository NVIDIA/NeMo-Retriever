# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph Pipeline Registry — manage, inspect, compare, and serialize golden pipeline graphs.

Provides a central :class:`GraphPipelineRegistry` that stores named graph
*blueprints* (factory functions + metadata).  Graphs built from the registry
can be inspected, diffed against each other, serialized to / loaded from JSON,
and configured with kwarg overrides — all without touching the code that
originally defined them.

A module-level :data:`default_registry` is provided for convenience so that
graph definitions scattered across the codebase can all register to a single
shared instance.

Quick-start::

    from nemo_retriever.graph.graph_pipeline_registry import default_registry

    @default_registry.register("my-pipeline", description="Demo pipeline")
    def _build():
        from nemo_retriever.graph import Graph
        return Graph() >> SomeOperator() >> AnotherOperator()

    graph = default_registry.build("my-pipeline")
    default_registry.print_graph("my-pipeline")
"""

from __future__ import annotations

import importlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qualified_name(cls: type) -> str:
    """Return the fully qualified ``module.ClassName`` string for *cls*."""
    module = cls.__module__ or "__main__"
    return f"{module}.{cls.__qualname__}"


def _import_class(qualified: str) -> type:
    """Import and return a class from its fully qualified dotted path."""
    module_path, _, class_name = qualified.rpartition(".")
    if not module_path:
        raise ImportError(f"Cannot import class from unqualified name: {qualified!r}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Module {module_path!r} has no attribute {class_name!r}")
    return cls


class _RegistryJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serializable types found in operator kwargs."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, type):
            return {"__type_ref__": _qualified_name(obj)}
        if callable(obj) and hasattr(obj, "__qualname__"):
            module = getattr(obj, "__module__", None) or ""
            return {"__callable_ref__": f"{module}.{obj.__qualname__}"}
        if isinstance(obj, Path):
            return {"__path__": str(obj)}
        if isinstance(obj, (set, frozenset)):
            return {"__set__": sorted(obj, key=str)}
        if isinstance(obj, bytes):
            return {"__bytes_len__": len(obj), "__repr__": repr(obj[:64])}
        if hasattr(obj, "__dict__"):
            safe_attrs = {}
            for k, v in obj.__dict__.items():
                if not k.startswith("_"):
                    try:
                        json.dumps(v, cls=_RegistryJSONEncoder)
                        safe_attrs[k] = v
                    except (TypeError, ValueError):
                        safe_attrs[k] = repr(v)
            return {
                "__object__": _qualified_name(type(obj)),
                "__attrs__": safe_attrs,
            }
        return super().default(obj)


def _safe_serialize_value(value: Any) -> Any:
    """Best-effort conversion of *value* into something JSON-safe."""
    try:
        json.dumps(value, cls=_RegistryJSONEncoder)
        return value
    except (TypeError, ValueError, OverflowError):
        return repr(value)


# ---------------------------------------------------------------------------
# Graph walking / introspection utilities
# ---------------------------------------------------------------------------


def walk_nodes(graph: Graph) -> Iterator[Tuple[Node, int]]:
    """Yield ``(node, depth)`` for every unique node via depth-first traversal."""
    visited: Set[int] = set()

    def _dfs(node: Node, depth: int) -> Iterator[Tuple[Node, int]]:
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        yield node, depth
        for child in node.children:
            yield from _dfs(child, depth + 1)

    for root in graph.roots:
        yield from _dfs(root, 0)


def collect_nodes(graph: Graph) -> List[Node]:
    """Return an ordered list of all unique nodes in the graph."""
    return [node for node, _ in walk_nodes(graph)]


def node_count(graph: Graph) -> int:
    """Return the total number of unique nodes in the graph."""
    return len(collect_nodes(graph))


def max_depth(graph: Graph) -> int:
    """Return the maximum depth (longest root-to-leaf path) of the graph."""
    return max((d for _, d in walk_nodes(graph)), default=0)


def find_node(graph: Graph, name: str) -> Optional[Node]:
    """Return the first node whose ``name`` matches *name*, or ``None``."""
    for node, _ in walk_nodes(graph):
        if node.name == name:
            return node
    return None


def find_nodes(graph: Graph, name: str) -> List[Node]:
    """Return every node whose ``name`` matches *name*."""
    return [node for node, _ in walk_nodes(graph) if node.name == name]


def leaf_nodes(graph: Graph) -> List[Node]:
    """Return all leaf nodes (nodes with no children)."""
    return [node for node in collect_nodes(graph) if not node.children]


def get_node_kwargs(graph: Graph, name: str) -> Dict[str, Any]:
    """Return the ``operator_kwargs`` for the first node named *name*.

    Raises ``KeyError`` if no node matches.
    """
    node = find_node(graph, name)
    if node is None:
        raise KeyError(f"No node named {name!r} in graph")
    return dict(node.operator_kwargs)


def list_all_kwargs(graph: Graph) -> Dict[str, Dict[str, Any]]:
    """Return ``{node_name: operator_kwargs}`` for every node in the graph."""
    return {node.name: dict(node.operator_kwargs) for node in collect_nodes(graph)}


# ---------------------------------------------------------------------------
# Pretty-print / inspection
# ---------------------------------------------------------------------------


def format_graph_tree(
    graph: Graph,
    *,
    show_kwargs: bool = False,
    show_class: bool = True,
    max_value_width: int = 120,
) -> str:
    """Return a human-readable tree representation of the graph.

    Parameters
    ----------
    graph
        The graph to format.
    show_kwargs
        Display each node's ``operator_kwargs`` beneath it.
    show_class
        Show the fully qualified operator class next to the node name.
    max_value_width
        Truncate kwarg value reprs longer than this.
    """
    lines: List[str] = []
    visited: Set[int] = set()

    def _resource_marker(node: Node) -> str:
        try:
            from nemo_retriever.graph.cpu_operator import CPUOperator
            from nemo_retriever.graph.gpu_operator import GPUOperator

            if isinstance(node.operator, GPUOperator):
                return " [GPU]"
            if isinstance(node.operator, CPUOperator):
                return " [CPU]"
        except ImportError:
            pass
        return ""

    def _render(node: Node, prefix: str, is_last: bool, is_root: bool) -> None:
        nid = id(node)
        if nid in visited:
            connector = "" if is_root else ("└── " if is_last else "├── ")
            lines.append(f"{prefix}{connector}↻ {node.name} (back-ref)")
            return
        visited.add(nid)

        connector = "" if is_root else ("└── " if is_last else "├── ")
        marker = _resource_marker(node)
        class_info = f"  ({_qualified_name(node.operator_class)})" if show_class else ""
        lines.append(f"{prefix}{connector}{node.name}{marker}{class_info}")

        if show_kwargs and node.operator_kwargs:
            kw_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
            for key, val in sorted(node.operator_kwargs.items()):
                val_repr = repr(val) if not isinstance(val, str) else f"'{val}'"
                if len(val_repr) > max_value_width:
                    val_repr = val_repr[: max_value_width - 3] + "..."
                lines.append(f"{kw_prefix}  ╰ {key} = {val_repr}")

        child_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
        for i, child in enumerate(node.children):
            _render(child, child_prefix, is_last=(i == len(node.children) - 1), is_root=False)

    for i, root in enumerate(graph.roots):
        if i > 0:
            lines.append("")
        _render(root, "", is_last=(i == len(graph.roots) - 1), is_root=True)

    return "\n".join(lines)


def format_node_details(node: Node) -> str:
    """Return a detailed multi-line description of a single node."""
    lines = [
        f"Node: {node.name}",
        f"  Operator class : {_qualified_name(node.operator_class)}",
        f"  Children       : {[c.name for c in node.children]}",
        f"  Kwargs ({len(node.operator_kwargs)}):",
    ]
    for key, val in sorted(node.operator_kwargs.items()):
        val_repr = repr(val)
        if len(val_repr) > 200:
            val_repr = val_repr[:197] + "..."
        lines.append(f"    {key:30s} = {val_repr}")
    return "\n".join(lines)


def format_graph_summary(graph: Graph) -> str:
    """Return a concise summary: node count, depth, root/leaf names."""
    nodes = collect_nodes(graph)
    leaves = [n for n in nodes if not n.children]
    root_names = [r.name for r in graph.roots]
    leaf_names = [n.name for n in leaves]
    return (
        f"Graph Summary\n"
        f"  Roots ({len(root_names)}) : {root_names}\n"
        f"  Leaves ({len(leaf_names)}): {leaf_names}\n"
        f"  Total nodes    : {len(nodes)}\n"
        f"  Max depth      : {max_depth(graph)}"
    )


def format_full_report(graph: Graph, *, show_kwargs: bool = True) -> str:
    """Return a complete inspection report: summary + tree + per-node details."""
    sections: List[str] = [
        format_graph_summary(graph),
        "",
        format_graph_tree(graph, show_kwargs=show_kwargs),
        "",
    ]
    for node in collect_nodes(graph):
        sections.append(format_node_details(node))
        sections.append("")
    return "\n".join(sections)


def print_graph(graph: Graph, *, show_kwargs: bool = True) -> None:
    """Print a full graph inspection to stdout."""
    print(format_full_report(graph, show_kwargs=show_kwargs))


# ---------------------------------------------------------------------------
# Configuration update
# ---------------------------------------------------------------------------


def update_node_kwargs(
    graph: Graph,
    node_name: str,
    updates: Dict[str, Any],
    *,
    all_matches: bool = False,
) -> int:
    """Update ``operator_kwargs`` for node(s) matching *node_name* in-place.

    Parameters
    ----------
    graph
        The graph to modify.
    node_name
        Name of the target node(s).
    updates
        ``{kwarg_key: new_value}`` pairs to merge in.
    all_matches
        If ``True``, update every matching node.  Otherwise update only the
        first match and raise ``KeyError`` if none is found.

    Returns
    -------
    int
        Number of nodes updated.
    """
    if all_matches:
        targets = find_nodes(graph, node_name)
    else:
        target = find_node(graph, node_name)
        if target is None:
            raise KeyError(f"No node named {node_name!r} found in graph")
        targets = [target]

    for node in targets:
        node.operator_kwargs.update(updates)
    return len(targets)


def remove_node_kwargs(
    graph: Graph,
    node_name: str,
    keys: Sequence[str],
    *,
    all_matches: bool = False,
) -> int:
    """Remove specific kwarg keys from node(s) matching *node_name*.

    Returns the number of nodes modified.  Missing keys are silently ignored.
    """
    if all_matches:
        targets = find_nodes(graph, node_name)
    else:
        target = find_node(graph, node_name)
        if target is None:
            raise KeyError(f"No node named {node_name!r} found in graph")
        targets = [target]

    for node in targets:
        for key in keys:
            node.operator_kwargs.pop(key, None)
    return len(targets)


def replace_node_kwargs(
    graph: Graph,
    node_name: str,
    new_kwargs: Dict[str, Any],
    *,
    all_matches: bool = False,
) -> int:
    """Replace the entire ``operator_kwargs`` dict for matching node(s).

    Returns the number of nodes modified.
    """
    if all_matches:
        targets = find_nodes(graph, node_name)
    else:
        target = find_node(graph, node_name)
        if target is None:
            raise KeyError(f"No node named {node_name!r} found in graph")
        targets = [target]

    for node in targets:
        node.operator_kwargs.clear()
        node.operator_kwargs.update(new_kwargs)
    return len(targets)


# ---------------------------------------------------------------------------
# Graph comparison / diff
# ---------------------------------------------------------------------------


@dataclass
class NodeDiff:
    """Differences between two nodes at corresponding positions."""

    position: str
    node_a_name: str
    node_b_name: str
    name_changed: bool = False
    class_changed: bool = False
    class_a: str = ""
    class_b: str = ""
    kwargs_added: Dict[str, Any] = field(default_factory=dict)
    kwargs_removed: Dict[str, Any] = field(default_factory=dict)
    kwargs_changed: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    children_a_only: List[str] = field(default_factory=list)
    children_b_only: List[str] = field(default_factory=list)


@dataclass
class GraphDiff:
    """Full diff result between two graphs."""

    identical: bool
    structural_match: bool
    node_count_a: int
    node_count_b: int
    roots_a: List[str]
    roots_b: List[str]
    node_diffs: List[NodeDiff] = field(default_factory=list)
    nodes_only_in_a: List[str] = field(default_factory=list)
    nodes_only_in_b: List[str] = field(default_factory=list)

    def format(self) -> str:
        """Return a human-readable diff report."""
        lines: List[str] = []
        sep = "=" * 72
        lines.append(sep)
        lines.append("GRAPH COMPARISON REPORT")
        lines.append(sep)
        lines.append(f"  Identical        : {self.identical}")
        lines.append(f"  Structural match : {self.structural_match}")
        lines.append(f"  Nodes (A / B)    : {self.node_count_a} / {self.node_count_b}")
        lines.append(f"  Roots (A)        : {self.roots_a}")
        lines.append(f"  Roots (B)        : {self.roots_b}")

        if self.nodes_only_in_a:
            lines.append(f"\n  Nodes only in A: {self.nodes_only_in_a}")
        if self.nodes_only_in_b:
            lines.append(f"  Nodes only in B: {self.nodes_only_in_b}")

        if self.node_diffs:
            lines.append("")
            lines.append("-" * 72)
            lines.append("NODE DIFFS")
            lines.append("-" * 72)
            for nd in self.node_diffs:
                lines.append(f"\n  Position: {nd.position}")
                if nd.name_changed:
                    lines.append(f"    Name     : {nd.node_a_name!r} -> {nd.node_b_name!r}")
                else:
                    lines.append(f"    Node     : {nd.node_a_name!r}")
                if nd.class_changed:
                    lines.append(f"    Class    : {nd.class_a} -> {nd.class_b}")
                if nd.kwargs_added:
                    lines.append("    + Added kwargs:")
                    for k, v in sorted(nd.kwargs_added.items()):
                        lines.append(f"        {k} = {repr(v)}")
                if nd.kwargs_removed:
                    lines.append("    - Removed kwargs:")
                    for k, v in sorted(nd.kwargs_removed.items()):
                        lines.append(f"        {k} = {repr(v)}")
                if nd.kwargs_changed:
                    lines.append("    ~ Changed kwargs:")
                    for k, (old, new) in sorted(nd.kwargs_changed.items()):
                        lines.append(f"        {k}: {repr(old)} -> {repr(new)}")
                if nd.children_a_only:
                    lines.append(f"    Children only in A: {nd.children_a_only}")
                if nd.children_b_only:
                    lines.append(f"    Children only in B: {nd.children_b_only}")

        if self.identical:
            lines.append("\nGraphs are identical.")
        lines.append(sep)
        return "\n".join(lines)


def _diff_kwargs(kwargs_a: dict, kwargs_b: dict) -> Tuple[dict, dict, dict]:
    """Return ``(added, removed, changed)`` between two kwarg dicts."""
    all_keys = set(kwargs_a) | set(kwargs_b)
    added: dict = {}
    removed: dict = {}
    changed: dict = {}
    for key in sorted(all_keys):
        in_a = key in kwargs_a
        in_b = key in kwargs_b
        if in_a and not in_b:
            removed[key] = kwargs_a[key]
        elif in_b and not in_a:
            added[key] = kwargs_b[key]
        else:
            try:
                equal = kwargs_a[key] == kwargs_b[key]
            except Exception:
                equal = repr(kwargs_a[key]) == repr(kwargs_b[key])
            if not equal:
                changed[key] = (kwargs_a[key], kwargs_b[key])
    return added, removed, changed


def diff_graphs(graph_a: Graph, graph_b: Graph) -> GraphDiff:
    """Compute a structural + configuration diff between two graphs.

    Performs a parallel DFS walk and compares node names, operator classes,
    operator kwargs, and child topology at each corresponding position.
    """
    nodes_a = collect_nodes(graph_a)
    nodes_b = collect_nodes(graph_b)
    names_a = {n.name for n in nodes_a}
    names_b = {n.name for n in nodes_b}

    result = GraphDiff(
        identical=True,
        structural_match=True,
        node_count_a=len(nodes_a),
        node_count_b=len(nodes_b),
        roots_a=[r.name for r in graph_a.roots],
        roots_b=[r.name for r in graph_b.roots],
        nodes_only_in_a=sorted(names_a - names_b),
        nodes_only_in_b=sorted(names_b - names_a),
    )

    if result.nodes_only_in_a or result.nodes_only_in_b:
        result.identical = False
    if len(graph_a.roots) != len(graph_b.roots):
        result.structural_match = False
        result.identical = False

    visited_pairs: Set[Tuple[int, int]] = set()

    def _compare(node_a: Node, node_b: Node, path: str) -> None:
        pair = (id(node_a), id(node_b))
        if pair in visited_pairs:
            return
        visited_pairs.add(pair)

        nd = NodeDiff(position=path, node_a_name=node_a.name, node_b_name=node_b.name)
        has_diff = False

        if node_a.name != node_b.name:
            nd.name_changed = True
            has_diff = True

        cls_a = _qualified_name(node_a.operator_class)
        cls_b = _qualified_name(node_b.operator_class)
        if cls_a != cls_b:
            nd.class_changed = True
            nd.class_a = cls_a
            nd.class_b = cls_b
            has_diff = True

        added, removed, changed = _diff_kwargs(node_a.operator_kwargs, node_b.operator_kwargs)
        if added or removed or changed:
            nd.kwargs_added = added
            nd.kwargs_removed = removed
            nd.kwargs_changed = changed
            has_diff = True

        children_a_names = [c.name for c in node_a.children]
        children_b_names = [c.name for c in node_b.children]
        if children_a_names != children_b_names:
            nd.children_a_only = [n for n in children_a_names if n not in children_b_names]
            nd.children_b_only = [n for n in children_b_names if n not in children_a_names]
            has_diff = True
            result.structural_match = False

        if has_diff:
            result.identical = False
            result.node_diffs.append(nd)

        children_b_map = {c.name: c for c in node_b.children}
        for child_a in node_a.children:
            child_b = children_b_map.get(child_a.name)
            if child_b is not None:
                _compare(child_a, child_b, f"{path} -> {child_a.name}")

    for i, (ra, rb) in enumerate(zip(graph_a.roots, graph_b.roots)):
        _compare(ra, rb, f"root[{i}]/{ra.name}")

    return result


def print_diff(graph_a: Graph, graph_b: Graph) -> None:
    """Print a human-readable diff between two graphs to stdout."""
    print(diff_graphs(graph_a, graph_b).format())


# ---------------------------------------------------------------------------
# Serialization / deserialization
# ---------------------------------------------------------------------------


def _serialize_node(node: Node) -> dict:
    """Serialize a single node to a JSON-compatible dict."""
    safe_kwargs = {}
    for k, v in node.operator_kwargs.items():
        safe_kwargs[k] = _safe_serialize_value(v)
    return {
        "name": node.name,
        "operator_class": _qualified_name(node.operator_class),
        "operator_kwargs": safe_kwargs,
        "children": [_serialize_node(child) for child in node.children],
    }


def serialize_graph(graph: Graph) -> dict:
    """Serialize a graph to a JSON-compatible dictionary.

    The result can be passed to :func:`json.dumps` (with the
    :class:`_RegistryJSONEncoder`) and later restored via
    :func:`deserialize_graph`.
    """
    return {
        "roots": [_serialize_node(root) for root in graph.roots],
        "metadata": {
            "node_count": node_count(graph),
            "max_depth": max_depth(graph),
            "serialized_at": datetime.now(timezone.utc).isoformat(),
        },
    }


class _PlaceholderOperator(AbstractOperator):
    """Stand-in used when the real operator class cannot be instantiated during deserialization."""

    def __init__(self, original_class: str = "", original_kwargs: Optional[dict] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._original_class = original_class
        self._original_kwargs = original_kwargs or {}

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"PlaceholderOperator for {self._original_class!r} cannot process data. "
            f"The original operator class could not be instantiated."
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _restore_special_values(kwargs: dict) -> dict:
    """Walk a kwargs dict and restore ``__type_ref__``, ``__path__``, etc."""
    cleaned: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, dict):
            if "__type_ref__" in v:
                try:
                    cleaned[k] = _import_class(v["__type_ref__"])
                except ImportError:
                    cleaned[k] = v
                continue
            if "__callable_ref__" in v:
                try:
                    cleaned[k] = _import_class(v["__callable_ref__"])
                except ImportError:
                    cleaned[k] = v
                continue
            if "__path__" in v:
                cleaned[k] = Path(v["__path__"])
                continue
            if "__set__" in v:
                cleaned[k] = set(v["__set__"])
                continue
        cleaned[k] = v
    return cleaned


def _deserialize_node(data: dict) -> Node:
    """Reconstruct a :class:`Node` from its serialized dict."""
    cls = _import_class(data["operator_class"])
    raw_kwargs = data.get("operator_kwargs", {})
    cleaned = _restore_special_values(raw_kwargs)

    try:
        op = cls(**cleaned)
    except Exception:
        op = _PlaceholderOperator(original_class=data["operator_class"], original_kwargs=cleaned)

    node = Node(op, name=data.get("name"), operator_class=cls, operator_kwargs=cleaned)
    for child_data in data.get("children", []):
        child_node = _deserialize_node(child_data)
        node.children.append(child_node)
    return node


def deserialize_graph(data: dict) -> Graph:
    """Reconstruct a :class:`Graph` from a dict produced by :func:`serialize_graph`."""
    graph = Graph()
    for root_data in data.get("roots", []):
        root_node = _deserialize_node(root_data)
        graph.roots.append(root_node)
    return graph


def save_graph(graph: Graph, path: Union[str, Path], *, indent: int = 2) -> Path:
    """Serialize *graph* and write it to a JSON file at *path*.

    Returns the resolved :class:`Path` that was written.
    """
    path = Path(path)
    payload = serialize_graph(graph)
    path.write_text(json.dumps(payload, cls=_RegistryJSONEncoder, indent=indent, default=repr))
    return path


def load_graph(path: Union[str, Path]) -> Graph:
    """Load a graph from a JSON file produced by :func:`save_graph`."""
    path = Path(path)
    payload = json.loads(path.read_text())
    return deserialize_graph(payload)


def clone_graph(graph: Graph) -> Graph:
    """Create a structural deep-copy of *graph* by round-tripping through serialization.

    This produces new ``Node`` / operator instances so modifications to the
    clone do not affect the original.
    """
    return deserialize_graph(serialize_graph(graph))


# ---------------------------------------------------------------------------
# Blueprint — metadata wrapper for a registered graph
# ---------------------------------------------------------------------------


@dataclass
class GraphBlueprint:
    """A named, versioned graph definition held in the registry."""

    name: str
    graph_factory: Callable[[], Graph]
    description: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def build(self) -> Graph:
        """Construct a fresh :class:`Graph` from the stored factory."""
        return self.graph_factory()

    def info(self) -> str:
        """Return a concise multi-line info string (builds the graph once to inspect it)."""
        graph = self.build()
        tag_str = ", ".join(self.tags) if self.tags else "(none)"
        return (
            f"Blueprint: {self.name}\n"
            f"  Version     : {self.version}\n"
            f"  Tags        : {tag_str}\n"
            f"  Description : {self.description}\n"
            f"  Created at  : {self.created_at}\n"
            f"  Updated at  : {self.updated_at}\n"
            f"  {format_graph_summary(graph)}"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class GraphPipelineRegistry:
    """Central registry for golden pipeline graph definitions.

    Stores :class:`GraphBlueprint` objects keyed by name.  Supports
    decorator and imperative registration, building fresh graph instances,
    inspection / pretty-printing, diffing between graphs, kwarg overrides,
    and JSON serialization / deserialization of the entire registry.

    Usage::

        registry = GraphPipelineRegistry()

        @registry.register("my-pipeline", description="Demo", version="1.0")
        def _build():
            return Graph() >> SomeOperator() >> AnotherOperator()

        graph = registry.build("my-pipeline")
        registry.print_graph("my-pipeline")
    """

    def __init__(self) -> None:
        self._blueprints: OrderedDict[str, GraphBlueprint] = OrderedDict()

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        description: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Callable[[Callable[[], Graph]], Callable[[], Graph]]:
        """Decorator that registers a graph factory function.

        Example::

            @registry.register("pdf-extract", description="PDF extraction pipeline")
            def _build():
                return Graph() >> PDFSplitActor() >> PDFExtractionActor()
        """

        def decorator(factory: Callable[[], Graph]) -> Callable[[], Graph]:
            if name in self._blueprints and not overwrite:
                raise ValueError(f"Graph {name!r} is already registered. Pass overwrite=True to replace it.")
            self._blueprints[name] = GraphBlueprint(
                name=name,
                graph_factory=factory,
                description=description,
                version=version,
                tags=tags or [],
            )
            return factory

        return decorator

    def register_graph(
        self,
        name: str,
        factory: Callable[[], Graph],
        *,
        description: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Programmatically register a graph factory (non-decorator form)."""
        if name in self._blueprints and not overwrite:
            raise ValueError(f"Graph {name!r} is already registered. Pass overwrite=True to replace it.")
        self._blueprints[name] = GraphBlueprint(
            name=name,
            graph_factory=factory,
            description=description,
            version=version,
            tags=tags or [],
        )

    def unregister(self, name: str) -> GraphBlueprint:
        """Remove and return the blueprint for *name*.

        Raises ``KeyError`` if *name* is not registered.
        """
        if name not in self._blueprints:
            raise KeyError(f"No graph registered under {name!r}")
        return self._blueprints.pop(name)

    # -- retrieval ----------------------------------------------------------

    def get_blueprint(self, name: str) -> GraphBlueprint:
        """Return the :class:`GraphBlueprint` for *name*.

        Raises ``KeyError`` if not found.
        """
        if name not in self._blueprints:
            raise KeyError(f"No graph registered under {name!r}")
        return self._blueprints[name]

    def build(self, name: str) -> Graph:
        """Build and return a fresh :class:`Graph` from the named blueprint."""
        return self.get_blueprint(name).build()

    def list_names(self) -> List[str]:
        """Return all registered graph names in insertion order."""
        return list(self._blueprints.keys())

    def list_blueprints(self, *, tag: Optional[str] = None) -> List[GraphBlueprint]:
        """Return all blueprints, optionally filtered by *tag*."""
        bps = list(self._blueprints.values())
        if tag is not None:
            bps = [bp for bp in bps if tag in bp.tags]
        return bps

    def __contains__(self, name: str) -> bool:
        return name in self._blueprints

    def __len__(self) -> int:
        return len(self._blueprints)

    def __iter__(self) -> Iterator[str]:
        return iter(self._blueprints)

    def __repr__(self) -> str:
        names = self.list_names()
        return f"GraphPipelineRegistry(graphs={names})"

    # -- inspection ---------------------------------------------------------

    def print_graph(self, name: str, *, show_kwargs: bool = True) -> None:
        """Build and pretty-print the named graph with full details."""
        bp = self.get_blueprint(name)
        print(bp.info())
        print()
        graph = bp.build()
        print(format_graph_tree(graph, show_kwargs=show_kwargs))
        print()
        for node in collect_nodes(graph):
            print(format_node_details(node))
            print()

    def print_summary(self) -> None:
        """Print a compact table of every registered graph."""
        if not self._blueprints:
            print("(registry is empty)")
            return
        header = f"{'Name':35s} {'Version':10s} {'Nodes':>6s} {'Depth':>6s}  {'Tags'}"
        print(header)
        print("-" * len(header))
        for bp in self._blueprints.values():
            graph = bp.build()
            nc = node_count(graph)
            d = max_depth(graph)
            tag_str = ", ".join(bp.tags) if bp.tags else ""
            print(f"{bp.name:35s} {bp.version:10s} {nc:>6d} {d:>6d}  {tag_str}")

    def get_graph_info(self, name: str) -> str:
        """Return the full inspection report for a named graph as a string."""
        graph = self.build(name)
        bp = self.get_blueprint(name)
        return bp.info() + "\n\n" + format_full_report(graph)

    # -- comparison ---------------------------------------------------------

    def diff(self, name_a: str, name_b: str) -> GraphDiff:
        """Build both named graphs and return a :class:`GraphDiff`."""
        return diff_graphs(self.build(name_a), self.build(name_b))

    def print_diff(self, name_a: str, name_b: str) -> None:
        """Print a human-readable diff between two registered graphs."""
        print(self.diff(name_a, name_b).format())

    # -- configuration overrides --------------------------------------------

    def build_with_overrides(self, name: str, overrides: Dict[str, Dict[str, Any]]) -> Graph:
        """Build a graph and apply kwarg overrides to named nodes.

        Parameters
        ----------
        name
            Registered graph name.
        overrides
            ``{node_name: {kwarg_key: new_value, ...}}`` — each matching
            node's ``operator_kwargs`` are updated with the given values.
        """
        graph = self.build(name)
        for node_name, updates in overrides.items():
            update_node_kwargs(graph, node_name, updates, all_matches=True)
        return graph

    # -- serialization (registry-wide) --------------------------------------

    def save_all(self, path: Union[str, Path], *, indent: int = 2) -> Path:
        """Serialize every registered graph to a single JSON file.

        The file contains ``{name: {roots, metadata, blueprint}}`` for each
        registered graph.  Returns the resolved path.
        """
        path = Path(path)
        payload: Dict[str, Any] = {}
        for name, bp in self._blueprints.items():
            graph = bp.build()
            entry = serialize_graph(graph)
            entry["blueprint"] = {
                "description": bp.description,
                "version": bp.version,
                "tags": bp.tags,
                "created_at": bp.created_at,
                "updated_at": bp.updated_at,
            }
            payload[name] = entry
        path.write_text(json.dumps(payload, cls=_RegistryJSONEncoder, indent=indent, default=repr))
        return path

    def load_all(self, path: Union[str, Path], *, overwrite: bool = False) -> List[str]:
        """Load graphs from a JSON file produced by :meth:`save_all`.

        Each loaded graph is registered as a factory that deserializes the
        stored structure.  Returns the list of graph names loaded.
        """
        path = Path(path)
        payload = json.loads(path.read_text())
        loaded: List[str] = []
        for name, entry in payload.items():
            bp_meta = entry.get("blueprint", {})
            graph_data = {k: v for k, v in entry.items() if k != "blueprint"}

            def _factory(_gd: dict = graph_data) -> Graph:
                return deserialize_graph(_gd)

            self.register_graph(
                name,
                _factory,
                description=bp_meta.get("description", ""),
                version=bp_meta.get("version", "1.0.0"),
                tags=bp_meta.get("tags", []),
                overwrite=overwrite,
            )
            loaded.append(name)
        return loaded

    def save_graph(self, name: str, path: Union[str, Path], *, indent: int = 2) -> Path:
        """Serialize a single named graph to a JSON file."""
        graph = self.build(name)
        bp = self.get_blueprint(name)
        payload = serialize_graph(graph)
        payload["blueprint"] = {
            "description": bp.description,
            "version": bp.version,
            "tags": bp.tags,
            "created_at": bp.created_at,
            "updated_at": bp.updated_at,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, cls=_RegistryJSONEncoder, indent=indent, default=repr))
        return path

    def load_graph(self, path: Union[str, Path], *, name: Optional[str] = None, overwrite: bool = False) -> str:
        """Load a single graph from a JSON file and register it.

        If *name* is not provided, the blueprint name stored in the file is
        used (falls back to the file stem).  Returns the registered name.
        """
        path = Path(path)
        payload = json.loads(path.read_text())
        bp_meta = payload.get("blueprint", {})
        graph_data = {k: v for k, v in payload.items() if k != "blueprint"}
        resolved_name = name or bp_meta.get("name") or path.stem

        def _factory(_gd: dict = graph_data) -> Graph:
            return deserialize_graph(_gd)

        self.register_graph(
            resolved_name,
            _factory,
            description=bp_meta.get("description", ""),
            version=bp_meta.get("version", "1.0.0"),
            tags=bp_meta.get("tags", []),
            overwrite=overwrite,
        )
        return resolved_name


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

default_registry = GraphPipelineRegistry()
