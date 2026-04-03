# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for graph_pipeline_registry: walking, inspection, diff, serialization, and registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.graph.graph_pipeline_registry import (
    GraphBlueprint,
    GraphDiff,
    GraphPipelineRegistry,
    _PlaceholderOperator,
    _import_class,
    _qualified_name,
    _safe_serialize_value,
    clone_graph,
    collect_nodes,
    deserialize_graph,
    diff_graphs,
    find_node,
    find_nodes,
    format_full_report,
    format_graph_summary,
    format_graph_tree,
    format_node_details,
    get_node_kwargs,
    leaf_nodes,
    list_all_kwargs,
    load_graph,
    max_depth,
    node_count,
    print_diff,
    print_graph,
    remove_node_kwargs,
    replace_node_kwargs,
    save_graph,
    serialize_graph,
    update_node_kwargs,
    walk_nodes,
)


# ---------------------------------------------------------------------------
# Operator stubs (mirrors test_pipeline_graph.py conventions)
# ---------------------------------------------------------------------------


class AddOp(AbstractOperator):
    def __init__(self, value: int = 1) -> None:
        super().__init__()
        self.value = value

    def preprocess(self, data: Any, **kw: Any) -> Any:
        return data

    def process(self, data: Any, **kw: Any) -> Any:
        return data + self.value

    def postprocess(self, data: Any, **kw: Any) -> Any:
        return data


class MulOp(AbstractOperator):
    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor

    def preprocess(self, data: Any, **kw: Any) -> Any:
        return data

    def process(self, data: Any, **kw: Any) -> Any:
        return data * self.factor

    def postprocess(self, data: Any, **kw: Any) -> Any:
        return data


class AppendOp(AbstractOperator):
    def __init__(self, suffix: str = "_out") -> None:
        super().__init__()
        self.suffix = suffix

    def preprocess(self, data: Any, **kw: Any) -> Any:
        return data

    def process(self, data: Any, **kw: Any) -> Any:
        return str(data) + self.suffix

    def postprocess(self, data: Any, **kw: Any) -> Any:
        return data


# ---------------------------------------------------------------------------
# Helpers for building test graphs
# ---------------------------------------------------------------------------


def _linear_graph() -> Graph:
    """A -> B -> C  (add 1 -> mul 2 -> add 10)."""
    return Graph() >> AddOp(1) >> MulOp(2) >> AddOp(10)


def _fan_out_graph() -> Graph:
    """Root with two children."""
    root = Node(AddOp(1), name="Root")
    left = Node(MulOp(2), name="Left")
    right = Node(MulOp(3), name="Right")
    root.add_child(left)
    root.add_child(right)
    g = Graph()
    g.roots.append(root)
    return g


def _multi_root_graph() -> Graph:
    g = Graph()
    g.add_root(Node(AddOp(10), name="R1"))
    g.add_root(Node(MulOp(5), name="R2"))
    return g


# =====================================================================
# Helper function tests
# =====================================================================


class TestQualifiedName:
    def test_returns_dotted_path(self):
        result = _qualified_name(AddOp)
        assert result.endswith(".AddOp")
        assert "test_graph_pipeline_registry" in result

    def test_builtin_type(self):
        assert _qualified_name(int) == "builtins.int"


class TestImportClass:
    def test_round_trips(self):
        qn = _qualified_name(AddOp)
        cls = _import_class(qn)
        assert cls is AddOp

    def test_raises_for_unqualified(self):
        with pytest.raises(ImportError, match="unqualified"):
            _import_class("AddOp")

    def test_raises_for_missing_attr(self):
        with pytest.raises(ImportError, match="has no attribute"):
            _import_class("builtins.NoSuchClass")

    def test_raises_for_bad_module(self):
        with pytest.raises(ImportError):
            _import_class("no_such_module_xyz.Foo")


class TestSafeSerialize:
    def test_primitive_passes_through(self):
        assert _safe_serialize_value(42) == 42
        assert _safe_serialize_value("hello") == "hello"

    def test_non_serializable_becomes_repr(self):
        result = _safe_serialize_value(object())
        assert isinstance(result, str)
        assert "object" in result


# =====================================================================
# Graph walking tests
# =====================================================================


class TestWalkNodes:
    def test_linear(self):
        graph = _linear_graph()
        pairs = list(walk_nodes(graph))
        assert len(pairs) == 3
        depths = [d for _, d in pairs]
        assert depths == [0, 1, 2]

    def test_fan_out(self):
        graph = _fan_out_graph()
        pairs = list(walk_nodes(graph))
        assert len(pairs) == 3
        names = [n.name for n, _ in pairs]
        assert names[0] == "Root"
        assert set(names[1:]) == {"Left", "Right"}

    def test_multi_root(self):
        graph = _multi_root_graph()
        pairs = list(walk_nodes(graph))
        assert len(pairs) == 2
        assert all(d == 0 for _, d in pairs)

    def test_empty_graph(self):
        assert list(walk_nodes(Graph())) == []

    def test_shared_child_visited_once(self):
        root_a = Node(AddOp(1), name="A")
        root_b = Node(AddOp(2), name="B")
        shared = Node(MulOp(3), name="Shared")
        root_a.add_child(shared)
        root_b.add_child(shared)
        g = Graph()
        g.roots.extend([root_a, root_b])
        nodes = list(walk_nodes(g))
        names = [n.name for n, _ in nodes]
        assert names.count("Shared") == 1


class TestCollectNodes:
    def test_returns_nodes(self):
        graph = _linear_graph()
        nodes = collect_nodes(graph)
        assert len(nodes) == 3
        assert all(isinstance(n, Node) for n in nodes)

    def test_empty(self):
        assert collect_nodes(Graph()) == []


class TestNodeCount:
    def test_linear(self):
        assert node_count(_linear_graph()) == 3

    def test_fan_out(self):
        assert node_count(_fan_out_graph()) == 3

    def test_empty(self):
        assert node_count(Graph()) == 0


class TestMaxDepth:
    def test_linear(self):
        assert max_depth(_linear_graph()) == 2

    def test_fan_out(self):
        assert max_depth(_fan_out_graph()) == 1

    def test_single_node(self):
        g = Graph()
        g.add_root(Node(AddOp()))
        assert max_depth(g) == 0

    def test_empty(self):
        assert max_depth(Graph()) == 0


class TestFindNode:
    def test_finds_existing(self):
        graph = _linear_graph()
        node = find_node(graph, "MulOp")
        assert node is not None
        assert node.name == "MulOp"

    def test_returns_none_missing(self):
        assert find_node(_linear_graph(), "NoSuchNode") is None

    def test_returns_first_match(self):
        g = Graph() >> AddOp(1) >> AddOp(2)
        node = find_node(g, "AddOp")
        assert node is not None
        assert node.operator_kwargs.get("value") == 1


class TestFindNodes:
    def test_finds_all(self):
        g = Graph() >> AddOp(1) >> AddOp(2)
        nodes = find_nodes(g, "AddOp")
        assert len(nodes) == 2

    def test_empty_result(self):
        assert find_nodes(_linear_graph(), "Missing") == []


class TestLeafNodes:
    def test_linear(self):
        graph = _linear_graph()
        leaves = leaf_nodes(graph)
        assert len(leaves) == 1

    def test_fan_out(self):
        graph = _fan_out_graph()
        leaves = leaf_nodes(graph)
        assert len(leaves) == 2
        assert {n.name for n in leaves} == {"Left", "Right"}


class TestGetNodeKwargs:
    def test_returns_kwargs(self):
        graph = _linear_graph()
        kw = get_node_kwargs(graph, "MulOp")
        assert kw["factor"] == 2

    def test_raises_for_missing(self):
        with pytest.raises(KeyError, match="No node named"):
            get_node_kwargs(_linear_graph(), "Missing")

    def test_returns_copy(self):
        graph = _linear_graph()
        kw = get_node_kwargs(graph, "MulOp")
        kw["factor"] = 999
        assert find_node(graph, "MulOp").operator_kwargs["factor"] == 2


class TestListAllKwargs:
    def test_returns_all(self):
        graph = _linear_graph()
        all_kw = list_all_kwargs(graph)
        # Two AddOp nodes share the same name key, so dict has 2 unique names
        assert len(all_kw) == 2
        assert "MulOp" in all_kw
        assert "AddOp" in all_kw


# =====================================================================
# Pretty-print / inspection tests
# =====================================================================


class TestFormatGraphTree:
    def test_linear_tree(self):
        tree = format_graph_tree(_linear_graph())
        assert "AddOp" in tree
        assert "MulOp" in tree
        assert "└──" in tree

    def test_show_kwargs(self):
        tree = format_graph_tree(_linear_graph(), show_kwargs=True)
        assert "value" in tree
        assert "factor" in tree

    def test_no_class(self):
        tree = format_graph_tree(_linear_graph(), show_class=False)
        assert "AddOp" in tree
        assert "test_graph_pipeline_registry" not in tree

    def test_fan_out_tree(self):
        tree = format_graph_tree(_fan_out_graph())
        assert "Root" in tree
        assert "Left" in tree
        assert "Right" in tree

    def test_multi_root_tree(self):
        tree = format_graph_tree(_multi_root_graph())
        assert "R1" in tree
        assert "R2" in tree

    def test_empty_graph(self):
        assert format_graph_tree(Graph()) == ""

    def test_back_ref_shown(self):
        root = Node(AddOp(1), name="A")
        child = Node(MulOp(2), name="B")
        root.add_child(child)
        child.add_child(root)  # cycle
        g = Graph()
        g.roots.append(root)
        tree = format_graph_tree(g)
        assert "back-ref" in tree

    def test_long_value_truncated(self):
        op = AppendOp(suffix="x" * 200)
        g = Graph() >> op
        tree = format_graph_tree(g, show_kwargs=True, max_value_width=50)
        assert "..." in tree


class TestFormatNodeDetails:
    def test_contains_name_and_class(self):
        node = Node(AddOp(5), name="MyAdd")
        details = format_node_details(node)
        assert "MyAdd" in details
        assert "AddOp" in details
        assert "value" in details

    def test_long_value_truncated(self):
        node = Node(AppendOp(suffix="y" * 300))
        details = format_node_details(node)
        assert "..." in details


class TestFormatGraphSummary:
    def test_summary_contents(self):
        summary = format_graph_summary(_linear_graph())
        assert "Graph Summary" in summary
        assert "Roots" in summary
        assert "Leaves" in summary
        assert "Total nodes" in summary
        assert "Max depth" in summary


class TestFormatFullReport:
    def test_includes_summary_and_tree(self):
        report = format_full_report(_linear_graph())
        assert "Graph Summary" in report
        assert "└──" in report
        assert "Operator class" in report


class TestPrintGraph:
    def test_prints_to_stdout(self, capsys):
        print_graph(_linear_graph())
        captured = capsys.readouterr()
        assert "Graph Summary" in captured.out
        assert "AddOp" in captured.out


# =====================================================================
# Configuration update tests
# =====================================================================


class TestUpdateNodeKwargs:
    def test_updates_first_match(self):
        graph = _linear_graph()
        count = update_node_kwargs(graph, "MulOp", {"factor": 10})
        assert count == 1
        assert find_node(graph, "MulOp").operator_kwargs["factor"] == 10

    def test_adds_new_key(self):
        graph = _linear_graph()
        update_node_kwargs(graph, "MulOp", {"new_key": "hello"})
        assert find_node(graph, "MulOp").operator_kwargs["new_key"] == "hello"

    def test_raises_for_missing(self):
        with pytest.raises(KeyError, match="No node named"):
            update_node_kwargs(_linear_graph(), "Missing", {"x": 1})

    def test_all_matches(self):
        g = Graph() >> AddOp(1) >> AddOp(2)
        count = update_node_kwargs(g, "AddOp", {"value": 99}, all_matches=True)
        assert count == 2
        for node in find_nodes(g, "AddOp"):
            assert node.operator_kwargs["value"] == 99

    def test_all_matches_none_found(self):
        g = _linear_graph()
        count = update_node_kwargs(g, "Missing", {"x": 1}, all_matches=True)
        assert count == 0


class TestRemoveNodeKwargs:
    def test_removes_key(self):
        graph = _linear_graph()
        find_node(graph, "MulOp").operator_kwargs["extra"] = "remove_me"
        count = remove_node_kwargs(graph, "MulOp", ["extra"])
        assert count == 1
        assert "extra" not in find_node(graph, "MulOp").operator_kwargs

    def test_missing_key_ignored(self):
        graph = _linear_graph()
        remove_node_kwargs(graph, "MulOp", ["no_such_key"])

    def test_raises_for_missing_node(self):
        with pytest.raises(KeyError):
            remove_node_kwargs(_linear_graph(), "Missing", ["x"])


class TestReplaceNodeKwargs:
    def test_replaces_entirely(self):
        graph = _linear_graph()
        replace_node_kwargs(graph, "MulOp", {"new_factor": 7})
        kw = find_node(graph, "MulOp").operator_kwargs
        assert kw == {"new_factor": 7}
        assert "factor" not in kw

    def test_raises_for_missing(self):
        with pytest.raises(KeyError):
            replace_node_kwargs(_linear_graph(), "Missing", {})


# =====================================================================
# Graph diff tests
# =====================================================================


class TestDiffKwargs:
    def test_identical(self):
        result = diff_graphs(_linear_graph(), _linear_graph())
        assert result.identical is True
        assert result.structural_match is True
        assert result.node_diffs == []

    def test_kwarg_change(self):
        a = _linear_graph()
        b = _linear_graph()
        update_node_kwargs(b, "MulOp", {"factor": 99})
        result = diff_graphs(a, b)
        assert result.identical is False
        assert result.structural_match is True
        changed = [nd for nd in result.node_diffs if nd.kwargs_changed]
        assert len(changed) == 1
        assert "factor" in changed[0].kwargs_changed
        old, new = changed[0].kwargs_changed["factor"]
        assert old == 2
        assert new == 99

    def test_kwarg_added(self):
        a = _linear_graph()
        b = _linear_graph()
        update_node_kwargs(b, "MulOp", {"new_param": True})
        result = diff_graphs(a, b)
        assert result.identical is False
        added = [nd for nd in result.node_diffs if nd.kwargs_added]
        assert len(added) == 1
        assert "new_param" in added[0].kwargs_added

    def test_kwarg_removed(self):
        a = _linear_graph()
        b = _linear_graph()
        remove_node_kwargs(b, "MulOp", ["factor"])
        result = diff_graphs(a, b)
        assert result.identical is False
        removed = [nd for nd in result.node_diffs if nd.kwargs_removed]
        assert len(removed) == 1
        assert "factor" in removed[0].kwargs_removed


class TestDiffStructural:
    def test_different_root_count(self):
        a = _linear_graph()
        b = _multi_root_graph()
        result = diff_graphs(a, b)
        assert result.identical is False
        assert result.structural_match is False

    def test_different_children(self):
        a = _fan_out_graph()
        b_root = Node(AddOp(1), name="Root")
        b_root.add_child(Node(MulOp(2), name="Left"))
        b = Graph()
        b.roots.append(b_root)
        result = diff_graphs(a, b)
        assert result.identical is False
        assert result.structural_match is False
        child_diffs = [nd for nd in result.node_diffs if nd.children_a_only]
        assert len(child_diffs) >= 1

    def test_class_change(self):
        a = Graph() >> AddOp(1)
        b = Graph() >> MulOp(1)
        result = diff_graphs(a, b)
        assert result.identical is False
        class_diffs = [nd for nd in result.node_diffs if nd.class_changed]
        assert len(class_diffs) == 1

    def test_name_change(self):
        a = Graph()
        a.roots.append(Node(AddOp(1), name="Alpha"))
        b = Graph()
        b.roots.append(Node(AddOp(1), name="Beta"))
        result = diff_graphs(a, b)
        assert result.identical is False
        name_diffs = [nd for nd in result.node_diffs if nd.name_changed]
        assert len(name_diffs) == 1


class TestDiffNodes:
    def test_nodes_only_in_a(self):
        a = Graph() >> AddOp(1) >> MulOp(2)
        b = Graph() >> AddOp(1)
        result = diff_graphs(a, b)
        assert "MulOp" in result.nodes_only_in_a

    def test_nodes_only_in_b(self):
        a = Graph() >> AddOp(1)
        b = Graph() >> AddOp(1) >> MulOp(2)
        result = diff_graphs(a, b)
        assert "MulOp" in result.nodes_only_in_b


class TestGraphDiffFormat:
    def test_format_identical(self):
        result = diff_graphs(_linear_graph(), _linear_graph())
        text = result.format()
        assert "GRAPH COMPARISON REPORT" in text
        assert "Identical" in text
        assert "identical" in text.lower()

    def test_format_with_diffs(self):
        a = _linear_graph()
        b = _linear_graph()
        update_node_kwargs(b, "MulOp", {"factor": 99})
        text = diff_graphs(a, b).format()
        assert "Changed kwargs" in text
        assert "factor" in text

    def test_format_with_added(self):
        a = _linear_graph()
        b = _linear_graph()
        update_node_kwargs(b, "MulOp", {"new_key": True})
        text = diff_graphs(a, b).format()
        assert "Added kwargs" in text

    def test_format_with_removed(self):
        a = _linear_graph()
        b = _linear_graph()
        remove_node_kwargs(b, "MulOp", ["factor"])
        text = diff_graphs(a, b).format()
        assert "Removed kwargs" in text


class TestPrintDiff:
    def test_prints_to_stdout(self, capsys):
        print_diff(_linear_graph(), _linear_graph())
        captured = capsys.readouterr()
        assert "GRAPH COMPARISON REPORT" in captured.out


# =====================================================================
# Serialization / deserialization tests
# =====================================================================


class TestSerializeDeserialize:
    def test_round_trip_linear(self):
        original = _linear_graph()
        data = serialize_graph(original)
        restored = deserialize_graph(data)
        assert node_count(restored) == node_count(original)
        names_orig = [n.name for n in collect_nodes(original)]
        names_rest = [n.name for n in collect_nodes(restored)]
        assert names_orig == names_rest

    def test_round_trip_fan_out(self):
        original = _fan_out_graph()
        restored = deserialize_graph(serialize_graph(original))
        assert node_count(restored) == 3
        root = restored.roots[0]
        assert root.name == "Root"
        assert len(root.children) == 2

    def test_metadata_present(self):
        data = serialize_graph(_linear_graph())
        assert "metadata" in data
        assert data["metadata"]["node_count"] == 3
        assert data["metadata"]["max_depth"] == 2
        assert "serialized_at" in data["metadata"]

    def test_kwargs_preserved(self):
        original = _linear_graph()
        restored = deserialize_graph(serialize_graph(original))
        node = find_node(restored, "MulOp")
        assert node.operator_kwargs["factor"] == 2

    def test_operator_class_restored(self):
        original = _linear_graph()
        restored = deserialize_graph(serialize_graph(original))
        node = find_node(restored, "MulOp")
        assert node.operator_class is MulOp

    def test_restored_graph_executes(self):
        original = _linear_graph()
        restored = deserialize_graph(serialize_graph(original))
        assert restored.execute(5) == original.execute(5)

    def test_diff_after_round_trip(self):
        original = _linear_graph()
        restored = deserialize_graph(serialize_graph(original))
        result = diff_graphs(original, restored)
        assert result.identical is True


class TestSaveLoadGraph:
    def test_save_and_load(self, tmp_path):
        graph = _linear_graph()
        path = tmp_path / "graph.json"
        returned = save_graph(graph, path)
        assert returned == path
        assert path.exists()

        restored = load_graph(path)
        assert node_count(restored) == 3
        assert restored.execute(5) == graph.execute(5)

    def test_json_is_valid(self, tmp_path):
        path = tmp_path / "graph.json"
        save_graph(_linear_graph(), path)
        data = json.loads(path.read_text())
        assert "roots" in data

    def test_save_with_path_object(self, tmp_path):
        path = Path(tmp_path / "test.json")
        save_graph(_linear_graph(), path)
        assert path.exists()

    def test_save_with_string_path(self, tmp_path):
        path = str(tmp_path / "test.json")
        save_graph(_linear_graph(), path)
        assert Path(path).exists()


class TestCloneGraph:
    def test_clone_is_independent(self):
        original = _linear_graph()
        cloned = clone_graph(original)
        update_node_kwargs(cloned, "MulOp", {"factor": 999})
        assert find_node(original, "MulOp").operator_kwargs["factor"] == 2

    def test_clone_executes_same(self):
        original = _linear_graph()
        cloned = clone_graph(original)
        assert cloned.execute(5) == original.execute(5)

    def test_clone_structure(self):
        original = _fan_out_graph()
        cloned = clone_graph(original)
        assert node_count(cloned) == node_count(original)
        assert len(cloned.roots[0].children) == 2


class TestPlaceholderOperator:
    def test_preprocess_passthrough(self):
        op = _PlaceholderOperator(original_class="SomeClass")
        assert op.preprocess("data") == "data"

    def test_postprocess_passthrough(self):
        op = _PlaceholderOperator(original_class="SomeClass")
        assert op.postprocess("data") == "data"

    def test_process_raises(self):
        op = _PlaceholderOperator(original_class="SomeClass")
        with pytest.raises(RuntimeError, match="PlaceholderOperator"):
            op.process("data")


class TestSpecialValueSerialization:
    def test_path_round_trip(self, tmp_path):
        node = Node(AddOp(1), name="Test")
        node.operator_kwargs["some_path"] = Path("/usr/local/bin")
        data = serialize_graph(Graph())
        data["roots"] = [
            {
                "name": "Test",
                "operator_class": _qualified_name(AddOp),
                "operator_kwargs": {"value": 1, "some_path": {"__path__": "/usr/local/bin"}},
                "children": [],
            }
        ]
        restored = deserialize_graph(data)
        kw = restored.roots[0].operator_kwargs
        assert isinstance(kw["some_path"], Path)
        assert str(kw["some_path"]) == "/usr/local/bin"

    def test_set_round_trip(self):
        data = {
            "roots": [
                {
                    "name": "Test",
                    "operator_class": _qualified_name(AddOp),
                    "operator_kwargs": {"value": 1, "tags": {"__set__": ["a", "b", "c"]}},
                    "children": [],
                }
            ]
        }
        restored = deserialize_graph(data)
        assert restored.roots[0].operator_kwargs["tags"] == {"a", "b", "c"}

    def test_type_ref_round_trip(self):
        data = {
            "roots": [
                {
                    "name": "Test",
                    "operator_class": _qualified_name(AddOp),
                    "operator_kwargs": {"value": 1, "cls": {"__type_ref__": _qualified_name(MulOp)}},
                    "children": [],
                }
            ]
        }
        restored = deserialize_graph(data)
        assert restored.roots[0].operator_kwargs["cls"] is MulOp


# =====================================================================
# GraphBlueprint tests
# =====================================================================


class TestGraphBlueprint:
    def test_build(self):
        bp = GraphBlueprint(name="test", graph_factory=_linear_graph)
        graph = bp.build()
        assert isinstance(graph, Graph)
        assert node_count(graph) == 3

    def test_build_returns_fresh_instance(self):
        bp = GraphBlueprint(name="test", graph_factory=_linear_graph)
        g1 = bp.build()
        g2 = bp.build()
        assert g1 is not g2

    def test_info(self):
        bp = GraphBlueprint(
            name="test-bp",
            graph_factory=_linear_graph,
            description="A test blueprint",
            version="2.0.0",
            tags=["test", "example"],
        )
        info = bp.info()
        assert "test-bp" in info
        assert "2.0.0" in info
        assert "test, example" in info
        assert "A test blueprint" in info
        assert "Graph Summary" in info

    def test_default_timestamps(self):
        bp = GraphBlueprint(name="t", graph_factory=_linear_graph)
        assert bp.created_at is not None
        assert bp.updated_at is not None


# =====================================================================
# GraphPipelineRegistry tests
# =====================================================================


class TestRegistryRegistration:
    def test_decorator_register(self):
        reg = GraphPipelineRegistry()

        @reg.register("test-pipe", description="test", version="1.0.0")
        def _build():
            return _linear_graph()

        assert "test-pipe" in reg
        assert len(reg) == 1

    def test_imperative_register(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("my-graph", _linear_graph, description="desc")
        assert "my-graph" in reg

    def test_duplicate_raises(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("dup", _linear_graph)
        with pytest.raises(ValueError, match="already registered"):
            reg.register_graph("dup", _linear_graph)

    def test_duplicate_decorator_raises(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("dup", _linear_graph)
        with pytest.raises(ValueError, match="already registered"):

            @reg.register("dup")
            def _build():
                return _linear_graph()

    def test_overwrite(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("ow", _linear_graph)
        reg.register_graph("ow", _fan_out_graph, overwrite=True)
        graph = reg.build("ow")
        assert node_count(graph) == 3
        assert graph.roots[0].name == "Root"

    def test_overwrite_decorator(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("ow", _linear_graph)

        @reg.register("ow", overwrite=True)
        def _build():
            return _fan_out_graph()

        assert reg.build("ow").roots[0].name == "Root"

    def test_unregister(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("gone", _linear_graph)
        bp = reg.unregister("gone")
        assert isinstance(bp, GraphBlueprint)
        assert "gone" not in reg

    def test_unregister_missing_raises(self):
        reg = GraphPipelineRegistry()
        with pytest.raises(KeyError, match="No graph registered"):
            reg.unregister("missing")

    def test_tags(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph, tags=["pdf"])
        reg.register_graph("b", _fan_out_graph, tags=["text"])
        assert len(reg.list_blueprints(tag="pdf")) == 1
        assert reg.list_blueprints(tag="pdf")[0].name == "a"
        assert len(reg.list_blueprints(tag="text")) == 1
        assert len(reg.list_blueprints(tag="missing")) == 0


class TestRegistryRetrieval:
    def test_build(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("test", _linear_graph)
        graph = reg.build("test")
        assert isinstance(graph, Graph)
        assert node_count(graph) == 3

    def test_build_missing_raises(self):
        reg = GraphPipelineRegistry()
        with pytest.raises(KeyError):
            reg.build("missing")

    def test_get_blueprint(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("bp", _linear_graph, description="desc", version="3.0")
        bp = reg.get_blueprint("bp")
        assert bp.name == "bp"
        assert bp.description == "desc"
        assert bp.version == "3.0"

    def test_get_blueprint_missing_raises(self):
        reg = GraphPipelineRegistry()
        with pytest.raises(KeyError):
            reg.get_blueprint("missing")

    def test_list_names(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        reg.register_graph("b", _fan_out_graph)
        assert reg.list_names() == ["a", "b"]

    def test_list_blueprints(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph, tags=["x"])
        reg.register_graph("b", _fan_out_graph, tags=["y"])
        assert len(reg.list_blueprints()) == 2
        assert len(reg.list_blueprints(tag="x")) == 1

    def test_contains(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        assert "a" in reg
        assert "b" not in reg

    def test_len(self):
        reg = GraphPipelineRegistry()
        assert len(reg) == 0
        reg.register_graph("a", _linear_graph)
        assert len(reg) == 1

    def test_iter(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        reg.register_graph("b", _fan_out_graph)
        assert list(reg) == ["a", "b"]

    def test_repr(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        assert "GraphPipelineRegistry" in repr(reg)
        assert "a" in repr(reg)


class TestRegistryInspection:
    def test_print_graph(self, capsys):
        reg = GraphPipelineRegistry()
        reg.register_graph("test", _linear_graph, description="A test")
        reg.print_graph("test")
        captured = capsys.readouterr()
        assert "Blueprint: test" in captured.out
        assert "AddOp" in captured.out

    def test_print_summary(self, capsys):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph, tags=["x"])
        reg.register_graph("b", _fan_out_graph, tags=["y"])
        reg.print_summary()
        captured = capsys.readouterr()
        assert "a" in captured.out
        assert "b" in captured.out
        assert "x" in captured.out

    def test_print_summary_empty(self, capsys):
        reg = GraphPipelineRegistry()
        reg.print_summary()
        captured = capsys.readouterr()
        assert "empty" in captured.out

    def test_get_graph_info(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("test", _linear_graph, description="info test")
        info = reg.get_graph_info("test")
        assert "Blueprint: test" in info
        assert "Graph Summary" in info
        assert "Operator class" in info


class TestRegistryDiff:
    def test_diff_by_name(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        reg.register_graph("b", _fan_out_graph)
        result = reg.diff("a", "b")
        assert isinstance(result, GraphDiff)
        assert result.identical is False

    def test_print_diff(self, capsys):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        reg.register_graph("b", _linear_graph)
        reg.print_diff("a", "b")
        captured = capsys.readouterr()
        assert "GRAPH COMPARISON REPORT" in captured.out


class TestRegistryOverrides:
    def test_build_with_overrides(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("test", _linear_graph)
        graph = reg.build_with_overrides("test", {"MulOp": {"factor": 100}})
        assert find_node(graph, "MulOp").operator_kwargs["factor"] == 100

    def test_original_unmodified(self):
        reg = GraphPipelineRegistry()
        reg.register_graph("test", _linear_graph)
        reg.build_with_overrides("test", {"MulOp": {"factor": 100}})
        fresh = reg.build("test")
        assert find_node(fresh, "MulOp").operator_kwargs["factor"] == 2


class TestRegistrySerialization:
    def test_save_all_and_load_all(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph, description="linear", tags=["l"])
        reg.register_graph("b", _fan_out_graph, description="fan", tags=["f"])

        path = tmp_path / "all.json"
        reg.save_all(path)
        assert path.exists()

        reg2 = GraphPipelineRegistry()
        loaded = reg2.load_all(path)
        assert set(loaded) == {"a", "b"}
        assert node_count(reg2.build("a")) == 3
        assert node_count(reg2.build("b")) == 3

    def test_load_all_overwrite(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        path = tmp_path / "all.json"
        reg.save_all(path)

        reg.load_all(path, overwrite=True)
        assert "a" in reg

    def test_load_all_no_overwrite_raises(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("a", _linear_graph)
        path = tmp_path / "all.json"
        reg.save_all(path)

        with pytest.raises(ValueError, match="already registered"):
            reg.load_all(path, overwrite=False)

    def test_save_graph_single(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("single", _linear_graph, description="single test")
        path = tmp_path / "single.json"
        reg.save_graph("single", path)
        assert path.exists()

        data = json.loads(path.read_text())
        assert "blueprint" in data
        assert data["blueprint"]["description"] == "single test"

    def test_load_graph_single(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("orig", _linear_graph, description="for load")
        path = tmp_path / "orig.json"
        reg.save_graph("orig", path)

        reg2 = GraphPipelineRegistry()
        name = reg2.load_graph(path)
        assert name in reg2
        graph = reg2.build(name)
        assert node_count(graph) == 3

    def test_load_graph_custom_name(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("orig", _linear_graph)
        path = tmp_path / "orig.json"
        reg.save_graph("orig", path)

        reg2 = GraphPipelineRegistry()
        name = reg2.load_graph(path, name="custom-name")
        assert name == "custom-name"
        assert "custom-name" in reg2

    def test_load_graph_overwrite(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("x", _linear_graph)
        path = tmp_path / "x.json"
        reg.save_graph("x", path)
        reg.load_graph(path, name="x", overwrite=True)
        assert "x" in reg

    def test_load_graph_no_overwrite_raises(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph("x", _linear_graph)
        path = tmp_path / "x.json"
        reg.save_graph("x", path)
        with pytest.raises(ValueError, match="already registered"):
            reg.load_graph(path, name="x", overwrite=False)

    def test_blueprint_metadata_preserved(self, tmp_path):
        reg = GraphPipelineRegistry()
        reg.register_graph(
            "meta",
            _linear_graph,
            description="metadata test",
            version="5.0.0",
            tags=["alpha", "beta"],
        )
        path = tmp_path / "meta.json"
        reg.save_all(path)

        reg2 = GraphPipelineRegistry()
        reg2.load_all(path)
        bp = reg2.get_blueprint("meta")
        assert bp.description == "metadata test"
        assert bp.version == "5.0.0"
        assert bp.tags == ["alpha", "beta"]


# =====================================================================
# Edge cases and integration
# =====================================================================


class TestEdgeCases:
    def test_empty_graph_serialize(self):
        g = Graph()
        data = serialize_graph(g)
        assert data["roots"] == []
        restored = deserialize_graph(data)
        assert restored.roots == []

    def test_empty_graph_diff(self):
        result = diff_graphs(Graph(), Graph())
        assert result.identical is True

    def test_single_node_graph(self):
        g = Graph()
        g.add_root(Node(AddOp(42), name="Solo"))
        assert node_count(g) == 1
        assert max_depth(g) == 0
        assert leaf_nodes(g) == collect_nodes(g)

        data = serialize_graph(g)
        restored = deserialize_graph(data)
        assert node_count(restored) == 1
        assert restored.execute(0) == [42]

    def test_deep_chain(self):
        g = Graph()
        g >> AddOp(1) >> AddOp(2) >> AddOp(3) >> AddOp(4) >> AddOp(5)
        assert node_count(g) == 5
        assert max_depth(g) == 4

        restored = deserialize_graph(serialize_graph(g))
        assert restored.execute(0) == g.execute(0)

    def test_full_workflow(self, tmp_path):
        """Register, build, override, diff, save, load, diff again."""
        reg = GraphPipelineRegistry()
        reg.register_graph("base", _linear_graph, description="baseline", version="1.0")

        base = reg.build("base")
        modified = reg.build_with_overrides("base", {"MulOp": {"factor": 10}})

        result = diff_graphs(base, modified)
        assert result.identical is False
        assert result.structural_match is True

        path = tmp_path / "workflow.json"
        reg.save_all(path)

        reg2 = GraphPipelineRegistry()
        reg2.load_all(path)
        restored = reg2.build("base")

        assert node_count(restored) == 3
        assert restored.execute(5) == base.execute(5)

    def test_registry_decorator_returns_original_function(self):
        reg = GraphPipelineRegistry()

        @reg.register("fn-test")
        def my_factory():
            return _linear_graph()

        assert callable(my_factory)
        assert my_factory() is not None
        assert isinstance(my_factory(), Graph)
