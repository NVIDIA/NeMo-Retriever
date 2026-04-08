# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Node, Graph, >> chaining (including auto-wrap), and Executors."""

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.graph import FileListLoaderOperator, MultiTypeExtractOperator, UDFOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.executor import AbstractExecutor, InprocessExecutor, RayDataExecutor
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.params import ExtractParams
from nemo_retriever.utils.ray_resource_hueristics import Resources


# ---------------------------------------------------------------------------
# Concrete operator stubs for testing
# ---------------------------------------------------------------------------
class AddOperator(AbstractOperator):
    """Adds a fixed value to numeric data."""

    def __init__(self, value: int = 1) -> None:
        super().__init__()
        self.value = value

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self.value

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class MultiplyOperator(AbstractOperator):
    """Multiplies data by a fixed factor."""

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data * self.factor

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class AppendOperator(AbstractOperator):
    """Appends a suffix to string data."""

    def __init__(self, suffix: str = "_out") -> None:
        super().__init__()
        self.suffix = suffix

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return str(data) + self.suffix

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class ParamsHolderOperator(AbstractOperator):
    """Operator that stores its constructor arg on a private attribute."""

    def __init__(self, params: dict[str, int]) -> None:
        super().__init__()
        self._params = params

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self._params["value"]

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class CPUAdaptiveAddOperator(AbstractOperator, CPUOperator):
    def __init__(self, value: int = 1) -> None:
        super().__init__()
        self.value = value

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self.value

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class GPUAdaptiveAddOperator(AbstractOperator, GPUOperator):
    def __init__(self, value: int = 1) -> None:
        super().__init__()
        self.value = value

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self.value

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class AdaptiveAddOperator(ArchetypeOperator):
    _cpu_variant_class = CPUAdaptiveAddOperator
    _gpu_variant_class = GPUAdaptiveAddOperator

    def __init__(self, value: int = 1) -> None:
        super().__init__(value=value)
        self.value = value


# =====================================================================
# Node tests
# =====================================================================
class TestNode:
    def test_create_node(self):
        op = AddOperator(5)
        node = Node(op)
        assert node.operator is op
        assert node.name == "AddOperator"
        assert node.children == []

    def test_create_node_custom_name(self):
        node = Node(AddOperator(), name="my_adder")
        assert node.name == "my_adder"

    def test_node_infers_operator_kwargs_from_instance(self):
        node = Node(AddOperator(5))
        assert node.operator_kwargs == {"value": 5}

    def test_node_infers_private_constructor_attrs(self):
        node = Node(ParamsHolderOperator({"value": 9}))
        assert node.operator_kwargs == {"params": {"value": 9}}

    def test_node_rejects_non_operator(self):
        with pytest.raises(TypeError, match="operator must be an AbstractOperator"):
            Node("not_an_operator")

    def test_add_child(self):
        parent = Node(AddOperator())
        child = Node(MultiplyOperator())
        returned = parent.add_child(child)
        assert returned is child
        assert child in parent.children
        assert len(parent.children) == 1

    def test_add_child_auto_wraps_operator(self):
        parent = Node(AddOperator())
        op = MultiplyOperator(3)
        returned = parent.add_child(op)
        assert isinstance(returned, Node)
        assert returned.operator is op
        assert returned in parent.children

    def test_add_child_rejects_invalid_type(self):
        parent = Node(AddOperator())
        with pytest.raises(TypeError, match="Expected a Node, Graph, or AbstractOperator"):
            parent.add_child("not_a_node")

    def test_add_multiple_children(self):
        parent = Node(AddOperator())
        c1 = Node(MultiplyOperator(2))
        c2 = Node(MultiplyOperator(3))
        parent.add_child(c1)
        parent.add_child(c2)
        assert parent.children == [c1, c2]

    def test_repr(self):
        parent = Node(AddOperator(), name="A")
        child = Node(MultiplyOperator(), name="B")
        parent.add_child(child)
        assert "A" in repr(parent)
        assert "B" in repr(parent)


# =====================================================================
# >> operator tests (Node >> Node, Node >> Operator)
# =====================================================================
class TestNodeRshiftChaining:
    def test_simple_chain(self):
        a = Node(AddOperator(), name="A")
        b = Node(MultiplyOperator(), name="B")
        result = a >> b
        assert isinstance(result, Graph)
        assert result.roots == [a]
        assert b in a.children

    def test_triple_chain(self):
        a = Node(AddOperator(1), name="A")
        b = Node(AddOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        a >> b >> c
        assert a.children == [b]
        assert b.children == [c]
        assert c.children == []

    def test_long_chain(self):
        nodes = [Node(AddOperator(i), name=f"N{i}") for i in range(5)]
        nodes[0] >> nodes[1] >> nodes[2] >> nodes[3] >> nodes[4]
        for i in range(4):
            assert nodes[i].children == [nodes[i + 1]]
        assert nodes[4].children == []

    def test_fan_out(self):
        root = Node(AddOperator(), name="root")
        left = Node(MultiplyOperator(2), name="left")
        right = Node(MultiplyOperator(3), name="right")
        root.add_child(left)
        root.add_child(right)
        assert root.children == [left, right]

    def test_diamond_shape(self):
        a = Node(AddOperator(1), name="A")
        b = Node(AddOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        d = Node(AddOperator(4), name="D")
        a.add_child(b)
        a.add_child(c)
        b.add_child(d)
        c.add_child(d)
        assert a.children == [b, c]
        assert b.children == [d]
        assert c.children == [d]

    def test_rshift_returns_graph_with_root(self):
        a = Node(AddOperator(), name="A")
        b = Node(AddOperator(), name="B")
        c = Node(AddOperator(), name="C")
        graph = a >> b
        assert isinstance(graph, Graph)
        assert graph.roots == [a]
        graph2 = graph >> c
        assert graph2 is graph  # same Graph object
        assert graph2.roots == [a]
        assert b.children == [c]

    def test_node_rshift_operator_auto_wraps(self):
        """Node >> AbstractOperator should auto-wrap the operator in a Node."""
        n = Node(AddOperator(1), name="A")
        op = MultiplyOperator(2)
        result = n >> op
        assert isinstance(result, Graph)
        assert result.roots == [n]
        child = n.children[0]
        assert child.operator is op

    def test_node_rshift_operator_chain(self):
        """Node >> op1 >> op2 should chain all three."""
        n = Node(AddOperator(1), name="A")
        op1 = MultiplyOperator(2)
        op2 = AddOperator(10)
        graph = n >> op1 >> op2
        assert isinstance(graph, Graph)
        assert graph.roots == [n]
        assert len(n.children) == 1
        mid = n.children[0]
        assert mid.operator is op1
        assert len(mid.children) == 1
        assert mid.children[0].operator is op2


# =====================================================================
# >> operator tests (Operator >> Operator)
# =====================================================================
class TestOperatorRshiftChaining:
    def test_operator_rshift_operator(self):
        """op_a >> op_b should create two Nodes and return a Graph."""
        op_a = AddOperator(1)
        op_b = MultiplyOperator(2)
        result = op_a >> op_b
        assert isinstance(result, Graph)
        assert result.roots[0].operator is op_a
        assert result.roots[0].children[0].operator is op_b

    def test_operator_rshift_node(self):
        """operator >> Node should auto-wrap the operator and return a Graph."""
        op = AddOperator(1)
        n = Node(MultiplyOperator(2), name="B")
        result = op >> n
        assert isinstance(result, Graph)
        assert result.roots[0].children[0] is n

    def test_operator_triple_chain(self):
        """op_a >> op_b >> op_c should chain all three via Graph."""
        op_a = AddOperator(1)
        op_b = MultiplyOperator(2)
        op_c = AddOperator(10)
        graph = op_a >> op_b >> op_c
        assert isinstance(graph, Graph)
        assert graph.roots[0].operator is op_a
        leaf = graph.roots[0].children[0].children[0]
        assert leaf.operator is op_c


# =====================================================================
# Graph tests
# =====================================================================
class TestGraph:
    def test_create_empty_graph(self):
        g = Graph()
        assert g.roots == []

    def test_add_root(self):
        g = Graph()
        node = Node(AddOperator())
        returned = g.add_root(node)
        assert returned is node
        assert g.roots == [node]

    def test_add_root_auto_wraps_operator(self):
        g = Graph()
        op = AddOperator(5)
        returned = g.add_root(op)
        assert isinstance(returned, Node)
        assert returned.operator is op
        assert returned in g.roots

    def test_add_root_rejects_invalid_type(self):
        g = Graph()
        with pytest.raises(TypeError, match="Expected a Node, Graph, or AbstractOperator"):
            g.add_root("not_a_node")

    def test_multiple_roots(self):
        g = Graph()
        r1 = Node(AddOperator(10), name="R1")
        r2 = Node(AddOperator(20), name="R2")
        g.add_root(r1)
        g.add_root(r2)
        assert g.roots == [r1, r2]

    def test_add_chain(self):
        g = Graph()
        a = Node(AddOperator(1), name="A")
        b = Node(AddOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        g.add_chain(a, b, c)
        assert g.roots == [a]
        assert a.children == [b]
        assert b.children == [c]

    def test_add_chain_with_operators(self):
        """add_chain should accept bare operators and auto-wrap them."""
        g = Graph()
        op_a = AddOperator(1)
        op_b = MultiplyOperator(2)
        g.add_chain(op_a, op_b)
        assert len(g.roots) == 1
        root = g.roots[0]
        assert root.operator is op_a
        assert len(root.children) == 1
        assert root.children[0].operator is op_b

    def test_add_chain_mixed(self):
        """add_chain should accept a mix of Nodes and operators."""
        g = Graph()
        n = Node(AddOperator(1), name="A")
        op = MultiplyOperator(2)
        g.add_chain(n, op)
        assert g.roots[0] is n
        assert n.children[0].operator is op

    def test_add_chain_empty(self):
        g = Graph()
        g.add_chain()
        assert g.roots == []

    def test_add_chain_single_node(self):
        g = Graph()
        a = Node(AddOperator())
        g.add_chain(a)
        assert g.roots == [a]
        assert a.children == []

    def test_graph_rshift_adds_root_when_empty(self):
        """graph >> op should add op as root when graph is empty."""
        g = Graph()
        op = AddOperator(5)
        result = g >> op
        assert result is g  # returns self
        assert len(g.roots) == 1
        assert g.roots[0].operator is op

    def test_graph_rshift_chains_to_leaves(self):
        """graph >> op should chain op to the tail node."""
        g = Graph()
        root = g.add_root(AddOperator(1))
        g._tail = root
        op = MultiplyOperator(2)
        result = g >> op
        assert result is g
        assert len(root.children) == 1
        assert root.children[0].operator is op

    def test_graph_rshift_multiple_leaves(self):
        """graph >> op should add op as child of the tail."""
        g = Graph()
        root = Node(AddOperator(1))
        left = Node(MultiplyOperator(2))
        right = Node(MultiplyOperator(3))
        root.add_child(left)
        root.add_child(right)
        g.add_root(root)
        g._tail = right  # set tail to right
        new_op = AddOperator(100)
        result = g >> new_op
        assert result is g
        # Appended to tail (right)
        assert len(right.children) == 1
        assert right.children[0].operator is new_op

    def test_graph_rshift_sequential(self):
        """graph >> op1 >> op2 should build a chain."""
        g = Graph()
        g >> AddOperator(1) >> MultiplyOperator(2) >> AddOperator(10)
        assert len(g.roots) == 1
        root = g.roots[0]
        assert isinstance(root.operator, AddOperator)
        mid = root.children[0]
        assert isinstance(mid.operator, MultiplyOperator)
        leaf = mid.children[0]
        assert isinstance(leaf.operator, AddOperator)
        assert leaf.operator.value == 10

    def test_repr(self):
        g = Graph()
        g.add_root(Node(AddOperator(), name="A"))
        assert "A" in repr(g)


# =====================================================================
# Graph.execute tests
# =====================================================================
class TestGraphExecute:
    def test_resolve_returns_clone_with_concrete_operator_class(self):
        g = Graph() >> AdaptiveAddOperator(5)

        resolved = g.resolve(Resources(cpu_count=8, gpu_count=0))

        assert resolved is not g
        assert resolved.roots[0].operator_class is CPUAdaptiveAddOperator
        assert g.roots[0].operator_class is AdaptiveAddOperator

    def test_execute_resolves_archetypes_locally(self, monkeypatch):
        resources = Resources(cpu_count=8, gpu_count=0)
        monkeypatch.setattr("nemo_retriever.graph.operator_resolution.gather_local_resources", lambda: resources)
        monkeypatch.setattr("nemo_retriever.graph.operator_archetype.gather_local_resources", lambda: resources)

        g = Graph() >> AdaptiveAddOperator(5)

        assert g.execute(7) == [12]

    def test_single_node(self):
        g = Graph()
        g.add_root(Node(AddOperator(10)))
        results = g.execute(5)
        assert results == [15]

    def test_linear_chain(self):
        a = Node(AddOperator(1), name="A")
        b = Node(MultiplyOperator(2), name="B")
        a >> b
        g = Graph()
        g.add_root(a)
        results = g.execute(3)
        assert results == [8]

    def test_triple_chain(self):
        a = Node(AddOperator(1))
        b = Node(MultiplyOperator(2))
        c = Node(AddOperator(10))
        a >> b >> c
        g = Graph()
        g.add_root(a)
        results = g.execute(5)
        assert results == [22]

    def test_fan_out(self):
        root = Node(AddOperator(1))
        left = Node(MultiplyOperator(2))
        right = Node(MultiplyOperator(3))
        root >> left
        root >> right
        g = Graph()
        g.add_root(root)
        results = g.execute(4)
        assert sorted(results) == [10, 15]

    def test_fan_out_with_continued_chain(self):
        root = Node(AddOperator(1))
        left = Node(MultiplyOperator(2))
        right = Node(MultiplyOperator(3))
        leaf_left = Node(AddOperator(100))
        root >> left >> leaf_left
        root >> right
        g = Graph()
        g.add_root(root)
        results = g.execute(0)
        assert sorted(results) == [3, 102]

    def test_multiple_roots(self):
        g = Graph()
        g.add_root(Node(AddOperator(10)))
        g.add_root(Node(MultiplyOperator(5)))
        results = g.execute(2)
        assert sorted(results) == [10, 12]

    def test_execute_with_string_data(self):
        a = Node(AppendOperator("_A"))
        b = Node(AppendOperator("_B"))
        a >> b
        g = Graph()
        g.add_root(a)
        results = g.execute("start")
        assert results == ["start_A_B"]

    def test_execute_with_add_chain(self):
        a = Node(AddOperator(1))
        b = Node(MultiplyOperator(3))
        c = Node(AddOperator(5))
        g = Graph()
        g.add_chain(a, b, c)
        results = g.execute(2)
        assert results == [14]

    def test_diamond_execute(self):
        a = Node(AddOperator(1))
        b = Node(MultiplyOperator(2))
        c = Node(MultiplyOperator(3))
        d = Node(AddOperator(100))
        a >> b
        a >> c
        b >> d
        c >> d
        g = Graph()
        g.add_root(a)
        results = g.execute(1)
        assert sorted(results) == [104, 106]

    def test_execute_graph_built_with_rshift(self):
        """Graph built entirely with >> should execute correctly."""
        g = Graph()
        g >> AddOperator(1) >> MultiplyOperator(3) >> AddOperator(5)
        results = g.execute(2)
        # 2 -> +1=3 -> *3=9 -> +5=14
        assert results == [14]

    def test_execute_auto_wrapped_add_chain(self):
        """add_chain with bare operators should execute correctly."""
        g = Graph()
        g.add_chain(AddOperator(10), MultiplyOperator(2))
        results = g.execute(5)
        # 5 -> +10=15 -> *2=30
        assert results == [30]

    def test_custom_udf_operator_in_chain(self):
        """Ensure user-defined function operator works in a larger graph chain."""

        # UDF: multiply by 4 (in process), then add 3, then append suffix
        def multiply_by_four(x):
            return x * 4

        udf = UDFOperator(multiply_by_four, name="MultiplyByFour")
        a = AddOperator(3)
        b = MultiplyOperator(2)
        c = AppendOperator("_done")

        graph = udf >> a >> b >> c

        # 1 -> *4=4 -> +3=7 -> *2=14 -> _done
        results = graph.execute(1)
        assert results == ["14_done"]


# =====================================================================
# MultiTypeExtractOperator tests
# =====================================================================
class TestMultiTypeExtractOperator:
    def test_group_files_by_type(self):
        """Test file grouping logic."""

        op = MultiTypeExtractOperator()

        # Mock folder with mixed files
        files = [
            "/folder/test.pdf",
            "/folder/image.png",
            "/folder/text.txt",
            "/folder/page.html",
            "/folder/audio.mp3",
            "/folder/video.mp4",
            "/folder/unknown.xyz",
        ]

        grouped = op.preprocess(files)

        assert grouped["pdf"] == ["/folder/test.pdf"]
        assert grouped["image"] == ["/folder/image.png"]
        assert grouped["text"] == ["/folder/text.txt"]
        assert grouped["html"] == ["/folder/page.html"]
        assert grouped["audio"] == ["/folder/audio.mp3"]
        assert grouped["video"] == ["/folder/video.mp4"]

    def test_preprocess_folder_path(self):
        """Test preprocessing with folder path."""
        from unittest.mock import patch
        from pathlib import Path

        op = MultiTypeExtractOperator()

        with patch("pathlib.Path.rglob") as mock_rglob, patch("pathlib.Path.is_file", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ):

            mock_rglob.return_value = [Path("/folder/file.pdf")]
            grouped = op.preprocess("/folder")

            assert grouped["pdf"] == ["/folder/file.pdf"]

    def test_process_empty_groups(self):
        """Test process with no files."""
        op = MultiTypeExtractOperator()
        grouped = {"pdf": [], "image": [], "text": [], "html": [], "audio": [], "video": []}
        result = op.process(grouped)
        assert result == []

    def test_detection_pipeline_resolves_suboperators_through_archetype_resolution(self, monkeypatch):
        from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractCPUActor
        from nemo_retriever.utils.ray_resource_hueristics import Resources

        calls = []

        class _IdentityStage:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self, data):
                return data

        def _fake_resolve(operator_class, resources, operator_kwargs=None):
            calls.append((operator_class.__name__, resources))
            return _IdentityStage

        monkeypatch.setattr("nemo_retriever.graph.multi_type_extract_operator.resolve_operator_class", _fake_resolve)
        monkeypatch.setattr(
            "nemo_retriever.graph.multi_type_extract_operator.gather_local_resources",
            lambda: Resources(cpu_count=8, gpu_count=1),
        )

        op = MultiTypeExtractCPUActor(
            extraction_mode="image",
            extract_params=ExtractParams(
                method="ocr",
                extract_text=True,
                extract_tables=True,
                use_table_structure=True,
                extract_charts=True,
                use_graphic_elements=True,
                extract_infographics=True,
            ),
        )

        batch_df = pd.DataFrame({"page_image": ["x"]})
        result = op._run_detection_pipeline(batch_df)

        pd.testing.assert_frame_equal(result, batch_df)
        assert [name for name, _resources in calls] == [
            "PageElementDetectionActor",
            "TableStructureActor",
            "GraphicElementsActor",
            "OCRActor",
        ]
        assert len({id(resources) for _name, resources in calls}) == 1

    def test_parse_pipeline_resolves_nemotron_parse_through_archetype_resolution(self, monkeypatch):
        from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractCPUActor
        from nemo_retriever.utils.ray_resource_hueristics import Resources

        calls = []

        class _IdentityStage:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self, data):
                return data

        monkeypatch.setattr(
            "nemo_retriever.graph.multi_type_extract_operator.DocToPdfConversionActor.run",
            lambda self, data: data,
        )
        monkeypatch.setattr(
            "nemo_retriever.graph.multi_type_extract_operator.PDFSplitActor.run",
            lambda self, data: data,
        )

        def _fake_resolve(operator_class, resources, operator_kwargs=None):
            calls.append((operator_class.__name__, resources))
            return _IdentityStage

        monkeypatch.setattr("nemo_retriever.graph.multi_type_extract_operator.resolve_operator_class", _fake_resolve)
        monkeypatch.setattr(
            "nemo_retriever.graph.multi_type_extract_operator.gather_local_resources",
            lambda: Resources(cpu_count=8, gpu_count=1),
        )

        op = MultiTypeExtractCPUActor(
            extraction_mode="pdf",
            extract_params=ExtractParams(method="nemotron_parse"),
        )

        batch_df = pd.DataFrame({"path": ["/tmp/test.pdf"]})
        result = op._run_pdf_pipeline(batch_df)

        pd.testing.assert_frame_equal(result, batch_df)
        assert [name for name, _resources in calls] == ["NemotronParseActor"]


class TestFileListLoaderOperator:
    def test_loads_files_into_path_and_bytes_dataframe(self, tmp_path) -> None:
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text("alpha", encoding="utf-8")
        second.write_text("beta", encoding="utf-8")

        op = FileListLoaderOperator()
        result = op.run([str(first), str(second)])

        assert list(result.columns) == ["path", "bytes"]
        assert result["path"].tolist() == [str(first.resolve()), str(second.resolve())]
        assert result["bytes"].tolist() == [b"alpha", b"beta"]

    def test_returns_empty_dataframe_for_missing_files(self, tmp_path) -> None:
        op = FileListLoaderOperator()
        result = op.run([str(tmp_path / "missing.txt")])

        assert list(result.columns) == ["path", "bytes"]
        assert result.empty


# =====================================================================
# AbstractExecutor tests
# =====================================================================
class TestAbstractExecutor:
    def test_cannot_instantiate_directly(self):
        g = Graph()
        with pytest.raises(TypeError):
            AbstractExecutor(g)

    def test_rejects_non_graph(self):
        class ConcreteExecutor(AbstractExecutor):
            def ingest(self, data, **kw):
                return None

        with pytest.raises(TypeError, match="graph must be a Graph"):
            ConcreteExecutor("not_a_graph")

    def test_concrete_subclass(self):
        class ConcreteExecutor(AbstractExecutor):
            def ingest(self, data, **kw):
                return self.graph.execute(data, **kw)

        g = Graph()
        g.add_chain(AddOperator(1), MultiplyOperator(3))
        executor = ConcreteExecutor(g)
        assert executor.ingest(2) == [9]  # (2+1)*3


# =====================================================================
# RayDataExecutor tests
# =====================================================================
class TestRayDataExecutor:
    def test_inherits_abstract_executor(self):
        assert issubclass(RayDataExecutor, AbstractExecutor)

    def test_instantiation(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = RayDataExecutor(g)
        assert executor.graph is g

    def test_linearize_empty(self):
        g = Graph()
        assert RayDataExecutor._linearize(g) == []

    def test_linearize_single_node(self):
        g = Graph()
        n = Node(AddOperator(1), name="A")
        g.add_root(n)
        result = RayDataExecutor._linearize(g)
        assert result == [n]

    def test_linearize_chain(self):
        g = Graph()
        a = Node(AddOperator(1), name="A")
        b = Node(MultiplyOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        a >> b >> c
        g.add_root(a)
        result = RayDataExecutor._linearize(g)
        assert result == [a, b, c]

    def test_linearize_rejects_multiple_roots(self):
        g = Graph()
        g.add_root(Node(AddOperator(1)))
        g.add_root(Node(AddOperator(2)))
        with pytest.raises(ValueError, match="single-root"):
            RayDataExecutor._linearize(g)

    def test_linearize_rejects_fan_out(self):
        g = Graph()
        root = Node(AddOperator(1))
        root >> Node(AddOperator(2))
        root >> Node(AddOperator(3))
        g.add_root(root)
        with pytest.raises(ValueError, match="fan-out"):
            RayDataExecutor._linearize(g)

    def test_node_overrides_stored(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        overrides = {"AddOperator": {"batch_size": 16, "num_gpus": 0.5}}
        executor = RayDataExecutor(g, node_overrides=overrides)
        assert executor._node_overrides == overrides

    def test_ingest_rejects_invalid_data_type(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = RayDataExecutor(g)
        with pytest.raises(TypeError, match="data must be"):
            executor.ingest(12345)


# ---------------------------------------------------------------------------
# InprocessExecutor tests
# ---------------------------------------------------------------------------
class TestInprocessExecutor:
    def test_inherits_abstract_executor(self):
        assert issubclass(InprocessExecutor, AbstractExecutor)

    def test_instantiation(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = InprocessExecutor(g)
        assert executor.graph is g

    def test_rejects_non_graph(self):
        with pytest.raises(TypeError, match="graph must be a Graph"):
            InprocessExecutor("not_a_graph")

    def test_linearize_empty(self):
        g = Graph()
        assert InprocessExecutor._linearize(g) == []

    def test_linearize_single_node(self):
        g = Graph()
        n = Node(AddOperator(1), name="A")
        g.add_root(n)
        result = InprocessExecutor._linearize(g)
        assert result == [n]

    def test_linearize_chain(self):
        g = Graph()
        a = Node(AddOperator(1), name="A")
        b = Node(MultiplyOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        a >> b >> c
        g.add_root(a)
        result = InprocessExecutor._linearize(g)
        assert result == [a, b, c]

    def test_linearize_rejects_multiple_roots(self):
        g = Graph()
        g.add_root(Node(AddOperator(1)))
        g.add_root(Node(AddOperator(2)))
        with pytest.raises(ValueError, match="single-root"):
            InprocessExecutor._linearize(g)

    def test_linearize_rejects_fan_out(self):
        g = Graph()
        root = Node(AddOperator(1))
        root >> Node(AddOperator(2))
        root >> Node(AddOperator(3))
        g.add_root(root)
        with pytest.raises(ValueError, match="fan-out"):
            InprocessExecutor._linearize(g)

    def test_ingest_dataframe(self):
        """Test ingest with a pandas DataFrame passed directly."""
        import pandas as pd

        g = Graph()
        n_add = Node(
            AddOperator(10),
            name="Add10",
            operator_class=AddOperator,
            operator_kwargs={"value": 10},
        )
        n_mul = Node(
            MultiplyOperator(3),
            name="Mul3",
            operator_class=MultiplyOperator,
            operator_kwargs={"factor": 3},
        )
        n_add >> n_mul
        g.add_root(n_add)

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))
        # AddOperator adds 10 -> DataFrame({"val": [15]})
        # MultiplyOperator multiplies by 3 -> DataFrame({"val": [45]})
        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 45

    def test_ingest_single_chain(self):
        """Test a single-operator graph with a DataFrame."""
        import pandas as pd

        g = Graph()
        n = Node(
            AddOperator(7),
            name="Add7",
            operator_class=AddOperator,
            operator_kwargs={"value": 7},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [3]}))
        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 10

    def test_ingest_rejects_invalid_data_type(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = InprocessExecutor(g)
        with pytest.raises(TypeError, match="data must be"):
            executor.ingest(12345)

    def test_ingest_file_paths(self, tmp_path):
        """Test ingest loads files from paths into a DataFrame with bytes/path columns."""
        import pandas as pd

        # Create a simple operator that returns the DataFrame as-is
        class IdentityOperator(AbstractOperator):
            def preprocess(self, data, **kw):
                return data

            def process(self, data, **kw):
                return data

            def postprocess(self, data, **kw):
                return data

        # Write a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        g = Graph()
        n = Node(
            IdentityOperator(),
            name="Identity",
            operator_class=IdentityOperator,
            operator_kwargs={},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest([str(test_file)])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["bytes"] == b"hello world"
        assert "path" in result.columns

    def test_ingest_glob_pattern(self, tmp_path):
        """Test ingest expands glob patterns."""
        import pandas as pd

        class IdentityOperator(AbstractOperator):
            def preprocess(self, data, **kw):
                return data

            def process(self, data, **kw):
                return data

            def postprocess(self, data, **kw):
                return data

        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")

        g = Graph()
        n = Node(
            IdentityOperator(),
            name="Identity",
            operator_class=IdentityOperator,
            operator_kwargs={},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest([str(tmp_path / "*.txt")])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_uses_operator_kwargs_for_construction(self):
        """Test that InprocessExecutor constructs operators from operator_kwargs, not the instance."""
        import pandas as pd

        g = Graph()
        # The instance has value=1, but operator_kwargs says value=100
        n = Node(
            AddOperator(1),
            name="Add",
            operator_class=AddOperator,
            operator_kwargs={"value": 100},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))
        # Should use value=100 from operator_kwargs, not value=1 from instance
        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 105

    def test_infers_operator_kwargs_for_construction(self):
        """Test that InprocessExecutor can reconstruct from operator instance state."""
        import pandas as pd

        g = Graph()
        g.add_root(Node(AddOperator(7), name="Add"))

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))

        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 12

    def test_infers_private_constructor_attrs_for_construction(self):
        """Test reconstruction when the constructor arg is stored on a private attribute."""
        import pandas as pd

        g = Graph()
        g.add_root(Node(ParamsHolderOperator({"value": 8}), name="ParamsHolder"))

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))

        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 13
