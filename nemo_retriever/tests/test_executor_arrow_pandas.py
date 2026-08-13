# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Ray's Arrow-to-pandas operator boundary."""

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from ray.data.block import BlockAccessor
from ray.data import DataContext
from ray.data.extensions import TensorArray

from nemo_retriever.graph.executor import (
    _ArrowPandasOperatorAdapter,
    _preserves_pandas_output,
    _requires_stable_pandas_blocks,
)
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows, explode_content_to_rows
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.extract.txt.ray_data import TextChunkCPUActor
from nemo_retriever.operators.graph_ops.custom_operator import UDFOperator


class _PassthroughOperator(AbstractOperator):
    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def test_adapter_compacts_sliced_nested_arrow_columns() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "metadata": {
                    "has_text": True,
                    "source_path": "document.pdf",
                    "error": None,
                },
                "text": f"page {page_number}",
            }
            for page_number in range(4)
        ]
    )

    result = _ArrowPandasOperatorAdapter(_PassthroughOperator, {})(table.slice(2, 1))
    roundtripped = pa.Table.from_pandas(result, preserve_index=False)

    roundtripped.validate(full=True)
    assert isinstance(result.dtypes["text"], pd.ArrowDtype)


def test_adapter_preserves_ray_pandas_conversion_policy() -> None:
    context = DataContext.get_current()
    original = context.enable_arrow_backed_pandas_conversion
    context.enable_arrow_backed_pandas_conversion = False
    try:
        result = _ArrowPandasOperatorAdapter(_PassthroughOperator, {})(pa.table({"value": [1, 2]}))
    finally:
        context.enable_arrow_backed_pandas_conversion = original

    assert not isinstance(result.dtypes["value"], pd.ArrowDtype)


def test_adapter_normalizes_pickled_object_columns_for_pandas_row_operations() -> None:
    table = BlockAccessor.batch_to_block(
        pd.DataFrame(
            {
                "text": ["first", "second", "third"],
                "tables": [np.array([], dtype=object) for _ in range(3)],
            }
        )
    )

    result = _ArrowPandasOperatorAdapter(_PassthroughOperator, {})(table)

    assert result["tables"].dtype == object
    assert all(isinstance(value, np.ndarray) and value.size == 0 for value in result["tables"])
    assert result.apply(lambda row: row["text"], axis=1).tolist() == ["first", "second", "third"]


def test_adapter_preserves_numeric_numpy_arrays() -> None:
    table = BlockAccessor.batch_to_block(pd.DataFrame({"embedding": [np.array([0.1, 0.2])]}))

    result = _ArrowPandasOperatorAdapter(_PassthroughOperator, {})(table)

    assert isinstance(result.iloc[0]["embedding"], np.ndarray)
    np.testing.assert_array_equal(result.iloc[0]["embedding"], np.array([0.1, 0.2]))


def test_adapter_does_not_mutate_output_block_context() -> None:
    context = DataContext.get_current()
    original_arrow_format = context.batch_to_block_arrow_format
    original_tensor_casting = context.enable_tensor_extension_casting
    context.batch_to_block_arrow_format = True
    context.enable_tensor_extension_casting = True
    try:
        _ArrowPandasOperatorAdapter(_PassthroughOperator, {})(pa.table({"value": [1, 2]}))

        assert context.batch_to_block_arrow_format
        assert context.enable_tensor_extension_casting
    finally:
        context.batch_to_block_arrow_format = original_arrow_format
        context.enable_tensor_extension_casting = original_tensor_casting


def test_only_opted_in_operators_preserve_pandas_output() -> None:
    opted_in = UDFOperator(lambda frame: frame, preserve_pandas_output=True)
    default = UDFOperator(lambda frame: frame)

    assert _preserves_pandas_output(type(opted_in), opted_in.get_constructor_kwargs())
    assert not _preserves_pandas_output(type(default), default.get_constructor_kwargs())
    assert _preserves_pandas_output(_PassthroughOperator, {"preserve_pandas_output": True})


def test_text_chunk_operator_preserves_heterogeneous_pandas_output() -> None:
    assert _preserves_pandas_output(TextChunkCPUActor, {})


def test_only_opted_in_graphs_require_stable_pandas_blocks() -> None:
    stable_graph = Graph() >> UDFOperator(lambda frame: frame, preserve_pandas_output=True)
    default_graph = Graph() >> UDFOperator(lambda frame: frame)
    text_chunk_graph = Graph() >> TextChunkCPUActor()

    assert _requires_stable_pandas_blocks([stable_graph.roots[0]])
    assert _requires_stable_pandas_blocks([text_chunk_graph.roots[0]])
    assert not _requires_stable_pandas_blocks([default_graph.roots[0]])


def test_adapter_preserves_multimodal_content_semantics() -> None:
    table = BlockAccessor.batch_to_block(
        pd.DataFrame(
            {
                "text": ["page text"],
                "table": [np.array([{"text": "table text"}], dtype=object)],
                "chart": [np.array([{"text": "chart text"}], dtype=object)],
            }
        )
    )

    page_result = _ArrowPandasOperatorAdapter(
        UDFOperator,
        {
            "fn": partial(collapse_content_to_page_rows, modality="text"),
            "name": "CollapseContentToPageRows",
        },
    )(table)
    element_result = _ArrowPandasOperatorAdapter(
        UDFOperator,
        {
            "fn": partial(explode_content_to_rows, modality="text"),
            "name": "ExplodeContentToRows",
        },
    )(table)

    assert page_result["text"].tolist() == ["page text\n\ntable text\n\nchart text"]
    assert element_result["text"].tolist() == ["page text", "table text", "chart text"]
    assert element_result["_content_type"].tolist() == ["text", "table", "chart"]
    pa.Table.from_pandas(page_result, preserve_index=False).validate(full=True)
    pa.Table.from_pandas(element_result, preserve_index=False).validate(full=True)


def test_explode_normalizes_numpy_bbox_for_pandas_blocks() -> None:
    result = explode_content_to_rows(
        pd.DataFrame(
            {
                "text": ["page text"],
                "table": [[{"text": "cell text", "bbox_xyxy_norm": np.array([0.1, 0.2, 0.8, 0.9])}]],
            }
        ),
        modality="text",
    )

    assert result.loc[result["_content_type"] == "table", "_bbox_xyxy_norm"].iloc[0] == [
        0.1,
        0.2,
        0.8,
        0.9,
    ]
    assert isinstance(result.iloc[0]["table"], list)


def test_explode_normalizes_ray_tensor_bbox_for_pandas_blocks() -> None:
    bbox = TensorArray([np.array([0.1, 0.2, 0.8, 0.9])])[0]
    result = explode_content_to_rows(
        pd.DataFrame({"text": ["page text"], "table": [[{"text": "cell text", "bbox_xyxy_norm": bbox}]]}),
        modality="text",
    )

    value = result.loc[result["_content_type"] == "table", "_bbox_xyxy_norm"].iloc[0]
    assert value == [0.1, 0.2, 0.8, 0.9]
    assert not isinstance(value, np.ndarray)


def test_explode_normalizes_bbox_when_content_columns_are_absent() -> None:
    bbox = TensorArray([np.array([0.1, 0.2, 0.8, 0.9])])[0]
    result = explode_content_to_rows(pd.DataFrame({"text": ["page text"], "_bbox_xyxy_norm": [bbox]}))

    assert result["_bbox_xyxy_norm"].dtype == object
    assert result.iloc[0]["_bbox_xyxy_norm"] == [0.1, 0.2, 0.8, 0.9]
    assert not hasattr(result.iloc[0]["_bbox_xyxy_norm"], "to_numpy")


def test_explode_leaves_non_dataframe_input_unchanged() -> None:
    batch = [{"text": "page text"}]

    assert explode_content_to_rows(batch) is batch


def test_explode_text_image_does_not_boolean_evaluate_numpy_bbox() -> None:
    result = explode_content_to_rows(
        pd.DataFrame(
            {
                "text": ["page text"],
                "page_image": [{"image_b64": "page_b64"}],
                "table": [[{"text": "cell text", "bbox_xyxy_norm": np.array([0.1, 0.2, 0.8, 0.9])}]],
            }
        ),
        modality="text_image",
    )

    table_bbox = result.loc[result["_content_type"] == "table", "_bbox_xyxy_norm"].iloc[0]
    assert table_bbox == [0.1, 0.2, 0.8, 0.9]
    assert not isinstance(table_bbox, np.ndarray)


def test_collapse_returns_iterrows_safe_page_rows() -> None:
    table = BlockAccessor.batch_to_block(
        pd.DataFrame(
            {
                "text": ["page text"],
                "table": [np.array([], dtype=object)],
                "chart": [np.array([], dtype=object)],
                "images": [np.array([], dtype=object)],
            }
        )
    )

    result = _ArrowPandasOperatorAdapter(
        UDFOperator,
        {
            "fn": partial(collapse_content_to_page_rows, modality="text"),
            "name": "CollapseContentToPageRows",
            "preserve_pandas_output": True,
        },
    )(table)

    rows = list(result.iterrows())
    assert len(rows) == 1
    assert rows[0][1]["text"] == "page text"
    assert result["_embed_modality"].tolist() == ["text"]
