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

from nemo_retriever.graph.executor import _ArrowPandasOperatorAdapter
from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows, explode_content_to_rows
from nemo_retriever.operators.abstract_operator import AbstractOperator
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
    assert result.apply(lambda row: row["text"], axis=1).tolist() == ["first", "second", "third"]


def test_adapter_preserves_page_and_element_content_semantics_for_sliced_pdf_rows() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "text": f"page {page_number}",
                "table": [{"text": f"table {page_number}"}],
                "chart": [{"text": f"chart {page_number}"}],
                "metadata": {"source_path": "document.pdf", "error": None},
            }
            for page_number in range(3)
        ]
    ).slice(2, 1)

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

    assert page_result["text"].tolist() == ["page 2\n\ntable 2\n\nchart 2"]
    assert element_result["text"].tolist() == ["page 2", "table 2", "chart 2"]
    assert element_result["_content_type"].tolist() == ["text", "table", "chart"]
    pa.Table.from_pandas(page_result, preserve_index=False).validate(full=True)
    pa.Table.from_pandas(element_result, preserve_index=False).validate(full=True)
