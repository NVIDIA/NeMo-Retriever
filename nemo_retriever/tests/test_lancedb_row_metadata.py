import ast
import json
from types import SimpleNamespace

import numpy as np

from nemo_retriever.common.vdb.lancedb_bulk import _build_lancedb_rows_from_df
from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_row


def test_build_lancedb_row_persists_normalized_content_type() -> None:
    row = SimpleNamespace(
        path="/tmp/doc_a.pdf",
        page_number=7,
        metadata={"embedding": [0.1, 0.2], "source_path": "/tmp/doc_a.pdf"},
        text="table text",
        _content_type="table_caption",
    )

    row_out = build_lancedb_row(row)

    assert row_out is not None
    metadata = json.loads(row_out["metadata"])
    assert metadata["_content_type"] == "table"


def test_build_lancedb_row_normalizes_arrow_backed_arrays() -> None:
    row = SimpleNamespace(
        path="/tmp/doc_arrays.pdf",
        page_number=2,
        metadata={"embedding": np.array([0.1, 0.2], dtype=np.float32)},
        text="table text",
        table=np.array([{"text": "first"}, {"text": "second"}], dtype=object),
        chart=np.array([], dtype=object),
        infographic=np.array([], dtype=object),
        _bbox_xyxy_norm=np.array([0.1, 0.2, 0.8, 0.9]),
    )

    row_out = build_lancedb_row(row)

    assert row_out is not None
    assert row_out["vector"] == [np.float32(0.1), np.float32(0.2)]
    assert json.loads(row_out["bbox_xyxy_norm"]) == [0.1, 0.2, 0.8, 0.9]
    assert json.loads(row_out["metadata"])["ocr_table_detections"] == 2


def test_build_lancedb_rows_from_df_persists_normalized_content_type() -> None:
    rows = [
        {
            "path": "/tmp/doc_b.pdf",
            "page_number": 3,
            "text": "chart text",
            "_content_type": "chart_caption",
            "metadata": {"embedding": [0.3, 0.4], "source_path": "/tmp/doc_b.pdf"},
        }
    ]

    row_out = _build_lancedb_rows_from_df(rows)

    assert len(row_out) == 1
    metadata = ast.literal_eval(row_out[0]["metadata"])
    assert metadata["_content_type"] == "chart"
