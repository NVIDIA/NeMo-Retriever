import ast
import json
from types import SimpleNamespace

from nemo_retriever.vector_store.lancedb_store import _build_lancedb_rows_from_df
from nemo_retriever.vector_store.lancedb_utils import build_lancedb_row


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
