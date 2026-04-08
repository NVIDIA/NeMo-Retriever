# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-level tests for StoreOperator at multiple pipeline positions."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pandas as pd

from nemo_retriever.graph import InprocessExecutor, StoreOperator, UDFOperator
from nemo_retriever.params import StoreParams


def _make_tiny_png_b64(width: int = 4, height: int = 4, color=(255, 0, 0)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_post_ocr_df(b64: str) -> pd.DataFrame:
    """Build a single-row DataFrame matching post-OCR pipeline output."""
    return pd.DataFrame(
        [
            {
                "path": "/docs/test.pdf",
                "page_number": 1,
                "page_image": {"image_b64": b64},
                "text": "Sample page text",
                "table": [{"text": "col1|col2", "image_b64": b64}],
                "chart": [{"text": "chart data", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9]}],
                "infographic": [],
                "images": [{"image_b64": b64}],
            }
        ]
    )


class TestStoreOperatorInGraph:
    def test_store_operator_runs_in_graph(self, tmp_path: Path):
        """StoreOperator works end-to-end through InprocessExecutor,
        including kwargs reconstruction (the Ray worker serialization contract)."""
        b64 = _make_tiny_png_b64()
        df = _make_post_ocr_df(b64)

        params = StoreParams(storage_uri=str(tmp_path))
        graph = UDFOperator(lambda x: x, name="Identity") >> StoreOperator(params=params)
        result = InprocessExecutor(graph, show_progress=False).ingest(df)

        assert (tmp_path / "test" / "page_1.png").exists()
        assert (tmp_path / "test" / "page_1_table_0.png").exists()
        assert (tmp_path / "test" / "page_1_image_0.png").exists()
        assert "stored_image_uri" in result.iloc[0]["page_image"]

    def test_two_store_nodes_different_positions(self, tmp_path: Path):
        """Store at position A sees different content than store at position B.

        Graph: store(stage_a) >> add_table >> store(stage_b)

        stage_a should have only page image (no table yet).
        stage_b should have page image AND the table added between stores.
        """
        b64 = _make_tiny_png_b64()
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "page_image": {"image_b64": b64},
                    "text": "",
                    "table": [],
                    "chart": [],
                    "infographic": [],
                    "images": [],
                }
            ]
        )

        def _add_table_column(df: pd.DataFrame) -> pd.DataFrame:
            b64 = _make_tiny_png_b64(color=(0, 255, 0))
            df = df.copy()
            df["table"] = [[{"text": "added later", "image_b64": b64}]]
            return df

        stage_a = tmp_path / "stage_a"
        stage_b = tmp_path / "stage_b"
        params_a = StoreParams(storage_uri=str(stage_a), strip_base64=False)
        params_b = StoreParams(storage_uri=str(stage_b), strip_base64=False)

        graph = (
            StoreOperator(params=params_a)
            >> UDFOperator(_add_table_column, name="AddTable")
            >> StoreOperator(params=params_b)
        )
        InprocessExecutor(graph, show_progress=False).ingest(df)

        # stage_a: only page image, no table content existed yet
        assert (stage_a / "test" / "page_1.png").exists()
        assert not list(stage_a.rglob("*_table_*"))

        # stage_b: page image + table file (added by UDF between stores)
        assert (stage_b / "test" / "page_1.png").exists()
        assert (stage_b / "test" / "page_1_table_0.png").exists()
