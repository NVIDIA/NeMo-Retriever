# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying all pipeline actors inherit from AbstractOperator."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.graph.abstract_operator import AbstractOperator


# ---------------------------------------------------------------------------
# 1. PDFSplitActor
# ---------------------------------------------------------------------------
class TestPDFSplitActor:
    def _make(self):
        from nemo_retriever.pdf.split import PDFSplitActor

        return PDFSplitActor()

    def test_inherits(self):
        from nemo_retriever.pdf.split import PDFSplitActor

        assert issubclass(PDFSplitActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"], "path": ["/tmp/a.pdf"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"], "path": ["/tmp/a.pdf"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.pdf.split.split_pdf_batch")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"page": [1]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.pdf.split.split_pdf_batch")
    def test_call_delegates_to_run(self, mock_fn):
        expected = pd.DataFrame({"page": [1]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"x"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 2. PDFExtractionActor
# ---------------------------------------------------------------------------
class TestPDFExtractionActor:
    def _make(self):
        from nemo_retriever.pdf.extract import PDFExtractionActor

        return PDFExtractionActor(method="pdfium")

    def test_inherits(self):
        from nemo_retriever.pdf.extract import PDFExtractionActor

        assert issubclass(PDFExtractionActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.pdf.extract.pdf_extraction")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"bytes": [b"x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.pdf.extract.pdf_extraction")
    def test_call_delegates_to_run(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"x"]}))
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.pdf.extract.pdf_extraction", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"], "path": ["/tmp/a.pdf"]})
        result = actor(df)
        assert isinstance(result, list)
        record = result[0]
        assert record["metadata"]["error"]["type"] == "RuntimeError"

    def test_pdfium_output_can_have_empty_text_without_ocr_flag(self):
        from nemo_retriever.pdf.extract import PDFExtractionActor
        from nemo_retriever.pdf.split import PDFSplitActor

        pdf_path = Path("/raid/data/jp20/1312679.pdf")
        if not pdf_path.exists():
            pytest.skip(f"External regression fixture not available: {pdf_path}")

        source_df = pd.DataFrame({"path": [str(pdf_path)], "bytes": [pdf_path.read_bytes()]})
        split_df = PDFSplitActor()(source_df)

        result = PDFExtractionActor(
            method="pdfium",
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
        )(split_df.head(5))

        first_page = result[result["page_number"] == 1].iloc[0]
        metadata = first_page["metadata"]

        assert first_page["text"] == ""
        assert metadata["has_text"] is False
        assert metadata["needs_ocr_for_text"] is False
        assert metadata["error"] is None


# ---------------------------------------------------------------------------
# 3. PageElementDetectionActor
# ---------------------------------------------------------------------------
class TestPageElementDetectionActor:
    def _make(self):
        from nemo_retriever.page_elements.page_elements import PageElementDetectionActor

        return PageElementDetectionActor(invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.page_elements.page_elements import PageElementDetectionActor

        assert issubclass(PageElementDetectionActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.page_elements.page_elements.detect_page_elements_v3")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"page_elements_v3": ["det"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.page_elements.page_elements.detect_page_elements_v3")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"page_elements_v3": ["det"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"page_image": ["x"]}))
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.page_elements.page_elements.detect_page_elements_v3", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "page_elements_v3" in result.columns


# ---------------------------------------------------------------------------
# 4. GraphicElementsActor
# ---------------------------------------------------------------------------
class TestGraphicElementsActor:
    def _make(self):
        from nemo_retriever.chart.chart_detection import GraphicElementsActor

        return GraphicElementsActor(
            graphic_elements_invoke_url="http://fake",
            ocr_invoke_url="http://fake",
        )

    def test_inherits(self):
        from nemo_retriever.chart.chart_detection import GraphicElementsActor

        assert issubclass(GraphicElementsActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.chart.chart_detection.graphic_elements_ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"chart": [[]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.chart.chart_detection.graphic_elements_ocr_page_elements", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "graphic_elements_ocr_v1" in result.columns


# ---------------------------------------------------------------------------
# 5. TableStructureActor
# ---------------------------------------------------------------------------
class TestTableStructureActor:
    def _make(self):
        from nemo_retriever.table.table_detection import TableStructureActor

        return TableStructureActor(
            table_structure_invoke_url="http://fake",
            ocr_invoke_url="http://fake",
        )

    def test_inherits(self):
        from nemo_retriever.table.table_detection import TableStructureActor

        assert issubclass(TableStructureActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.table.table_detection.table_structure_ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"table": [[]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.table.table_detection.table_structure_ocr_page_elements", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "table_structure_ocr_v1" in result.columns


# ---------------------------------------------------------------------------
# 6. OCRActor
# ---------------------------------------------------------------------------
class TestOCRActor:
    def _make(self):
        from nemo_retriever.ocr.ocr import OCRActor

        return OCRActor(ocr_invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.ocr.ocr import OCRActor

        assert issubclass(OCRActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.ocr.ocr.ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"ocr_v1": ["res"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.ocr.ocr.ocr_page_elements", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "ocr_v1" in result.columns


# ---------------------------------------------------------------------------
# 7. NemotronParseActor
# ---------------------------------------------------------------------------
class TestNemotronParseActor:
    def _make(self):
        from nemo_retriever.parse.nemotron_parse import NemotronParseActor

        return NemotronParseActor(nemotron_parse_invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.parse.nemotron_parse import NemotronParseActor

        assert issubclass(NemotronParseActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.parse.nemotron_parse.nemotron_parse_pages")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"nemotron_parse_v1_2": ["res"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.parse.nemotron_parse.nemotron_parse_pages", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "nemotron_parse_v1_2" in result.columns


# ---------------------------------------------------------------------------
# 8. TextChunkActor
# ---------------------------------------------------------------------------
class TestTextChunkActor:
    def _make(self):
        from nemo_retriever.txt.ray_data import TextChunkActor

        return TextChunkActor()

    def test_inherits(self):
        from nemo_retriever.txt.ray_data import TextChunkActor

        assert issubclass(TextChunkActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    def test_process_empty_df(self):
        actor = self._make()
        df = pd.DataFrame()
        result = actor.process(df)
        assert result.empty

    @patch("nemo_retriever.txt.split.split_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk1"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"text": ["hello world"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 9. ImageLoadActor
# ---------------------------------------------------------------------------
class TestImageLoadActor:
    def _make(self):
        from nemo_retriever.image.ray_data import ImageLoadActor

        return ImageLoadActor()

    def test_inherits(self):
        from nemo_retriever.image.ray_data import ImageLoadActor

        assert issubclass(ImageLoadActor, AbstractOperator)

    def test_preprocess_empty(self):
        actor = self._make()
        result = actor.preprocess(pd.DataFrame())
        assert result.empty

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"path": ["/tmp/a.png"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.image.ray_data.image_bytes_to_pages_df")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"path": ["/tmp/a.png"], "page_number": [0]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"img"], "path": ["/tmp/a.png"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.image.ray_data.image_bytes_to_pages_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"path": ["/tmp/a.png"], "page_number": [0]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"img"], "path": ["/tmp/a.png"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 10. TxtSplitActor
# ---------------------------------------------------------------------------
class TestTxtSplitActor:
    def _make(self):
        from nemo_retriever.txt.ray_data import TxtSplitActor

        return TxtSplitActor()

    def test_inherits(self):
        from nemo_retriever.txt.ray_data import TxtSplitActor

        assert issubclass(TxtSplitActor, AbstractOperator)

    def test_preprocess_empty(self):
        actor = self._make()
        result = actor.preprocess(pd.DataFrame())
        assert list(result.columns) == ["text", "path", "page_number", "metadata"]

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.txt.ray_data.txt_bytes_to_chunks_df")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.txt"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"hello"], "path": ["/a.txt"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.txt.ray_data.txt_bytes_to_chunks_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.txt"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"hello"], "path": ["/a.txt"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 11. HtmlSplitActor
# ---------------------------------------------------------------------------
class TestHtmlSplitActor:
    def _make(self):
        from nemo_retriever.html.ray_data import HtmlSplitActor

        return HtmlSplitActor()

    def test_inherits(self):
        from nemo_retriever.html.ray_data import HtmlSplitActor

        assert issubclass(HtmlSplitActor, AbstractOperator)

    def test_preprocess_empty(self):
        actor = self._make()
        result = actor.preprocess(pd.DataFrame())
        assert list(result.columns) == ["text", "path", "page_number", "metadata"]

    @patch("nemo_retriever.html.ray_data.html_bytes_to_chunks_df")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.html"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"<p>hi</p>"], "path": ["/a.html"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.html.ray_data.html_bytes_to_chunks_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.html"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"<p>hi</p>"], "path": ["/a.html"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 12. _BatchEmbedActor
# ---------------------------------------------------------------------------
class TestBatchEmbedActor:
    def _make(self):
        from nemo_retriever.params import EmbedParams
        from nemo_retriever.text_embed.operators import _BatchEmbedActor

        params = EmbedParams(model_name="test-model", embed_invoke_url="http://fake")
        return _BatchEmbedActor(params=params)

    def test_inherits(self):
        from nemo_retriever.text_embed.operators import _BatchEmbedActor

        assert issubclass(_BatchEmbedActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.text_embed.operators.embed_text_main_text_embed")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"], "embedding": [[0.1, 0.2]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"text": ["hello"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.text_embed.operators.embed_text_main_text_embed")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"], "embedding": [[0.1, 0.2]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"text": ["hello"]}))
        pd.testing.assert_frame_equal(result, expected)
