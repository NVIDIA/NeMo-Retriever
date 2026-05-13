# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying all pipeline actors inherit from AbstractOperator."""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    @patch("nemo_retriever.page_elements.cpu_actor.detect_page_elements_v3")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"page_elements_v3": ["det"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.page_elements.cpu_actor.detect_page_elements_v3")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"page_elements_v3": ["det"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"page_image": ["x"]}))
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.page_elements.cpu_actor.detect_page_elements_v3", side_effect=RuntimeError("boom"))
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

    @patch("nemo_retriever.chart.cpu_actor.graphic_elements_ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"chart": [[]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.chart.cpu_actor.graphic_elements_ocr_page_elements", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "graphic_elements_ocr_v1" in result.columns


# ---------------------------------------------------------------------------
# 4b. GraphicElementsActor (GPU variant) default OCR tests
# ---------------------------------------------------------------------------
class TestGraphicElementsGPUActor:
    def test_init_signature_uses_ocr_version_selector(self):
        from nemo_retriever.chart.gpu_actor import GraphicElementsActor as GPUActor

        assert set(inspect.signature(GPUActor.__init__).parameters) == {
            "self",
            "graphic_elements_invoke_url",
            "ocr_invoke_url",
            "invoke_url",
            "api_key",
            "request_timeout_s",
            "remote_max_pool_workers",
            "remote_max_retries",
            "remote_max_429_retries",
            "inference_batch_size",
            "ocr_version",
            "ocr_lang",
        }

    def test_init_with_no_kwargs_defaults_to_local_ocr_v2(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.chart.gpu_actor import GraphicElementsActor as GPUActor

        mock_graphic = MagicMock()
        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronGraphicElementsV1", mock_graphic)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)

        actor = GPUActor()

        assert actor._graphic_elements_invoke_url == ""
        assert actor._ocr_invoke_url == ""
        mock_graphic.assert_called_once_with()
        mock_ocr_v2.assert_called_once_with(lang="multi")
        assert actor._nim_client is None

    def test_init_can_explicitly_use_local_ocr_v1(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.chart.gpu_actor import GraphicElementsActor as GPUActor

        mock_graphic = MagicMock()
        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronGraphicElementsV1", mock_graphic)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)

        actor = GPUActor(ocr_version="v1")

        assert actor._ocr_invoke_url == ""
        mock_graphic.assert_called_once_with()
        mock_ocr_v2.assert_called_once_with(lang="v1")
        assert actor._nim_client is None


# ---------------------------------------------------------------------------
# 5. TableStructureActor
# ---------------------------------------------------------------------------
class TestTableStructureActor:
    def _make(self):
        from nemo_retriever.table.table_detection import TableStructureActor

        return TableStructureActor(table_structure_invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.table.table_detection import TableStructureActor

        assert issubclass(TableStructureActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.table.cpu_actor.table_structure_ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"table": [[]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.table.cpu_actor.table_structure_ocr_page_elements", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "table_structure_ocr_v1" in result.columns


# ---------------------------------------------------------------------------
# 5b. TableStructureActor (GPU variant) regression tests
# ---------------------------------------------------------------------------
class TestTableStructureGPUActor:
    """Regression tests for the GPU variant of TableStructureActor.

    The GPU variant lives in nemo_retriever.table.gpu_actor and is selected
    by the archetype resolver when GPUs are available and no CPU-only
    endpoint is configured. Prior to this fix, its __init__ referenced
    ``self._ocr_invoke_url`` without ever assigning it, raising
    ``AttributeError`` for any non-CPU dispatch. These tests pin the
    contract so the regression cannot reappear.
    """

    def test_init_signature_uses_ocr_version_selector(self):
        from nemo_retriever.table.gpu_actor import TableStructureActor as GPUActor

        assert set(inspect.signature(GPUActor.__init__).parameters) == {
            "self",
            "table_structure_invoke_url",
            "ocr_invoke_url",
            "ocr_version",
            "ocr_lang",
            "invoke_url",
            "api_key",
            "table_output_format",
            "request_timeout_s",
            "remote_max_pool_workers",
            "remote_max_retries",
            "remote_max_429_retries",
        }

    def test_init_with_no_kwargs_defaults_to_local_ocr_v2(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.table.gpu_actor import TableStructureActor as GPUActor

        mock_ts = MagicMock()
        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronTableStructureV1", mock_ts)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)

        actor = GPUActor()

        assert actor._table_structure_invoke_url == ""
        assert actor._ocr_invoke_url == ""
        mock_ts.assert_called_once_with()
        mock_ocr_v2.assert_called_once_with(lang="multi")
        assert actor._nim_client is None

    def test_init_can_explicitly_use_local_ocr_v1(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.table.gpu_actor import TableStructureActor as GPUActor

        mock_ts = MagicMock()
        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronTableStructureV1", mock_ts)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)

        actor = GPUActor(ocr_version="v1")

        assert actor._ocr_invoke_url == ""
        mock_ts.assert_called_once_with()
        mock_ocr_v2.assert_called_once_with(lang="v1")
        assert actor._nim_client is None

    def test_init_with_ocr_invoke_url_skips_local_ocr(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.table.gpu_actor import TableStructureActor as GPUActor

        mock_ts = MagicMock()
        mock_ocr = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronTableStructureV1", mock_ts)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr)

        actor = GPUActor(ocr_invoke_url="http://ocr.example/v1/cv/nvidia/nemotron-ocr-v1")

        assert actor._ocr_invoke_url == "http://ocr.example/v1/cv/nvidia/nemotron-ocr-v1"
        assert actor._ocr_model is None
        mock_ocr.assert_not_called()
        mock_ts.assert_called_once_with()
        assert actor._nim_client is not None

    def test_init_with_both_urls_skips_all_local_models(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.table.gpu_actor import TableStructureActor as GPUActor

        mock_ocr = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr)

        actor = GPUActor(
            table_structure_invoke_url="http://ts.example/v1",
            ocr_invoke_url="http://ocr.example/v1",
        )

        assert actor._table_structure_model is None
        assert actor._ocr_model is None
        mock_ocr.assert_not_called()
        assert actor._nim_client is not None

    def test_init_strips_whitespace_from_ocr_invoke_url(self):
        from nemo_retriever.table.gpu_actor import TableStructureActor as GPUActor

        actor = GPUActor(
            table_structure_invoke_url="http://ts.example/v1",
            ocr_invoke_url="  http://ocr.example/v1  ",
        )

        assert actor._ocr_invoke_url == "http://ocr.example/v1"

    def test_init_treats_none_ocr_invoke_url_as_empty(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.table.gpu_actor import TableStructureActor as GPUActor

        mock_ocr = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr)

        actor = GPUActor(
            table_structure_invoke_url="http://ts.example/v1",
            ocr_invoke_url=None,
        )

        assert actor._ocr_invoke_url == ""
        mock_ocr.assert_called_once_with(lang="multi")


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

    @patch("nemo_retriever.ocr.cpu_ocr.ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"ocr": ["res"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.ocr.cpu_ocr.ocr_page_elements", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "ocr" in result.columns

    def test_local_gpu_actor_defaults_to_v2_multi(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.ocr.gpu_ocr import OCRActor as OCRGPUActor

        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)
        actor = OCRGPUActor()

        mock_ocr_v2.assert_called_once_with(lang="multi")
        assert actor._nim_client is None

    def test_local_gpu_actor_passes_v2_ocr_lang(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.ocr.gpu_ocr import OCRActor as OCRGPUActor

        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)
        actor = OCRGPUActor(ocr_lang="english")

        mock_ocr_v2.assert_called_once_with(lang="english")
        assert actor._nim_client is None

    def test_local_gpu_actor_uses_v2_legacy_mode_for_v1(self, monkeypatch):
        import nemo_retriever.model.local as local_models
        from nemo_retriever.ocr.gpu_ocr import OCRActor as OCRGPUActor

        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)
        actor = OCRGPUActor(ocr_version="v1")

        mock_ocr_v2.assert_called_once_with(lang="v1")
        assert actor._nim_client is None


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

    @patch("nemo_retriever.text_embed.cpu_operator.embed_text_main_text_embed")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"], "embedding": [[0.1, 0.2]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"text": ["hello"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.text_embed.cpu_operator.embed_text_main_text_embed")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"], "embedding": [[0.1, 0.2]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"text": ["hello"]}))
        pd.testing.assert_frame_equal(result, expected)
