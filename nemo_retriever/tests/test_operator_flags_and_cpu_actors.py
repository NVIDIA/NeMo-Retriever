# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPUOperator/CPUOperator flags and CPU-only actor variants."""

from unittest.mock import patch

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.cpu_operator import CPUOperator


# ---------------------------------------------------------------------------
# GPUOperator / CPUOperator flag tests
# ---------------------------------------------------------------------------
class TestGPUOperatorFlag:
    def test_is_standalone_class(self):
        assert isinstance(GPUOperator(), GPUOperator)

    def test_gpu_operators_have_flag(self):
        from nemo_retriever.page_elements.page_elements import PageElementDetectionGPUActor
        from nemo_retriever.chart.chart_detection import GraphicElementsGPUActor
        from nemo_retriever.table.table_detection import TableStructureGPUActor
        from nemo_retriever.ocr.ocr import OCRGPUActor
        from nemo_retriever.parse.nemotron_parse import NemotronParseGPUActor
        from nemo_retriever.text_embed.operators import _BatchEmbedGPUActor
        from nemo_retriever.caption.caption import CaptionGPUActor
        from nemo_retriever.infographic.infographic_detection import InfographicDetectionGPUActor
        from nemo_retriever.rerank.rerank import NemotronRerankGPUActor
        from nemo_retriever.text_embed.text_embed import TextEmbedGPUActor

        assert issubclass(PageElementDetectionGPUActor, GPUOperator)
        assert issubclass(GraphicElementsGPUActor, GPUOperator)
        assert issubclass(TableStructureGPUActor, GPUOperator)
        assert issubclass(OCRGPUActor, GPUOperator)
        assert issubclass(NemotronParseGPUActor, GPUOperator)
        assert issubclass(_BatchEmbedGPUActor, GPUOperator)
        assert issubclass(CaptionGPUActor, GPUOperator)
        assert issubclass(InfographicDetectionGPUActor, GPUOperator)
        assert issubclass(NemotronRerankGPUActor, GPUOperator)
        assert issubclass(TextEmbedGPUActor, GPUOperator)

    def test_gpu_operators_are_not_cpu(self):
        from nemo_retriever.page_elements.page_elements import PageElementDetectionGPUActor

        assert not issubclass(PageElementDetectionGPUActor, CPUOperator)


class TestCPUOperatorFlag:
    def test_is_standalone_class(self):
        assert isinstance(CPUOperator(), CPUOperator)

    def test_cpu_operators_have_flag(self):
        from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionCPUActor
        from nemo_retriever.pdf.split import PDFSplitCPUActor
        from nemo_retriever.pdf.extract import PDFExtractionCPUActor
        from nemo_retriever.txt.ray_data import TextChunkCPUActor, TxtSplitCPUActor
        from nemo_retriever.image.ray_data import ImageLoadCPUActor
        from nemo_retriever.html.ray_data import HtmlSplitCPUActor
        from nemo_retriever.graph.content_operators import ExplodeContentActor
        from nemo_retriever.audio.asr_actor import ASRCPUActor
        from nemo_retriever.caption.caption import CaptionCPUActor
        from nemo_retriever.infographic.infographic_detection import InfographicDetectionCPUActor
        from nemo_retriever.rerank.rerank import NemotronRerankCPUActor

        assert issubclass(DocToPdfConversionCPUActor, CPUOperator)
        assert issubclass(PDFSplitCPUActor, CPUOperator)
        assert issubclass(PDFExtractionCPUActor, CPUOperator)
        assert issubclass(TextChunkCPUActor, CPUOperator)
        assert issubclass(TxtSplitCPUActor, CPUOperator)
        assert issubclass(ImageLoadCPUActor, CPUOperator)
        assert issubclass(HtmlSplitCPUActor, CPUOperator)
        assert issubclass(ExplodeContentActor, CPUOperator)
        assert issubclass(ASRCPUActor, CPUOperator)
        assert issubclass(CaptionCPUActor, CPUOperator)
        assert issubclass(InfographicDetectionCPUActor, CPUOperator)
        assert issubclass(NemotronRerankCPUActor, CPUOperator)

    def test_cpu_operators_are_not_gpu(self):
        from nemo_retriever.pdf.split import PDFSplitCPUActor

        assert not issubclass(PDFSplitCPUActor, GPUOperator)

    def test_public_actor_names_are_archetypes(self):
        from nemo_retriever.audio.asr_actor import ASRActor
        from nemo_retriever.caption.caption import CaptionActor
        from nemo_retriever.chart.chart_detection import GraphicElementsActor
        from nemo_retriever.ocr.ocr import OCRActor
        from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
        from nemo_retriever.table.table_detection import TableStructureActor

        assert issubclass(ASRActor, ArchetypeOperator)
        assert issubclass(CaptionActor, ArchetypeOperator)
        assert issubclass(GraphicElementsActor, ArchetypeOperator)
        assert issubclass(OCRActor, ArchetypeOperator)
        assert issubclass(PageElementDetectionActor, ArchetypeOperator)
        assert issubclass(TableStructureActor, ArchetypeOperator)
        assert not issubclass(ASRActor, CPUOperator)
        assert not issubclass(CaptionActor, GPUOperator)

    def test_all_operators_are_abstract_operator(self):
        from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor
        from nemo_retriever.audio.asr_actor import ASRActor
        from nemo_retriever.audio.chunk_actor import MediaChunkActor
        from nemo_retriever.caption.caption import CaptionActor
        from nemo_retriever.infographic.infographic_detection import InfographicDetectionActor
        from nemo_retriever.rerank.rerank import NemotronRerankActor
        from nemo_retriever.text_embed.text_embed import TextEmbedActor
        from nemo_retriever.pdf.split import PDFSplitActor
        from nemo_retriever.page_elements.page_elements import PageElementDetectionActor

        assert issubclass(DocToPdfConversionActor, AbstractOperator)
        assert issubclass(ASRActor, AbstractOperator)
        assert issubclass(MediaChunkActor, AbstractOperator)
        assert issubclass(CaptionActor, AbstractOperator)
        assert issubclass(InfographicDetectionActor, AbstractOperator)
        assert issubclass(NemotronRerankActor, AbstractOperator)
        assert issubclass(PDFSplitActor, AbstractOperator)
        assert issubclass(PageElementDetectionActor, AbstractOperator)
        assert issubclass(TextEmbedActor, AbstractOperator)


# ---------------------------------------------------------------------------
# CPU-only actor variant tests
# ---------------------------------------------------------------------------
class TestPageElementDetectionCPUActor:
    def test_inherits_cpu_operator(self):
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        assert issubclass(PageElementDetectionCPUActor, CPUOperator)
        assert issubclass(PageElementDetectionCPUActor, AbstractOperator)
        assert not issubclass(PageElementDetectionCPUActor, GPUOperator)

    def test_uses_default_invoke_url(self):
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        actor = PageElementDetectionCPUActor()
        assert actor._model is None
        assert "nemotron-page-elements-v3" in actor.detect_kwargs["invoke_url"]

    def test_creates_with_custom_invoke_url(self):
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        actor = PageElementDetectionCPUActor(invoke_url="http://custom")
        assert actor._model is None
        assert actor.detect_kwargs["invoke_url"] == "http://custom"

    def test_preprocess_passthrough(self):
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        actor = PageElementDetectionCPUActor(invoke_url="http://fake")
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.page_elements.cpu_actor.detect_page_elements_v3")
    def test_process(self, mock_fn):
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        expected = pd.DataFrame({"page_elements_v3": ["det"]})
        mock_fn.return_value = expected
        actor = PageElementDetectionCPUActor(invoke_url="http://fake")
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)


class TestGraphicElementsCPUActor:
    def test_inherits_cpu_operator(self):
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        assert issubclass(GraphicElementsCPUActor, CPUOperator)
        assert not issubclass(GraphicElementsCPUActor, GPUOperator)

    def test_uses_default_urls(self):
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        actor = GraphicElementsCPUActor()
        assert actor._graphic_elements_model is None
        assert actor._ocr_model is None
        assert "nemotron-graphic-elements-v1" in actor._graphic_elements_invoke_url
        assert "nemotron-ocr-v1" in actor._ocr_invoke_url

    def test_creates_with_custom_urls(self):
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        actor = GraphicElementsCPUActor(
            graphic_elements_invoke_url="http://custom1",
            ocr_invoke_url="http://custom2",
        )
        assert actor._graphic_elements_invoke_url == "http://custom1"
        assert actor._ocr_invoke_url == "http://custom2"

    @patch("nemo_retriever.chart.cpu_actor.graphic_elements_ocr_page_elements")
    def test_process(self, mock_fn):
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        expected = pd.DataFrame({"chart": [[]]})
        mock_fn.return_value = expected
        actor = GraphicElementsCPUActor()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)


class TestTableStructureCPUActor:
    def test_inherits_cpu_operator(self):
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        assert issubclass(TableStructureCPUActor, CPUOperator)
        assert not issubclass(TableStructureCPUActor, GPUOperator)

    def test_uses_default_urls(self):
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        actor = TableStructureCPUActor()
        assert actor._table_structure_model is None
        assert actor._ocr_model is None
        assert "nemotron-table-structure-v1" in actor._table_structure_invoke_url
        assert "nemotron-ocr-v1" in actor._ocr_invoke_url

    def test_creates_with_custom_urls(self):
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        actor = TableStructureCPUActor(
            table_structure_invoke_url="http://custom1",
            ocr_invoke_url="http://custom2",
        )
        assert actor._table_structure_invoke_url == "http://custom1"
        assert actor._ocr_invoke_url == "http://custom2"

    @patch("nemo_retriever.table.cpu_actor.table_structure_ocr_page_elements")
    def test_process(self, mock_fn):
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        expected = pd.DataFrame({"table": [[]]})
        mock_fn.return_value = expected
        actor = TableStructureCPUActor()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)


class TestOCRCPUActor:
    def test_inherits_cpu_operator(self):
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        assert issubclass(OCRCPUActor, CPUOperator)
        assert not issubclass(OCRCPUActor, GPUOperator)

    def test_uses_default_invoke_url(self):
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        actor = OCRCPUActor()
        assert actor._model is None
        assert "nemotron-ocr-v1" in actor.ocr_kwargs["invoke_url"]

    def test_creates_with_custom_invoke_url(self):
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        actor = OCRCPUActor(ocr_invoke_url="http://custom")
        assert actor._model is None
        assert actor.ocr_kwargs["invoke_url"] == "http://custom"

    @patch("nemo_retriever.ocr.cpu_ocr.ocr_page_elements")
    def test_process(self, mock_fn):
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        expected = pd.DataFrame({"ocr_v1": ["res"]})
        mock_fn.return_value = expected
        actor = OCRCPUActor(ocr_invoke_url="http://fake")
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)


class TestNemotronParseCPUActor:
    def test_inherits_cpu_operator(self):
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        assert issubclass(NemotronParseCPUActor, CPUOperator)
        assert not issubclass(NemotronParseCPUActor, GPUOperator)

    def test_uses_default_invoke_url(self):
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        actor = NemotronParseCPUActor()
        assert actor._model is None
        assert "integrate.api.nvidia.com" in actor._invoke_url

    def test_creates_with_custom_invoke_url(self):
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        actor = NemotronParseCPUActor(nemotron_parse_invoke_url="http://custom")
        assert actor._model is None
        assert actor._invoke_url == "http://custom"

    @patch("nemo_retriever.ocr.cpu_parse.nemotron_parse_page_elements")
    def test_process(self, mock_fn):
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        expected = pd.DataFrame({"nemotron_parse_v1_2": ["res"]})
        mock_fn.return_value = expected
        actor = NemotronParseCPUActor(nemotron_parse_invoke_url="http://fake")
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)


class TestBatchEmbedCPUActor:
    def _make_params(self):
        from nemo_retriever.params import EmbedParams

        return EmbedParams(model_name="test-model", embed_invoke_url="http://fake")

    def test_inherits_cpu_operator(self):
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        assert issubclass(_BatchEmbedCPUActor, CPUOperator)
        assert not issubclass(_BatchEmbedCPUActor, GPUOperator)

    @patch("nemo_retriever.model.create_local_embedder")
    def test_uses_local_embedder_when_no_remote_endpoint(self, mock_create):
        """No HTTP endpoint → load local HF embedder (see cpu_operator / embed URL normalization)."""
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor
        from nemo_retriever.params import EmbedParams

        mock_model = object()
        mock_create.return_value = mock_model
        actor = _BatchEmbedCPUActor(params=EmbedParams(model_name="test-model"))
        mock_create.assert_called_once()
        assert actor._model is mock_model
        assert actor._kwargs.get("embedding_endpoint") is None
        assert actor._kwargs.get("embed_invoke_url") is None

    def test_creates_with_custom_invoke_url(self):
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        actor = _BatchEmbedCPUActor(params=self._make_params())
        assert actor._model is None
        assert actor._kwargs["embedding_endpoint"] == "http://fake"

    @patch("nemo_retriever.text_embed.cpu_operator.embed_text_main_text_embed")
    def test_process(self, mock_fn):
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        expected = pd.DataFrame({"text": ["hello"], "embedding": [[0.1, 0.2]]})
        mock_fn.return_value = expected
        actor = _BatchEmbedCPUActor(params=self._make_params())
        result = actor.process(pd.DataFrame({"text": ["hello"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    def test_preprocess_passthrough(self):
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        actor = _BatchEmbedCPUActor(params=self._make_params())
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    def test_postprocess_passthrough(self):
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        actor = _BatchEmbedCPUActor(params=self._make_params())
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)
