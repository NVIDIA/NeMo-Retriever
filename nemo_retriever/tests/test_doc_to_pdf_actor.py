# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor, convert_to_pdf_bytes, convert_batch_to_pdf


class TestConvertToPdfBytes:
    @patch("nemo_retriever.utils.convert.to_pdf.shutil.which", return_value=None)
    def test_raises_when_libreoffice_missing(self, _mock_which):
        with pytest.raises(FileNotFoundError, match="LibreOffice is required"):
            convert_to_pdf_bytes(b"fake docx content", ".docx")

    @patch("nemo_retriever.utils.convert.to_pdf.shutil.which", return_value=None)
    def test_skips_check_for_pdf(self, _mock_which):
        """PDF passthrough should not require LibreOffice."""
        result = convert_to_pdf_bytes(b"%PDF-1.4 content", ".pdf")
        assert result == b"%PDF-1.4 content"


class TestConvertBatchToPdf:
    @patch("nemo_retriever.utils.convert.to_pdf.shutil.which", return_value=None)
    def test_libreoffice_missing_propagates_through_batch(self, _mock_which):
        """FileNotFoundError must not be swallowed by the batch error handler."""
        df = pd.DataFrame({"bytes": [b"fake"], "path": ["/tmp/test.docx"]})
        with pytest.raises(FileNotFoundError, match="LibreOffice is required"):
            convert_batch_to_pdf(df)


class TestDocToPdfConversionActor:
    def test_inherits_abstract_operator(self):
        assert issubclass(DocToPdfConversionActor, AbstractOperator)

    def test_instantiation(self):
        actor = DocToPdfConversionActor()
        assert isinstance(actor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = DocToPdfConversionActor()
        df = pd.DataFrame({"bytes": [b"fake"], "path": ["/tmp/test.docx"]})
        result = actor.preprocess(df)
        pd.testing.assert_frame_equal(result, df)

    def test_postprocess_passthrough(self):
        actor = DocToPdfConversionActor()
        df = pd.DataFrame({"bytes": [b"fake"], "path": ["/tmp/test.pdf"]})
        result = actor.postprocess(df)
        pd.testing.assert_frame_equal(result, df)

    @patch("nemo_retriever.utils.convert.to_pdf.convert_batch_to_pdf")
    def test_process_calls_convert(self, mock_convert):
        expected = pd.DataFrame({"bytes": [b"pdf"], "path": ["/tmp/test.pdf"]})
        mock_convert.return_value = expected
        actor = DocToPdfConversionActor()
        df = pd.DataFrame({"bytes": [b"docx"], "path": ["/tmp/test.docx"]})
        result = actor.process(df)
        mock_convert.assert_called_once_with(df)
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.utils.convert.to_pdf.convert_batch_to_pdf")
    def test_run_chains_preprocess_process_postprocess(self, mock_convert):
        expected = pd.DataFrame({"bytes": [b"pdf"], "path": ["/tmp/out.pdf"]})
        mock_convert.return_value = expected
        actor = DocToPdfConversionActor()
        df = pd.DataFrame({"bytes": [b"docx"], "path": ["/tmp/test.docx"]})
        result = actor.run(df)
        mock_convert.assert_called_once_with(df)
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.utils.convert.to_pdf.convert_batch_to_pdf")
    def test_call_delegates_to_run(self, mock_convert):
        expected = pd.DataFrame({"bytes": [b"pdf"], "path": ["/tmp/out.pdf"]})
        mock_convert.return_value = expected
        actor = DocToPdfConversionActor()
        df = pd.DataFrame({"bytes": [b"docx"], "path": ["/tmp/test.docx"]})
        result = actor(df)
        mock_convert.assert_called_once_with(df)
        pd.testing.assert_frame_equal(result, expected)
