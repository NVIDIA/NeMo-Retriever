# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.operators.extract.pdf.extract import build_pdf_extraction_kwargs, pdf_extraction


@pytest.mark.parametrize(("value", "expected"), [(None, True), (True, True), (False, False)])
def test_unresolved_auto_keeps_image_crops_outside_graph_ingestor(value, expected):
    assert build_pdf_extraction_kwargs(ExtractParams(extract_images=value))["extract_images"] is expected


@pytest.mark.parametrize(("method", "renders"), [("pdfium_hybrid", True), ("ocr", True), ("pdfium", False)])
def test_native_text_page_keeps_page_render_when_page_elements_or_ocr_run(method, renders):
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    with (
        patch("nemo_retriever.operators.extract.pdf.extract._is_scanned_page", return_value=False),
        patch("nemo_retriever.operators.extract.pdf.extract.extract_image_like_objects_from_pdfium_page") as crops,
    ):
        result = pdf_extraction(
            pd.DataFrame([{"bytes": buf.getvalue(), "path": "t.pdf", "page_number": 1}]),
            extract_text=True,
            extract_images=False,
            method=method,
        )

    page_image = result.iloc[0].get("page_image")
    assert bool(isinstance(page_image, dict) and page_image.get("image_b64")) is renders
    crops.assert_not_called()
