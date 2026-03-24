# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image captioning pipeline stage."""

import base64
import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _make_test_png_b64(size: tuple[int, int] = (64, 64)) -> str:
    img = Image.new("RGB", size, color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_page_df(num_images=2, captioned=False):
    b64 = _make_test_png_b64()
    images = [
        {"bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "text": "done" if captioned else "", "image_b64": b64}
        for _ in range(num_images)
    ]
    return pd.DataFrame([{"text": "page", "images": images, "tables": [], "charts": [], "infographics": []}])


def test_caption_images_writes_back():
    from nemo_retriever.caption.caption import caption_images

    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["cap1", "cap2"]
    result = caption_images(_make_page_df(), model=mock_model)
    assert result.iloc[0]["images"][0]["text"] == "cap1"
    assert result.iloc[0]["images"][1]["text"] == "cap2"


def test_caption_images_skips_already_captioned():
    from nemo_retriever.caption.caption import caption_images

    mock_model = MagicMock()
    result = caption_images(_make_page_df(captioned=True), model=mock_model)
    mock_model.caption_batch.assert_not_called()
    assert result.iloc[0]["images"][0]["text"] == "done"


@patch("nemo_retriever.pdf.extract.extract_image_like_objects_from_pdfium_page")
def test_pdf_extraction_populates_images(mock_extract):
    _ext = pytest.importorskip("nemo_retriever.pdf.extract")
    pdfium = pytest.importorskip("pypdfium2")

    mock_img = MagicMock(image=_make_test_png_b64(), bbox=(10, 20, 100, 200), max_width=612, max_height=792)
    mock_extract.return_value = [mock_img]

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    result = _ext.pdf_extraction(
        pd.DataFrame([{"bytes": buf.getvalue(), "path": "t.pdf", "page_number": 1}]), extract_images=True
    )
    images = result.iloc[0]["images"]
    assert len(images) == 1
    assert images[0]["text"] == ""
    assert abs(images[0]["bbox_xyxy_norm"][0] - 10 / 612) < 1e-6


def test_explode_includes_captioned_images():
    from nemo_retriever.ingest_modes.inprocess import explode_content_to_rows

    b64 = _make_test_png_b64()
    df = pd.DataFrame(
        [
            {
                "text": "page",
                "page_image": {"image_b64": b64},
                "images": [{"text": "a dog", "bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "image_b64": b64}],
                "tables": [],
                "charts": [],
                "infographics": [],
            }
        ]
    )
    result = explode_content_to_rows(df, content_columns=("table", "chart", "infographic", "images"))
    assert len(result) == 2  # page text + image caption

    # Default columns exclude images
    result2 = explode_content_to_rows(df)
    assert len(result2) == 1


def test_context_text_prepended_to_prompt():
    from nemo_retriever.caption.caption import caption_images

    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["captioned with context"]

    df = _make_page_df(num_images=1)
    df.at[0, "text"] = "The quick brown fox jumps over the lazy dog."

    result = caption_images(df, model=mock_model, context_text_max_chars=100)

    assert result.iloc[0]["images"][0]["text"] == "captioned with context"
    # The prompt passed to caption_batch should contain the page text.
    call_kwargs = mock_model.caption_batch.call_args[1]
    assert "quick brown fox" in call_kwargs["prompt"]
    assert "Text near this image:" in call_kwargs["prompt"]


def test_caption_images_skips_small_images():
    from nemo_retriever.caption.caption import caption_images

    tiny_b64 = _make_test_png_b64(size=(1, 1))
    images = [{"bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "text": "", "image_b64": tiny_b64}]
    df = pd.DataFrame([{"text": "page", "images": images, "tables": [], "charts": [], "infographics": []}])

    mock_model = MagicMock()
    result = caption_images(df, model=mock_model)
    mock_model.caption_batch.assert_not_called()
    assert result.iloc[0]["images"][0]["text"] == ""
