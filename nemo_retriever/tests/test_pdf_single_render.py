# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image/shape crops are cut from the page-element render instead of a second pdfium render."""

import base64
import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pdfium = pytest.importorskip("pypdfium2")
from PIL import Image  # noqa: E402

from nemo_retriever.common.api.util.pdf import pdfium as _pdfium  # noqa: E402
from nemo_retriever.operators.extract.pdf import extract as _extract  # noqa: E402

# Max mean absolute difference (0-255) between a crop cut from the resized page-element render
# and the same crop cut from a native scale-1 render.
CROP_TOLERANCE = 8.0

# 612x792 renders at scale 1024/792 (> 1, crops are downscaled); 2000x1500 at 1024/2000 (< 1, upscaled).
PAGE_SIZES = [(612, 792), (2000, 1500)]
ROTATIONS = [0, 90, 180, 270]


def _make_pdf(width: float, height: float, rotation: int) -> bytes:
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(width, height)
    ramp = np.linspace(0, 255, 256, dtype=np.uint8)
    pixels = np.stack(
        [np.tile(ramp, (256, 1)), np.tile(ramp[:, None], (1, 256)), np.full((256, 256), 128, np.uint8)], -1
    )
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(Image.fromarray(pixels)))
    image.set_matrix(pdfium.PdfMatrix().scale(width / 3, height / 3).translate(width / 4, height / 2))
    page.insert_obj(image)
    page.gen_content()
    page.set_rotation(rotation)
    buf = io.BytesIO()
    pdf.save(buf)
    pdf.close()
    return buf.getvalue()


def _decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(int)


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("size", PAGE_SIZES)
def test_crops_match_native_render(size, rotation):
    page = pdfium.PdfDocument(_make_pdf(*size, rotation))[0]
    native = _pdfium.pdfium_pages_to_numpy([page], render_dpi=72, rotation=rotation)[0][0]
    bboxes = _pdfium.extract_merged_images_from_pdfium_page(page)
    expected = [(bbox, crop) for bbox in bboxes if (crop := _pdfium.crop_image(native, bbox)) is not None]

    page_rgb = _extract._render_page_rgb(page, dpi=200, render_mode="fit_to_model")
    crops = _pdfium.extract_image_like_objects_from_pdfium_page(page, page_rgb)

    assert [tuple(c.bbox) for c in crops] == [tuple(b) for b, _ in expected]
    for c, (_, crop) in zip(crops, expected):
        got = _decode(c.image)
        assert got.shape == crop.shape
        assert np.abs(got - crop.astype(int)).mean() <= CROP_TOLERANCE


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_crops_without_page_image_render_at_scale_one(rotation):
    page = pdfium.PdfDocument(_make_pdf(612, 792, rotation))[0]
    native = _pdfium.pdfium_pages_to_numpy([page], render_dpi=72, rotation=rotation)[0][0]
    bboxes = _pdfium.extract_merged_images_from_pdfium_page(page)
    expected = [crop for bbox in bboxes if (crop := _pdfium.crop_image(native, bbox)) is not None]

    crops = _pdfium.extract_image_like_objects_from_pdfium_page(page)

    assert len(crops) == len(expected)
    for c, crop in zip(crops, expected):
        assert np.array_equal(_decode(c.image), crop.astype(int))


@pytest.mark.parametrize("rotation", [0, 180])
@pytest.mark.parametrize("size", PAGE_SIZES)
def test_upright_pages_yield_a_crop(size, rotation):
    page = pdfium.PdfDocument(_make_pdf(*size, rotation))[0]
    page_rgb = _extract._render_page_rgb(page, dpi=200, render_mode="fit_to_model")
    assert len(_pdfium.extract_image_like_objects_from_pdfium_page(page, page_rgb)) == 1


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("size", PAGE_SIZES)
def test_pdf_extraction_renders_each_page_once(size, rotation):
    pdf_bytes = _make_pdf(*size, rotation)
    real_render = pdfium.PdfPage.render
    with patch.object(pdfium.PdfPage, "render", autospec=True, side_effect=real_render) as render:
        result = _extract.pdf_extraction(
            pd.DataFrame([{"bytes": pdf_bytes, "path": "t.pdf", "page_number": 1}]),
            extract_text=True,
            extract_images=True,
            extract_tables=True,
        )
    assert render.call_count == 1

    page = pdfium.PdfDocument(pdf_bytes)[0]
    assert result.iloc[0]["page_image"] == _extract._render_page_to_base64(page)
