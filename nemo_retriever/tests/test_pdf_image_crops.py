# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDF image crops land on the image at every page rotation and reuse the page-element render."""

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

# Max mean absolute difference (0-255) between crops cut from renders at different scales.
CROP_TOLERANCE = 8.0

# 612x792 renders at scale 1024/792 (> 1, crops are downscaled); 2000x1500 at 1024/2000 (< 1, upscaled).
PAGE_SIZES = [(612, 792), (2000, 1500)]
ROTATIONS = [0, 90, 180, 270]


def _make_pdf(width: float, height: float, rotation: int, origin: float = 0.0) -> bytes:
    """One off-center RGB ramp image on a white page, so a misplaced crop shows white."""
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(width, height)
    ramp = np.linspace(0, 255, 256, dtype=np.uint8)
    pixels = np.stack(
        [np.tile(ramp, (256, 1)), np.tile(ramp[:, None], (1, 256)), np.full((256, 256), 128, np.uint8)], -1
    )
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(Image.fromarray(pixels)))
    matrix = pdfium.PdfMatrix().scale(width / 3, height / 3).translate(origin + width / 4, origin + height / 2)
    image.set_matrix(matrix)
    page.insert_obj(image)
    page.gen_content()
    if origin:
        page.set_mediabox(origin, origin, origin + width, origin + height)
    page.set_rotation(rotation)
    buf = io.BytesIO()
    pdf.save(buf)
    pdf.close()
    return buf.getvalue()


def _page(*args, **kwargs):
    return pdfium.PdfDocument(_make_pdf(*args, **kwargs))[0]


def _decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(int)


def _drawn_bbox(page) -> tuple:
    """Box of the non-white pixels in a scale-1 render, i.e. where a viewer sees the image."""
    rendered = page.render(scale=1).to_numpy()[:, :, :3]
    ys, xs = np.where((rendered < 250).any(-1))
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


@pytest.mark.parametrize("origin", [0.0, 100.0])
@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("size", PAGE_SIZES)
def test_crop_lands_on_the_image(size, rotation, origin):
    page = _page(*size, rotation, origin=origin)
    page_rgb = _extract._render_page_rgb(page, dpi=200, render_mode="fit_to_model")

    (crop,) = _pdfium.extract_image_like_objects_from_pdfium_page(page, page_rgb)

    assert np.allclose(crop.bbox, _drawn_bbox(page), atol=1)
    assert (crop.max_width, crop.max_height) == (page.get_width(), page.get_height())
    pixels = _decode(crop.image)
    assert pixels.shape == (crop.height, crop.width, 3)
    assert (pixels < 250).any(-1).mean() > 0.98


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_crop_is_upright_as_displayed(rotation):
    upright = _decode(_pdfium.extract_image_like_objects_from_pdfium_page(_page(612, 792, 0))[0].image)
    rotated = _decode(_pdfium.extract_image_like_objects_from_pdfium_page(_page(612, 792, rotation))[0].image)

    expected = np.rot90(upright, k=-(rotation // 90))  # PDF rotation is clockwise.
    assert rotated.shape == expected.shape
    assert np.abs(rotated - expected).mean() <= CROP_TOLERANCE


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("size", PAGE_SIZES)
def test_crops_from_the_page_render_match_a_scale_one_render(size, rotation):
    page = _page(*size, rotation)
    page_rgb = _extract._render_page_rgb(page, dpi=200, render_mode="fit_to_model")

    from_page_render = _pdfium.extract_image_like_objects_from_pdfium_page(page, page_rgb)
    from_scale_one = _pdfium.extract_image_like_objects_from_pdfium_page(page)

    assert [c.bbox for c in from_page_render] == [c.bbox for c in from_scale_one]
    for got, expected in zip(from_page_render, from_scale_one):
        assert np.abs(_decode(got.image) - _decode(expected.image)).mean() <= CROP_TOLERANCE


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
