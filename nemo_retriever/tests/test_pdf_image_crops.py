# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDF image crops land where PDFium draws the image, for every page box and rotation."""

import base64
import ctypes
import io

import numpy as np
import pytest

pdfium = pytest.importorskip("pypdfium2")
import pypdfium2.raw as pdfium_c  # noqa: E402
from PIL import Image  # noqa: E402

from nemo_retriever.common.api.util.pdf import pdfium as _pdfium  # noqa: E402

# Where the image is drawn, in PDF user space: (left, bottom, right, top).
IMAGE_RECT = (250.0, 300.0, 450.0, 420.0)

# (page-tree attributes, page attributes) for each way a PDF can define the rendered page box.
PAGE_BOXES = {
    "mediabox": (b"", b"/MediaBox [0 0 612 792]"),
    "offset_mediabox": (b"", b"/MediaBox [100 100 712 892]"),
    "inherited_mediabox": (b"/MediaBox [0 0 2000 1500]", b""),
    "inherited_inset_cropbox": (b"/MediaBox [0 0 612 792] /CropBox [50 40 562 752]", b""),
    "cropbox_outside_mediabox": (b"", b"/MediaBox [0 0 612 792] /CropBox [-50 -50 562 742]"),
}
ROTATIONS = [0, 90, 180, 270]


def _make_pdf(page_box: str, rotation: int, in_form: bool = False, rects=(IMAGE_RECT,), extra_content=b"") -> bytes:
    """One page drawing a 16x16 four-colour image at each of ``rects``, directly or inside a Form XObject."""
    pages_attrs, page_attrs = PAGE_BOXES[page_box]
    draw_image = b" ".join(
        b"q %g 0 0 %g %g %g cm /Im1 Do Q" % (right - left, top - bottom, left, bottom)
        for left, bottom, right, top in rects
    )
    quadrants = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], np.uint8)
    pixels = quadrants.repeat(8, axis=0).repeat(8, axis=1).tobytes()
    image = b"<< /Type /XObject /Subtype /Image /Width 16 /Height 16 /ColorSpace /DeviceRGB /BitsPerComponent 8"
    image += b" /Length %d >>\nstream\n%s\nendstream" % (len(pixels), pixels)
    image_resources = b"<< /XObject << /Im1 6 0 R >> >>"
    if in_form:
        page_content, page_resources = b"/Fm1 Do", b"<< /XObject << /Fm1 5 0 R >> >>"
    else:
        page_content, page_resources = draw_image + b" " + extra_content, image_resources
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 %s >>" % pages_attrs,
        b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources %s /Rotate %d %s >>"
        % (page_resources, rotation, page_attrs),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(page_content), page_content),
        b"<< /Type /XObject /Subtype /Form /BBox [-5000 -5000 5000 5000] /Resources %s /Length %d >>\nstream\n%s\n"
        b"endstream" % (image_resources, len(draw_image), draw_image),
        image,
    ]
    pdf, offsets = b"%PDF-1.7\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return pdf


def _page(*args, **kwargs):
    return pdfium.PdfDocument(_make_pdf(*args, **kwargs))[0]


def _expected_bbox(page, rect=IMAGE_RECT) -> list:
    """``rect`` mapped by PDFium's own page-to-device transform at scale 1."""
    width, height = page.get_width(), page.get_height()
    corners = []
    for x, y in ((rect[0], rect[1]), (rect[2], rect[3])):
        device_x, device_y = ctypes.c_int(), ctypes.c_int()
        # 10 device pixels per point, so rounding stays well under the 1 pt tolerance.
        pdfium_c.FPDF_PageToDevice(page, 0, 0, round(width * 10), round(height * 10), 0, x, y, device_x, device_y)
        corners.append((device_x.value / 10, device_y.value / 10))
    (ax, ay), (bx, by) = corners
    return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]


def _decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(int)


@pytest.mark.parametrize("in_form", [False, True])
@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("page_box", PAGE_BOXES)
def test_crop_lands_on_the_image(page_box, rotation, in_form):
    page = _page(page_box, rotation, in_form=in_form)

    (crop,) = _pdfium.extract_image_like_objects_from_pdfium_page(page)

    assert np.allclose(crop.bbox, _expected_bbox(page), atol=1)
    assert (crop.max_width, crop.max_height) == (page.get_width(), page.get_height())
    pixels = _decode(crop.image)
    assert pixels.shape == (crop.height, crop.width, 3)
    assert (pixels < 250).any(-1).mean() > 0.98


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("page_box", PAGE_BOXES)
def test_nested_image_box_lands_on_the_image(page_box, rotation):
    page = _page(page_box, rotation, in_form=True)

    (image,) = _pdfium.extract_nested_simple_images_from_pdfium_page(page)

    assert np.allclose(image.bbox, _expected_bbox(page), atol=1)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_crop_is_upright_as_displayed(rotation):
    upright = _decode(_pdfium.extract_image_like_objects_from_pdfium_page(_page("mediabox", 0))[0].image)
    rotated = _decode(_pdfium.extract_image_like_objects_from_pdfium_page(_page("mediabox", rotation))[0].image)

    expected = np.rot90(upright, k=-(rotation // 90))  # PDF rotation is clockwise.
    assert rotated.shape == expected.shape
    assert np.abs(rotated - expected).mean() <= 8


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_off_page_image_does_not_stretch_the_visible_crop(rotation):
    visible, off_page = (100.0, 5.0, 200.0, 95.0), (50.0, -200.0, 300.0, -5.0)
    page = _page("mediabox", rotation, rects=(visible, off_page))

    (crop,) = _pdfium.extract_image_like_objects_from_pdfium_page(page)

    assert np.allclose(crop.bbox, _expected_bbox(page, visible), atol=1)


def test_thin_table_rules_stay_in_their_shape_group():
    rules = b" ".join(b"72 %g 468 0.4 re f" % y for y in (300.2, 308.2, 316.2, 324.2))
    page = _page("mediabox", 0, rects=(), extra_content=rules)

    assert _pdfium.extract_merged_shapes_from_pdfium_page(page) == [[72, 467, 540, 491]]
