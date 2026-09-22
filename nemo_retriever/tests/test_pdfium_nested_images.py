# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for raster images nested in PDF Form XObjects."""

from pathlib import Path

import pytest


def _pdf_with_decode_variants() -> bytes:
    raw_pixels = bytes(range(100))
    form_stream = b"q 10 0 0 10 10 10 cm /Im1 Do Q\nq 10 0 0 10 30 10 cm /Im2 Do Q"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] "
            b"/Resources << /XObject << /Fm1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length 12 >>\nstream\nq /Fm1 Do Q\nendstream",
        (
            b"<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] "
            b"/Resources << /XObject << /Im1 6 0 R /Im2 7 0 R >> >> /Length "
            + str(len(form_stream)).encode()
            + b" >>\nstream\n"
            + form_stream
            + b"\nendstream"
        ),
        (
            b"<< /Type /XObject /Subtype /Image /Width 10 /Height 10 /ColorSpace /DeviceGray "
            b"/BitsPerComponent 8 /Decode [0 1] /Length 100 >>\nstream\n" + raw_pixels + b"\nendstream"
        ),
        (
            b"<< /Type /XObject /Subtype /Image /Width 10 /Height 10 /ColorSpace /DeviceGray "
            b"/BitsPerComponent 8 /Decode [1 0] /Length 100 >>\nstream\n" + raw_pixels + b"\nendstream"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())

    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
    pdf.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode())
    return bytes(pdf)


def test_extract_nested_form_images_decodes_each_placement(monkeypatch) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

    original_get_bitmap = pdfium.PdfImage.get_bitmap
    decode_count = 0

    def tracked_get_bitmap(self, *args, **kwargs):
        nonlocal decode_count
        decode_count += 1
        return original_get_bitmap(self, *args, **kwargs)

    monkeypatch.setattr(pdfium.PdfImage, "get_bitmap", tracked_get_bitmap)

    fixture = Path(__file__).resolve().parents[2] / "data" / "test-page-form.pdf"
    document = pdfium.PdfDocument(fixture)
    page = document[0]

    try:
        images = extract_nested_simple_images_from_pdfium_page(page)
    finally:
        page.close()
        document.close()

    assert len(images) == 18
    assert all(0 <= image.bbox[0] < image.bbox[2] <= image.max_width for image in images)
    assert all(0 <= image.bbox[1] < image.bbox[3] <= image.max_height for image in images)
    assert {(image.width, image.height) for image in images} == {(256, 256), (512, 512)}
    assert len({image.image for image in images}) == 5
    assert decode_count == 18


def test_decode_state_is_not_reused_and_source_budget_is_aggregate() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

    document = pdfium.PdfDocument(_pdf_with_decode_variants())
    page = document[0]
    try:
        images = extract_nested_simple_images_from_pdfium_page(page)
        assert len(images) == 2
        assert images[0].image != images[1].image

        with pytest.raises(RuntimeError, match="raw source-byte limit"):
            extract_nested_simple_images_from_pdfium_page(page, max_source_bytes=199)
    finally:
        page.close()
        document.close()


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max_images": 17}, "count exceeds"),
        ({"max_decoded_pixels": 1}, "decoded-pixel limit"),
        ({"max_source_bytes": 1}, "raw source-byte limit"),
    ],
)
def test_extract_nested_form_images_enforces_predecode_budgets(monkeypatch, limits, message) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

    fixture = Path(__file__).resolve().parents[2] / "data" / "test-page-form.pdf"
    document = pdfium.PdfDocument(fixture)
    page = document[0]
    original_get_bitmap = pdfium.PdfImage.get_bitmap
    decode_count = 0

    def tracked_get_bitmap(self, *args, **kwargs):
        nonlocal decode_count
        decode_count += 1
        return original_get_bitmap(self, *args, **kwargs)

    monkeypatch.setattr(pdfium.PdfImage, "get_bitmap", tracked_get_bitmap)
    try:
        with pytest.raises(RuntimeError, match=message):
            extract_nested_simple_images_from_pdfium_page(page, **limits)
    finally:
        page.close()
        document.close()

    if "max_images" not in limits:
        assert decode_count == 0


@pytest.mark.parametrize(
    "position",
    [
        (-20.0, 10.0, -10.0, 20.0),
        (110.0, 10.0, 120.0, 20.0),
        (10.0, -20.0, 20.0, -10.0),
        (10.0, 110.0, 20.0, 120.0),
        (10.1, 10.1, 10.2, 10.2),
    ],
)
def test_visible_page_bbox_rejects_off_page_and_degenerate_boxes(position) -> None:
    from nemo_retriever.common.api.util.pdf.pdfium import _visible_page_bbox

    assert _visible_page_bbox(position, 100.0, 100.0) is None


def test_visible_page_bbox_clips_partial_intersection() -> None:
    from nemo_retriever.common.api.util.pdf.pdfium import _visible_page_bbox

    assert _visible_page_bbox((-10.0, 20.0, 30.0, 120.0), 100.0, 100.0) == [0, 0, 30, 80]


@pytest.mark.parametrize(
    "limits",
    [
        {"max_images": 0},
        {"max_decoded_pixels": 0},
        {"max_source_bytes": 0},
    ],
)
def test_extract_nested_form_images_rejects_nonpositive_budgets(limits) -> None:
    from nemo_retriever.common.api.util.pdf.pdfium import extract_nested_simple_images_from_pdfium_page

    with pytest.raises(ValueError, match="must all be positive"):
        extract_nested_simple_images_from_pdfium_page(object(), **limits)
