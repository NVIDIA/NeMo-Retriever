# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for rerank image recovery helpers."""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

from nemo_retriever.common.io.image_store import (
    image_mime_type_from_uri,
    load_image_b64_from_uri,
    render_page_image_b64,
    validate_image_uri,
)


class TestLoadImageB64FromUri:
    def test_round_trip(self, tmp_path: Path):
        raw = b"fake image bytes"
        dest = tmp_path / "image.bin"
        dest.write_bytes(raw)

        result = load_image_b64_from_uri(dest.as_uri())

        assert result is not None
        assert base64.b64decode(result) == raw

    def test_missing_file_returns_none(self):
        assert load_image_b64_from_uri("file:///nonexistent/path/image.png") is None

    def test_validated_image_round_trip(self, tmp_path: Path):
        raw = b"fake image bytes"
        dest = tmp_path / "image.png"
        dest.write_bytes(raw)

        result = load_image_b64_from_uri(dest.as_uri(), max_bytes=len(raw), validate=True)

        assert result is not None
        assert base64.b64decode(result) == raw

    def test_oversized_image_returns_none(self, tmp_path: Path):
        dest = tmp_path / "image.png"
        dest.write_bytes(b"too large")

        assert load_image_b64_from_uri(dest.as_uri(), max_bytes=3, validate=True) is None

    def test_rejects_non_image_local_path(self, tmp_path: Path):
        dest = tmp_path / "data.csv"
        dest.write_bytes(b"a,b,c")

        assert load_image_b64_from_uri(dest.as_uri(), validate=True) is None

    def test_rejects_private_http_host(self):
        assert load_image_b64_from_uri("http://127.0.0.1/image.png", validate=True) is None


def test_image_mime_type_from_uri():
    assert image_mime_type_from_uri("https://example.com/chart.jpg?download=1") == "image/jpeg"
    assert image_mime_type_from_uri("/data/image.unknown") == "image/png"


def test_validate_image_uri():
    assert validate_image_uri("https://example.com/image.png") is True
    assert validate_image_uri("http://169.254.169.254/latest/meta-data/") is False


class TestRenderPageImageB64:
    def test_renders_one_indexed_pdf_page(self, monkeypatch):
        closed = {"value": False}

        class _FakePdfDocument:
            def __init__(self, path):
                self.path = path

            def __getitem__(self, index):
                return f"page-{index}"

            def close(self):
                closed["value"] = True

        fake_pdfium = types.SimpleNamespace(PdfDocument=_FakePdfDocument)
        fake_extract = types.ModuleType("nemo_retriever.operators.extract.pdf.extract")

        def _render_page_to_base64(page, *, dpi):
            assert page == "page-1"
            assert dpi == 123
            return {"image_b64": "rendered"}

        fake_extract._render_page_to_base64 = _render_page_to_base64
        monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
        monkeypatch.setitem(sys.modules, "nemo_retriever.operators.extract.pdf.extract", fake_extract)

        assert render_page_image_b64("/tmp/doc.pdf", 2, dpi=123) == "rendered"
        assert closed["value"] is True

    def test_negative_page_returns_none(self, monkeypatch):
        class _FakePdfDocument:
            def __init__(self, path):
                self.path = path

            def __getitem__(self, index):
                raise AssertionError("negative page should not be read")

            def close(self):
                pass

        fake_pdfium = types.SimpleNamespace(PdfDocument=_FakePdfDocument)
        fake_extract = types.ModuleType("nemo_retriever.operators.extract.pdf.extract")
        fake_extract._render_page_to_base64 = lambda page, *, dpi: {"image_b64": "rendered"}
        monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
        monkeypatch.setitem(sys.modules, "nemo_retriever.operators.extract.pdf.extract", fake_extract)

        assert render_page_image_b64("/tmp/doc.pdf", 0) is None
