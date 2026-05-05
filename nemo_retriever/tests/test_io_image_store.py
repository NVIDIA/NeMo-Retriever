# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for rerank image recovery helpers."""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

from nemo_retriever.io.image_store import load_image_b64_from_uri, render_page_image_b64


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
        fake_extract = types.ModuleType("nemo_retriever.pdf.extract")

        def _render_page_to_base64(page, *, dpi):
            assert page == "page-1"
            assert dpi == 123
            return {"image_b64": "rendered"}

        fake_extract._render_page_to_base64 = _render_page_to_base64
        monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
        monkeypatch.setitem(sys.modules, "nemo_retriever.pdf.extract", fake_extract)

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
        fake_extract = types.ModuleType("nemo_retriever.pdf.extract")
        fake_extract._render_page_to_base64 = lambda page, *, dpi: {"image_b64": "rendered"}
        monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
        monkeypatch.setitem(sys.modules, "nemo_retriever.pdf.extract", fake_extract)

        assert render_page_image_b64("/tmp/doc.pdf", 0) is None
