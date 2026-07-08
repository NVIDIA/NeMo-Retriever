# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LiteVLMClient — image loading, prompt building, and generation."""

import base64
import struct
import zlib
from unittest.mock import MagicMock, patch

from nemo_retriever.models.llm.clients.vlm_litellm import (
    _MAX_IMAGE_BYTES,
    _build_multimodal_rag_prompt,
    _load_image_as_base64,
    _mime_type_from_uri,
    _validate_http_uri,
    _validate_local_path,
)
from nemo_retriever.models.llm.types import MultimodalChunk


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_png(size: int = 64) -> bytes:
    """Return a minimal valid 1×1 PNG."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr)
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", ihdr_crc)
    idat_data = zlib.compress(b"\x00\xff\xff\xff")
    idat_crc = zlib.crc32(b"IDAT" + idat_data)
    idat_chunk = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND")
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return sig + ihdr_chunk + idat_chunk + iend_chunk


# ── _mime_type_from_uri ───────────────────────────────────────────────────────


class TestMimeTypeFromUri:
    def test_png(self) -> None:
        assert _mime_type_from_uri("/data/img/chart.png") == "image/png"

    def test_jpeg(self) -> None:
        assert _mime_type_from_uri("/data/img/photo.jpg") == "image/jpeg"

    def test_webp(self) -> None:
        assert _mime_type_from_uri("https://host/img.webp") == "image/webp"

    def test_unknown_falls_back_to_png(self) -> None:
        assert _mime_type_from_uri("/data/img/noext") == "image/png"


# ── _validate_http_uri ────────────────────────────────────────────────────────


class TestValidateHttpUri:
    def test_public_url_allowed(self) -> None:
        assert _validate_http_uri("https://example.com/image.png") is True

    def test_aws_metadata_blocked(self) -> None:
        assert _validate_http_uri("http://169.254.169.254/latest/meta-data/") is False

    def test_gcp_metadata_blocked(self) -> None:
        assert _validate_http_uri("http://metadata.google.internal/computeMetadata/v1/") is False

    def test_private_ip_blocked(self) -> None:
        assert _validate_http_uri("http://192.168.1.100/image.png") is False

    def test_loopback_blocked(self) -> None:
        assert _validate_http_uri("http://127.0.0.1/image.png") is False


# ── _validate_local_path ──────────────────────────────────────────────────────


class TestValidateLocalPath:
    def test_png_allowed(self, tmp_path) -> None:
        p = tmp_path / "chart.png"
        p.write_bytes(b"")
        assert _validate_local_path(str(p)) is True

    def test_jpeg_allowed(self, tmp_path) -> None:
        p = tmp_path / "photo.jpeg"
        p.write_bytes(b"")
        assert _validate_local_path(str(p)) is True

    def test_non_image_rejected(self, tmp_path) -> None:
        p = tmp_path / "passwd"
        p.write_bytes(b"root:x:0:0")
        assert _validate_local_path(str(p)) is False

    def test_text_file_rejected(self, tmp_path) -> None:
        p = tmp_path / "data.txt"
        p.write_bytes(b"hello")
        assert _validate_local_path(str(p)) is False


# ── _load_image_as_base64 ─────────────────────────────────────────────────────


class TestLoadImageAsBase64:
    def test_local_png_loads(self, tmp_path) -> None:
        p = tmp_path / "img.png"
        png = _make_png()
        p.write_bytes(png)
        result = _load_image_as_base64(str(p))
        assert result == base64.b64encode(png).decode("ascii")

    def test_file_scheme_stripped(self, tmp_path) -> None:
        p = tmp_path / "img.png"
        p.write_bytes(_make_png())
        result = _load_image_as_base64(f"file://{p}")
        assert result is not None

    def test_missing_file_returns_none(self, tmp_path) -> None:
        result = _load_image_as_base64(str(tmp_path / "missing.png"))
        assert result is None

    def test_oversized_local_file_skipped(self, tmp_path) -> None:
        p = tmp_path / "big.png"
        p.write_bytes(b"x")
        with patch("os.path.getsize", return_value=_MAX_IMAGE_BYTES + 1):
            result = _load_image_as_base64(str(p))
        assert result is None

    def test_non_image_extension_rejected(self, tmp_path) -> None:
        p = tmp_path / "data.csv"
        p.write_bytes(b"a,b,c")
        result = _load_image_as_base64(str(p))
        assert result is None

    def test_blocked_ssrf_host_returns_none(self) -> None:
        result = _load_image_as_base64("http://169.254.169.254/latest/meta-data/")
        assert result is None

    def test_http_oversized_returns_none(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers.get.return_value = str(_MAX_IMAGE_BYTES + 1)
        mock_resp.read.return_value = b"x" * (_MAX_IMAGE_BYTES + 1)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _load_image_as_base64("https://example.com/huge.png")
        assert result is None

    def test_boto3_missing_returns_none(self) -> None:
        with patch.dict("sys.modules", {"boto3": None}):
            result = _load_image_as_base64("s3://bucket/img.png")
        assert result is None


# ── _build_multimodal_rag_prompt ──────────────────────────────────────────────


class TestBuildMultimodalRagPrompt:
    _SYS = "You are a helpful assistant."

    def test_empty_chunks_produces_no_context_block(self) -> None:
        msgs = _build_multimodal_rag_prompt("q?", [], formatted_rag_system_prompt=self._SYS)
        texts = [b["text"] for b in msgs[1]["content"] if b.get("type") == "text"]
        assert any("no context" in t for t in texts)

    def test_text_only_chunk(self) -> None:
        chunks = [MultimodalChunk(text="Some fact.", content_type="text")]
        msgs = _build_multimodal_rag_prompt("q?", chunks, formatted_rag_system_prompt=self._SYS)
        img_blocks = [b for b in msgs[1]["content"] if b.get("type") == "image_url"]
        assert img_blocks == []

    def test_visual_chunk_with_valid_image(self, tmp_path) -> None:
        p = tmp_path / "chart.png"
        p.write_bytes(_make_png())
        chunks = [MultimodalChunk(text="Bar chart.", image_uri=str(p), content_type="chart")]
        msgs = _build_multimodal_rag_prompt("q?", chunks, formatted_rag_system_prompt=self._SYS)
        img_blocks = [b for b in msgs[1]["content"] if b.get("type") == "image_url"]
        assert len(img_blocks) == 1
        assert img_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_mime_type_inferred_for_jpeg(self, tmp_path) -> None:
        p = tmp_path / "photo.jpg"
        p.write_bytes(_make_png())  # content doesn't matter for MIME test
        chunks = [MultimodalChunk(text="Photo.", image_uri=str(p), content_type="image")]
        msgs = _build_multimodal_rag_prompt("q?", chunks, formatted_rag_system_prompt=self._SYS)
        img_blocks = [b for b in msgs[1]["content"] if b.get("type") == "image_url"]
        assert img_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_failed_image_load_falls_back_to_text(self, tmp_path) -> None:
        chunks = [MultimodalChunk(text="Caption.", image_uri="/nonexistent/img.png", content_type="chart")]
        msgs = _build_multimodal_rag_prompt("q?", chunks, formatted_rag_system_prompt=self._SYS)
        img_blocks = [b for b in msgs[1]["content"] if b.get("type") == "image_url"]
        assert img_blocks == []
        texts = " ".join(b["text"] for b in msgs[1]["content"] if b.get("type") == "text")
        assert "Caption." in texts

    def test_question_appended_at_end(self) -> None:
        msgs = _build_multimodal_rag_prompt("What happened?", [], formatted_rag_system_prompt=self._SYS)
        last_text = msgs[1]["content"][-1]["text"]
        assert "What happened?" in last_text

    def test_system_prompt_role(self) -> None:
        msgs = _build_multimodal_rag_prompt("q?", [], formatted_rag_system_prompt=self._SYS)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == self._SYS
