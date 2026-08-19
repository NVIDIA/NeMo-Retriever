# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LiteVLMClient — image loading, prompt building, and generation."""

import struct
import zlib

from nemo_retriever.models.llm.clients.vlm_litellm import _build_multimodal_rag_prompt
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
