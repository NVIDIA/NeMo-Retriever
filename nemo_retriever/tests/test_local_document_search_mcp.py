# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from tests.test_local_document_search_cli import _write_manifest


def test_mcp_ask_payload_reuses_existing_index(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("The renewal date is May 7.", encoding="utf-8")
    stat = doc.stat()
    index = tmp_path / "index"
    _write_manifest(
        index,
        corpus_root=corpus,
        documents=[
            {
                "path": str(doc),
                "relative_path": "guide.txt",
                "input_type": "txt",
                "extension": ".txt",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "page_count": None,
            }
        ],
    )

    from nemo_retriever.local import document_search
    from nemo_retriever.local.mcp_server import local_document_ask_payload

    monkeypatch.setattr(
        document_search,
        "_table_info",
        lambda *args, **kwargs: {
            "readable": True,
            "uri": str(index / "lancedb"),
            "table": "local-documents",
            "table_exists": True,
            "row_count": 1,
            "error": None,
        },
    )
    monkeypatch.setattr(document_search, "_validate_local_inference_available", lambda: None)
    monkeypatch.setattr(
        document_search,
        "_run_ingestion_for_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reindex")),
    )

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def query(self, query, top_k):
            return [{"text": "The renewal date is May 7.", "source_id": str(doc), "metadata": "{}"}]

    monkeypatch.setitem(sys.modules, "nemo_retriever.retriever", SimpleNamespace(Retriever=_FakeRetriever))

    payload = local_document_ask_payload(input_path=str(corpus), query="renewal date", index=str(index))

    assert payload["ok"] is True
    assert payload["index_action"] == "reused"
    assert payload["reused_index"] is True
    assert payload["evidence"][0]["chunk_text"] == "The renewal date is May 7."


def test_mcp_status_payload_returns_structured_error_for_missing_index(tmp_path):
    from nemo_retriever.local.mcp_server import local_document_status_payload

    payload = local_document_status_payload(index=str(tmp_path / "missing-index"))

    assert payload["ok"] is False
    assert payload["command"] == "status"
    assert payload["error"]["code"] == "index_not_initialized"


def test_mcp_default_index_is_stable_hash_of_resolved_input_path(tmp_path):
    from nemo_retriever.local.mcp_server import _default_index_for_input_path

    corpus = tmp_path / "docs"
    corpus.mkdir()

    first = _default_index_for_input_path(str(corpus))
    second = _default_index_for_input_path(str(corpus / ".." / "docs"))

    assert first == second
    assert first.parent.as_posix() == ".nemo-retriever"
    assert first.name.startswith("local-index-")
    assert len(first.name.removeprefix("local-index-")) == 12


def test_mcp_ask_uses_hashed_index_when_index_is_omitted(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    doc = corpus / "guide.txt"
    doc.write_text("The renewal date is May 7.", encoding="utf-8")

    from nemo_retriever.local import document_search
    from nemo_retriever.local import mcp_server

    expected_index = mcp_server._default_index_for_input_path(str(corpus))
    seen = {}

    def fake_ask_documents(**kwargs):
        seen.update(kwargs)
        return {
            "command": "ask",
            "ok": True,
            "answer": None,
            "answer_generation": "not_configured",
            "query": kwargs["query"],
            "input_path": str(kwargs["input_path"]),
            "resolved_input_path": str(corpus.resolve()),
            "corpus_root": str(corpus.resolve()),
            "index_path": str(kwargs["index"]),
            "index_action": "created",
            "indexed_now": True,
            "reused_index": False,
            "reindexed": False,
            "reindex_reasons": ["index_not_initialized"],
            "evidence": [],
            "index_metadata": {},
            "warnings": [],
        }

    monkeypatch.setattr(document_search, "ask_documents", fake_ask_documents)
    monkeypatch.setattr(mcp_server, "ask_documents", fake_ask_documents)

    payload = mcp_server.local_document_ask_payload(input_path=str(corpus), query="renewal date")

    assert payload["ok"] is True
    assert seen["index"] == expected_index


def test_mcp_server_imports_and_exposes_tool_registration():
    from nemo_retriever.local import mcp_server

    assert mcp_server.mcp is not None
    assert callable(mcp_server.local_document_ask_payload)
    assert callable(mcp_server.local_document_search_payload)
    assert callable(mcp_server.local_document_status_payload)


def test_mcp_uses_writable_hf_modules_cache_when_default_is_bad(tmp_path, monkeypatch):
    from nemo_retriever.local import mcp_server

    bad_cache_file = tmp_path / "not-a-directory"
    bad_cache_file.write_text("blocked", encoding="utf-8")
    fallback_base = tmp_path / "mcp-cache"
    monkeypatch.setenv("HF_MODULES_CACHE", str(bad_cache_file))
    monkeypatch.setenv("NEMO_RETRIEVER_MCP_CACHE_DIR", str(fallback_base))

    mcp_server._ensure_writable_hf_modules_cache()

    expected = fallback_base / "huggingface" / "modules"
    assert os.environ["HF_MODULES_CACHE"] == str(expected)
    assert expected.is_dir()
