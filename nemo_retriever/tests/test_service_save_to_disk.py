# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 tests for ``ServiceIngestor.save_to_disk()``.

``save_to_disk`` is a *client-side* concern in service run_mode: the
worker has no view into the caller's filesystem, so the ingestor calls
``GET /v1/ingest/status/{id}`` after each ``document_complete`` event
and writes the returned ``result_data`` to disk.
"""

from __future__ import annotations

import gzip
import io
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from nemo_retriever.service_ingestor import ServiceIngestor


def test_save_to_disk_requires_output_directory() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="output_directory"):
        ing.save_to_disk()


def test_save_to_disk_rejects_unsupported_compression(tmp_path: Path) -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="compression"):
        ing.save_to_disk(output_directory=str(tmp_path), compression="bzip2")


def test_save_to_disk_creates_target_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out"
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.save_to_disk(output_directory=str(target))
    assert target.exists()
    assert ing._save_to_disk_dir == target
    assert ing._save_to_disk_compression == "gzip"


def test_save_to_disk_compression_none(tmp_path: Path) -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.save_to_disk(output_directory=str(tmp_path), compression=None)
    assert ing._save_to_disk_compression is None


# ----------------------------------------------------------------------
# _save_document_to_disk(): exercised with a fake urlopen
# ----------------------------------------------------------------------


@contextmanager
def _stub_status_response(body: dict[str, Any]):
    """Patch ``urllib.request.urlopen`` to yield a single JSON response."""

    class _FakeResp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    payload = json.dumps(body).encode("utf-8")
    with patch("urllib.request.urlopen", return_value=_FakeResp(payload)) as mock:
        yield mock


def test_save_document_writes_gzip_json_by_default(tmp_path: Path) -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.save_to_disk(output_directory=str(tmp_path))
    rows = [{"page": 1, "text": "hello"}, {"page": 2, "text": "world"}]

    with _stub_status_response({"id": "doc-123", "result_data": rows}):
        out = ing._save_document_to_disk("doc-123")

    assert out == tmp_path / "doc-123.json.gz"
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        body = json.load(fh)
    assert body == {"document_id": "doc-123", "rows": rows}


def test_save_document_writes_plain_json_when_compression_none(tmp_path: Path) -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.save_to_disk(output_directory=str(tmp_path), compression=None)
    rows = [{"chunk": "a"}, {"chunk": "b"}]

    with _stub_status_response({"id": "doc-1", "result_data": rows}):
        out = ing._save_document_to_disk("doc-1")

    assert out == tmp_path / "doc-1.json"
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body == {"document_id": "doc-1", "rows": rows}


def test_save_document_handles_empty_result_data(tmp_path: Path) -> None:
    """A document with no extracted rows still produces an artifact."""
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.save_to_disk(output_directory=str(tmp_path), compression=None)

    with _stub_status_response({"id": "empty", "result_data": []}):
        out = ing._save_document_to_disk("empty")

    body = json.loads(out.read_text(encoding="utf-8"))
    assert body == {"document_id": "empty", "rows": []}


def test_save_document_rejects_empty_id(tmp_path: Path) -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.save_to_disk(output_directory=str(tmp_path))
    with pytest.raises(ValueError, match="empty document_id"):
        ing._save_document_to_disk("")


def test_save_document_without_enabling_raises(tmp_path: Path) -> None:
    """Calling the helper without first calling save_to_disk() is a programming error."""
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(RuntimeError, match="save_to_disk was never enabled"):
        ing._save_document_to_disk("x")


def test_save_document_authorisation_header_sent_when_token_present(tmp_path: Path) -> None:
    ing = ServiceIngestor(base_url="http://example:7670", api_token="sekret")
    ing.save_to_disk(output_directory=str(tmp_path), compression=None)

    with _stub_status_response({"result_data": []}) as mock:
        ing._save_document_to_disk("doc-x")

    # First positional arg is the urllib Request object; verify header.
    request = mock.call_args.args[0]
    assert request.headers.get("Authorization") == "Bearer sekret"
