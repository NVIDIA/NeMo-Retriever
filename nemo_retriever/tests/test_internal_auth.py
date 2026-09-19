# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated internal-token normalization regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from nemo_retriever.service.auth import _is_gateway_proxy_path, internal_auth_headers
from nemo_retriever.service.config import load_config
from nemo_retriever.service.services.pipeline_executor import (
    _DEFAULT_VECTORDB_WRITE_TIMEOUT_S,
    _post_records_to_vectordb,
)
from nemo_retriever.service.services.pipeline_pool import DocumentWriteContext


def test_internal_auth_headers_strip_secret_whitespace() -> None:
    assert internal_auth_headers("  internal-secret\n") == {"X-NRL-Internal-Token": "internal-secret"}
    assert internal_auth_headers(" \n\t") == {}


def test_gateway_handoff_includes_parameterized_sidecar_delete() -> None:
    assert _is_gateway_proxy_path("/v1/ingest/sidecar")
    assert _is_gateway_proxy_path("/v1/ingest/sidecar/sidecar-id")
    assert not _is_gateway_proxy_path("/v1/ingest/sidecars")


def test_load_config_strips_internal_token_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "service.yaml"
    config_path.write_text("vectordb:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("NRL_INTERNAL_VDB_TOKEN", " internal-secret\n")

    assert load_config(str(config_path)).vectordb.internal_api_token == "internal-secret"


def test_pipeline_vectordb_request_uses_normalized_internal_header() -> None:
    captured: dict[str, Any] = {}

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def urlopen(request: Any, timeout: int) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    with patch("urllib.request.urlopen", urlopen):
        _post_records_to_vectordb(
            [[{"text": "row"}]],
            "http://vectordb:7671",
            "document.txt",
            internal_api_token=" internal-secret\n",
            context=DocumentWriteContext(),
        )

    assert captured["timeout"] == _DEFAULT_VECTORDB_WRITE_TIMEOUT_S
    assert captured["request"].get_header("X-nrl-internal-token") == "internal-secret"
