# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAPI ``info.version`` must track the running service release."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.version import get_service_api_version


def test_openapi_version_matches_service_api_version() -> None:
    app = create_app(ServiceConfig(mode="gateway"))

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert schema["info"]["version"] == get_service_api_version()
    assert schema["info"]["version"] != "26.8.0"


def test_openapi_version_respects_service_version_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVER_SERVICE_VERSION", "26.08")
    app = create_app(ServiceConfig(mode="gateway"))

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert schema["info"]["version"] == "26.08"
    assert get_service_api_version() == "26.08"
