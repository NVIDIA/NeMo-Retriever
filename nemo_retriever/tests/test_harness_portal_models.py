# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_models_catalog_includes_both_supported_parse_versions() -> None:
    from nemo_retriever.harness.portal.app import _AVAILABLE_MODELS

    parse_model_ids = {model["id"] for model in _AVAILABLE_MODELS if model["type"] == "document-parser"}

    assert parse_model_ids == {
        "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        "nvidia/NVIDIA-Nemotron-Parse-2.0",
    }


@pytest.mark.parametrize(
    "model_id",
    [
        "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        "nvidia/NVIDIA-Nemotron-Parse-2.0",
    ],
)
def test_parse_request_accepts_supported_models(model_id: str) -> None:
    from nemo_retriever.harness.portal.app import ParseTestRequest

    request = ParseTestRequest(model_id=model_id, image_b64="aW1hZ2U=")

    assert request.model_id == model_id


def test_parse_request_rejects_non_parse_model_before_loading() -> None:
    from nemo_retriever.harness.portal.app import ParseTestRequest

    with pytest.raises(ValidationError):
        ParseTestRequest(model_id="nvidia/llama-nemotron-embed-1b-v2", image_b64="aW1hZ2U=")
