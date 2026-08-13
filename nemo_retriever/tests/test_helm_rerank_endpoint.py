# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm wiring for remote reranker configuration."""

import pytest
import yaml

from tests.test_helm_shared_results import _render


@pytest.mark.parametrize(
    ("topology_args", "expected_modes"),
    (((), {"standalone"}), (("--set", "topology.mode=split"), {"gateway", "realtime", "batch"})),
)
def test_rerank_endpoint_is_rendered_in_each_service_config(topology_args, expected_modes) -> None:
    url = "http://reranker.example:8000/v1/ranking"
    model = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    documents = _render(
        *topology_args,
        "--set-string",
        f"serviceConfig.nimEndpoints.rerankInvokeUrl={url}",
        "--set-string",
        f"serviceConfig.nimEndpoints.rerankModelName={model}",
    )

    configs = [
        yaml.safe_load(document["data"]["retriever-service.yaml"])
        for document in documents
        if document.get("kind") == "ConfigMap" and "retriever-service.yaml" in document.get("data", {})
    ]

    assert {config["mode"] for config in configs} == expected_modes
    for config in configs:
        assert config["nim_endpoints"]["rerank_invoke_url"] == url
        assert config["nim_endpoints"]["rerank_model_name"] == model
