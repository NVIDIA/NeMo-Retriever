# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.service.config import NimEndpointsConfig
from nemo_retriever.service.services.pipeline_executor import build_embed_params


def test_build_embed_params_returns_none_without_endpoint() -> None:
    nim = NimEndpointsConfig()
    assert build_embed_params(nim) is None


def test_build_embed_params_from_nim_config() -> None:
    nim = NimEndpointsConfig(
        embed_invoke_url="http://embed-nim/v1/embeddings",
        embed_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
        api_key="k",
    )
    ep = build_embed_params(nim)
    assert ep is not None
    assert ep.embed_invoke_url == "http://embed-nim/v1/embeddings"
    assert ep.model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert ep.embed_model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert ep.api_key == "k"
