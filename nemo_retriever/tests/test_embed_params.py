# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for EmbedParams modality validation and IMAGE_MODALITIES constant.
"""

import warnings

import pytest

from nemo_retriever.common.params.models import EmbedParams, IMAGE_MODALITIES
from nemo_retriever.common.params.utils import build_embed_option_kwargs, route_embed_model_kwargs


def test_image_text_alias_is_rejected():
    """'image_text' should be rejected so users must pass the canonical 'text_image'."""
    with pytest.raises(ValueError, match="text_image"):
        EmbedParams(
            embed_modality="image_text",
            text_elements_modality="image_text",
            structured_elements_modality="image_text",
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("text", "text"),
        ("image", "image"),
        ("text_image", "text_image"),
        (None, None),
    ],
)
def test_normalize_modality_passthrough(value, expected):
    """Allowed modality values pass through unchanged."""
    kwargs = {}
    if value is not None:
        kwargs["embed_modality"] = value
    kwargs["text_elements_modality"] = value
    kwargs["structured_elements_modality"] = value

    params = EmbedParams(**kwargs)

    if value is not None:
        assert params.embed_modality == expected
    assert params.text_elements_modality == expected
    assert params.structured_elements_modality == expected


def test_image_modalities_constant():
    """IMAGE_MODALITIES contains only canonical image-bearing modalities."""
    assert IMAGE_MODALITIES == {"image", "text_image"}
    assert isinstance(IMAGE_MODALITIES, frozenset)


def test_build_embed_option_kwargs_defers_remote_model_provider_prefix():
    kwargs = build_embed_option_kwargs(
        "https://litellm.example.com/v1/embeddings",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_model_provider_prefix="nvidia",
    )

    assert kwargs["model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert kwargs["embed_model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert kwargs["embed_model_provider_prefix"] == "nvidia"

    routed = route_embed_model_kwargs(kwargs)

    assert routed["model_name"] == "nvidia/nvidia/llama-nemotron-embed-vl-1b-v2"
    assert routed["embed_model_name"] == "nvidia/nvidia/llama-nemotron-embed-vl-1b-v2"
    assert "embed_model_provider_prefix" not in routed


def test_build_embed_option_kwargs_leaves_model_unchanged_without_prefix():
    kwargs = build_embed_option_kwargs(
        "https://integrate.api.nvidia.com/v1/embeddings",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
    )

    assert kwargs["model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert kwargs["embed_model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"


def test_build_embed_option_kwargs_prefix_supports_other_vendor_namespaces():
    kwargs = build_embed_option_kwargs(
        "https://litellm.example.com/v1/embeddings",
        "mistral/embed-small",
        embed_model_provider_prefix="acme",
    )

    assert kwargs["model_name"] == "mistral/embed-small"
    assert kwargs["embed_model_name"] == "mistral/embed-small"
    assert kwargs["embed_model_provider_prefix"] == "acme"


def test_build_embed_option_kwargs_prefix_supports_bare_model_name():
    kwargs = build_embed_option_kwargs(
        "https://litellm.example.com/v1/embeddings",
        "nv-embedqa-e5-v5",
        embed_model_provider_prefix="nvidia",
    )

    assert kwargs["model_name"] == "nv-embedqa-e5-v5"
    assert kwargs["embed_model_name"] == "nv-embedqa-e5-v5"
    assert kwargs["embed_model_provider_prefix"] == "nvidia"


def test_build_embed_option_kwargs_retains_prefix_until_endpoint_is_resolved():
    kwargs = build_embed_option_kwargs(
        None,
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_model_provider_prefix="nvidia",
    )

    assert kwargs["model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert kwargs["embed_model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert kwargs["embed_model_provider_prefix"] == "nvidia"


def test_build_embed_option_kwargs_retains_prefix_when_model_is_omitted():
    kwargs = build_embed_option_kwargs(
        "https://inference-api.nvidia.com/v1",
        None,
        embed_model_provider_prefix="nvidia",
    )

    assert kwargs == {
        "embed_invoke_url": "https://inference-api.nvidia.com/v1",
        "embedding_endpoint": "https://inference-api.nvidia.com/v1",
        "embed_model_provider_prefix": "nvidia",
    }


# ===================================================================
# embed_granularity
# ===================================================================


class TestEmbedParamsGranularity:
    def test_default_is_element(self):
        params = EmbedParams()
        assert params.embed_granularity == "element"

    def test_page_accepted(self):
        params = EmbedParams(embed_granularity="page")
        assert params.embed_granularity == "page"

    def test_invalid_value_rejected(self):
        with pytest.raises(Exception):
            EmbedParams(embed_granularity="invalid")

    def test_warning_on_per_type_modality_with_page(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EmbedParams(
                embed_granularity="page",
                text_elements_modality="image",
            )
            assert len(w) == 1
            assert "ignored" in str(w[0].message).lower()

    def test_no_warning_on_element_granularity_with_overrides(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EmbedParams(
                embed_granularity="element",
                text_elements_modality="image",
                structured_elements_modality="text_image",
            )
            assert len(w) == 0
