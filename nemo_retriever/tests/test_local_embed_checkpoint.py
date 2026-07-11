# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dropping in a local embedding checkpoint directory.

Two mechanisms are exercised, both scoped to on-disk checkpoints only so that
registered Hub ids keep their existing behavior:

- The revision pin is bypassed for local model directories only when the
  caller explicitly opts into local-path loading.
- ``create_local_embedder`` routes a local dir to the VL or text embedder based
  on an *explicit* ``vl``/``text`` declaration (arg or ``NRL_LOCAL_EMBED_ARCH``),
  failing loudly rather than inferring.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_retriever.models import (
    LOCAL_EMBED_ARCH_ENV,
    create_local_embedder,
    create_local_query_embedder,
    resolve_embed_model_use_vl,
)
from nemo_retriever.models.hf_model_registry import get_hf_revision

# ---------------------------------------------------------------------------
# Lock: get_hf_revision is bypassed for local dirs, unchanged for Hub ids
# ---------------------------------------------------------------------------


def test_local_dir_requires_explicit_revision_pin_bypass(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="No pinned HuggingFace revision"):
        get_hf_revision(str(tmp_path))

    assert get_hf_revision(str(tmp_path), allow_local_path=True) is None
    assert get_hf_revision(str(tmp_path), strict=True, allow_local_path=True) is None


def test_local_dir_without_model_config_does_not_bypass_revision_pin(tmp_path):
    with pytest.raises(ValueError, match="No pinned HuggingFace revision"):
        get_hf_revision(str(tmp_path), allow_local_path=True)


def test_registered_hub_id_still_pinned():
    assert get_hf_revision("nvidia/llama-nemotron-embed-1b-v2") == "b4caa8456edd360b3b4e938d94ed4398dd437fad"


def test_unregistered_hub_id_still_raises():
    with pytest.raises(ValueError, match="No pinned HuggingFace revision"):
        get_hf_revision("some-org/not-registered")


def test_unregistered_hub_id_non_strict_returns_none():
    assert get_hf_revision("some-org/not-registered", strict=False) is None


# ---------------------------------------------------------------------------
# Routing: local dir -> VL or text embedder by explicit declaration
# ---------------------------------------------------------------------------


def test_resolver_routes_registered_vl_id_without_arch(_patch_embedders):
    assert resolve_embed_model_use_vl("nvidia/llama-nemotron-embed-vl-1b-v2") is True


def test_resolver_routes_registered_text_id_without_arch(_patch_embedders):
    assert resolve_embed_model_use_vl("nvidia/llama-nemotron-embed-1b-v2") is False


def test_resolver_routes_local_dir_from_arch(tmp_path, _patch_embedders):
    assert resolve_embed_model_use_vl(str(tmp_path), model_arch="vl") is True
    assert resolve_embed_model_use_vl(str(tmp_path), model_arch="text") is False


@pytest.fixture(autouse=True)
def _patch_embedders(monkeypatch):
    """Stub the four embedder classes so no real model is loaded.

    Mirrors test_create_local_embedder.py: the ``model.local`` package exposes
    classes lazily, so inject fake submodules directly into ``sys.modules``.
    """
    fake_text_vllm = MagicMock(name="LlamaNemotronEmbed1BV2Embedder")
    fake_text_hf = MagicMock(name="LlamaNemotronEmbed1BV2HFEmbedder")
    fake_vl_hf = MagicMock(name="LlamaNemotronEmbedVL1BV2Embedder")
    fake_vl_vllm = MagicMock(name="LlamaNemotronEmbedVL1BV2VLLMEmbedder")

    text_mod = ModuleType("nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder")
    text_mod.LlamaNemotronEmbed1BV2Embedder = fake_text_vllm

    text_hf_mod = ModuleType("nemo_retriever.models.local.llama_nemotron_embed_1b_v2_hf_embedder")
    text_hf_mod.LlamaNemotronEmbed1BV2HFEmbedder = fake_text_hf

    vl_mod = ModuleType("nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder")
    vl_mod.LlamaNemotronEmbedVL1BV2Embedder = fake_vl_hf
    vl_mod.LlamaNemotronEmbedVL1BV2VLLMEmbedder = fake_vl_vllm

    monkeypatch.setitem(sys.modules, "nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder", text_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.models.local.llama_nemotron_embed_1b_v2_hf_embedder", text_hf_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder", vl_mod)
    monkeypatch.delenv(LOCAL_EMBED_ARCH_ENV, raising=False)

    yield fake_text_vllm, fake_text_hf, fake_vl_hf, fake_vl_vllm


def test_local_dir_arch_vl_routes_to_vl_vllm(tmp_path, _patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    result = create_local_embedder(str(tmp_path), model_arch="vl")  # default backend vllm
    fake_vl_vllm.assert_called_once()
    assert fake_vl_vllm.call_args.kwargs["model_id"] == str(tmp_path)
    assert result is fake_vl_vllm.return_value


def test_local_dir_arch_vl_hf_routes_to_vl_hf(tmp_path, _patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    result = create_local_embedder(str(tmp_path), backend="hf", model_arch="vl")
    fake_vl_hf.assert_called_once()
    assert result is fake_vl_hf.return_value


def test_local_dir_arch_text_hf_routes_to_text_hf(tmp_path, _patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    result = create_local_embedder(str(tmp_path), backend="hf", model_arch="text")
    fake_text_hf.assert_called_once()
    assert fake_text_hf.call_args.kwargs["model_id"] == str(tmp_path)
    assert result is fake_text_hf.return_value


def test_local_dir_arch_text_vllm_routes_to_text_vllm(tmp_path, _patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    result = create_local_embedder(str(tmp_path), model_arch="text")
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_local_dir_without_arch_fails_loud(tmp_path, _patch_embedders):
    with pytest.raises(ValueError, match=LOCAL_EMBED_ARCH_ENV):
        create_local_embedder(str(tmp_path), backend="hf")


def test_local_dir_invalid_arch_fails_loud(tmp_path, _patch_embedders):
    with pytest.raises(ValueError, match=LOCAL_EMBED_ARCH_ENV):
        create_local_embedder(str(tmp_path), backend="hf", model_arch="multimodal")


def test_local_dir_arch_from_env(tmp_path, monkeypatch, _patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    monkeypatch.setenv(LOCAL_EMBED_ARCH_ENV, "vl")
    result = create_local_embedder(str(tmp_path), backend="hf")
    fake_vl_hf.assert_called_once()
    assert result is fake_vl_hf.return_value


def test_explicit_arg_overrides_env(tmp_path, monkeypatch, _patch_embedders):
    _, fake_text_hf, fake_vl_hf, _ = _patch_embedders
    monkeypatch.setenv(LOCAL_EMBED_ARCH_ENV, "vl")
    create_local_embedder(str(tmp_path), backend="hf", model_arch="text")
    fake_text_hf.assert_called_once()
    fake_vl_hf.assert_not_called()


def test_query_embedder_forwards_arch_for_local_dir(tmp_path, _patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    result = create_local_query_embedder(str(tmp_path), backend="hf", model_arch="vl")
    fake_vl_hf.assert_called_once()
    assert result is fake_vl_hf.return_value


def test_query_embedder_local_dir_without_arch_fails_loud(tmp_path, _patch_embedders):
    with pytest.raises(ValueError, match=LOCAL_EMBED_ARCH_ENV):
        create_local_query_embedder(str(tmp_path), backend="hf")


# ---------------------------------------------------------------------------
# Registered Hub ids: routing is unchanged (no arch needed)
# ---------------------------------------------------------------------------


def test_registered_id_ignores_arch_and_uses_allowlist(_patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    # The VL default id routes to VL regardless of any arch hint.
    result = create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2")
    fake_vl_vllm.assert_called_once()
    assert result is fake_vl_vllm.return_value
