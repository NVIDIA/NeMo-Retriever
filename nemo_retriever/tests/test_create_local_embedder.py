# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.model.create_local_embedder factory."""

import sys
import warnings
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_retriever.model import create_local_embedder, create_local_query_embedder


@pytest.fixture(autouse=True)
def _patch_embedders(monkeypatch):
    """Prevent real model downloads by stubbing all four embedder classes.

    The ``nemo_retriever.model.local`` package uses a custom ``__getattr__``
    that only exposes specific class names — not submodule names.  Because
    ``monkeypatch.setattr`` resolves each path segment via ``getattr``, it
    cannot traverse to the submodule.  We work around this by injecting fake
    modules directly into ``sys.modules``, which Python checks first when
    handling ``from … import`` statements.
    """
    fake_text_vllm = MagicMock(name="LlamaNemotronEmbed1BV2Embedder")
    fake_text_hf = MagicMock(name="LlamaNemotronEmbed1BV2HFEmbedder")
    fake_vl_hf = MagicMock(name="LlamaNemotronEmbedVL1BV2Embedder")
    fake_vl_vllm = MagicMock(name="LlamaNemotronEmbedVL1BV2VLLMEmbedder")

    text_mod = ModuleType("nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder")
    text_mod.LlamaNemotronEmbed1BV2Embedder = fake_text_vllm

    text_hf_mod = ModuleType("nemo_retriever.model.local.llama_nemotron_embed_1b_v2_hf_embedder")
    text_hf_mod.LlamaNemotronEmbed1BV2HFEmbedder = fake_text_hf

    vl_mod = ModuleType("nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder")
    vl_mod.LlamaNemotronEmbedVL1BV2Embedder = fake_vl_hf
    vl_mod.LlamaNemotronEmbedVL1BV2VLLMEmbedder = fake_vl_vllm

    monkeypatch.setitem(sys.modules, "nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder", text_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.model.local.llama_nemotron_embed_1b_v2_hf_embedder", text_hf_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder", vl_mod)

    yield fake_text_vllm, fake_text_hf, fake_vl_hf, fake_vl_vllm


# ---------------------------------------------------------------------------
# create_local_embedder — text model (non-VL)
# ---------------------------------------------------------------------------


def test_default_returns_text_vllm_embedder(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    result = create_local_embedder()
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_none_model_name_returns_text_embedder(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    result = create_local_embedder(None)
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_alias_resolved_to_text_embedder(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    result = create_local_embedder("nemo_retriever_v1")
    call_kwargs = fake_text_vllm.call_args
    assert call_kwargs.kwargs["model_id"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert result is fake_text_vllm.return_value


def test_text_model_explicit_vllm_backend(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    result = create_local_embedder(backend="vllm")
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_text_model_hf_backend_returns_hf_embedder(_patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    result = create_local_embedder(backend="hf")
    fake_text_hf.assert_called_once()
    assert result is fake_text_hf.return_value


def test_kwargs_forwarded_to_text_vllm_embedder(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    create_local_embedder(
        device="cuda:1",
        hf_cache_dir="/tmp/cache",
        gpu_memory_utilization=0.6,
        normalize=False,
        max_length=4096,
    )
    kw = fake_text_vllm.call_args.kwargs
    assert kw["device"] == "cuda:1"
    assert kw["hf_cache_dir"] == "/tmp/cache"
    assert kw["gpu_memory_utilization"] == 0.6
    assert kw["normalize"] is False
    assert kw["max_length"] == 4096


def test_kwargs_forwarded_to_text_hf_embedder(_patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    create_local_embedder(
        backend="hf",
        device="cuda:0",
        hf_cache_dir="/models",
        normalize=False,
        max_length=512,
    )
    kw = fake_text_hf.call_args.kwargs
    assert kw["device"] == "cuda:0"
    assert kw["hf_cache_dir"] == "/models"
    assert kw["normalize"] is False
    assert kw["max_length"] == 512


def test_unknown_model_passes_through(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    create_local_embedder("custom-org/my-embed-model")
    kw = fake_text_vllm.call_args.kwargs
    assert kw["model_id"] == "custom-org/my-embed-model"


# ---------------------------------------------------------------------------
# create_local_embedder — VL model
# ---------------------------------------------------------------------------


def test_vl_model_default_returns_vllm_embedder(_patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    result = create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2")
    fake_vl_vllm.assert_called_once()
    assert result is fake_vl_vllm.return_value


def test_vl_short_alias_returns_vllm_embedder(_patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    result = create_local_embedder("llama-nemotron-embed-vl-1b-v2")
    fake_vl_vllm.assert_called_once()
    assert result is fake_vl_vllm.return_value


def test_vl_model_hf_backend_returns_hf_embedder(_patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    result = create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="hf")
    fake_vl_hf.assert_called_once()
    assert result is fake_vl_hf.return_value


def test_kwargs_forwarded_to_vl_hf_embedder(_patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    create_local_embedder(
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        backend="hf",
        device="cuda:0",
        hf_cache_dir="/models",
    )
    kw = fake_vl_hf.call_args.kwargs
    assert kw["device"] == "cuda:0"
    assert kw["hf_cache_dir"] == "/models"
    assert kw["model_id"] == "nvidia/llama-nemotron-embed-vl-1b-v2"


# ---------------------------------------------------------------------------
# Invalid backend
# ---------------------------------------------------------------------------


def test_invalid_backend_raises_value_error(_patch_embedders):
    with pytest.raises(ValueError, match="backend must be"):
        create_local_embedder(backend="auto")


def test_invalid_backend_raises_for_vl(_patch_embedders):
    with pytest.raises(ValueError, match="backend must be"):
        create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="auto")


# ---------------------------------------------------------------------------
# create_local_query_embedder
# ---------------------------------------------------------------------------


def test_query_embedder_defaults_to_hf(_patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    result = create_local_query_embedder()
    fake_text_hf.assert_called_once()
    assert result is fake_text_hf.return_value


def test_query_embedder_explicit_hf(_patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    result = create_local_query_embedder(backend="hf")
    fake_text_hf.assert_called_once()
    assert result is fake_text_hf.return_value


def test_query_embedder_vllm_uses_vllm_embedder(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    result = create_local_query_embedder(backend="vllm")
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_query_embedder_invalid_backend_raises(_patch_embedders):
    with pytest.raises(ValueError, match="backend must be"):
        create_local_query_embedder(backend="auto")


def test_query_embedder_vl_hf_uses_hf_vl(_patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    result = create_local_query_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="hf")
    fake_vl_hf.assert_called_once()
    assert result is fake_vl_hf.return_value


def test_query_embedder_vl_vllm_uses_vllm_vl(_patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    result = create_local_query_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="vllm")
    fake_vl_vllm.assert_called_once()
    assert result is fake_vl_vllm.return_value


# ---------------------------------------------------------------------------
# Real-class smoke test (requires torch; skipped if not installed)
# ---------------------------------------------------------------------------


def test_llama_nemotron_text_embedder_deprecates_device(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    import importlib

    monkeypatch.delitem(
        sys.modules,
        "nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder",
        raising=False,
    )
    mod = importlib.import_module("nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder")
    monkeypatch.setattr(
        "nemo_retriever.text_embed.vllm.create_vllm_llm",
        MagicMock(return_value=MagicMock()),
    )
    Embed = mod.LlamaNemotronEmbed1BV2Embedder

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always", category=DeprecationWarning)
        Embed(device="cuda:0")
    assert any("no longer uses 'device'" in str(w.message) for w in wrec)

    with warnings.catch_warnings(record=True) as wrec2:
        warnings.simplefilter("always", category=DeprecationWarning)
        Embed(device=None)
    assert not any("no longer uses 'device'" in str(w.message) for w in wrec2)
