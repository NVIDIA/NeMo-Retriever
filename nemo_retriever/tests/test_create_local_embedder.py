# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.model.create_local_embedder factory."""

import sys
import warnings
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_retriever.model import create_local_embedder


@pytest.fixture(autouse=True)
def _patch_embedders(monkeypatch):
    """Prevent real model downloads by stubbing both embedder classes.

    The ``nemo_retriever.model.local`` package uses a custom ``__getattr__``
    that only exposes specific class names — not submodule names.  Because
    ``monkeypatch.setattr`` resolves each path segment via ``getattr``, it
    cannot traverse to the submodule.  We work around this by injecting fake
    modules directly into ``sys.modules``, which Python checks first when
    handling ``from … import`` statements.
    """
    fake_text = MagicMock(name="LlamaNemotronEmbed1BV2Embedder")
    fake_vl = MagicMock(name="LlamaNemotronEmbedVL1BV2Embedder")

    text_mod = ModuleType("nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder")
    text_mod.LlamaNemotronEmbed1BV2Embedder = fake_text

    vl_mod = ModuleType("nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder")
    vl_mod.LlamaNemotronEmbedVL1BV2Embedder = fake_vl

    monkeypatch.setitem(sys.modules, "nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder", text_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder", vl_mod)

    yield fake_text, fake_vl


def test_default_returns_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    result = create_local_embedder()
    fake_text.assert_called_once()
    assert result is fake_text.return_value


def test_none_model_name_returns_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    result = create_local_embedder(None)
    fake_text.assert_called_once()
    assert result is fake_text.return_value


def test_alias_resolved_to_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    result = create_local_embedder("nemo_retriever_v1")
    call_kwargs = fake_text.call_args
    assert call_kwargs.kwargs["model_id"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert result is fake_text.return_value


def test_vl_model_returns_vl_embedder(_patch_embedders):
    _, fake_vl = _patch_embedders
    result = create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2")
    fake_vl.assert_called_once()
    assert result is fake_vl.return_value


def test_vl_short_alias_returns_vl_embedder(_patch_embedders):
    _, fake_vl = _patch_embedders
    result = create_local_embedder("llama-nemotron-embed-vl-1b-v2")
    fake_vl.assert_called_once()
    assert result is fake_vl.return_value


def test_kwargs_forwarded_to_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    create_local_embedder(
        device="cuda:1",
        hf_cache_dir="/tmp/cache",
        gpu_memory_utilization=0.6,
        normalize=False,
        max_length=4096,
    )
    kw = fake_text.call_args.kwargs
    assert kw["device"] == "cuda:1"
    assert kw["hf_cache_dir"] == "/tmp/cache"
    assert kw["gpu_memory_utilization"] == 0.6
    assert kw["normalize"] is False
    assert kw["max_length"] == 4096


def test_kwargs_forwarded_to_vl_embedder(_patch_embedders):
    _, fake_vl = _patch_embedders
    create_local_embedder(
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        device="cuda:0",
        hf_cache_dir="/models",
    )
    kw = fake_vl.call_args.kwargs
    assert kw["device"] == "cuda:0"
    assert kw["hf_cache_dir"] == "/models"
    assert kw["model_id"] == "nvidia/llama-nemotron-embed-vl-1b-v2"


def test_unknown_model_passes_through(_patch_embedders):
    fake_text, _ = _patch_embedders
    create_local_embedder("custom-org/my-embed-model")
    kw = fake_text.call_args.kwargs
    assert kw["model_id"] == "custom-org/my-embed-model"


def test_llama_nemotron_text_embedder_deprecates_device(monkeypatch):
    # Autouse _patch_embedders shadows this submodule with a MagicMock; load the real class.
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
