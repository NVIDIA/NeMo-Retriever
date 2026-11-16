# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os.path
from types import SimpleNamespace

import huggingface_hub
import pytest
from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision, resolve_embed_model_spec
from nemo_retriever.models.hf_model_registry import HF_MODEL_REVISIONS


def _write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _write_prompt_config(tmp_path, prompts):
    path = tmp_path / "config_sentence_transformers.json"
    path.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")
    return path


def _write_sentence_config(tmp_path, **config):
    path = tmp_path / "sentence_bert_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _text_config(**overrides):
    config = {
        "model_type": "llama_bidirec",
        "architectures": ["LlamaBidirectionalModel"],
        "hidden_size": 2048,
        "pooling": "avg",
    }
    config.update(overrides)
    return config


def _ministral_config(**overrides):
    config = {
        "model_type": "ministral3",
        "architectures": ["Ministral3Model"],
        "hidden_size": 2048,
        "is_causal": False,
        "pooling": "avg",
    }
    config.update(overrides)
    return config


def _gemma_config(**overrides):
    config = {
        "model_type": "gemma3_text",
        "architectures": ["Gemma3TextModel"],
        "hidden_size": 768,
        "use_bidirectional_attention": True,
    }
    config.update(overrides)
    return config


def _gemma2_config(**overrides):
    config = {
        "model_type": "embedding_gemma2",
        "architectures": ["EmbeddingGemma2Model"],
        "text_config": {
            "model_type": "embedding_gemma2_text",
            "hidden_size": 512,
            "embedding_dim": 768,
            "max_position_embeddings": 262144,
        },
    }
    config.update(overrides)
    return config


def _vl_config(**overrides):
    config = {
        "model_type": "llama_nemotron_vl",
        "architectures": ["LlamaNemotronVLModel"],
        "llm_config": {"hidden_size": 2048},
        "pooling": "avg",
    }
    config.update(overrides)
    return config


def test_local_text_checkpoint_is_resolved_from_model_type(tmp_path):
    _write_config(tmp_path, _text_config())
    _write_sentence_config(tmp_path, max_seq_length=8192)

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.family == "text"
    assert spec.output_dimension == 2048
    assert spec.revision is None
    assert spec.requires_vllm is False
    assert spec.max_input_tokens == 8192


@pytest.mark.parametrize("path_style", ["absolute", "relative", "tilde"])
def test_local_template_uses_same_path_resolution_as_checkpoint(tmp_path, monkeypatch, path_style):
    from nemo_retriever.models.inference.retrieval_template import load_retrieval_template

    _write_config(tmp_path, _text_config())
    template = "{{ messages[0].content }} {{ messages[1].content }}"
    (tmp_path / "chat_template.jinja").write_text(template, encoding="utf-8")
    if path_style == "tilde":
        model_id = "~/" + tmp_path.name
        expanduser = os.path.expanduser
        monkeypatch.setattr(
            os.path, "expanduser", lambda value: str(tmp_path.parent) if value == "~" else expanduser(value)
        )
    elif path_style == "relative":
        monkeypatch.chdir(tmp_path.parent)
        model_id = "./" + tmp_path.name
    else:
        model_id = str(tmp_path)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *args, **kwargs: pytest.fail("Local checkpoint must not access Hub")
    )
    load_retrieval_template.cache_clear()
    try:
        assert resolve_embed_model_spec(model_id).revision is None
        assert load_retrieval_template(model_id, None) == template
    finally:
        load_retrieval_template.cache_clear()


def test_vl_checkpoint_uses_declared_document_input_limit(tmp_path):
    _write_config(tmp_path, _vl_config(p_max_length=4096))

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.max_input_tokens == 4096


@pytest.mark.parametrize(
    ("config", "family", "quantization"),
    [
        pytest.param(
            _vl_config(quantization_config={"quant_method": "modelopt", "quant_algo": "FP8"}),
            "vl",
            "FP8",
            id="vl-fp8",
        ),
        pytest.param(
            _text_config(quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"}),
            "text",
            "NVFP4",
            id="text-nvfp4",
        ),
        pytest.param(
            _ministral_config(quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"}),
            "text",
            "NVFP4",
            id="ministral3-nvfp4",
        ),
    ],
)
def test_local_modelopt_checkpoint_requires_vllm(tmp_path, config, family, quantization):
    _write_config(tmp_path, config)

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.family == family
    assert spec.quantization == quantization
    assert spec.requires_vllm is True


def test_local_checkpoint_rejects_unsupported_architecture(tmp_path):
    _write_config(tmp_path, {"model_type": "bert"})

    with pytest.raises(ValueError, match="supported dense embed model types"):
        resolve_embed_model_spec(str(tmp_path))


def test_text_reranker_architecture_is_rejected(tmp_path):
    config = _text_config(architectures=["LlamaBidirectionalForSequenceClassification"])
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="unsupported architectures"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize(
    ("config", "family"),
    [
        pytest.param(_text_config(hidden_size=4096), "text", id="text"),
        pytest.param(_ministral_config(hidden_size=4096), "text", id="ministral3"),
        pytest.param(_vl_config(llm_config={"hidden_size": 4096}), "vl", id="vl"),
    ],
)
def test_4096_wide_embedder_is_supported(tmp_path, config, family):
    _write_config(tmp_path, config)

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.family == family
    assert spec.output_dimension == 4096


def test_invalid_embedding_dimension_is_rejected(tmp_path):
    _write_config(tmp_path, _text_config(hidden_size=0))

    with pytest.raises(ValueError, match="invalid embedding dimension 0"):
        resolve_embed_model_spec(str(tmp_path))


def test_ministral_checkpoint_is_resolved_with_prompt_metadata(tmp_path):
    _write_config(tmp_path, _ministral_config())
    _write_prompt_config(
        tmp_path,
        {
            "query": "query: ",
            "document": "passage: ",
        },
    )

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.family == "text"
    assert spec.output_dimension == 2048
    assert spec.query_prefix == "query: "
    assert spec.document_prefix == "passage: "
    assert spec.requires_vllm is False


def test_ministral_checkpoint_rejects_wrong_architecture(tmp_path):
    _write_config(
        tmp_path,
        _ministral_config(architectures=["Mistral3ForConditionalGeneration"]),
    )

    with pytest.raises(ValueError, match="unsupported architectures"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize("is_causal", [True, None])
def test_ministral_checkpoint_requires_explicit_non_causal_config(tmp_path, is_causal):
    _write_config(tmp_path, _ministral_config(is_causal=is_causal))

    with pytest.raises(ValueError, match="dense Ministral3 embedding profiles require is_causal=false"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize("config,max_tokens", [(_gemma_config(), 2048), (_gemma2_config(), 8192)])
def test_gemma_checkpoint_is_resolved_with_prompt_metadata(tmp_path, config, max_tokens):
    _write_config(tmp_path, config)
    _write_prompt_config(
        tmp_path,
        {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        },
    )
    if config["model_type"] == "gemma3_text":
        _write_sentence_config(tmp_path, max_seq_length=max_tokens)
    else:
        _write_sentence_config(tmp_path, module_output_name="token_embeddings")

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.family == "text"
    assert spec.output_dimension == 768
    assert spec.query_prefix == "task: search result | query: "
    assert spec.document_prefix == "title: none | text: "
    assert spec.max_input_tokens == max_tokens
    assert spec.requires_vllm is True
    assert spec.query_prefix_declared is True
    assert spec.document_prefix_declared is True


def test_gemma2_checkpoint_without_optional_metadata(tmp_path):
    _write_config(tmp_path, _gemma2_config())

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.output_dimension == 768
    assert spec.max_input_tokens == 8192
    assert spec.query_prefix == "query: "
    assert spec.document_prefix == "passage: "
    assert spec.query_prefix_declared is False
    assert spec.document_prefix_declared is False


@pytest.mark.parametrize("text_config", [None, {}, {"model_type": "gemma3_text", "embedding_dim": 768}])
def test_gemma2_checkpoint_requires_text_config(tmp_path, text_config):
    _write_config(tmp_path, _gemma2_config(text_config=text_config))

    with pytest.raises(ValueError, match="requires an embedding_gemma2_text text_config"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize("dimension", [None, False, 0, -1, "768"])
def test_gemma2_checkpoint_requires_embedding_dimension(tmp_path, dimension):
    config = _gemma2_config()
    config["text_config"]["embedding_dim"] = dimension
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="a positive embedding_dim is required"):
        resolve_embed_model_spec(str(tmp_path))


def test_gemma2_checkpoint_rejects_wrong_architecture(tmp_path):
    _write_config(tmp_path, _gemma2_config(architectures=["Gemma3TextModel"]))

    with pytest.raises(ValueError, match="unsupported architectures"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize("max_seq_length,expected", [(4096, 4096), (16384, 8192)])
def test_gemma2_checkpoint_caps_declared_input_limit(tmp_path, max_seq_length, expected):
    _write_config(tmp_path, _gemma2_config())
    _write_sentence_config(tmp_path, max_seq_length=max_seq_length)

    assert resolve_embed_model_spec(str(tmp_path)).max_input_tokens == expected


@pytest.mark.parametrize("max_seq_length", [False, 0, -1, "8192"])
def test_gemma2_checkpoint_rejects_invalid_input_limit(tmp_path, max_seq_length):
    _write_config(tmp_path, _gemma2_config())
    _write_sentence_config(tmp_path, max_seq_length=max_seq_length)

    with pytest.raises(ValueError, match="invalid supported input limit"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize("use_bidirectional_attention", [False, None])
def test_gemma_checkpoint_requires_bidirectional_attention(tmp_path, use_bidirectional_attention):
    _write_config(tmp_path, _gemma_config(use_bidirectional_attention=use_bidirectional_attention))

    with pytest.raises(ValueError, match="Gemma 3 embedding profiles require use_bidirectional_attention=true"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize("config", [_gemma_config(), _gemma2_config()])
def test_gemma_checkpoint_rejects_hf_backend(tmp_path, config):
    from nemo_retriever.models.embed_model_spec import validate_embed_model_backend

    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="requires backend='vllm'"):
        validate_embed_model_backend(resolve_embed_model_spec(str(tmp_path)), "hf")


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "model_type": "mistral3",
                "architectures": ["Mistral3ForConditionalGeneration"],
                "hidden_size": 2048,
                "is_causal": True,
                "pooling": "avg",
            },
            id="mistral3-conditional-generation",
        ),
        pytest.param(
            {
                "model_type": "ministral3_bidirec",
                "architectures": ["Ministral3BidirectionalModel"],
                "hidden_size": 2048,
                "is_causal": False,
                "pooling": "avg",
            },
            id="nemo-automodel-bidirectional",
        ),
    ],
)
def test_other_mistral_checkpoint_formats_remain_rejected(tmp_path, config):
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="unsupported (model_type|architectures)"):
        resolve_embed_model_spec(str(tmp_path))


def test_checkpoint_prompt_metadata_is_resolved(tmp_path):
    _write_config(tmp_path, _text_config(hidden_size=4096))
    _write_prompt_config(
        tmp_path,
        {
            "query": "Instruct: Retrieve relevant passages\nQuery: ",
            "document": "",
        },
    )

    spec = resolve_embed_model_spec(str(tmp_path))

    assert spec.query_prefix == "Instruct: Retrieve relevant passages\nQuery: "
    assert spec.document_prefix == ""
    assert spec.query_prefix_declared is True
    assert spec.document_prefix_declared is True


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_text_config(pooling="last"), id="text"),
        pytest.param(_ministral_config(pooling="last"), id="ministral3"),
    ],
)
def test_non_average_pooling_is_rejected(tmp_path, config):
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="unsupported pooling 'last'"):
        resolve_embed_model_spec(str(tmp_path))


def test_local_directory_requires_config_json(tmp_path):
    with pytest.raises(ValueError, match="does not contain config.json"):
        resolve_embed_model_spec(str(tmp_path))


@pytest.mark.parametrize(
    ("model_id", "config", "family", "dimension", "requires_vllm"),
    [
        ("google/embeddinggemma-2", _gemma2_config(), "text", 768, True),
        ("google/embeddinggemma-300m", _gemma_config(), "text", 768, True),
        ("nvidia/llama-3.2-nv-embedqa-1b-v2", _text_config(), "text", 2048, False),
        ("nvidia/llama-nemotron-embed-1b-v2", _text_config(), "text", 2048, False),
        ("nvidia/llama-nemotron-embed-vl-1b-v2", _vl_config(), "vl", 2048, False),
        (
            "nvidia/llama-nemotron-embed-vl-1b-v2-fp8",
            _vl_config(quantization_config={"quant_method": "modelopt", "quant_algo": "FP8"}),
            "vl",
            2048,
            True,
        ),
        ("nvidia/llama-embed-nemotron-8b", _text_config(hidden_size=4096), "text", 4096, False),
        ("nvidia/llama-nv-embed-reasoning-3b", _text_config(hidden_size=3072), "text", 3072, False),
        ("nvidia/Nemotron-3-Embed-1B-BF16", _ministral_config(), "text", 2048, False),
        ("nvidia/Nemotron-3-Embed-8B-BF16", _ministral_config(hidden_size=4096), "text", 4096, False),
        (
            "nvidia/Nemotron-3-Embed-1B-NVFP4",
            _ministral_config(quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"}),
            "text",
            2048,
            True,
        ),
    ],
)
def test_supported_hub_models_resolve_immutable_revisions(
    monkeypatch,
    tmp_path,
    model_id,
    config,
    family,
    dimension,
    requires_vllm,
):
    _write_config(tmp_path, config)
    resolved_sha = "c" * 40
    model_info_calls = []

    def model_info(_self, requested_model_id, revision=None):
        model_info_calls.append((requested_model_id, revision))
        return SimpleNamespace(sha=resolved_sha)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(tmp_path / "config.json"))

    spec = resolve_embed_model_spec(model_id)

    assert spec.family == family
    assert spec.output_dimension == dimension
    assert spec.requires_vllm is requires_vllm
    if model_id in HF_MODEL_REVISIONS:
        assert spec.revision == HF_MODEL_REVISIONS[model_id]
        assert model_info_calls == []
    else:
        assert spec.revision == resolved_sha
        assert model_info_calls == [(model_id, None)]


def test_compatible_custom_hub_model_is_pinned_before_config_load(monkeypatch, tmp_path):
    _write_config(tmp_path, _vl_config())
    resolved_sha = "c" * 40
    calls = {"downloads": []}

    def model_info(_self, model_id, revision=None):
        calls["model_info"] = (model_id, revision)
        return SimpleNamespace(sha=resolved_sha)

    def download(**kwargs):
        calls["downloads"].append(kwargs)
        return str(tmp_path / "config.json")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    cache_dir = str(tmp_path / "hub-cache")
    spec = resolve_embed_model_spec("acme/fine-tuned-nemotron", hf_cache_dir=cache_dir)

    assert spec.family == "vl"
    assert spec.revision == resolved_sha
    assert calls["model_info"] == ("acme/fine-tuned-nemotron", None)
    assert {call["filename"] for call in calls["downloads"]} == {
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
    }
    assert all(call["revision"] == resolved_sha for call in calls["downloads"])
    assert all(call["cache_dir"] == cache_dir for call in calls["downloads"])


def test_revision_only_resolution_avoids_config_download_for_registered_model(monkeypatch):
    model_id = "nvidia/llama-nemotron-embed-vl-1b-v2"

    def fail_download(*args, **kwargs):
        raise AssertionError("revision-only resolution must not download config.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_download)

    assert resolve_embed_model_revision(model_id, None) == HF_MODEL_REVISIONS[model_id]
