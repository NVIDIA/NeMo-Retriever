# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""N3.5 local defaults, protected inputs, scoring, and lifecycle contracts."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from nemo_retriever import models
from nemo_retriever.models.local.nemotron35_rerank import Nemotron35Reranker, Nemotron35RerankerHF, rerank_messages
from nemo_retriever.operators.rerank import (
    NemotronRerankCPUActor,
    NemotronRerankGPUActor,
    _default_rerank_invoke_url,
    _rerank_batch,
    rerank_hits,
)
from nemo_retriever.service.config import LocalRerankConfig

MODEL = models.NEMOTRON_35_RERANK_MODEL


@pytest.mark.parametrize("name", [None, "", MODEL, "Nemotron-3.5-Rerank-8B-BF16"])
@pytest.mark.parametrize("backend,constructor", [("vllm", "Nemotron35Reranker"), ("hf", "Nemotron35RerankerHF")])
def test_factory_defaults_and_aliases(name: str | None, backend: str, constructor: str) -> None:
    with patch(f"nemo_retriever.models.local.nemotron35_rerank.{constructor}") as loader:
        assert models.create_local_reranker(name, backend=backend) is loader.return_value
    assert models.DEFAULT_LOCAL_RERANK_MODEL == MODEL
    assert models.is_vl_rerank_model(MODEL)


@pytest.mark.parametrize(
    "backend,constructor",
    [
        ("vllm", "nemo_retriever.models.local.nemotron_rerank_vl_v2.NemotronRerankVLV2VLLM"),
        ("hf", "nemo_retriever.models.local.nemotron_rerank_vl_v2_hf.NemotronRerankVLV2"),
    ],
)
def test_explicit_legacy_vl_model_preserved(backend: str, constructor: str) -> None:
    with patch(constructor) as loader:
        assert models.create_local_reranker(models.VL_RERANK_MODEL, backend=backend) is loader.return_value
        assert loader.call_args.kwargs["model_name"] == models.VL_RERANK_MODEL


def test_explicit_legacy_text_model_preserved() -> None:
    legacy = "nvidia/llama-nemotron-rerank-1b-v2"
    with patch("nemo_retriever.models.local.nemotron_rerank_v2.NemotronRerankV2") as loader:
        assert models.create_local_reranker(legacy) is loader.return_value
        assert loader.call_args.kwargs["model_name"] == legacy


def test_gpu_default_is_local35_cpu_default_remains_hosted1b(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    with patch.object(models, "create_local_reranker") as loader:
        actor = NemotronRerankGPUActor()
        assert loader.call_args.kwargs["model_name"] == MODEL
        assert actor._kwargs["model_name"] == MODEL
    cpu = NemotronRerankCPUActor()
    assert cpu._kwargs["model_name"] == models.VL_RERANK_MODEL
    assert cpu._kwargs["rerank_invoke_url"].endswith("/llama-nemotron-rerank-vl-1b-v2/reranking")
    with pytest.raises(ValueError, match="explicit remote endpoint"):
        _default_rerank_invoke_url(MODEL)


def test_graph_and_cli_defer_omitted_model_to_resolved_actor() -> None:
    from nemo_retriever.cli.query.options import DEFAULT_RERANK_MODEL
    from nemo_retriever.graph.retriever import Retriever
    from nemo_retriever.query.options import QueryRerankOptions
    from nemo_retriever.query.workflow import _build_rerank_kwargs

    assert DEFAULT_RERANK_MODEL == MODEL
    assert Retriever(rerank=True)._merge_rerank_actor_kwargs()["model_name"] is None
    assert _build_rerank_kwargs(QueryRerankOptions(enabled=True)) == {}
    assert _build_rerank_kwargs(QueryRerankOptions(enabled=True, reranker_model_name=MODEL))["model_name"] == MODEL
    remote = _build_rerank_kwargs(QueryRerankOptions(enabled=True, reranker_invoke_url="http://nim/v1/ranking"))
    assert "model_name" not in remote


def test_beir_local_default_preserves_legacy_remote_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.tools.recall.beir import BeirConfig

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = BeirConfig("/tmp/index", "documents", None, "vidore_hf", "test")
    assert config.reranker_model_name is None
    legacy = "nvidia/llama-nemotron-rerank-1b-v2"
    for explicit in (None, MODEL):
        kwargs = {"model_name": explicit, "remote_default_model_name": legacy}
        with patch.object(models, "create_local_reranker") as loader:
            NemotronRerankGPUActor(**kwargs)
            assert loader.call_args.kwargs["model_name"] == MODEL
        if explicit is None:
            cpu = NemotronRerankCPUActor(**kwargs)
            assert cpu._kwargs["model_name"] == legacy
            assert cpu._kwargs["rerank_invoke_url"].endswith("/llama-nemotron-rerank-1b-v2/reranking")
        else:
            cpu = NemotronRerankCPUActor(**kwargs, rerank_invoke_url="http://nim/v1/ranking")
            assert cpu._kwargs["model_name"] == MODEL


def test_agentic_enable_flag_preserves_omitted_model_identity() -> None:
    from nemo_retriever.query.options import QueryRequest, QueryRerankOptions
    from nemo_retriever.query.workflow import build_agentic_config

    config = build_agentic_config(QueryRequest(query="q", rerank=QueryRerankOptions(enabled=True)))
    assert config.rerank is True and config.reranker is None
    remote = build_agentic_config(
        QueryRequest(
            query="q",
            rerank=QueryRerankOptions(
                enabled=True,
                reranker_invoke_url="http://nim/v1/ranking",
            ),
        )
    )
    assert remote.rerank is True and remote.reranker is None
    assert remote.reranker_endpoint == "http://nim/v1/ranking"


@pytest.mark.parametrize("name,expected", [(MODEL, 10240), (models.VL_RERANK_MODEL, 512)])
def test_model_aware_service_context_and_explicit_override(name: str, expected: int) -> None:
    assert LocalRerankConfig(model_name=name).max_length == expected
    assert LocalRerankConfig(model_name=name, max_length=256).max_length == 256
    assert LocalRerankConfig().model_name == MODEL
    assert LocalRerankConfig().enabled is False


@pytest.mark.parametrize("name,expected", [(MODEL, 10240), (models.VL_RERANK_MODEL, 512)])
def test_operator_context_defaults_and_explicit_override(name: str, expected: int) -> None:
    model = MagicMock()
    model.model_name = name
    model.score_pairs.return_value = [1.0]
    frame = pd.DataFrame({"query": ["q"], "text": ["d"]})
    _rerank_batch(frame, model=model)
    assert model.score_pairs.call_args.kwargs["max_length"] == expected
    _rerank_batch(frame, model=model, max_length=256)
    assert model.score_pairs.call_args.kwargs["max_length"] == 256
    model.score.return_value = [1.0]
    rerank_hits("q", [{"text": "d"}], model=model)
    assert model.score.call_args.kwargs["max_length"] == expected


def test_checkpoint_messages_delegate_escaping_to_template() -> None:
    image = object()
    messages = rerank_messages("<q>[x]", "<d>[y]", image)
    assert messages[0] == {"role": "query", "content": [{"type": "text", "text": "<q>[x]"}]}
    assert messages[1]["content"] == [{"type": "text", "text": "<d>[y]"}, {"type": "image", "image": image}]


@pytest.mark.parametrize("model_class", [Nemotron35RerankerHF, Nemotron35Reranker])
@pytest.mark.parametrize("kwargs", [{"max_length": 0}, {"max_length": 10241}, {"batch_size": 0}, {"images_b64": []}])
def test_invalid_input_limits_before_loading(model_class: type, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        object.__new__(model_class).score_pairs([("q", "d")], **kwargs)


@pytest.mark.parametrize("model_class", [Nemotron35RerankerHF, Nemotron35Reranker])
def test_empty_input_does_not_load(model_class: type) -> None:
    assert object.__new__(model_class).score_pairs([]) == []


def test_document_only_truncation_protects_query_and_full_image() -> None:
    parser = object.__new__(Nemotron35RerankerHF)
    parser._processor = MagicMock()
    parser._processor.tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
    parser._processor.tokenizer.decode.side_effect = lambda tokens: "".join(tokens)
    parser._processor.apply_chat_template.side_effect = lambda conversation, **kwargs: (
        conversation[0]["content"][0]["text"] + conversation[1]["content"][0]["text"]
    )
    image = object()

    def features(conversations: list) -> dict:
        conversation = conversations[0]
        assert conversation[0]["content"][0]["text"] == "query"
        assert conversation[1]["content"][1]["image"] is image
        return {"attention_mask": torch.ones((1, 8 + len(conversation[1]["content"][0]["text"])))}

    parser._features = features
    assert parser._fit_document("query", "abcdefghijkl", image, 12)[1]["content"][0]["text"] == "abcd"
    with pytest.raises(ValueError, match="query and image"):
        parser._fit_document("query", "doc", image, 7)


@pytest.mark.parametrize("has_image", [False, True])
def test_long_document_search_does_not_reprocess_image(has_image: bool) -> None:
    reranker = object.__new__(Nemotron35RerankerHF)
    reranker._processor = MagicMock()
    tokenizer = reranker._processor.tokenizer
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
    tokenizer.decode.side_effect = lambda tokens: "".join(tokens)
    image = object() if has_image else None

    def render(conversation: list, **kwargs: object) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": False}
        assert conversation[0]["content"][0]["text"] == "query"
        if has_image:
            assert conversation[1]["content"][1]["image"] is image
        return "query:" + conversation[1]["content"][0]["text"] + ("[IMG]" if has_image else "")

    reranker._processor.apply_chat_template.side_effect = render
    image_token_cost = 100 if has_image else 0
    reranker._features = MagicMock(
        side_effect=lambda conversations: {
            "attention_mask": torch.ones(
                (1, len(render(conversations[0], tokenize=False, add_generation_prompt=False)) + image_token_cost)
            )
        }
    )
    document = "x" * 16000
    budget = 10240
    result = reranker._fit_document("query", document, image, budget)
    assert len(render(result, tokenize=False, add_generation_prompt=False)) + image_token_cost == budget
    assert result[1]["content"][0]["text"] == document[: budget - 6 - (105 if has_image else 0)]
    reranker._features.assert_called_once()
    # The binary search did actual text-only probes, not repeated image passes.
    assert reranker._processor.apply_chat_template.call_count >= 14


@pytest.mark.parametrize("model_class", [Nemotron35RerankerHF, Nemotron35Reranker])
def test_scoring_still_checks_actual_expanded_length(model_class: type) -> None:
    reranker = object.__new__(model_class)
    reranker._device = "cpu"
    reranker._model = MagicMock()
    reranker._features = MagicMock(return_value=BatchFeature({"attention_mask": torch.ones((1, 13))}))
    reranker._processor = MagicMock()
    with pytest.raises(RuntimeError, match="exceed"):
        reranker._score_conversations([rerank_messages("query", "doc", object())], 12)
    reranker._model.assert_not_called()
    reranker._model.encode.assert_not_called()


def test_processor_disables_implicit_special_tokens_and_truncation() -> None:
    parser = object.__new__(Nemotron35RerankerHF)
    parser._processor = MagicMock()
    parser._features([rerank_messages("q", "d")])
    assert parser._processor.apply_chat_template.call_args.kwargs["processor_kwargs"] == {
        "padding": True,
        "truncation": False,
        "add_special_tokens": False,
    }


@pytest.mark.parametrize("bad_key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_hf_load_fails_closed_on_weight_mismatch(bad_key: str) -> None:
    with (
        patch("nemo_retriever.models.local.nemotron35_rerank.configure_global_hf_cache_base", return_value="/cache"),
        patch("transformers.AutoProcessor.from_pretrained"),
        patch("transformers.AutoModelForSequenceClassification.from_pretrained") as loader,
    ):
        loader.return_value = (MagicMock(), {bad_key: ["bad-weight"]})
        with pytest.raises(RuntimeError, match="did not load exactly"):
            Nemotron35RerankerHF(device="cpu")
        assert loader.call_args.kwargs["revision"] == "f88f8d9104726747791283ab156d2fcdb29e90a7"
        assert loader.call_args.kwargs["cache_dir"] == "/cache"


def test_vllm_cuda_error_precedes_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with patch("huggingface_hub.snapshot_download") as download:
        with pytest.raises(RuntimeError, match="requires CUDA"):
            Nemotron35Reranker()
        download.assert_not_called()


def test_vllm_trained_head_is_mapped_and_required() -> None:
    from nemo_retriever.models.local.nemotron35_rerank_vllm import (
        NRLNemotron35RerankModel,
        TransformersMultiModalForSequenceClassification,
    )

    parser = object.__new__(NRLNemotron35RerankModel)
    tensor = torch.ones((1, 2))
    with patch.object(
        TransformersMultiModalForSequenceClassification, "load_weights", return_value={"classifier.weight"}
    ) as load:
        assert parser.load_weights([("language_model.score.weight", tensor)]) == {"classifier.weight"}
        assert list(load.call_args.args[0])[0][0] == "classifier.weight"
    with patch.object(TransformersMultiModalForSequenceClassification, "load_weights", return_value=set()):
        with pytest.raises(RuntimeError, match="trained classification head"):
            parser.load_weights([])


@pytest.mark.parametrize("model_class", [Nemotron35RerankerHF, Nemotron35Reranker])
def test_owned_images_close_even_on_inference_failure(model_class: type) -> None:
    parser = object.__new__(model_class)
    parser._processor = MagicMock()
    parser._model = MagicMock()
    parser._fit_document = MagicMock(return_value=[])
    parser._score_conversations = MagicMock(side_effect=RuntimeError("inference failed"))
    with patch("nemo_retriever.models.local.nemotron35_rerank.decode_base64_image") as decode:
        with pytest.raises(RuntimeError, match="inference failed"):
            parser.score_pairs([("q", "d")], images_b64=["image"])
        decode.return_value.close.assert_called_once()


@pytest.mark.parametrize("bad_scores", [[float("nan")], [], [1.0, 2.0]])
def test_nonfinite_or_misaligned_scores_fail(bad_scores: list[float]) -> None:
    with pytest.raises(RuntimeError, match="missing or nonfinite"):
        Nemotron35RerankerHF._validate_scores(bad_scores, 1)


def test_vllm_rejects_changed_expanded_input_length() -> None:
    parser = object.__new__(Nemotron35Reranker)
    parser._processor = MagicMock()
    parser._features = MagicMock(return_value={"attention_mask": torch.ones((1, 8))})
    parser._model = MagicMock()
    parser._model.encode.return_value = [
        SimpleNamespace(prompt_token_ids=[1], outputs=SimpleNamespace(data=torch.tensor(1.0)))
    ]
    with pytest.raises(RuntimeError, match="changed expanded input length"):
        parser._score_conversations([rerank_messages("q", "d")], 10240)


@pytest.mark.parametrize("failure", [False, True])
def test_vllm_explicit_shutdown_and_idempotence(failure: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    parser = object.__new__(Nemotron35Reranker)
    engine = parser._model = MagicMock()
    parser._processor = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if failure:
        engine.llm_engine.engine_core.shutdown.side_effect = RuntimeError("shutdown failed")
        with pytest.raises(RuntimeError, match="shutdown failed"):
            parser.unload()
        assert parser._model is engine
    else:
        parser.unload()
        parser.unload()
        engine.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=30.0)
        assert parser._model is None and parser._processor is None


@pytest.mark.parametrize("previous", [None, "fork", "spawn"])
@pytest.mark.parametrize("failure", [False, True])
def test_vllm_startup_scopes_spawn(previous: str | None, failure: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if previous is None:
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    else:
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", previous)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def load(**kwargs: object) -> MagicMock:
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert kwargs["pooler_config"] == {"seq_pooling_type": "MEAN", "use_activation": False}
        if failure:
            raise RuntimeError("startup failed")
        return MagicMock()

    with (
        patch("nemo_retriever.models.local.nemotron35_rerank.configure_global_hf_cache_base", return_value="/cache"),
        patch("huggingface_hub.snapshot_download", return_value="/snapshot"),
        patch("transformers.AutoProcessor.from_pretrained"),
        patch("nemo_retriever.models.inference.nemotron35_rerank_vllm.register_nemotron35_rerank_vllm"),
        patch("vllm.LLM", side_effect=load),
    ):
        if failure:
            with pytest.raises(RuntimeError, match="startup failed"):
                Nemotron35Reranker()
        else:
            Nemotron35Reranker()
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == previous


def test_hf_padding_positions_are_independent_of_left_padding() -> None:
    parser = object.__new__(Nemotron35RerankerHF)
    parser._device = "cpu"
    parser._model = MagicMock()
    parser._model.return_value.logits = torch.tensor([[1.0], [2.0]])
    parser._features = MagicMock(
        return_value=BatchFeature(
            {
                "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
                "input_ids": torch.ones((2, 4), dtype=torch.long),
            }
        )
    )
    assert parser._score_conversations([[], []], 10240) == [1.0, 2.0]
    assert parser._model.call_args.kwargs["position_ids"].tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]
