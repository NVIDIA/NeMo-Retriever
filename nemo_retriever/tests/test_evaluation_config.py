# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`nemo_retriever.tools.evaluation.config`.

Focus: the fail-fast contract in :func:`_normalize_config` that guards
``build_eval_chain`` / ``build_eval_pipeline`` from silently collapsing
heterogeneous-judge configs to a single judge, plus the ``num_retries``
plumbing contract from the judge config block down to the constructed
operator / ``LLMJudge``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_retriever.tools.evaluation.config import _normalize_config, build_eval_chain, build_eval_pipeline
from nemo_retriever.tools.evaluation.runner import _resolved_generation_config, run_eval_sweep


def _make_multi_judge_config() -> dict:
    """New ``models`` + ``evaluations`` schema with two distinct judges."""
    return {
        "models": {
            "gen-a": {"model": "provider/gen-a", "api_key": "k"},
            "gen-b": {"model": "provider/gen-b", "api_key": "k"},
            "judge-x": {"model": "provider/judge-x", "api_key": "k"},
            "judge-y": {"model": "provider/judge-y", "api_key": "k"},
        },
        "evaluations": [
            {"generator": "gen-a", "judge": "judge-x"},
            {"generator": "gen-b", "judge": "judge-y"},
        ],
    }


def _make_single_judge_new_schema_config() -> dict:
    """New schema with one judge shared across multiple generators."""
    return {
        "models": {
            "gen-a": {"model": "provider/gen-a", "api_key": "k"},
            "gen-b": {"model": "provider/gen-b", "api_key": "k"},
            "judge-x": {"model": "provider/judge-x", "api_key": "k"},
        },
        "evaluations": [
            {"generator": "gen-a", "judge": "judge-x"},
            {"generator": "gen-b", "judge": "judge-x"},
        ],
    }


def _make_legacy_config() -> dict:
    """Legacy ``generators`` + ``judge`` schema."""
    return {
        "generators": [
            {"name": "gen-a", "model": "provider/gen-a", "api_key": "k"},
            {"name": "gen-b", "model": "provider/gen-b", "api_key": "k"},
        ],
        "judge": {"name": "judge-x", "model": "provider/judge-x", "api_key": "k"},
    }


def test_normalize_config_multi_judge_raises() -> None:
    """Multi-judge configs must fail fast instead of silently collapsing.

    Previously ``_normalize_config`` logged a warning and kept the first
    judge, which meant ``build_eval_chain`` / ``build_eval_pipeline``
    scored every generator against that single judge without any error.
    """
    config = _make_multi_judge_config()

    with pytest.raises(ValueError) as exc_info:
        _normalize_config(config)

    message = str(exc_info.value)
    assert "run_eval_sweep" in message, "error must point users at the correct API for heterogeneous judges"
    assert "judge-x" in message and "judge-y" in message, "error must list the distinct judges that conflict"
    assert "2 distinct judges" in message, "error must report how many judges collided"


def test_normalize_config_single_judge_new_schema_passes() -> None:
    """Single-judge configs in the new schema must normalise cleanly."""
    config = _make_single_judge_new_schema_config()

    normalized = _normalize_config(config)

    assert "generators" in normalized
    assert "judge" in normalized
    assert normalized["judge"]["model"] == "provider/judge-x"
    gen_names = {g["name"] for g in normalized["generators"]}
    assert gen_names == {"gen-a", "gen-b"}


def test_normalize_config_legacy_schema_passes() -> None:
    """Legacy ``generators`` + ``judge`` configs must still normalise.

    The legacy schema has a scalar ``judge`` by construction, so it can
    never trigger the multi-judge fail-fast path.
    """
    config = _make_legacy_config()

    normalized = _normalize_config(config)

    assert "models" in normalized
    assert "evaluations" in normalized
    assert "gen-a" in normalized["models"]
    assert "gen-b" in normalized["models"]
    assert "judge-x" in normalized["models"]

    eval_judges = {e["judge"] for e in normalized["evaluations"]}
    assert eval_judges == {"judge-x"}


def _make_minimal_legacy_config_with_judge_retries(num_retries: int) -> dict:
    """Minimal legacy-schema config carrying an explicit judge.num_retries."""
    return {
        "generators": [
            {"name": "gen-a", "model": "provider/gen-a", "api_key": "k"},
        ],
        "judge": {
            "name": "judge-x",
            "model": "provider/judge-x",
            "api_key": "k",
            "num_retries": num_retries,
        },
        "retrieval": {"type": "file", "file_path": "dummy.json"},
        "dataset": {"source": "dummy.csv"},
        "execution": {"top_k": 5, "max_workers": 2},
    }


def test_build_eval_chain_forwards_judge_num_retries() -> None:
    """``judge.num_retries`` from YAML must reach ``JudgingOperator``.

    Before this fix the ``JudgingOperator`` constructor did not accept
    ``num_retries`` at all, so any value a user put in the judge block was
    silently dropped and the operator always ran with the default ``3``.
    """
    config = _make_minimal_legacy_config_with_judge_retries(num_retries=9)

    with (
        patch("nemo_retriever.tools.evaluation.retrieval_loader.RetrievalLoaderOperator"),
        patch("nemo_retriever.tools.evaluation.generation.QAGenerationOperator"),
        patch("nemo_retriever.operators.graph_ops.scoring_operator.ScoringOperator"),
        patch("nemo_retriever.tools.evaluation.judging.JudgingOperator") as mock_judge_op,
    ):
        mock_judge_op.return_value = MagicMock()

        build_eval_chain(config)

        mock_judge_op.assert_called_once()
        assert mock_judge_op.call_args.kwargs["num_retries"] == 9


def test_build_eval_chain_defaults_judge_num_retries_when_absent() -> None:
    """When ``judge.num_retries`` is omitted, the default ``3`` must be passed."""
    config = _make_minimal_legacy_config_with_judge_retries(num_retries=3)
    config["judge"].pop("num_retries")

    with (
        patch("nemo_retriever.tools.evaluation.retrieval_loader.RetrievalLoaderOperator"),
        patch("nemo_retriever.tools.evaluation.generation.QAGenerationOperator"),
        patch("nemo_retriever.operators.graph_ops.scoring_operator.ScoringOperator"),
        patch("nemo_retriever.tools.evaluation.judging.JudgingOperator") as mock_judge_op,
    ):
        mock_judge_op.return_value = MagicMock()

        build_eval_chain(config)

        assert mock_judge_op.call_args.kwargs["num_retries"] == 3


def test_build_eval_chain_forwards_generation_comparison_settings() -> None:
    config = _make_minimal_legacy_config_with_judge_retries(num_retries=3)
    config["generators"][0].update({"top_p": 0.8, "reasoning_enabled": False})

    with (
        patch("nemo_retriever.tools.evaluation.retrieval_loader.RetrievalLoaderOperator"),
        patch("nemo_retriever.tools.evaluation.generation.QAGenerationOperator") as mock_generation_op,
        patch("nemo_retriever.operators.graph_ops.scoring_operator.ScoringOperator"),
        patch("nemo_retriever.tools.evaluation.judging.JudgingOperator"),
    ):
        build_eval_chain(config)

    assert mock_generation_op.call_args.kwargs["top_p"] == 0.8
    assert mock_generation_op.call_args.kwargs["reasoning_enabled"] is False


def test_resolved_generation_config_is_effective_and_secret_free() -> None:
    model_config = {
        "model": "provider/model",
        "api_key": "must-not-be-emitted",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 512,
        "reasoning_enabled": True,
        "extra_params": {"authorization": "must-not-be-emitted"},
    }
    eval_config = {
        "temperature": 0.4,
        "max_tokens": 1024,
        "reasoning_enabled": False,
    }

    metadata = _resolved_generation_config(eval_config, model_config)

    assert metadata == {
        "model": "provider/model",
        "temperature": 0.4,
        "top_p": 0.7,
        "max_tokens": 1024,
        "reasoning_enabled": False,
    }
    assert "api_key" not in metadata
    assert "extra_params" not in metadata


def test_build_eval_pipeline_forwards_judge_num_retries() -> None:
    """``judge.num_retries`` from YAML must reach ``LLMJudge.from_kwargs``.

    This is the sibling path to :func:`build_eval_chain` -- same bug
    shape, different construction surface.
    """
    config = _make_minimal_legacy_config_with_judge_retries(num_retries=11)

    with (
        patch("nemo_retriever.tools.evaluation.retrievers.FileRetriever"),
        patch("nemo_retriever.models.llm.clients.LiteLLMClient"),
        patch("nemo_retriever.tools.evaluation.orchestrator.QAEvalPipeline"),
        patch("nemo_retriever.models.llm.clients.LLMJudge.from_kwargs") as mock_from_kwargs,
    ):
        mock_from_kwargs.return_value = MagicMock()

        build_eval_pipeline(config)

        mock_from_kwargs.assert_called_once()
        assert mock_from_kwargs.call_args.kwargs["num_retries"] == 11


def test_build_eval_pipeline_forwards_generation_comparison_settings() -> None:
    config = _make_minimal_legacy_config_with_judge_retries(num_retries=3)
    config["generators"][0].update({"top_p": 0.6, "reasoning_enabled": False})

    with (
        patch("nemo_retriever.tools.evaluation.retrievers.FileRetriever"),
        patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs") as mock_client,
        patch("nemo_retriever.tools.evaluation.orchestrator.QAEvalPipeline"),
        patch("nemo_retriever.models.llm.clients.LLMJudge.from_kwargs"),
    ):
        build_eval_pipeline(config)

    assert mock_client.call_args.kwargs["top_p"] == 0.6
    assert mock_client.call_args.kwargs["reasoning_enabled"] is False


def test_run_eval_sweep_persists_effective_generation_config(tmp_path) -> None:
    config = {
        "models": {
            "generator": {
                "model": "provider/model",
                "api_key": None,
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 512,
                "reasoning_enabled": True,
            },
            "judge": {"model": "provider/judge", "api_key": None},
        },
        "evaluations": [
            {
                "generator": "generator",
                "judge": "judge",
                "temperature": 0.4,
                "reasoning_enabled": False,
            }
        ],
        "dataset": {"source": "csv:qa.csv"},
    }
    pipeline = MagicMock()
    pipeline.evaluate.return_value = {"summary": {"total_completed": 1}}

    with (
        patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs") as mock_client,
        patch("nemo_retriever.models.llm.clients.LLMJudge.from_kwargs"),
        patch(
            "nemo_retriever.tools.evaluation.orchestrator.QAEvalPipeline",
            return_value=pipeline,
        ),
    ):
        results = run_eval_sweep(
            config,
            [{"query": "q", "answer": "a"}],
            str(tmp_path),
            retriever=SimpleNamespace(file_path="retrieval.json"),
        )

    with open(results[0]["output_path"], encoding="utf-8") as result_file:
        payload = json.load(result_file)

    assert payload["generation_config"] == {
        "model": "provider/model",
        "temperature": 0.4,
        "top_p": 0.7,
        "max_tokens": 512,
        "reasoning_enabled": False,
    }
    assert results[0]["generation_config"] == payload["generation_config"]
    assert mock_client.call_args.kwargs["reasoning_enabled"] is False
