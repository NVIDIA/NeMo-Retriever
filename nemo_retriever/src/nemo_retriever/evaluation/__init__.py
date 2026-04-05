# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QA evaluation framework for nemo_retriever.

Provides pluggable retrieval, generation, judging, and orchestration
components for measuring LLM answer quality given retrieved context.

The ``EvalOperator`` base class bridges ``graph.AbstractOperator`` into the
evaluation domain, enabling ``>>`` chaining, ``Graph.execute()``, and
executor compatibility for all evaluation operators.

Types, scoring, and ``EvalOperator`` are always available.
Modules that depend on ``litellm`` (generators, judges, generation,
judging, orchestrator, config) are lazy-loaded so that lightweight
consumers can use scoring without installing the ``[eval]`` extra::

    pip install nemo-retriever[eval]
"""

from nemo_retriever.evaluation.eval_operator import EvalOperator
from nemo_retriever.evaluation.scoring import score_dataframe
from nemo_retriever.evaluation.types import (
    AnswerJudge,
    GenerationResult,
    JudgeResult,
    LLMClient,
    QAQueryResult,
    RetrievalResult,
    RetrieverStrategy,
)

_LAZY_IMPORTS = {
    "QAGenerationOperator": "nemo_retriever.evaluation.generation",
    "JudgingOperator": "nemo_retriever.evaluation.judging",
    "ScoringOperator": "nemo_retriever.evaluation.scoring_operator",
    "RetrievalLoaderOperator": "nemo_retriever.evaluation.retrieval_loader",
    "LiteLLMClient": "nemo_retriever.evaluation.generators",
    "LLMJudge": "nemo_retriever.evaluation.judges",
    "QAEvalPipeline": "nemo_retriever.evaluation.orchestrator",
    "load_eval_config": "nemo_retriever.evaluation.config",
    "build_eval_chain": "nemo_retriever.evaluation.config",
    "build_eval_pipeline": "nemo_retriever.evaluation.config",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnswerJudge",
    "EvalOperator",
    "GenerationResult",
    "JudgeResult",
    "JudgingOperator",
    "LLMClient",
    "LLMJudge",
    "LiteLLMClient",
    "QAEvalPipeline",
    "QAGenerationOperator",
    "QAQueryResult",
    "RetrievalLoaderOperator",
    "RetrievalResult",
    "RetrieverStrategy",
    "ScoringOperator",
    "build_eval_chain",
    "build_eval_pipeline",
    "load_eval_config",
    "score_dataframe",
]
