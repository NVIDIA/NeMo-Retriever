# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pluggable configuration layer for evaluation pipelines.

Provides a single YAML/JSON file as the source of truth for an entire
evaluation run -- datasets, models, retrieval sources, scoring, and
execution parameters.

Primary entry points:
    ``load_eval_config(path)``       -- read and validate a config file
    ``build_eval_chain(config)``     -- single-model ``>>`` graph chain
    ``build_eval_pipeline(config)``  -- multi-model ``QAEvalPipeline``

Config schema (YAML example)::

    dataset:
      source: "csv:data/bo767_annotations.csv"
      ground_truth_dir: "tools/harness/data"
      query_column: "query"
      answer_column: "answer"
      limit: 0

    retrieval:
      type: "file"
      file_path: "data/retrieval_bo767_run_1"

    generators:
      - name: "nemotron"
        model: "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
        api_key: "${NVIDIA_API_KEY}"
        temperature: 0.0
        max_tokens: 4096

    judge:
      model: "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
      api_key: "${NVIDIA_API_KEY}"

    execution:
      top_k: 5
      max_workers: 8
      chunk_char_limit: 500
      include_chunks_in_results: true

    output:
      results_file: "data/qa_results.json"

Environment variables are expanded in string values: ``${VAR}`` resolves
to ``os.environ["VAR"]`` at load time. Secrets never live in the config.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.graph.pipeline_graph import Graph

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

_REQUIRED_SECTIONS = ("generators", "judge")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` in string values."""
    if isinstance(value, str):

        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                logger.warning("Environment variable %s is not set", var_name)
                return match.group(0)
            return env_val

        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_eval_config(path: str) -> dict:
    """Load eval config from YAML (``.yaml``/``.yml``) or JSON (``.json``) file.

    Supports ``${VAR}`` env var expansion in string values (recursive).
    YAML requires ``pyyaml`` (in ``[eval]`` extras). JSON uses stdlib.

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed and env-var-expanded configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is unsupported or required sections are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML config files. " "Install it: pip install nemo-retriever[eval]"
            ) from exc
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {suffix!r}. " "Use .yaml, .yml, or .json.")

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(raw).__name__}")

    config = _expand_env_vars(raw)

    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}")

    return config


def build_eval_chain(
    config: dict,
    model_name: str | None = None,
) -> "Graph":
    """Construct a ``>>`` chain from config for single-model evaluation.

    If *model_name* is specified, uses that generator from config.
    If ``None``, uses the first generator listed.

    Returns a :class:`Graph`::

        RetrievalLoaderOperator >> QAGenerationOperator >> JudgingOperator >> ScoringOperator

    Parameters
    ----------
    config : dict
        Parsed config from :func:`load_eval_config`.
    model_name : str, optional
        Generator name to use. Defaults to the first in ``config["generators"]``.

    Returns
    -------
    Graph
        A chainable graph ready for ``.execute(None)``.
    """
    from nemo_retriever.evaluation.generation import QAGenerationOperator
    from nemo_retriever.evaluation.judging import JudgingOperator
    from nemo_retriever.evaluation.retrieval_loader import RetrievalLoaderOperator
    from nemo_retriever.evaluation.scoring_operator import ScoringOperator

    generators = config["generators"]
    if not generators:
        raise ValueError("Config must have at least one generator")

    if model_name is not None:
        gen_cfg = next((g for g in generators if g.get("name") == model_name), None)
        if gen_cfg is None:
            available = [g.get("name", "?") for g in generators]
            raise ValueError(f"Generator {model_name!r} not found. Available: {available}")
    else:
        gen_cfg = generators[0]

    execution = config.get("execution", {})
    retrieval = config.get("retrieval", {})
    dataset = config.get("dataset", {})
    judge_cfg = config["judge"]

    retrieval_json = retrieval.get("file_path", "")
    ground_truth_csv = dataset.get("source", "")
    if ground_truth_csv.startswith("csv:"):
        ground_truth_csv = ground_truth_csv[4:]

    loader = RetrievalLoaderOperator(
        retrieval_json=retrieval_json,
        ground_truth_csv=ground_truth_csv,
        query_column=dataset.get("query_column", "query"),
        answer_column=dataset.get("answer_column", "answer"),
        top_k=execution.get("top_k", 5),
    )

    gen_op = QAGenerationOperator(
        model=gen_cfg["model"],
        api_base=gen_cfg.get("api_base"),
        api_key=gen_cfg.get("api_key"),
        temperature=gen_cfg.get("temperature", 0.0),
        max_tokens=gen_cfg.get("max_tokens", 4096),
        extra_params=gen_cfg.get("extra_params"),
        num_retries=gen_cfg.get("num_retries", 3),
        max_workers=execution.get("max_workers", 8),
    )

    judge_op = JudgingOperator(
        model=judge_cfg["model"],
        api_base=judge_cfg.get("api_base"),
        api_key=judge_cfg.get("api_key"),
        extra_params=judge_cfg.get("extra_params"),
        max_workers=execution.get("max_workers", 8),
    )

    scoring_op = ScoringOperator()

    return loader >> gen_op >> judge_op >> scoring_op


def build_eval_pipeline(config: dict) -> "QAEvalPipeline":
    """Construct a multi-model ``QAEvalPipeline`` from config.

    Uses all generators listed in config for multi-model sweeps.

    Parameters
    ----------
    config : dict
        Parsed config from :func:`load_eval_config`.

    Returns
    -------
    QAEvalPipeline
        A fully configured pipeline ready for ``.evaluate(qa_pairs)``
        or ``.process(df)``.
    """
    from nemo_retriever.evaluation.generators import LiteLLMClient
    from nemo_retriever.evaluation.judges import LLMJudge
    from nemo_retriever.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.evaluation.retrievers import FileRetriever

    generators = config["generators"]
    if not generators:
        raise ValueError("Config must have at least one generator")

    execution = config.get("execution", {})
    retrieval = config.get("retrieval", {})
    judge_cfg = config["judge"]

    retrieval_type = retrieval.get("type", "file")
    if retrieval_type == "file":
        retriever = FileRetriever(file_path=retrieval["file_path"])
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type!r}. " "Currently only 'file' is supported.")

    llm_clients: dict[str, LiteLLMClient] = {}
    for gen_cfg in generators:
        name = gen_cfg.get("name", gen_cfg["model"])
        llm_clients[name] = LiteLLMClient(
            model=gen_cfg["model"],
            api_base=gen_cfg.get("api_base"),
            api_key=gen_cfg.get("api_key"),
            temperature=gen_cfg.get("temperature", 0.0),
            max_tokens=gen_cfg.get("max_tokens", 4096),
            extra_params=gen_cfg.get("extra_params"),
            num_retries=gen_cfg.get("num_retries", 3),
        )

    judge = LLMJudge(
        model=judge_cfg["model"],
        api_base=judge_cfg.get("api_base"),
        api_key=judge_cfg.get("api_key"),
        extra_params=judge_cfg.get("extra_params"),
    )

    return QAEvalPipeline(
        retriever=retriever,
        llm_clients=llm_clients,
        judge=judge,
        top_k=execution.get("top_k", 5),
        max_workers=execution.get("max_workers", 8),
        include_chunks_in_results=execution.get("include_chunks_in_results", True),
        chunk_char_limit=execution.get("chunk_char_limit", 500),
    )
