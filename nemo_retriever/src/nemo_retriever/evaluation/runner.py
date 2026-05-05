# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable evaluation sweep runner.

Provides :func:`run_eval_sweep`, which iterates the ``evaluations`` list
in a parsed config, creates LLM clients and judges for each combo, runs
:class:`QAEvalPipeline`, and writes timestamped result JSONs.

This module is the library counterpart of the ``_run_config_sweep``
function that previously lived in ``tools/harness/run_qa_eval.py``
(now deleted; use ``retriever eval run`` instead).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.llm.types import RetrieverStrategy

logger = logging.getLogger(__name__)


def run_eval_sweep(
    config: dict,
    qa_pairs: list[dict],
    results_dir: str,
    *,
    retriever: "RetrieverStrategy | None" = None,
    dataset_label: str = "",
    on_run_complete: Callable[[dict, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Run all evaluations defined in *config* and write per-run result JSONs.

    Parameters
    ----------
    config : dict
        Parsed config from :func:`~nemo_retriever.evaluation.config.load_eval_config`.
    qa_pairs : list[dict]
        Ground-truth Q&A pairs (each must have ``"query"`` and ``"answer"``).
    results_dir : str
        Directory where timestamped result JSONs are written.
    retriever : RetrieverStrategy, optional
        Pre-built retriever.  When *None*, a
        :class:`~nemo_retriever.evaluation.retrievers.FileRetriever` is
        created from ``config["retrieval"]["file_path"]``.
    dataset_label : str
        Short label for the dataset (used in output filenames).
        When empty, derived from ``config["dataset"]["source"]``.
    on_run_complete : callable, optional
        ``on_run_complete(result_dict, total_queries)`` is called after
        each individual run completes.  Useful for per-run console output.

    Returns
    -------
    list[dict]
        One dict per run with keys ``label``, ``status`` (``"PASS"`` or
        ``"FAIL"``), ``output_path`` (or ``error``), and ``eval_results``
        (the full evaluation dict when status is PASS).
    """
    from nemo_retriever.llm.clients import LLMJudge, LiteLLMClient
    from nemo_retriever.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.evaluation.retrievers import FileRetriever

    models = config["models"]
    evaluations = config["evaluations"]
    execution = config.get("execution", {})
    retrieval_cfg = config.get("retrieval", {})
    dataset_cfg = config.get("dataset", {})

    default_runs = execution.get("runs", 1)
    top_k = execution.get("top_k", 5)
    max_workers = execution.get("max_workers", 8)
    default_timeout = execution.get("timeout", 120.0)

    if retriever is None:
        retrieval_file = retrieval_cfg.get("file_path", "")
        if not retrieval_file:
            raise ValueError("retrieval.file_path is required in config when no retriever is passed")
        retriever = FileRetriever(file_path=retrieval_file)

    retrieval_file = getattr(retriever, "file_path", "<custom retriever>")

    if not dataset_label:
        qa_source = dataset_cfg.get("source", "")
        raw = qa_source.split(":", 1)[-1] if ":" in qa_source else qa_source
        dataset_label = os.path.splitext(os.path.basename(raw))[0] if raw else "eval"

    total_runs = sum(e.get("runs", default_runs) for e in evaluations)
    results_log: list[dict[str, Any]] = []
    run_number = 0

    for eval_cfg in evaluations:
        gen_name = eval_cfg["generator"]
        judge_name = eval_cfg["judge"]
        n_runs = eval_cfg.get("runs", default_runs)

        gen_model_cfg = models[gen_name]
        judge_model_cfg = models[judge_name]

        from nemo_retriever.evaluation.config import check_unresolved_env

        check_unresolved_env(gen_model_cfg.get("api_key"), "api_key", f"generator '{gen_name}'")
        check_unresolved_env(judge_model_cfg.get("api_key"), "api_key", f"judge '{judge_name}'")

        client = LiteLLMClient.from_kwargs(
            model=gen_model_cfg["model"],
            api_base=gen_model_cfg.get("api_base"),
            api_key=gen_model_cfg.get("api_key"),
            temperature=eval_cfg.get("temperature", gen_model_cfg.get("temperature", 0.0)),
            top_p=eval_cfg.get("top_p", gen_model_cfg.get("top_p")),
            max_tokens=eval_cfg.get("max_tokens", gen_model_cfg.get("max_tokens", 4096)),
            extra_params=gen_model_cfg.get("extra_params"),
            num_retries=gen_model_cfg.get("num_retries", 3),
            timeout=gen_model_cfg.get("timeout", default_timeout),
        )
        judge = LLMJudge.from_kwargs(
            model=judge_model_cfg["model"],
            api_base=judge_model_cfg.get("api_base"),
            api_key=judge_model_cfg.get("api_key"),
            extra_params=judge_model_cfg.get("extra_params"),
            timeout=judge_model_cfg.get("timeout", default_timeout),
        )

        for run_idx in range(1, n_runs + 1):
            run_number += 1
            label = f"{gen_name} + {judge_name} (run {run_idx}/{n_runs})"
            logger.info("[%d/%d] %s", run_number, total_runs, label)

            try:
                pipeline = QAEvalPipeline(
                    retriever=retriever,
                    llm_clients={gen_name: client},
                    judge=judge,
                    top_k=top_k,
                    max_workers=max_workers,
                )
                eval_results = pipeline.evaluate(qa_pairs)

                gen_label = gen_name.replace("/", "_").replace(" ", "_")
                judge_label = judge_name.replace("/", "_").replace(" ", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if n_runs > 1:
                    fname = (
                        f"qa_results_{dataset_label}_{gen_label}_{judge_label}" f"_{run_idx}of{n_runs}_{timestamp}.json"
                    )
                else:
                    fname = f"qa_results_{dataset_label}_{gen_label}_{judge_label}" f"_{timestamp}.json"
                out_path = os.path.join(results_dir, fname)
                os.makedirs(results_dir, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "dataset": dataset_cfg.get("source", ""),
                            "retrieval_file": str(retrieval_file),
                            "top_k": top_k,
                            "generator": gen_name,
                            "judge": judge_name,
                            "run": run_idx,
                            "total_runs": n_runs,
                            "qa_results": eval_results,
                        },
                        f,
                        indent=2,
                    )
                logger.info("Results written to %s", out_path)
                result = {
                    "label": label,
                    "status": "PASS",
                    "output_path": out_path,
                    "eval_results": eval_results,
                }

            except Exception as exc:
                logger.error("Run failed: %s", exc)
                result = {
                    "label": label,
                    "status": "FAIL",
                    "error": str(exc),
                }

            results_log.append(result)
            if on_run_complete is not None:
                on_run_complete(result, len(qa_pairs))

    return results_log
