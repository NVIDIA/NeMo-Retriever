# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
QA evaluation pipeline orchestrator.

QAEvalPipeline wires together a retriever, one or more LLM clients, and a
judge to produce per-query and aggregate results across a full Q&A dataset.

It inherits from EvalOperator to provide graph-framework compatibility
(``>>``, ``run()``, ``Graph.execute()``), while preserving the legacy
``evaluate(qa_pairs)`` dict-based API.

Architecture:
    - QAEvalPipeline is a *molecule* (not an atom). It composes generation,
      judging, and scoring internally for multi-model sweeps.
    - ``evaluate()`` delegates to ``process()`` -- single codepath.
    - Not Ray-reconstructable (non-serializable constructor args).
      Use ``Graph.execute()`` or ``InprocessExecutor`` with instance reuse.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar, Optional

import pandas as pd

from nemo_retriever.evaluation.eval_operator import EvalOperator
from nemo_retriever.evaluation.scoring import (
    answer_in_context,
    classify_failure,
    token_f1,
)
from nemo_retriever.evaluation.types import (
    AnswerJudge,
    GenerationResult,
    JudgeResult,
    LLMClient,
    RetrieverStrategy,
)

logger = logging.getLogger(__name__)


class QAEvalPipeline(EvalOperator):
    """Orchestrates retrieval -> generation -> judging -> scoring.

    Handles multi-model by running one generation+judging+scoring pass
    per model, producing model-prefixed output columns.

    ``evaluate(qa_pairs)`` returns the legacy dict format for backward compat.
    ``process(df)`` provides the DataFrame-in/out interface for composition.

    This is a molecule operator -- not Ray-reconstructable due to
    non-serializable constructor args (retriever, llm_clients, judge).
    """

    required_columns: ClassVar[tuple[str, ...]] = ("query", "reference_answer", "context")
    output_columns: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        retriever: RetrieverStrategy,
        llm_clients: dict[str, LLMClient],
        judge: AnswerJudge,
        top_k: int = 5,
        max_workers: int = 8,
        include_chunks_in_results: bool = True,
        chunk_char_limit: int = 500,
    ):
        super().__init__()
        self.retriever = retriever
        self.llm_clients = llm_clients
        self.judge = judge
        self.top_k = top_k
        self.max_workers = max_workers
        self.include_chunks_in_results = include_chunks_in_results
        self.chunk_char_limit = chunk_char_limit

    # ------------------------------------------------------------------
    # Legacy dict API -- delegates to process()
    # ------------------------------------------------------------------

    def evaluate(self, qa_pairs: list[dict]) -> dict:
        """Evaluate all Q&A pairs and return aggregated results.

        This is the legacy dict-based API. Each dict must have a ``"query"``
        key and a reference answer under ``"reference_answer"`` or ``"answer"``
        (the latter for backward compatibility with ground truth CSVs).
        """
        df = self._prepare_dataframe(qa_pairs)
        result_df = self.process(df)
        return self._aggregate_from_df(result_df, qa_pairs)

    # ------------------------------------------------------------------
    # DataFrame-in/out (single codepath for all execution modes)
    # ------------------------------------------------------------------

    def process(self, data: Any, **kwargs: Any) -> Any:
        """Per-model generation+judging+scoring. Model-prefixed columns.

        Expects columns: ``query``, ``reference_answer``, ``context``
        (where ``context`` is a list[str] of retrieved chunks per row).

        For each configured model, runs generation then judging using the
        original client/judge objects, producing model-prefixed output
        columns with scoring metrics.
        """
        df = data

        prefixes: dict[str, str] = {}
        for name in self.llm_clients:
            prefix = _sanitize_prefix(name)
            if prefix in prefixes:
                raise ValueError(
                    f"Model names {prefixes[prefix]!r} and {name!r} produce "
                    f"the same column prefix {prefix!r}. Use distinct short names "
                    f"in llm_clients keys to avoid column collisions."
                )
            prefixes[prefix] = name

        out = df.copy()

        for model_name, client in self.llm_clients.items():
            prefix = _sanitize_prefix(model_name)

            # _client and _model_name are captured via default args to avoid
            # the late-binding closure bug across loop iterations. self.judge
            # is safe to access directly since it does not change per-iteration.
            def _process_row(row_tuple, _client=client, _model_name=model_name):
                _, row = row_tuple
                query = row["query"]
                ref = row["reference_answer"]
                ctx = row.get("context", [])
                if not isinstance(ctx, list):
                    ctx = []

                gen = _client.generate(query, ctx)
                answer_text = gen.answer if not gen.error else ""

                if gen.error == "thinking_truncated":
                    verdict = JudgeResult(
                        score=0,
                        reasoning="Skipped: thinking truncated",
                        error="thinking_truncated",
                    )
                else:
                    verdict = self.judge.judge(query, ref, answer_text)

                ref_in_chunks = answer_in_context(ref, ctx)
                tf1 = token_f1(ref, answer_text)
                judge_score_val = verdict.score if not verdict.error else None
                fm = classify_failure(
                    ref_in_chunks=ref_in_chunks,
                    judge_score=judge_score_val,
                    gen_error=gen.error,
                    candidate=answer_text,
                )
                return gen, verdict, ref_in_chunks, tf1, fm

            row_count = len(df)
            row_results: list = [None] * row_count
            done_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                row_futures = {pool.submit(_process_row, item): i for i, item in enumerate(df.iterrows())}
                for future in as_completed(row_futures):
                    idx = row_futures[future]
                    try:
                        row_results[idx] = future.result()
                    except Exception as exc:
                        logger.warning("Row %d failed for model %s: %s", idx, model_name, exc)
                        sentinel_gen = GenerationResult(
                            answer="",
                            latency_s=0.0,
                            model=model_name,
                            error=str(exc),
                        )
                        sentinel_judge = JudgeResult(score=0, reasoning="", error=str(exc))
                        row_results[idx] = (
                            sentinel_gen,
                            sentinel_judge,
                            False,
                            {"f1": 0.0, "exact_match": False},
                            "generation_miss",
                        )
                    done_count += 1
                    if done_count % 10 == 0 or done_count == row_count:
                        logger.info("Model %s: %d/%d rows completed", model_name, done_count, row_count)

            answers, latencies, models, gen_errors = [], [], [], []
            judge_scores, judge_reasonings, judge_errors = [], [], []
            aic_list, f1_list, fm_list = [], [], []

            for gen, verdict, aic, tf1, fm in row_results:
                answers.append(gen.answer)
                latencies.append(gen.latency_s)
                models.append(gen.model)
                gen_errors.append(gen.error)
                judge_scores.append(verdict.score)
                judge_reasonings.append(verdict.reasoning)
                judge_errors.append(verdict.error)
                aic_list.append(aic)
                f1_list.append(tf1.get("f1", 0.0))
                fm_list.append(fm)

            out[f"{prefix}_answer"] = answers
            out[f"{prefix}_latency_s"] = latencies
            out[f"{prefix}_model"] = models
            out[f"{prefix}_gen_error"] = gen_errors
            out[f"{prefix}_judge_score"] = judge_scores
            out[f"{prefix}_judge_reasoning"] = judge_reasonings
            out[f"{prefix}_judge_error"] = judge_errors
            out[f"{prefix}_answer_in_context"] = aic_list
            out[f"{prefix}_token_f1"] = f1_list
            out[f"{prefix}_failure_mode"] = fm_list

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_dataframe(self, qa_pairs: list[dict]) -> pd.DataFrame:
        """Convert qa_pairs to DataFrame with retrieval context.

        For each pair, retrieves chunks via self.retriever (threaded)
        and builds a DataFrame with: query, reference_answer, context.
        """
        total = len(qa_pairs)
        rows: list[Optional[dict]] = [None] * total
        counter = {"done": 0}
        lock = threading.Lock()

        def _retrieve(idx: int, pair: dict) -> tuple[int, dict]:
            query = pair["query"]
            reference = pair.get("reference_answer") or pair["answer"]
            retrieval = self.retriever.retrieve(query, self.top_k)
            return idx, {
                "query": query,
                "reference_answer": reference,
                "context": retrieval.chunks,
                "_retrieval_metadata": retrieval.metadata,
            }

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_retrieve, i, pair): i for i, pair in enumerate(qa_pairs)}
            for future in as_completed(futures):
                try:
                    idx, row_dict = future.result()
                    rows[idx] = row_dict
                except Exception as exc:
                    idx = futures[future]
                    pair = qa_pairs[idx]
                    logger.error("Retrieval for query [%d] failed: %r: %s", idx, pair.get("query", ""), exc)
                    rows[idx] = {
                        "query": pair.get("query", ""),
                        "reference_answer": pair.get("reference_answer") or pair.get("answer", ""),
                        "context": [],
                        "_retrieval_metadata": [],
                    }
                with lock:
                    counter["done"] += 1
                    done = counter["done"]
                    if done % 10 == 0 or done == total:
                        logger.info("Retrieval progress: %d/%d queries completed", done, total)

        valid_rows = [r for r in rows if r is not None]
        return pd.DataFrame(valid_rows)

    def _aggregate_from_df(self, df: pd.DataFrame, qa_pairs: list[dict]) -> dict:
        """Convert model-prefixed DataFrame to legacy dict format.

        Reconstructs the same output structure as the original _aggregate().
        """
        total_submitted = len(qa_pairs)

        scores_by_model: dict[str, list[int]] = defaultdict(list)
        latencies_by_model: dict[str, list[float]] = defaultdict(list)
        errors_by_model: dict[str, int] = defaultdict(int)
        f1_by_model: dict[str, list[float]] = defaultdict(list)
        failures_by_model: dict[str, Counter] = defaultdict(Counter)

        aic_count = 0
        per_query: list[dict] = []

        for _, row in df.iterrows():
            qr_dict: dict = {
                "query": row["query"],
                "reference_answer": row["reference_answer"],
                "retrieved_chunk_count": len(row.get("context", [])),
            }

            first_prefix = None
            row_aic = False
            generations_dict: dict = {}
            judgements_dict: dict = {}
            token_f1_dict: dict = {}
            failure_mode_dict: dict = {}

            for model_name in self.llm_clients:
                prefix = _sanitize_prefix(model_name)
                if first_prefix is None:
                    first_prefix = prefix

                aic_val = row.get(f"{prefix}_answer_in_context", False)
                if prefix == first_prefix:
                    row_aic = aic_val
                    qr_dict["answer_in_context"] = aic_val

                answer = row.get(f"{prefix}_answer", "")
                latency = row.get(f"{prefix}_latency_s", 0.0)
                model = row.get(f"{prefix}_model", "")
                gen_error = row.get(f"{prefix}_gen_error")
                j_score = row.get(f"{prefix}_judge_score", 0)
                j_reasoning = row.get(f"{prefix}_judge_reasoning", "")
                j_error = row.get(f"{prefix}_judge_error")
                tf1_val = row.get(f"{prefix}_token_f1", 0.0)
                fm_val = row.get(f"{prefix}_failure_mode", "")

                generations_dict[model_name] = {
                    "answer": answer,
                    "latency_s": round(latency, 3) if latency else 0.0,
                    "model": model,
                    "error": gen_error,
                }
                judgements_dict[model_name] = {
                    "score": j_score,
                    "reasoning": j_reasoning,
                    "error": j_error,
                }
                token_f1_dict[model_name] = {
                    "f1": tf1_val,
                }
                failure_mode_dict[model_name] = fm_val

                if j_error and j_score == 0:
                    errors_by_model[model_name] += 1
                else:
                    scores_by_model[model_name].append(j_score)

                if not gen_error:
                    latencies_by_model[model_name].append(latency)

                f1_by_model[model_name].append(tf1_val)
                failures_by_model[model_name][fm_val] += 1

            if row_aic:
                aic_count += 1

            if self.include_chunks_in_results:
                ctx = row.get("context", [])
                qr_dict["retrieved_chunks"] = [
                    _truncate(c, self.chunk_char_limit) for c in (ctx if isinstance(ctx, list) else [])
                ]
                qr_dict["retrieval_metadata"] = row.get("_retrieval_metadata", [])

            qr_dict["token_f1"] = token_f1_dict
            qr_dict["failure_mode"] = failure_mode_dict
            qr_dict["generations"] = generations_dict
            qr_dict["judgements"] = judgements_dict
            per_query.append(qr_dict)

        total_completed = len(df)
        dropped = total_submitted - total_completed

        by_model: dict[str, dict] = {}
        for name in self.llm_clients:
            scores = scores_by_model[name]
            latencies = latencies_by_model[name]
            dist: dict[str, int] = {str(k): 0 for k in range(1, 6)}
            for s in scores:
                dist[str(s)] = dist.get(str(s), 0) + 1
            by_model[name] = {
                "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "score_distribution": dist,
                "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                "scored_count": len(scores),
                "error_count": errors_by_model[name],
            }

        tier2: dict[str, dict] = {}
        for name in self.llm_clients:
            f1s = f1_by_model[name]
            tier2[name] = {
                "mean_token_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            }

        failure_breakdown: dict[str, dict[str, int]] = {}
        for name in self.llm_clients:
            failure_breakdown[name] = dict(failures_by_model[name])

        return {
            "summary": {
                "total_submitted": total_submitted,
                "total_completed": total_completed,
                "dropped_queries": dropped,
            },
            "tier1_retrieval": {
                "answer_in_context_rate": round(aic_count / total_completed, 4) if total_completed else 0.0,
                "answer_in_context_count": aic_count,
                "total": total_completed,
            },
            "tier2_programmatic": tier2,
            "tier3_llm_judge": by_model,
            "failure_breakdown": failure_breakdown,
            "per_query": per_query,
            "by_model": by_model,
        }


def _sanitize_prefix(name: str) -> str:
    """Convert a model name like 'nvidia_nim/llama-3.1-nemotron' to a column prefix."""
    short = name.rsplit("/", 1)[-1]
    return short.replace("-", "_").replace(".", "_")


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "..."
