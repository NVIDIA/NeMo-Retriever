"""
QA evaluation pipeline orchestrator.

QAEvalPipeline wires together a retriever, one or more LLM clients, and a
judge to produce per-query and aggregate results across a full Q&A dataset.

Concurrency:
    Queries are processed in parallel using ThreadPoolExecutor. All work is
    I/O-bound (API calls), so thread-based concurrency is appropriate and
    simpler than asyncio. Retrieval is called once per query and the result
    is reused across all LLMs, avoiding redundant VDB calls.

    Default max_workers=8. Set qa_max_workers in test_configs.yaml to tune.
    At 8 workers: 369 queries typically completes in under 15 minutes even
    with slow reasoning models.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from nv_ingest_harness.utils.qa.scoring import answer_in_context, classify_failure, token_f1
from nv_ingest_harness.utils.qa.types import (
    AnswerJudge,
    LLMClient,
    QAQueryResult,
    RetrieverStrategy,
)


class QAEvalPipeline:
    """
    Orchestrates retrieval -> generation -> judging for a Q&A dataset.

    Args:
        retriever: Any object satisfying the RetrieverStrategy protocol.
        llm_clients: Mapping of {name: LLMClient}. Each client is run for
                     every query; results are keyed by name in the output.
        judge: Any object satisfying the AnswerJudge protocol.
        top_k: Number of chunks to retrieve per query.
        max_workers: Thread pool size for concurrent query processing.
        include_chunks_in_results: Emit retrieved chunk text in per-query
            output for debuggability. Defaults to True.
        chunk_char_limit: Truncate each chunk to this many characters in
            the output JSON to control file size. 0 = no limit.
    """

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
        self.retriever = retriever
        self.llm_clients = llm_clients
        self.judge = judge
        self.top_k = top_k
        self.max_workers = max_workers
        self.include_chunks_in_results = include_chunks_in_results
        self.chunk_char_limit = chunk_char_limit

    def evaluate(self, qa_pairs: list[dict]) -> dict:
        """
        Evaluate all Q&A pairs and return aggregated results.

        Args:
            qa_pairs: List of dicts, each with at least "query" and "answer" keys.

        Returns:
            Dict of the form::

                {
                  "per_query": [QAQueryResult, ...],
                  "by_model": {
                    "nemotron_super_49b": {
                      "mean_score": 4.1,
                      "score_distribution": {"1": 0, "2": 5, ...},
                      "mean_latency_s": 6.2,
                      "error_count": 0,
                    },
                    ...
                  }
                }
        """
        total = len(qa_pairs)
        results: list[Optional[QAQueryResult]] = [None] * total
        counter = {"done": 0}
        lock = threading.Lock()

        def _process(idx: int, pair: dict) -> tuple[int, QAQueryResult]:
            query = pair["query"]
            reference = pair["answer"]

            retrieval = self.retriever.retrieve(query, self.top_k)

            query_result = QAQueryResult(
                query=query,
                reference_answer=reference,
                retrieval=retrieval,
            )

            ref_in_chunks = answer_in_context(reference, retrieval.chunks)
            query_result.answer_in_context = ref_in_chunks

            for name, client in self.llm_clients.items():
                gen = client.generate(query, retrieval.chunks)
                query_result.generations[name] = gen

                if gen.error == "thinking_truncated":
                    from nv_ingest_harness.utils.qa.types import JudgeResult

                    query_result.judgements[name] = JudgeResult(
                        score=0, reasoning="Skipped: thinking truncated", error="thinking_truncated",
                    )
                    query_result.token_f1[name] = {
                        "exact_match": False, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                    }
                    query_result.failure_mode[name] = "thinking_truncated"
                    continue

                answer_text = gen.answer if not gen.error else ""
                verdict = self.judge.judge(query, reference, answer_text)
                query_result.judgements[name] = verdict

                query_result.token_f1[name] = token_f1(reference, answer_text)

                judge_score = verdict.score if not verdict.error else None
                query_result.failure_mode[name] = classify_failure(
                    ref_in_chunks=ref_in_chunks,
                    judge_score=judge_score,
                    gen_error=gen.error,
                    candidate=answer_text,
                )

            return idx, query_result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_process, i, pair): i for i, pair in enumerate(qa_pairs)}

            for future in as_completed(futures):
                try:
                    idx, query_result = future.result()
                    results[idx] = query_result
                except Exception as exc:
                    idx = futures[future]
                    pair = qa_pairs[idx]
                    print(f"  ERROR processing query [{idx}]: {pair.get('query', '')!r}: {exc}")

                with lock:
                    counter["done"] += 1
                    done = counter["done"]
                    if done % 10 == 0 or done == total:
                        print(f"  Progress: {done}/{total} queries completed")

        return self._aggregate(results)

    def _aggregate(self, results: list[Optional[QAQueryResult]]) -> dict:
        """Compute per-model aggregate statistics from per-query results."""
        valid = [r for r in results if r is not None]
        total_submitted = len(results)
        dropped = total_submitted - len(valid)

        scores_by_model: dict[str, list[int]] = defaultdict(list)
        latencies_by_model: dict[str, list[float]] = defaultdict(list)
        errors_by_model: dict[str, int] = defaultdict(int)
        f1_by_model: dict[str, list[float]] = defaultdict(list)
        exact_by_model: dict[str, list[bool]] = defaultdict(list)
        failures_by_model: dict[str, Counter] = defaultdict(Counter)

        aic_count = 0

        for qr in valid:
            if qr.answer_in_context:
                aic_count += 1

            for name, verdict in qr.judgements.items():
                if verdict.error and verdict.score == 0:
                    errors_by_model[name] += 1
                else:
                    scores_by_model[name].append(verdict.score)

            for name, gen in qr.generations.items():
                if not gen.error:
                    latencies_by_model[name].append(gen.latency_s)

            for name, tf1 in qr.token_f1.items():
                f1_by_model[name].append(tf1.get("f1", 0.0))
                exact_by_model[name].append(tf1.get("exact_match", False))

            for name, fm in qr.failure_mode.items():
                failures_by_model[name][fm] += 1

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
            exacts = exact_by_model[name]
            tier2[name] = {
                "mean_exact_match": round(sum(exacts) / len(exacts), 4) if exacts else 0.0,
                "mean_token_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            }

        failure_breakdown: dict[str, dict[str, int]] = {}
        for name in self.llm_clients:
            failure_breakdown[name] = dict(failures_by_model[name])

        return {
            "summary": {
                "total_submitted": total_submitted,
                "total_completed": len(valid),
                "dropped_queries": dropped,
            },
            "tier1_retrieval": {
                "answer_in_context_rate": round(aic_count / len(valid), 4) if valid else 0.0,
                "answer_in_context_count": aic_count,
                "total": len(valid),
            },
            "tier2_programmatic": tier2,
            "tier3_llm_judge": by_model,
            "failure_breakdown": failure_breakdown,
            "per_query": [
                _query_result_to_dict(
                    r,
                    include_chunks=self.include_chunks_in_results,
                    chunk_char_limit=self.chunk_char_limit,
                )
                for r in valid
            ],
            "by_model": by_model,
        }


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "..."


def _query_result_to_dict(
    qr: QAQueryResult,
    include_chunks: bool = True,
    chunk_char_limit: int = 500,
) -> dict:
    """Serialize a QAQueryResult to a plain dict for JSON output."""
    result: dict = {
        "query": qr.query,
        "reference_answer": qr.reference_answer,
        "retrieved_chunk_count": len(qr.retrieval.chunks),
        "answer_in_context": qr.answer_in_context,
    }

    if qr.token_f1:
        result["token_f1"] = qr.token_f1
    if qr.failure_mode:
        result["failure_mode"] = qr.failure_mode

    if include_chunks:
        result["retrieved_chunks"] = [
            _truncate(c, chunk_char_limit) for c in qr.retrieval.chunks
        ]
        result["retrieval_metadata"] = qr.retrieval.metadata

    result["generations"] = {
        name: {
            "answer": gen.answer,
            "latency_s": round(gen.latency_s, 3),
            "model": gen.model,
            "error": gen.error,
        }
        for name, gen in qr.generations.items()
    }
    result["judgements"] = {
        name: {
            "score": j.score,
            "reasoning": j.reasoning,
            "error": j.error,
        }
        for name, j in qr.judgements.items()
    }

    return result
