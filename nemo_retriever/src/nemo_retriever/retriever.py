# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import pandas as pd

from nemo_retriever.model import VL_EMBED_MODEL, VL_RERANK_MODEL
from nemo_retriever.retriever_graph_utils import (
    filter_retrieval_kwargs,
    rerank_long_dataframe_to_hits,
)
from nemo_retriever.vdb.operators import RetrieveVdbOperator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nemo_retriever.llm.types import (
        AnswerJudge,
        AnswerResult,
        LLMClient,
        RetrievalResult,
    )


def _coerce_vdb_init(user: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``vdb_kwargs`` into :class:`RetrieveVdbOperator` constructor kwargs."""
    u = dict(user or {})
    if "vdb" in u or "vdb_op" in u:
        return u
    return {"vdb_op": "lancedb", "vdb_kwargs": u}


def _default_rerank_actor_kwargs() -> dict[str, Any]:
    return {
        "model_name": VL_RERANK_MODEL,
        "query_column": "query",
        "text_column": "text",
        "score_column": "rerank_score",
        "max_length": 10240,
        "batch_size": 32,
        "sort_results": False,
        "api_key": "",
    }


@dataclass
class Retriever:
    """Graph-based query helper: batch embed → VDB retrieve [→ Nemotron rerank].

    Configuration is passed through ``embed_kwargs`` (:class:`~nemo_retriever.params.EmbedParams`),
    ``vdb_kwargs`` (constructor kwargs for :class:`~nemo_retriever.vdb.operators.RetrieveVdbOperator`),
    and optional ``rerank_kwargs`` for :class:`~nemo_retriever.rerank.rerank.NemotronRerankActor`.

    See ``retriever.md`` for examples.
    """

    run_mode: Literal["local", "service"] = "local"
    """``local`` uses archetype batch embed resolution; ``service`` forces CPU HTTP embed."""

    top_k: int = 10
    rerank: bool = False
    """When ``True``, append :class:`~nemo_retriever.rerank.rerank.NemotronRerankActor` after retrieval."""

    graph: Any = None
    """Custom :class:`~nemo_retriever.graph.pipeline_graph.Graph`. When set, ``embed_kwargs`` /
    ``vdb_kwargs`` default-graph fields are ignored for construction (you still pass execute kwargs)."""

    embed_kwargs: dict[str, Any] = field(default_factory=dict)
    vdb_kwargs: dict[str, Any] = field(default_factory=dict)
    rerank_kwargs: dict[str, Any] = field(default_factory=dict)

    _cached_graph: Any = field(default=None, init=False, repr=False, compare=False)
    _cache_key: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.run_mode not in ("local", "service"):
            raise ValueError("run_mode must be 'local' or 'service'")

    def _merge_embed_params(self, extra: Optional[dict[str, Any]] = None) -> Any:
        from nemo_retriever.model import _LOCAL_INGEST_EMBED_BACKENDS, normalize_backend
        from nemo_retriever.params import EmbedParams

        base: dict[str, Any] = {
            "model_name": VL_EMBED_MODEL,
            "embed_model_name": VL_EMBED_MODEL,
            "input_type": "query",
            "text_column": "text",
            "inference_batch_size": 32,
            "embed_inference_batch_size": 32,
        }
        merged = {**base, **dict(self.embed_kwargs or {}), **dict(extra or {})}
        if "local_ingest_embed_backend" in merged and merged["local_ingest_embed_backend"] is not None:
            merged["local_ingest_embed_backend"] = normalize_backend(
                str(merged["local_ingest_embed_backend"]),
                _LOCAL_INGEST_EMBED_BACKENDS,
                field_name="local_ingest_embed_backend",
                default="vllm",
            )
        params = EmbedParams.model_validate(merged)
        if self.run_mode == "service":
            url = (params.embedding_endpoint or params.embed_invoke_url or "").strip()
            if not url:
                raise ValueError(
                    "run_mode='service' requires a non-empty HTTP embedding URL. "
                    "Set ``embedding_endpoint`` or ``embed_invoke_url`` inside ``embed_kwargs``."
                )
        return params

    def _merge_rerank_actor_kwargs(self) -> dict[str, Any]:
        return {**_default_rerank_actor_kwargs(), **dict(self.rerank_kwargs or {})}

    def _refine_factor(self) -> int:
        if not self.rerank:
            return 1
        return int(self._merge_rerank_actor_kwargs().get("refine_factor", 4))

    def _build_default_graph(self, *, embed_extra: Optional[dict[str, Any]] = None) -> Any:
        from nemo_retriever.rerank.rerank import NemotronRerankActor
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor
        from nemo_retriever.text_embed.operators import _BatchEmbedActor

        embed_params = self._merge_embed_params(embed_extra)
        if self.run_mode == "service":
            embed_op = _BatchEmbedCPUActor(params=embed_params)
        else:
            embed_op = _BatchEmbedActor(params=embed_params)

        vdb_init = _coerce_vdb_init(self.vdb_kwargs)
        retrieve = RetrieveVdbOperator(
            explode_for_rerank=self.rerank,
            **vdb_init,
        )

        chain = embed_op >> retrieve
        if self.rerank:
            rk = self._merge_rerank_actor_kwargs()
            rk.pop("refine_factor", None)
            chain = chain >> NemotronRerankActor(**rk)

        return chain

    def _get_graph(self, *, embed_extra: Optional[dict[str, Any]] = None) -> Any:
        if self.graph is not None:
            return self.graph

        key = (
            self.run_mode,
            self.rerank,
            json.dumps(self.vdb_kwargs, sort_keys=True, default=str),
            json.dumps(self.embed_kwargs, sort_keys=True, default=str),
            json.dumps(self.rerank_kwargs, sort_keys=True, default=str),
            json.dumps(embed_extra or {}, sort_keys=True, default=str),
        )
        if self._cached_graph is not None and self._cache_key == key:
            return self._cached_graph
        g = self._build_default_graph(embed_extra=embed_extra)
        self._cached_graph = g
        self._cache_key = key
        return g

    def _execute_queries_graph(
        self,
        query_texts: list[str],
        *,
        effective_top_k: int,
        retrieval_top_k: int,
        vdb_call_kwargs: Optional[dict[str, Any]],
        embed_extra: Optional[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        embed_params = self._merge_embed_params(embed_extra)
        text_col = str(embed_params.text_column)
        df = pd.DataFrame({text_col: query_texts})

        graph = self._get_graph(embed_extra=embed_extra)
        if not callable(getattr(graph, "resolve_for_local_execution", None)):
            raise TypeError("graph must provide resolve_for_local_execution() (e.g. pipeline_graph.Graph)")

        exec_kwargs: dict[str, Any] = {
            **filter_retrieval_kwargs(dict(vdb_call_kwargs or {})),
            "top_k": int(retrieval_top_k),
            "query_texts": query_texts,
        }
        resolved = graph.resolve_for_local_execution()
        leaves = resolved.execute(df, **exec_kwargs)
        if len(leaves) != 1:
            raise RuntimeError(
                f"Retriever query graph must yield exactly one leaf output; got {len(leaves)}. "
                "Use a linear graph or adjust your custom ``graph``."
            )
        out = leaves[0]

        if isinstance(out, pd.DataFrame):
            if not self.rerank:
                raise TypeError(
                    "Graph returned a DataFrame but ``rerank`` is False; expected list[list[dict]] from retrieval."
                )
            rk = self._merge_rerank_actor_kwargs()
            score_col = str(rk.get("score_column", "rerank_score"))
            return rerank_long_dataframe_to_hits(
                out, query_texts=query_texts, top_k=int(effective_top_k), score_column=score_col
            )
        if not isinstance(out, list):
            raise TypeError(f"Unexpected query graph output type: {type(out).__name__}")
        return out

    def query(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        return self.queries([query], top_k=top_k, vdb_kwargs=vdb_kwargs, embed_kwargs=embed_kwargs)[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        top_k: Optional[int] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[list[dict[str, Any]]]:
        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        effective_top_k = int(top_k) if top_k is not None else int(self.top_k)
        refine = self._refine_factor()
        retrieval_top_k = effective_top_k * refine if self.rerank else effective_top_k

        return self._execute_queries_graph(
            query_texts,
            effective_top_k=effective_top_k,
            retrieval_top_k=retrieval_top_k,
            vdb_call_kwargs=vdb_kwargs,
            embed_extra=embed_kwargs,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> "RetrievalResult":
        from nemo_retriever.llm.types import RetrievalResult

        hits = self.query(query, top_k=top_k, vdb_kwargs=vdb_kwargs, embed_kwargs=embed_kwargs)

        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []
        for hit in hits:
            chunks.append(str(hit.get("text", "")))
            metadata.append({k: v for k, v in hit.items() if k != "text"})
        return RetrievalResult(chunks=chunks, metadata=metadata)

    def retrieve_batch(
        self,
        queries: Sequence[str],
        *,
        top_k: Optional[int] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> list["RetrievalResult"]:
        from nemo_retriever.llm.types import RetrievalResult

        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        hits_per_query = self.queries(query_texts, top_k=top_k, vdb_kwargs=vdb_kwargs, embed_kwargs=embed_kwargs)

        results: list[RetrievalResult] = []
        for hits in hits_per_query:
            chunks = [str(hit.get("text", "")) for hit in hits]
            metadata = [{k: v for k, v in hit.items() if k != "text"} for hit in hits]
            results.append(RetrievalResult(chunks=chunks, metadata=metadata))
        return results

    def answer(
        self,
        query: str,
        *,
        llm: "LLMClient",
        judge: Optional["AnswerJudge"] = None,
        reference: Optional[str] = None,
        top_k: Optional[int] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> "AnswerResult":
        from nemo_retriever.llm.types import AnswerResult

        if judge is not None and reference is None:
            raise ValueError("judge requires reference")

        retrieved = self.retrieve(query, top_k=top_k, vdb_kwargs=vdb_kwargs, embed_kwargs=embed_kwargs)

        gen = llm.generate(query, retrieved.chunks)

        result = AnswerResult(
            query=query,
            answer=gen.answer,
            chunks=retrieved.chunks,
            metadata=retrieved.metadata,
            model=gen.model,
            latency_s=gen.latency_s,
            error=gen.error,
        )

        if gen.error is not None:
            return result

        if reference is None and judge is None:
            return result

        self._populate_scores(
            result,
            query=query,
            reference=reference,
            judge=judge,
            gen_error=gen.error,
        )
        return result

    def _populate_scores(
        self,
        result: "AnswerResult",
        *,
        query: str,
        reference: Optional[str],
        judge: Optional["AnswerJudge"],
        gen_error: Optional[str],
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from nemo_retriever.evaluation.scoring import (
            answer_in_context,
            classify_failure,
            token_f1,
        )

        def _scoring() -> tuple[Optional[bool], Optional[float], Optional[bool]]:
            if reference is None:
                return None, None, None
            aic = answer_in_context(reference, result.chunks)
            f1 = token_f1(reference, result.answer)
            return aic, float(f1.get("f1", 0.0)), bool(f1.get("exact_match", False))

        def _judging() -> tuple[Optional[int], Optional[str], Optional[str]]:
            if judge is None or reference is None:
                return None, None, None
            jr = judge.judge(query, reference, result.answer)
            return jr.score, jr.reasoning, jr.error

        with ThreadPoolExecutor(max_workers=2) as pool:
            scoring_future = pool.submit(_scoring)
            judge_future = pool.submit(_judging)
            aic, f1, em = scoring_future.result()
            judge_score, judge_reasoning, judge_error = judge_future.result()

        result.answer_in_context = aic
        result.token_f1 = f1
        result.exact_match = em
        result.judge_score = judge_score
        result.judge_reasoning = judge_reasoning
        result.judge_error = judge_error

        if reference is not None and aic is not None:
            result.failure_mode = classify_failure(
                ref_in_chunks=aic,
                judge_score=judge_score,
                gen_error=gen_error,
                candidate=result.answer,
            )

    def pipeline(self, *, top_k: Optional[int] = None) -> "RetrieverPipelineBuilder":
        effective_top_k = int(top_k) if top_k is not None else int(self.top_k)
        return RetrieverPipelineBuilder(self, top_k=effective_top_k)

    def generate_sql(self, query: str) -> str:
        from nemo_retriever.tabular_data.retrieval import generate_sql

        return generate_sql(query)


class RetrieverPipelineBuilder:
    """Fluent builder for live-RAG batch operator graphs.

    Returned from :meth:`Retriever.pipeline`.  Each builder method appends
    an :class:`~nemo_retriever.evaluation.eval_operator.EvalOperator` to an
    internal list; :meth:`run` composes them into a graph via the existing
    ``>>`` chaining and executes it on a DataFrame built from the provided
    queries.

    Example:
        >>> builder = retriever.pipeline()  # doctest: +SKIP
        >>> df = builder.generate(llm).score().judge(judge).run(  # doctest: +SKIP
        ...     queries=["q1", "q2"],
        ...     reference=["r1", "r2"],
        ... )
    """

    def __init__(self, retriever: "Retriever", *, top_k: int = 5) -> None:
        self._retriever = retriever
        self._top_k = int(top_k)
        self._steps: list[Any] = []

    def with_retrieval(self, *, top_k: int) -> "RetrieverPipelineBuilder":
        """Override the ``top_k`` used for the live retrieval source."""
        self._top_k = int(top_k)
        return self

    def generate(
        self,
        llm: Optional[Any] = None,
        /,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> "RetrieverPipelineBuilder":
        """Append a :class:`QAGenerationOperator` step.

        Accepts either a pre-built
        :class:`~nemo_retriever.llm.clients.LiteLLMClient` (whose transport
        and sampling params are unpacked onto the operator) or the flat
        ``model=..., api_base=..., ...`` kwargs forwarded to the operator
        constructor directly.

        Raises:
            ValueError: If neither ``llm`` nor ``model`` is provided.
        """
        from nemo_retriever.evaluation.generation import QAGenerationOperator

        if llm is None and model is None:
            raise ValueError("generate() requires either llm= or model=")

        if llm is not None:
            transport = llm.transport
            sampling = llm.sampling
            operator = QAGenerationOperator(
                model=transport.model,
                api_base=transport.api_base,
                api_key=transport.api_key,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                extra_params=dict(transport.extra_params) if transport.extra_params else None,
                num_retries=transport.num_retries,
                timeout=transport.timeout,
            )
        else:
            operator = QAGenerationOperator(model=model, **kwargs)

        self._steps.append(operator)
        return self

    def score(self) -> "RetrieverPipelineBuilder":
        """Append a :class:`ScoringOperator` step (Tier 1 + Tier 2)."""
        from nemo_retriever.evaluation.scoring_operator import ScoringOperator

        self._steps.append(ScoringOperator())
        return self

    def judge(
        self,
        judge: Optional[Any] = None,
        /,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> "RetrieverPipelineBuilder":
        """Append a :class:`JudgingOperator` step (Tier 3).

        Accepts either a pre-built
        :class:`~nemo_retriever.llm.clients.judge.LLMJudge` (whose transport params
        are unpacked onto the operator) or the flat ``model=...`` kwargs
        forwarded to the operator constructor.

        Raises:
            ValueError: If neither ``judge`` nor ``model`` is provided.
        """
        from nemo_retriever.evaluation.judging import JudgingOperator

        if judge is None and model is None:
            raise ValueError("judge() requires either judge= or model=")

        if judge is not None:
            transport = judge._client.transport
            operator = JudgingOperator(
                model=transport.model,
                api_base=transport.api_base,
                api_key=transport.api_key,
                extra_params=dict(transport.extra_params) if transport.extra_params else None,
                num_retries=transport.num_retries,
                timeout=transport.timeout,
            )
        else:
            operator = JudgingOperator(model=model, **kwargs)

        self._steps.append(operator)
        return self

    def run(
        self,
        queries: Any,
        *,
        reference: Any = None,
    ) -> "pd.DataFrame":
        """Execute the composed graph on ``queries``.

        Args:
            queries: A single query string, a list of query strings, or a
                pre-built ``pandas.DataFrame`` (which must contain a
                ``query`` column and, when judging/scoring, a
                ``reference_answer`` column).
            reference: Optional ground-truth answer(s).  Accepts a single
                string (applied to all queries), a list aligned with
                ``queries``, or ``None``.  Ignored when ``queries`` is
                already a DataFrame.

        Returns:
            A ``pandas.DataFrame`` with the columns contributed by each
            appended step (always ``query``, ``context``, and
            ``context_metadata``; plus ``answer``/``latency_s``/... when
            ``.generate()`` ran, and so on).

        Raises:
            ValueError: If ``reference`` is a list whose length does not
                match ``queries``.
        """
        import pandas as pd

        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator

        if isinstance(queries, str):
            query_list = [queries]
            df = pd.DataFrame({"query": query_list})
            if reference is not None:
                refs = reference if isinstance(reference, list) else [reference]
                if len(refs) != len(query_list):
                    raise ValueError("reference length must match queries length")
                df["reference_answer"] = refs
        elif isinstance(queries, list):
            df = pd.DataFrame({"query": list(queries)})
            if reference is not None:
                refs = reference if isinstance(reference, list) else [reference] * len(queries)
                if len(refs) != len(queries):
                    raise ValueError("reference length must match queries length")
                df["reference_answer"] = refs
        elif isinstance(queries, pd.DataFrame):
            df = queries.copy()
        else:
            raise TypeError("queries must be a str, list[str], or pandas.DataFrame; " f"got {type(queries).__name__}")

        retrieval_op = LiveRetrievalOperator(self._retriever, top_k=self._top_k)
        if not self._steps:
            out = retrieval_op.run(df)
        else:
            graph = retrieval_op
            for step in self._steps:
                graph = graph >> step
            # Linear live-RAG pipelines have exactly one leaf.
            leaves = graph.execute(df)
            if len(leaves) != 1:
                raise RuntimeError(f"Unexpected pipeline fan-out: got {len(leaves)} leaf outputs")
            out = leaves[0]

        # Expose the generation failure rate on ``df.attrs`` for downstream aggregators.
        if "gen_error" in out.columns and len(out) > 0:
            out.attrs["generation_failure_rate"] = float(out["gen_error"].notna().mean())

        return out


# Backward compatibility alias.
retriever = Retriever
