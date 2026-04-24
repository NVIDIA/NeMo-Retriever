# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence
from tqdm import tqdm

from nemo_retriever.model import VL_EMBED_MODEL, VL_RERANK_MODEL

if TYPE_CHECKING:
    import pandas as pd

    from nemo_retriever.llm.types import (
        AnswerJudge,
        AnswerResult,
        LLMClient,
        RetrievalResult,
    )

@dataclass
class Retriever:
    """VDB-agnostic query helper over vector databases.

    Retrieval pipeline
    ------------------
    1. Embed query strings with a local HuggingFace model.
    2. Search the configured vector database.
    3. Optionally rerank the results with ``nvidia/llama-nemotron-rerank-1b-v2``
       (NIM/vLLM endpoint or local HuggingFace model).

    Reranking
    ---------
    Set ``reranker`` to a model name (e.g.
    ``"nvidia/llama-nemotron-rerank-1b-v2"``) to enable post-retrieval
    reranking.  Results are re-sorted by the cross-encoder score and a
    ``"_rerank_score"`` key is added to each hit dict.

    Use ``reranker_endpoint`` to delegate to a running vLLM (>=0.14) or NIM
    server instead of loading the model locally::

        retriever = Retriever(
            vdb_op="lancedb",
            vdb_kwargs={"uri": "./kb", "table_name": "nv-ingest"},
            reranker="nvidia/llama-nemotron-rerank-1b-v2",
            reranker_endpoint="http://localhost:8000",
        )
        results = retriever.query("What is machine learning?")
    """

    vdb: Any | None = None
    vdb_op: Optional[str] = "lancedb"
    vdb_kwargs: dict[str, Any] = field(default_factory=dict)
    embedder: str = VL_EMBED_MODEL
    top_k: int = 10
    local_hf_device: Optional[str] = None
    local_hf_cache_dir: Optional[Path] = None
    local_hf_batch_size: int = 64
    # Reranking -----------------------------------------------------------
    reranker: Optional[bool] = False
    """True to enable reranking with the default model, will use the reranker_model_name as hf model"""
    reranker_model_name: Optional[str] = VL_RERANK_MODEL
    """HuggingFace model ID for local reranking (e.g. 'nvidia/llama-nemotron-rerank-1b-v2')."""
    reranker_endpoint: Optional[str] = None
    """Base URL of a vLLM / NIM ranking endpoint. Appends ``/v1/ranking`` unless already using ``/reranking``."""
    reranker_api_key: str = ""
    """Bearer token for the remote rerank endpoint."""
    reranker_max_length: int = 10240
    """Tokenizer truncation length for local reranking (max 8 192)."""
    reranker_batch_size: int = 32
    """GPU micro-batch size for local reranking."""
    reranker_refine_factor: int = 4
    """Number of candidates to rerank = top_k * reranker_refine_factor.
    Set to 1 to rerank only the top_k results."""
    rerank_modality: str = "text"
    """Reranking modality, typically matches embed_modality. Set to 'text_image'
    to enable multimodal reranking with images."""
    # Internal cache for the local rerank model (not part of the public API).
    _reranker_model: Any = field(default=None, init=False, repr=False, compare=False)
    # Internal cache for local HF embedders, keyed by model name.
    _embedder_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _merged_vdb_kwargs(
        self,
        *,
        top_k: int,
        vdb_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        merged = {**dict(self.vdb_kwargs or {}), **dict(vdb_kwargs or {})}
        merged["top_k"] = int(top_k)
        return merged

    def _get_local_embedder(self, model_name: str) -> Any:
        """Lazily load and cache the local HF embedder for *model_name*."""
        if model_name not in self._embedder_cache:
            from nemo_retriever.model import create_local_embedder

            cache_dir = str(self.local_hf_cache_dir) if self.local_hf_cache_dir else None
            self._embedder_cache[model_name] = create_local_embedder(
                model_name,
                device=self.local_hf_device,
                hf_cache_dir=cache_dir,
            )
        return self._embedder_cache[model_name]

    def _embed_queries_local_hf(self, query_texts: list[str], *, model_name: Optional[str] = None) -> list[list[float]]:
        from nemo_retriever.model import is_vl_embed_model

        model_name = str(model_name or self.embedder)
        embedder = self._get_local_embedder(model_name)

        if is_vl_embed_model(model_name):
            vectors = embedder.embed_queries(query_texts, batch_size=int(self.local_hf_batch_size))
        else:
            vectors = embedder.embed(["query: " + q for q in query_texts], batch_size=int(self.local_hf_batch_size))
        return vectors.detach().to("cpu").tolist()

    # ------------------------------------------------------------------
    # Reranking helpers
    # ------------------------------------------------------------------

    def _get_reranker_model(self) -> Any:
        """Lazily load and cache the local reranker model (text-only or VL)."""
        if self._reranker_model is None and self.reranker:
            from nemo_retriever.model import create_local_reranker

            cache_dir = str(self.local_hf_cache_dir) if self.local_hf_cache_dir else None
            self._reranker_model = create_local_reranker(
                model_name=self.reranker_model_name,
                device=self.local_hf_device,
                hf_cache_dir=cache_dir,
            )
        return self._reranker_model

    def _rerank_results(
        self,
        query_texts: list[str],
        results: list[list[dict[str, Any]]],
        *,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        """Rerank each per-query result list using the configured reranker."""
        from nemo_retriever.rerank import rerank_hits

        reranker_endpoint = (self.reranker_endpoint or "").strip() or None
        model = None if reranker_endpoint else self._get_reranker_model()

        reranked: list[list[dict[str, Any]]] = []
        for query, hits in tqdm(zip(query_texts, results), desc="Reranking", unit="query", total=len(query_texts)):
            reranked.append(
                rerank_hits(
                    query,
                    hits,
                    model=model,
                    invoke_url=reranker_endpoint,
                    model_name=str(self.reranker_model_name),
                    api_key=(self.reranker_api_key or "").strip(),
                    max_length=int(self.reranker_max_length),
                    batch_size=int(self.reranker_batch_size),
                    top_n=int(top_k),
                    modality=self.rerank_modality,
                )
            )
        return reranked

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval for a single query string.

        Args:
            query: The natural-language query.
            top_k: Per-call override of ``self.top_k``; passed as a local
                value so the instance attribute is never mutated.
            vdb_kwargs: Per-call VDB retrieval kwargs. These override
                instance-level ``self.vdb_kwargs`` for this call.
        """
        return self.queries([query], top_k=top_k, vdb_kwargs=vdb_kwargs)[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        top_k: Optional[int] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[list[dict[str, Any]]]:
        """Run retrieval for multiple query strings.

        If ``reranker`` is set on this instance the initial vector-search
        results are re-scored with ``nvidia/llama-nemotron-rerank-1b-v2``
        (or the configured endpoint) and returned sorted by cross-encoder
        score.  Each hit gains a ``"_rerank_score"`` key.

        The ``top_k`` kwarg is threaded through the search + rerank stack
        as a local value so concurrent callers never race on ``self.top_k``.
        """
        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        effective_top_k = int(top_k) if top_k is not None else int(self.top_k)
        retrieval_top_k = effective_top_k
        if self.reranker:
            retrieval_top_k = effective_top_k * int(self.reranker_refine_factor)
        merged_vdb_kwargs = self._merged_vdb_kwargs(top_k=retrieval_top_k, vdb_kwargs=vdb_kwargs)

        from nemo_retriever.vdb import (
            RetrieveVdbOperator,
            normalize_retrieval_results,
            search_vdb_with_vectors,
            supports_vector_search_vdb,
        )

        results: list[list[dict[str, Any]]] | None = None
        if self.vdb is None and supports_vector_search_vdb(self.vdb_op):
            resolved_embedder = str(
                merged_vdb_kwargs.get("query_model_name") or merged_vdb_kwargs.get("model_name") or self.embedder
            )
            vectors = self._embed_queries_local_hf(query_texts, model_name=resolved_embedder)
            raw_results = search_vdb_with_vectors(
                vdb_op=self.vdb_op,
                query_vectors=vectors,
                query_texts=query_texts,
                top_k=retrieval_top_k,
                vdb_kwargs=merged_vdb_kwargs,
            )
            results = normalize_retrieval_results(raw_results)

        if results is None:
            operator_kwargs: dict[str, Any] = {"vdb_kwargs": dict(self.vdb_kwargs or {})}
            if self.vdb is not None:
                operator_kwargs["vdb"] = self.vdb
            else:
                operator_kwargs["vdb_op"] = self.vdb_op

            operator = RetrieveVdbOperator(**operator_kwargs)
            results = operator.process(query_texts, **merged_vdb_kwargs)

        if self.reranker:
            results = self._rerank_results(query_texts, results, top_k=effective_top_k)

        return results

    # ------------------------------------------------------------------
    # Live RAG API (structured retrieval + generation)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        vdb_kwargs: Optional[dict[str, Any]] = None,
    ) -> "RetrievalResult":
        """Run retrieval for a single query and return a structured result.

        Thin adapter over :meth:`query` that reshapes the raw VDB hits
        into a :class:`~nemo_retriever.llm.RetrievalResult` with ``chunks``
        (the retrieved text, in rank order) and aligned ``metadata``
        (source, page_number, etc.).  Satisfies the
        :class:`~nemo_retriever.llm.RetrieverStrategy` Protocol.

        Args:
            query: The natural-language query.
            top_k: Per-call override of ``self.top_k``; passed as a local
                value so the instance attribute is never mutated.
            vdb_kwargs: Per-call VDB retrieval kwargs.

        Returns:
            A :class:`~nemo_retriever.llm.RetrievalResult` whose ``chunks``
            and ``metadata`` lists have the same length.

        Example:
            >>> retriever = Retriever(vdb_kwargs={"uri": "./kb", "table_name": "nv-ingest"})
            >>> result = retriever.retrieve("What is RAG?", top_k=3)
            >>> result.chunks[0][:40]  # doctest: +SKIP
            'Retrieval augmented generation combines...'
        """
        from nemo_retriever.llm.types import RetrievalResult

        hits = self.query(query, top_k=top_k, vdb_kwargs=vdb_kwargs)

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
    ) -> list["RetrievalResult"]:
        """Run retrieval for a batch of queries in one VDB call.

        Funnels the whole query list through :meth:`queries`, which delegates
        embedding and backend-specific search to the configured VDB.

        Args:
            queries: Iterable of natural-language query strings.  Order
                is preserved in the returned list.
            top_k: Per-call override of ``self.top_k``; passed as a local
                value so the instance attribute is never mutated.
            vdb_kwargs: Per-call VDB retrieval kwargs.

        Returns:
            A list of :class:`~nemo_retriever.llm.RetrievalResult`,
            aligned one-to-one with ``queries``.  Empty input returns an
            empty list.
        """

        from nemo_retriever.llm.types import RetrievalResult

        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        hits_per_query = self.queries(query_texts, top_k=top_k, vdb_kwargs=vdb_kwargs)

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
    ) -> "AnswerResult":
        """Run live RAG for a single query and optionally score the answer.

        Performs ``retrieve -> llm.generate`` and, when a ``reference`` answer
        (for token-level scoring) and/or a ``judge`` (for LLM-as-judge
        scoring) are supplied, fans those out concurrently on a small thread
        pool so the judge network call and the local token-F1 computation do
        not serialize.

        Scoring tiers that can be populated on the returned
        :class:`~nemo_retriever.llm.AnswerResult`:

          * Tier 1 -- ``answer_in_context`` (requires ``reference``)
          * Tier 2 -- ``token_f1``, ``exact_match`` (requires ``reference``)
          * Tier 3 -- ``judge_score``, ``judge_reasoning`` (requires ``judge``
            and ``reference``); also populates ``failure_mode``

        When generation fails the returned result has ``error`` populated
        and all scoring/judge fields remain ``None`` -- scoring is skipped
        to avoid misleading metrics on an empty answer.

        Args:
            query: Natural-language question.
            llm: Any object satisfying the
                :class:`~nemo_retriever.llm.LLMClient` Protocol (typically
                :class:`~nemo_retriever.llm.LiteLLMClient`).
            judge: Optional LLM-as-judge.  Requires ``reference``.
            reference: Ground-truth answer for token-F1 and judge scoring.
            top_k: Per-call override of ``self.top_k``.
            vdb_kwargs: Per-call VDB retrieval kwargs.

        Returns:
            An :class:`~nemo_retriever.llm.AnswerResult` carrying the
            generated answer, the retrieved context, and any scoring
            artefacts that were requested.

        Raises:
            ValueError: If ``judge`` is supplied without ``reference``.

        Example:
            >>> from nemo_retriever.llm import LiteLLMClient
            >>> retriever = Retriever(vdb_kwargs={"uri": "./kb", "table_name": "nv-ingest"})
            >>> llm = LiteLLMClient.from_kwargs(
            ...     model="nvidia_nim/meta/llama-3.3-70b-instruct",
            ... )
            >>> result = retriever.answer(  # doctest: +SKIP
            ...     "What did Q4 revenue look like?",
            ...     llm=llm,
            ... )
            >>> result.answer  # doctest: +SKIP
            'Revenue grew 12% YoY to $4.2B...'
        """
        from nemo_retriever.llm.types import AnswerResult

        if judge is not None and reference is None:
            raise ValueError("judge requires reference")

        retrieved = self.retrieve(query, top_k=top_k, vdb_kwargs=vdb_kwargs)

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
        """Populate scoring tiers on ``result`` in-place.

        Runs Tier-1 + Tier-2 (pure CPU, sub-millisecond) alongside the Tier-3
        judge API call (network-bound) on a two-worker thread pool so the
        judge latency is not extended by scoring.  After both complete,
        ``failure_mode`` is derived from the combined signals via
        :func:`~nemo_retriever.evaluation.scoring.classify_failure`.
        """
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
        """Return a fluent builder for a batch live-RAG operator graph.

        The builder composes existing evaluation operators -- live retrieval
        (via :class:`~nemo_retriever.evaluation.live_retrieval.LiveRetrievalOperator`),
        :class:`~nemo_retriever.evaluation.generation.QAGenerationOperator`,
        :class:`~nemo_retriever.evaluation.scoring_operator.ScoringOperator`,
        and :class:`~nemo_retriever.evaluation.judging.JudgingOperator` --
        using the existing ``>>`` chaining from
        :mod:`nemo_retriever.graph.pipeline_graph`.  No new graph primitives
        are introduced; this method is sugar for building and executing that
        graph against a list of queries.

        Steps are optional and independent.  Call only the ones you want, in
        any order (retrieval always runs first since it is the source).

        Args:
            top_k: Override ``self.top_k`` for retrieval within this
                pipeline.  Defaults to the instance attribute.

        Returns:
            A :class:`RetrieverPipelineBuilder` whose ``.run(queries)`` method
            executes the composed graph and returns a ``pandas.DataFrame``.

        Example:
            >>> from nemo_retriever.llm import LiteLLMClient, LLMJudge
            >>> retriever = Retriever(vdb_kwargs={"uri": "./kb", "table_name": "nv-ingest"})
            >>> llm = LiteLLMClient.from_kwargs(model="nvidia_nim/meta/llama-3.3-70b-instruct")
            >>> judge = LLMJudge.from_kwargs(model="nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")
            >>> df = (  # doctest: +SKIP
            ...     retriever.pipeline()
            ...     .generate(llm)
            ...     .score()
            ...     .judge(judge)
            ...     .run(queries=["What is RAG?"], reference=["Retrieval-augmented generation..."])
            ... )
        """
        effective_top_k = int(top_k) if top_k is not None else int(self.top_k)
        return RetrieverPipelineBuilder(self, top_k=effective_top_k)

    def generate_sql(self, query: str) -> str:
        """Generate a SQL query for a given natural language query."""
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
