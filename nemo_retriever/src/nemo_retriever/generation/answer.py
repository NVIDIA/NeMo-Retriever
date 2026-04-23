# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieve + generate as a DataFrame pipeline.

``answer`` chains :func:`~nemo_retriever.generation.retrieve` with
:class:`~nemo_retriever.evaluation.generation.QAGenerationOperator` so
callers get a one-shot ``queries -> answers`` path without having to
instantiate the operator themselves.  The LLM client is supplied
pre-built by the caller so transport details (``api_base``, ``api_key``,
``num_retries``, ``temperature``, ``top_p``, ``max_tokens``, etc.) are
owned by the caller and reused across rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    import pandas as pd

    from nemo_retriever.llm.clients import LiteLLMClient
    from nemo_retriever.retriever import Retriever


def answer(
    retriever: "Retriever",
    queries: Union[str, Sequence[str]],
    *,
    llm: "LiteLLMClient",
    reference: Optional[Union[str, Sequence[str]]] = None,
    top_k: Optional[int] = None,
    embedder: Optional[str] = None,
    lancedb_uri: Optional[str] = None,
    lancedb_table: Optional[str] = None,
) -> "pd.DataFrame":
    """Retrieve chunks and generate grounded answers for each query.

    Internally performs :func:`~nemo_retriever.generation.retrieve` to
    fetch per-query chunks, renames the ``chunks`` column to ``context``
    (the name :class:`QAGenerationOperator` requires), runs the
    operator, and returns a DataFrame with the following columns:

    * ``query``              -- original query string.
    * ``chunks``             -- retrieved text chunks (list[str]).
    * ``context``            -- same as ``chunks`` (alias preserved for
      downstream operators that expect this column name).
    * ``metadata``           -- per-chunk metadata dicts.
    * ``reference_answer``   -- present iff ``reference`` was supplied.
    * ``answer``             -- model-generated answer string.
    * ``model``              -- LLM model identifier actually used.
    * ``latency_s``          -- per-row generation latency in seconds.
    * ``gen_error``          -- ``Optional[str]`` error string; ``None``
      on success.

    Args:
        retriever: The :class:`Retriever` to use for chunk retrieval.
        queries: Single query string or iterable of query strings.
        llm: Pre-configured :class:`LiteLLMClient`; its transport /
            sampling settings are reused for every row.
        reference: Optional ground-truth answer(s).  If a single string
            is supplied with multiple queries, it is broadcast to every
            row; otherwise its length must match ``queries``.  When
            provided, a ``reference_answer`` column is attached so the
            returned DataFrame can be fed directly into
            :func:`~nemo_retriever.generation.score` or
            :func:`~nemo_retriever.generation.judge` without further
            reshaping.
        top_k: Per-call override of ``retriever.top_k``.
        embedder: Per-call embedder override.
        lancedb_uri: Per-call LanceDB URI override.
        lancedb_table: Per-call LanceDB table override.

    Returns:
        A :class:`pandas.DataFrame` with one row per query.  The input
        ``retriever`` and ``llm`` are not mutated.

    Raises:
        ValueError: If ``reference`` is a sequence whose length does not
            match the number of queries.
    """
    from nemo_retriever.evaluation.generation import QAGenerationOperator
    from nemo_retriever.generation.retrieve import retrieve

    query_list: list[str] = [queries] if isinstance(queries, str) else [str(q) for q in queries]

    df = retrieve(
        retriever,
        query_list,
        top_k=top_k,
        embedder=embedder,
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
    )

    # QAGenerationOperator requires a ``context`` column of list[str];
    # ``retrieve`` returns ``chunks`` with exactly that shape.  Alias
    # rather than rename so downstream callers can use either name.
    df = df.copy()
    df["context"] = df["chunks"]

    if reference is not None:
        references = _broadcast_reference(reference, len(query_list))
        df["reference_answer"] = references

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
    return operator.run(df)


def _broadcast_reference(reference: Union[str, Sequence[str]], n: int) -> list[str]:
    """Normalise ``reference`` to a list of length ``n``.

    Single-string inputs are broadcast to ``n`` copies (the common
    case where the caller has one ground-truth per call); sequence
    inputs must match ``n`` exactly.
    """
    if isinstance(reference, str):
        return [reference] * n
    refs = [str(r) for r in reference]
    if len(refs) != n:
        raise ValueError(f"reference length {len(refs)} does not match number of queries {n}")
    return refs
