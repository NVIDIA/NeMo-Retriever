# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DataFrame-returning retrieval entrypoint.

``retrieve`` is the boundary between the raw LanceDB-hit contract that
:meth:`~nemo_retriever.retriever.Retriever.queries` still speaks
(``list[list[dict]]``) and the DataFrame-native contract that the rest
of this package consumes.  Every downstream function (:func:`answer`,
:func:`score`, :func:`judge`, :func:`eval`) can be given the DataFrame
produced here and will enrich it with additional columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

if TYPE_CHECKING:
    import pandas as pd

    from nemo_retriever.retriever import Retriever


def retrieve(
    retriever: "Retriever",
    queries: Union[str, Sequence[str]],
    *,
    top_k: Optional[int] = None,
    embedder: Optional[str] = None,
    lancedb_uri: Optional[str] = None,
    lancedb_table: Optional[str] = None,
) -> "pd.DataFrame":
    """Run retrieval and return a DataFrame of hits.

    Thin adapter over :meth:`Retriever.queries <nemo_retriever.retriever.Retriever.queries>`
    that reshapes the raw LanceDB hits into a :class:`pandas.DataFrame`
    with one row per query and the columns ``query``, ``chunks``,
    ``metadata``.  ``chunks`` is a ``list[str]`` of retrieved text in
    rank order; ``metadata`` is an aligned ``list[dict]`` carrying
    everything else the hit dictionary contained (source, page number,
    distance scores, etc.).

    Args:
        retriever: The :class:`~nemo_retriever.retriever.Retriever`
            instance to query.  Not mutated.
        queries: A single query string or an iterable of query strings.
            Order is preserved in the returned DataFrame.
        top_k: Per-call override of ``retriever.top_k``.  Passed through
            as a local value so the instance attribute is never
            mutated.
        embedder: Per-call embedder override.
        lancedb_uri: Per-call LanceDB URI override.
        lancedb_table: Per-call LanceDB table override.

    Returns:
        A :class:`pandas.DataFrame` with columns ``[query, chunks,
        metadata]`` and one row per input query.  Empty input returns an
        empty DataFrame with the same columns.

    Example:
        >>> from nemo_retriever.retriever import Retriever
        >>> from nemo_retriever.generation import retrieve
        >>> r = Retriever(lancedb_uri="./kb")  # doctest: +SKIP
        >>> df = retrieve(r, ["What is RAG?", "What is NVIDIA?"])  # doctest: +SKIP
        >>> list(df.columns)  # doctest: +SKIP
        ['query', 'chunks', 'metadata']
    """
    import pandas as pd

    query_list: list[str] = [queries] if isinstance(queries, str) else [str(q) for q in queries]

    if not query_list:
        return pd.DataFrame({"query": [], "chunks": [], "metadata": []})

    hits_per_query = retriever.queries(
        query_list,
        top_k=top_k,
        embedder=embedder,
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
    )

    rows: list[dict[str, Any]] = []
    for q, hits in zip(query_list, hits_per_query):
        chunks = [str(hit.get("text", "")) for hit in hits]
        metadata = [{k: v for k, v in hit.items() if k != "text"} for hit in hits]
        rows.append({"query": q, "chunks": chunks, "metadata": metadata})

    return pd.DataFrame(rows, columns=["query", "chunks", "metadata"])
