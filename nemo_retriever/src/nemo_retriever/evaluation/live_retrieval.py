# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiveRetrievalOperator -- live LanceDB retrieval source for evaluation chains."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from nemo_retriever.evaluation.eval_operator import EvalOperator

if TYPE_CHECKING:
    from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)


class LiveRetrievalOperator(EvalOperator):
    """Live retrieval source for evaluation chains.

    Parallels :class:`~nemo_retriever.evaluation.retrieval_loader.RetrievalLoaderOperator`
    but pulls chunks from LanceDB on the fly via
    :meth:`Retriever.queries <nemo_retriever.retriever.Retriever.queries>`
    rather than loading them from a pre-computed retrieval JSON.  Used
    by :func:`nemo_retriever.generation.eval` to prepend retrieval to a
    DataFrame-in/out generation / scoring / judging graph.

    Input DataFrame must have a ``query`` column.  Adds ``context``
    (``list[str]`` of chunk texts per row) and ``context_metadata``
    (``list[dict]`` aligned with ``context``).

    Notes:
        This operator is **inprocess-only**.  The wrapped
        :class:`~nemo_retriever.retriever.Retriever` instance is held as an
        in-memory attribute rather than registered as a constructor kwarg,
        because a live embedder / reranker cache does not serialise for
        Ray fan-out.  Use
        :class:`~nemo_retriever.evaluation.retrieval_loader.RetrievalLoaderOperator`
        instead for distributed batch evaluation.

    Example:
        >>> from nemo_retriever.retriever import Retriever  # doctest: +SKIP
        >>> retriever = Retriever(lancedb_uri="./kb")  # doctest: +SKIP
        >>> op = LiveRetrievalOperator(retriever, top_k=5)  # doctest: +SKIP
        >>> import pandas as pd  # doctest: +SKIP
        >>> df = pd.DataFrame({"query": ["What is RAG?"]})  # doctest: +SKIP
        >>> enriched = op.process(df)  # doctest: +SKIP
        >>> list(enriched.columns)  # doctest: +SKIP
        ['query', 'context', 'context_metadata']
    """

    required_columns: ClassVar[tuple[str, ...]] = ("query",)
    output_columns: ClassVar[tuple[str, ...]] = ("context", "context_metadata")

    def __init__(self, retriever: "Retriever", *, top_k: int = 5) -> None:
        # ``retriever`` is not a serialisable constructor kwarg (embedders,
        # LanceDB handles, reranker model caches), so only ``top_k`` is
        # registered for get_constructor_kwargs().  This operator is
        # inprocess-only as documented above.
        super().__init__(top_k=int(top_k))
        self._retriever = retriever
        self._top_k = int(top_k)

    def process(self, data: Any, **kwargs: Any) -> Any:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} requires a pandas.DataFrame input, " f"got {type(data).__name__}")

        out = data.copy()
        query_texts = [str(q) for q in out["query"]]

        # One batched call instead of per-row iteration.  The Retriever
        # embeds all queries in a single NIM round trip and issues a
        # single LanceDB sweep, so an N-row DataFrame pays O(1) network
        # cost end-to-end rather than O(N).  ``Retriever.queries``
        # preserves input order so ``hits_per_query[i]`` aligns with
        # row ``i``.
        hits_per_query = self._retriever.queries(query_texts, top_k=self._top_k)

        if len(hits_per_query) != len(query_texts):
            raise RuntimeError(
                "Retriever.queries returned "
                f"{len(hits_per_query)} results for {len(query_texts)} queries; "
                "this violates the contract and points at a Retriever bug."
            )

        out["context"] = [[str(hit.get("text", "")) for hit in hits] for hits in hits_per_query]
        out["context_metadata"] = [
            [{k: v for k, v in hit.items() if k != "text"} for hit in hits] for hits in hits_per_query
        ]
        return out
