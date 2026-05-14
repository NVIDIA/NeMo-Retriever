# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator: fetch tabular entity descriptions from Neo4j into an embedding-ready DataFrame."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator


class TabularFetchEmbeddingsOp(AbstractOperator, CPUOperator):
    """Fetch tabular entity descriptions from Neo4j into an embedding-ready DataFrame.

    This operator ignores its input — it always queries Neo4j directly and
    returns a fresh DataFrame with columns:
    ``text``, ``_embed_modality``, ``path``, ``page_number``, ``metadata``.

    The output schema matches the format produced by the unstructured pipeline,
    so the standard :class:`~nemo_retriever.text_embed.operators._BatchEmbedActor`
    can be chained directly after this operator.

    When ``node_ids`` is provided, the query is restricted to those
    ``Table``/``Column`` ids. This is how the PATCH-driven incremental flow
    avoids re-embedding the whole database: the API handler already knows
    which nodes changed, so the operator fetches only those rows. Leaving
    ``node_ids`` as ``None`` returns every entity under ``database_name``.
    """

    def __init__(
        self,
        *,
        database_name: str,
        node_ids: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        normalized_ids: list[str] | None
        if node_ids is None:
            normalized_ids = None
        else:
            normalized_ids = [str(n) for n in node_ids if n is not None and str(n)]
            if not normalized_ids:
                normalized_ids = None
        super().__init__(database_name=database_name, node_ids=normalized_ids, **kwargs)
        self._database_name = database_name
        self._node_ids = normalized_ids

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        from nemo_retriever.tabular_data.ingestion.embeddings import fetch_tabular_embedding_dataframe

        return fetch_tabular_embedding_dataframe(
            database_name=self._database_name,
            node_ids=self._node_ids,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
