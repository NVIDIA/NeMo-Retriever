# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator: fetch tabular entity descriptions from Neo4j into an embedding-ready DataFrame."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator


class TabularFetchEmbeddingsOp(AbstractOperator, CPUOperator):
    """Fetch all tabular entity descriptions from Neo4j into an embedding-ready DataFrame.

    This operator ignores its input — it always queries Neo4j directly and
    returns a fresh DataFrame with columns:
    ``text``, ``_embed_modality``, ``path``, ``page_number``, ``metadata``.

    The output schema matches the format produced by the unstructured pipeline,
    so the standard :class:`~nemo_retriever.text_embed.operators._BatchEmbedActor`
    can be chained directly after this operator.
    """

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        from nemo_retriever.tabular_data.ingestion.embeddings import fetch_tabular_embedding_dataframe

        return fetch_tabular_embedding_dataframe()

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
