# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator: extract relational DB schema and store it in Neo4j."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.params import TabularExtractParams


class TabularSchemaExtractOp(AbstractOperator, CPUOperator):
    """Extract schema entities from a relational DB and write them to Neo4j.

    Combines two steps:
    1. Pull schema metadata (tables, columns, views, PKs, FKs) from the
       database via the :class:`~nemo_retriever.tabular_data.sql_database.SQLDatabase`
       connector stored in *tabular_params*.
    2. Write the extracted entities as graph nodes and relationships into Neo4j.

    The operator produces an empty DataFrame as output so it can be chained
    with downstream operators (e.g. :class:`TabularFetchEmbeddingsOp`) via
    ``>>``.  All meaningful state lives in Neo4j after this step.
    """

    def __init__(
        self,
        *,
        tabular_params: TabularExtractParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(tabular_params=tabular_params, **kwargs)
        self._tabular_params = tabular_params

    def preprocess(self, data: Any, **kwargs: Any) -> TabularExtractParams | None:
        if isinstance(data, TabularExtractParams):
            return data
        return self._tabular_params

    def process(self, data: TabularExtractParams | None, **kwargs: Any) -> pd.DataFrame:
        from nemo_retriever.tabular_data.ingestion.extract_data import (
            extract_tabular_db_data,
            store_relational_db_in_neo4j,
        )

        schema_data = extract_tabular_db_data(params=data)
        store_relational_db_in_neo4j(data=schema_data, dialect=data.connector.dialect)
        return pd.DataFrame()

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
