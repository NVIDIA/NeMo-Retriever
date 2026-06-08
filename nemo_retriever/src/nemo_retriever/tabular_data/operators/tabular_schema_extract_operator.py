# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator: extract relational DB schema and store it in Neo4j."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.common.params import TabularExtractParams


class TabularSchemaExtractOp(AbstractOperator, CPUOperator):
    """Extract schema entities from a relational DB and write them to Neo4j.

    Combines two steps:
    1. Pull schema metadata (tables, columns, views, PKs, FKs) from the
       database via the :class:`~nemo_retriever.tabular_data.sql_database.SQLDatabase`
       connector stored in *tabular_params*.
    2. Write the extracted entities as graph nodes and relationships into Neo4j.

    The operator returns ``(tables_df, columns_df)`` — concatenated across
    every ingested :class:`Schema` — carrying the post-ingest UUIDs written
    to Neo4j. The per-row ``table_schema`` column keeps schemas
    distinguishable. Downstream operators (notably
    :class:`TabularFetchEmbeddingsOp`) can build embedding text directly
    from this pair without a Neo4j round-trip.

    Returns ``(empty_df, empty_df)`` when there is nothing to ingest, so
    the chain still flows.
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

    def process(self, data: TabularExtractParams | None, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        from nemo_retriever.tabular_data.ingestion.extract_data import (
            extract_tabular_db_data,
            store_relational_db_in_neo4j,
        )

        empty = (pd.DataFrame(), pd.DataFrame())
        if data is None or data.connector is None:
            return empty

        schema_data = extract_tabular_db_data(params=data)
        schemas = store_relational_db_in_neo4j(data=schema_data, dialect=data.connector.dialect) or {}
        if not schemas:
            return empty

        tables = [s.tables_df for s in schemas.values() if s.tables_df is not None]
        columns = [s.columns_df for s in schemas.values() if s.columns_df is not None]
        tables_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        columns_df = pd.concat(columns, ignore_index=True) if columns else pd.DataFrame()
        return tables_df, columns_df

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
