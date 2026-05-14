# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator: extract relational DB schema and store it in Neo4j."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.params import TabularExtractParams

if TYPE_CHECKING:
    from nemo_retriever.tabular_data.ingestion.model.schema import Schema


class TabularSchemaExtractOp(AbstractOperator, CPUOperator):
    """Extract schema entities from a relational DB and write them to Neo4j.

    Combines two steps:
    1. Pull schema metadata (tables, columns, views, PKs, FKs) from the
       database via the :class:`~nemo_retriever.tabular_data.sql_database.SQLDatabase`
       connector stored in *tabular_params*.
    2. Write the extracted entities as graph nodes and relationships into Neo4j.

    The operator returns the ``{schema_name_lower: Schema}`` dict produced by
    the ingest step. Each :class:`Schema` already carries the post-ingest
    ``tables_df`` / ``columns_df`` with the UUIDs written to Neo4j, so
    downstream operators — notably :class:`TabularFetchEmbeddingsOp` — can
    build embedding text directly from it without a Neo4j round-trip.

    Returns an empty dict when there is nothing to ingest, so the chain
    still flows.
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

    def process(self, data: TabularExtractParams | None, **kwargs: Any) -> dict[str, "Schema"]:
        from nemo_retriever.tabular_data.ingestion.extract_data import (
            extract_tabular_db_data,
            store_relational_db_in_neo4j,
        )

        if data is None or data.connector is None:
            return {}

        schema_data = extract_tabular_db_data(params=data)
        return store_relational_db_in_neo4j(data=schema_data, dialect=data.connector.dialect) or {}

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
