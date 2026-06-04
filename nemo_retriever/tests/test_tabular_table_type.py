# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for persisting type through tabular ingestion normalization."""

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.schema import Schema
from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes
from nemo_retriever.tabular_data.ingestion.utils import normalize_tables


def test_normalize_tables_keeps_type():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "type": ["base table"],
        }
    )
    result = normalize_tables(raw)

    assert "type" in result.columns
    assert result["type"].iloc[0] == "base table"
    assert str(result["type"].dtype) == "category"


def test_normalize_tables_adds_type_when_absent():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
        }
    )
    result = normalize_tables(raw)

    assert "type" in result.columns
    assert result["type"].iloc[0] == TableTypes.BASE_TABLE
    assert str(result["type"].dtype) == "category"


def test_normalize_tables_accepts_legacy_table_type_column():
    raw = pd.DataFrame(
        {
            "table_schema": ["public", "public"],
            "table_name": ["orders", "orders_v"],
            "table_type": ["base table", "view"],
        }
    )
    result = normalize_tables(raw)

    assert "table_type" not in result.columns
    assert result["type"].tolist() == ["base table", "view"]


def test_normalize_tables_maps_materialized_view():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["mv_orders"],
            "type": ["materialized view"],
        }
    )
    result = normalize_tables(raw)
    assert result["type"].iloc[0] == TableTypes.MATERIALIZED_VIEW


def test_reset_tables_props_sets_type():
    tables_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "created": [pd.NA],
            "description": [pd.NA],
            "type": ["base table"],
            "id": ["table-uuid-1"],
        }
    )
    columns_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "column_name": ["id"],
            "ordinal_position": [1],
            "data_type": ["INTEGER"],
            "is_nullable": ["NO"],
            "description": [pd.NA],
            "id": ["col-uuid-1"],
        }
    )
    db_node = Neo4jNode(name="mydb", label="Database", props={"name": "mydb"})
    schema = Schema(
        db_node=db_node,
        schema_tables_df=tables_df,
        schema_columns_df=columns_df,
        is_creation_mode=True,
    )

    props = schema.tables_df.iloc[0]["props"]
    assert props["type"] == "base table"
    assert props["name"] == "orders"


def test_reset_tables_props_defaults_type_when_absent():
    tables_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "created": [pd.NA],
            "description": [pd.NA],
            "id": ["table-uuid-1"],
        }
    )
    columns_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "column_name": ["id"],
            "ordinal_position": [1],
            "data_type": ["INTEGER"],
            "is_nullable": ["NO"],
            "description": [pd.NA],
            "id": ["col-uuid-1"],
        }
    )
    db_node = Neo4jNode(name="mydb", label="Database", props={"name": "mydb"})
    schema = Schema(
        db_node=db_node,
        schema_tables_df=tables_df,
        schema_columns_df=columns_df,
        is_creation_mode=True,
    )

    props = schema.tables_df.iloc[0]["props"]
    assert props["type"] == TableTypes.BASE_TABLE


def test_create_table_node_does_not_set_type():
    db_node = Neo4jNode(name="mydb", label="Database", props={"name": "mydb"})
    schema = Schema(
        db_node=db_node,
        schema_tables_df=pd.DataFrame(columns=["table_schema", "table_name", "created", "description", "id"]),
        schema_columns_df=pd.DataFrame(
            columns=[
                "table_schema",
                "table_name",
                "column_name",
                "ordinal_position",
                "data_type",
                "is_nullable",
                "description",
                "id",
            ]
        ),
        is_creation_mode=False,
    )
    schema.create_schema_node("public")
    schema.create_table_node("orders", id="table-uuid-1")

    props = schema.get_table_node("orders").get_properties()
    assert "type" not in props


def test_create_table_node_sets_type():
    db_node = Neo4jNode(name="mydb", label="Database", props={"name": "mydb"})
    schema = Schema(
        db_node=db_node,
        schema_tables_df=pd.DataFrame(columns=["table_schema", "table_name", "created", "description", "id"]),
        schema_columns_df=pd.DataFrame(
            columns=[
                "table_schema",
                "table_name",
                "column_name",
                "ordinal_position",
                "data_type",
                "is_nullable",
                "description",
                "id",
            ]
        ),
        is_creation_mode=False,
    )
    schema.create_schema_node("public")
    schema.create_table_node("orders", id="table-uuid-1", type="VIEW")

    props = schema.get_table_node("orders").get_properties()
    assert props["type"] == "view"


def test_get_table_node_passes_type_from_dataframe():
    tables_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "created": [pd.NA],
            "description": [pd.NA],
            "type": ["materialized view"],
            "id": ["table-uuid-1"],
        }
    )
    columns_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "column_name": ["id"],
            "ordinal_position": [1],
            "data_type": ["INTEGER"],
            "is_nullable": ["NO"],
            "description": [pd.NA],
            "id": ["col-uuid-1"],
        }
    )
    db_node = Neo4jNode(name="mydb", label="Database", props={"name": "mydb"})
    schema = Schema(
        db_node=db_node,
        schema_tables_df=tables_df,
        schema_columns_df=columns_df,
        is_creation_mode=True,
    )
    schema.create_schema_node("public")

    props = schema.get_table_node("orders").get_properties()
    assert props["type"] == TableTypes.MATERIALIZED_VIEW
