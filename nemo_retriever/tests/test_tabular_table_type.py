# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for persisting table_type through tabular ingestion normalization."""

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.schema import Schema
from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes
from nemo_retriever.tabular_data.ingestion.utils import normalize_tables


def test_normalize_tables_keeps_table_type():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "table_type": ["base table"],
        }
    )
    result = normalize_tables(raw)

    assert "table_type" in result.columns
    assert result["table_type"].iloc[0] == "base table"
    assert str(result["table_type"].dtype) == "category"


def test_normalize_tables_adds_table_type_when_absent():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
        }
    )
    result = normalize_tables(raw)

    assert "table_type" in result.columns
    assert result["table_type"].iloc[0] == TableTypes.BASE_TABLE
    assert str(result["table_type"].dtype) == "category"


def test_normalize_tables_keeps_materialized_view():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["mv_orders"],
            "table_type": ["materialized view"],
        }
    )
    result = normalize_tables(raw)
    assert result["table_type"].iloc[0] == TableTypes.MATERIALIZED_VIEW


def test_normalize_tables_accepts_connector_table_type_column():
    """Connectors expose ``table_type`` from information_schema."""
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "table_type": ["base table"],
        }
    )
    result = normalize_tables(raw)

    assert result["table_type"].iloc[0] == TableTypes.BASE_TABLE


def test_normalize_tables_preserves_graph_columns():
    """Graph reload rows carry ``id`` / ``database`` alongside normalized fields."""
    raw = pd.DataFrame(
        {
            "database": ["mydb"],
            "table_schema": ["public"],
            "table_name": ["orders"],
            "id": ["table-uuid-1"],
            "created": ["2024-01-15T10:30:00Z"],
            "description": [pd.NA],
            "table_type": ["view"],
        }
    )
    result = normalize_tables(raw)

    assert result["id"].iloc[0] == "table-uuid-1"
    assert result["database"].iloc[0] == "mydb"
    assert result["table_type"].iloc[0] == TableTypes.VIEW
    assert result["created"].iloc[0] == pd.Timestamp("2024-01-15 10:30:00+00:00")


def test_reset_tables_props_sets_table_type():
    tables_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "created": [pd.NA],
            "description": [pd.NA],
            "table_type": ["base table"],
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
    assert props["table_type"] == "base table"
    assert props["name"] == "orders"


def test_reset_tables_props_defaults_table_type_when_absent():
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
    assert props["table_type"] == TableTypes.BASE_TABLE


def test_create_table_node_does_not_set_table_type():
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
    assert "table_type" not in props


def test_create_table_node_sets_table_type():
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
    schema.create_table_node("orders", id="table-uuid-1", table_type="VIEW")

    props = schema.get_table_node("orders").get_properties()
    assert props["table_type"] == "view"


def test_get_table_node_passes_table_type_from_dataframe():
    tables_df = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "created": [pd.NA],
            "description": [pd.NA],
            "table_type": ["materialized view"],
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
    assert props["table_type"] == TableTypes.MATERIALIZED_VIEW
