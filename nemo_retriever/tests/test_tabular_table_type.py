"""Tests for persisting table_type through tabular ingestion normalization."""

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.schema import Schema
from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes
from nemo_retriever.tabular_data.ingestion.utils import (
    _table_type_node_props,
    normalize_tables,
)


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
    assert str(result["table_type"].dtype) == "string"


def test_normalize_tables_omits_table_type_when_absent():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
        }
    )
    result = normalize_tables(raw)

    assert "table_type" not in result.columns


def test_table_types_normalize_materialized_view():
    assert TableTypes.normalize("MATERIALIZED VIEW") == TableTypes.MATERIALIZED_VIEW
    assert TableTypes.normalize("m") == TableTypes.MATERIALIZED_VIEW
    assert TableTypes.normalize("unknown_kind") == TableTypes.BASE_TABLE


def test_normalize_tables_maps_materialized_view():
    raw = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["mv_orders"],
            "table_type": ["materialized view"],
        }
    )
    result = normalize_tables(raw)
    assert result["table_type"].iloc[0] == TableTypes.MATERIALIZED_VIEW


def test_table_type_node_props():
    row_with_type = pd.Series({"table_type": "view"})
    assert _table_type_node_props(row_with_type) == {"type": "view"}

    row_without_type = pd.Series({"table_name": "t"})
    assert _table_type_node_props(row_without_type) == {}

    row_na = pd.Series({"table_type": pd.NA})
    assert _table_type_node_props(row_na) == {}


def test_reset_tables_props_sets_type():
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
    assert props["type"] == "base table"
    assert props["name"] == "orders"
