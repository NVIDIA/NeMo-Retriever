"""Unit tests for the tabular ingestion pipeline functions.

Covers the standalone functions used by the tabular ingestion operators:
extract from DB, normalize, store in Neo4j.

All external databases (DuckDB, Neo4j, etc.) are replaced with lightweight
in-process stubs so the tests run without any infrastructure.
"""

import pandas as pd

from nemo_retriever.params import TabularExtractParams
from nemo_retriever.tabular_data.sql_database import SQLDatabase
from nemo_retriever.tabular_data.ingestion.extract_data import (
    data_for_populate_tabular,
    extract_tabular_db_data,
    store_relational_db_in_neo4j,
)


# ── Fake DB row data matching the column names DuckDB connector returns ────────

_FAKE_TABLES = pd.DataFrame(
    {
        "table_schema": ["public"],
        "table_name": ["orders"],
    }
)

_FAKE_COLUMNS = pd.DataFrame(
    {
        "table_schema": ["public"],
        "table_name": ["orders"],
        "column_name": ["id"],
        "ordinal_position": [1],
        "data_type": ["INTEGER"],
        "is_nullable": ["NO"],
    }
)

_FAKE_VIEWS = pd.DataFrame(
    {
        "table_schema": ["public"],
        "table_name": ["v_orders"],
        "view_definition": ["SELECT * FROM orders"],
    }
)

# DuckDB connector returns these column names for PKs (database / table_schema, not *_name variants)
_FAKE_PKS = pd.DataFrame(
    {
        "table_schema": ["public"],
        "table_name": ["orders"],
        "column_name": ["id"],
        "ordinal_position": [1],
    }
)

# No FKs in this schema — empty DataFrame matching connector column set
_FAKE_FKS = pd.DataFrame(
    columns=[
        "table_schema",
        "table_name",
        "column_name",
        "referenced_schema",
        "referenced_table",
        "referenced_column",
    ]
)


class _DummyDuckDB(SQLDatabase):
    """Drop-in replacement for DuckDB that returns pre-canned DataFrames."""

    def __init__(self, connection_string: str) -> None:
        pass

    @property
    def dialect(self) -> str:
        return "duckdb"

    @property
    def database_name(self) -> str:
        return "mydb"

    def execute(self, sql: str, parameters=None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_tables(self) -> pd.DataFrame:
        return _FAKE_TABLES.copy()

    def get_columns(self) -> pd.DataFrame:
        return _FAKE_COLUMNS.copy()

    def get_views(self) -> pd.DataFrame:
        return _FAKE_VIEWS.copy()

    def get_pks(self) -> pd.DataFrame:
        return _FAKE_PKS.copy()

    def get_fks(self) -> pd.DataFrame:
        return _FAKE_FKS.copy()

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        return pd.DataFrame(columns=["end_time", "query_text"])

    def close(self) -> None:
        pass


# ── Tests ──────────────────────────────────────────────────────────────────────

EXPECTED_DATA_KEYS = {"database_name", "tables", "columns", "views", "pks", "fks", "queries"}
EXPECTED_DATAFRAME_KEYS = EXPECTED_DATA_KEYS - {"database_name"}


def test_pull_tabular_db_entities():
    """extract_tabular_db_data returns the expected schema dict, content, and empty dict when connector is None."""
    # ── with explicit connector in params ─────────────────────────────────────
    connector = _DummyDuckDB("dummy.duckdb")
    data = extract_tabular_db_data(params=TabularExtractParams(connector=connector))

    assert isinstance(data, dict)
    assert set(data.keys()) == EXPECTED_DATA_KEYS
    assert data["database_name"] == "mydb"
    for key in EXPECTED_DATAFRAME_KEYS:
        assert isinstance(data[key], pd.DataFrame), f"data['{key}'] must be a DataFrame"

    assert data["tables"]["table_name"].iloc[0] == "orders"
    assert data["columns"]["column_name"].iloc[0] == "id"
    assert data["views"]["table_name"].iloc[0] == "v_orders"
    assert data["fks"].empty
    assert data["queries"].empty

    # ── params=None returns an empty dict ─────────────────────────────────────
    data_default = extract_tabular_db_data(params=None)
    assert data_default == {}


# ── data_for_populate_tabular ──────────────────────────────────────────────────


def test_data_for_populate_tabular(monkeypatch):
    """data_for_populate_tabular returns all required keys as DataFrames and applies normalization."""
    raw_tables = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "owner": ["dba"],  # normalize_tables should drop this
        }
    )
    raw_columns = pd.DataFrame(
        {
            "table_schema": ["public"],
            "table_name": ["orders"],
            "column_name": ["id"],
            "ordinal_position": ["1"],  # string; normalize_columns should coerce to Int16
            "data_type": ["INTEGER"],
            "is_nullable": ["NO"],
        }
    )
    raw_views = pd.DataFrame(
        {"database": ["mydb"], "table_schema": ["public"], "table_name": ["v_orders"], "view_definition": ["SELECT 1"]}
    )
    raw_pks = pd.DataFrame(columns=["table_schema", "table_name", "column_name"])
    raw_fks = pd.DataFrame(columns=["table_schema", "table_name", "column_name"])
    raw_queries = pd.DataFrame(columns=["end_time", "query_text"])

    monkeypatch.setattr(
        "nemo_retriever.tabular_data.ingestion.extract_data.create_dataframe",
        lambda connector: (raw_tables, raw_columns, raw_views, raw_queries, raw_pks, raw_fks),
    )

    data = data_for_populate_tabular(_DummyDuckDB("dummy.duckdb"))

    # required keys; database_name is a str, the rest are DataFrames
    assert set(data.keys()) == EXPECTED_DATA_KEYS
    assert data["database_name"] == "mydb"
    for key in EXPECTED_DATAFRAME_KEYS:
        assert isinstance(data[key], pd.DataFrame)

    # normalize_tables: owner dropped, dtypes applied
    assert "owner" not in data["tables"].columns
    assert str(data["tables"]["table_name"].dtype) == "string"

    # normalize_columns: string "1" coerced to Int16
    assert str(data["columns"]["ordinal_position"].dtype) == "Int16"
    assert data["columns"]["ordinal_position"].iloc[0] == 1

    # views and queries are raw pass-through; pks/fks are empty
    assert len(data["views"]) == 1
    assert data["pks"].empty
    assert data["fks"].empty
    assert data["queries"].equals(raw_queries)


# ── store_relational_db_in_neo4j ───────────────────────────────────────────────


def test_store_relational_db_in_neo4j_delegates_to_populate(monkeypatch):
    """store_relational_db_in_neo4j calls populate_tabular_data with the exact args."""
    calls: list[dict] = []

    # Patch get_neo4j_conn *before* write_to_graph is imported so that the
    # module-level `conn = get_neo4j_conn()` calls in schemas_dal and db_dal
    # do not require NEO4J_URI to be set in the environment.
    monkeypatch.setattr(
        "nemo_retriever.tabular_data.neo4j.get_neo4j_conn",
        lambda: None,
    )
    monkeypatch.setattr(
        "nemo_retriever.tabular_data.ingestion.write_to_graph.populate_tabular_data",
        lambda data, num_workers, dialect: calls.append({"data": data, "num_workers": num_workers, "dialect": dialect}),
    )

    dummy_data = {
        "database_name": "mydb",
        **{k: pd.DataFrame() for k in ("tables", "columns", "views", "pks", "fks", "queries")},
    }
    store_relational_db_in_neo4j(data=dummy_data, dialect="duckdb")

    assert len(calls) == 1
    assert calls[0]["data"] is dummy_data
    assert calls[0]["num_workers"] == 4
    assert calls[0]["dialect"] == "duckdb"
