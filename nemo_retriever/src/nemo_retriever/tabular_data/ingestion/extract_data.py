# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.tabular_data.sql_database import SQLDatabase
from nemo_retriever.tabular_data.ingestion.utils import (
    normalize_tables,
    normalize_columns,
)


def create_dataframe(connector: SQLDatabase):
    """Extract raw schema DataFrames from any SQLDatabase connector."""
    tables = connector.get_tables()
    columns = connector.get_columns()
    views = connector.get_views()
    queries = connector.get_queries()
    pks = connector.get_pks()
    fks = connector.get_fks()
    return tables, columns, views, queries, pks, fks


def data_for_populate_tabular(connector: SQLDatabase):
    """Build the `data` dict expected by populate_tabular_data() from a SQLDatabase connector."""
    tables, columns, views, queries, pks, fks = create_dataframe(connector)
    tables = normalize_tables(tables)
    columns = normalize_columns(columns)
    data = {
        "database_name": connector.database_name,
        "tables": tables,
        "columns": columns,
        "views": views,
        "pks": pks,
        "fks": fks,
        "queries": queries,
    }
    # queries is not used by populate_tabular_data(); include if needed elsewhere
    return data


def extract_tabular_db_data(params=None):
    """Step 1 — Pull schema entities from the relational DB into a data dict.

    Args:
        params: TabularExtractParams instance. ``params.connector`` is used as
                the SQLDatabase connector. When omitted or when
                ``params.connector`` is ``None``, an empty data dict is returned.

    Returns:
        data dict with keys: tables, columns, views, pks, fks, queries.
    """
    if params is None or params.connector is None:
        return {}
    return data_for_populate_tabular(params.connector)


def store_relational_db_in_neo4j(data, dialect: str, num_workers: int = 4):
    """Step 2 — Write the extracted data dict as graph nodes into Neo4j.

    Args:
        data:       Data dict returned by extract_tabular_db_data().
        dialect:    SQL dialect used by the connector (e.g. "sqlite", "duckdb", "snowflake").
        num_workers: Worker count forwarded to populate_tabular_data.

    Returns:
        ``{schema_name_lower: Schema}`` dict produced during ingestion, so
        callers can recover the post-ingest ``tables_df`` / ``columns_df``
        (with the UUIDs assigned to each Table/Column node) without a
        round-trip back to Neo4j. Returns ``{}`` when *data* is empty.
    """
    if not data:
        return {}

    from nemo_retriever.tabular_data.ingestion.write_to_graph import (
        populate_tabular_data,
    )

    return populate_tabular_data(
        data,
        num_workers=num_workers,
        dialect=dialect,
    )
