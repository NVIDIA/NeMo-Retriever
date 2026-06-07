# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class Labels:
    DB = "Database"
    SCHEMA = "Schema"
    TABLE = "Table"
    COLUMN = "Column"
    SQL = "Sql"
    CUSTOM_ANALYSIS = "CustomAnalysis"

    LIST_OF_ALL = [
        DB,
        CUSTOM_ANALYSIS,
        SCHEMA,
        TABLE,
        COLUMN,
        SQL,
    ]


class TableTypes:
    """Canonical ``table_type`` on Neo4j ``Table`` nodes (Postgres ``information_schema`` style)."""

    VIEW = "view"
    MATERIALIZED_VIEW = "materialized view"
    BASE_TABLE = "base table"

    LIST_OF_ALL = [
        VIEW,
        MATERIALIZED_VIEW,
        BASE_TABLE,
    ]


class Edges:
    CONTAINS = "CONTAINS"
    FOREIGN_KEY = "FOREIGN_KEY"
    JOIN = "JOIN"
    UNION = "UNION"
    SQL = "SQL"
    HAS_SQL = "HAS_SQL"


class Props:
    """Edge/node property keys (used by utils_dal, node)."""

    JOIN = "join"
    UNION = "union"
    SQL_ID = "sql_id"
    ANALYSIS_ID = "analysis_id"
