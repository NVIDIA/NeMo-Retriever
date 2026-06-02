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
    """Canonical ``table_type`` / Neo4j ``Table.type`` values (Postgres-style)."""

    VIEW = "view"
    MATERIALIZED_VIEW = "materialized view"
    BASE_TABLE = "base table"

    LIST_OF_ALL = [
        VIEW,
        MATERIALIZED_VIEW,
        BASE_TABLE,
    ]

    _ALIASES: dict[str, str] = {
        "view": VIEW,
        "materialized view": MATERIALIZED_VIEW,
        "materialized_view": MATERIALIZED_VIEW,
        "matview": MATERIALIZED_VIEW,
        "base table": BASE_TABLE,
        "base_table": BASE_TABLE,
        "table": BASE_TABLE,
        # Postgres pg_class.relkind
        "r": BASE_TABLE,
        "v": VIEW,
        "m": MATERIALIZED_VIEW,
    }

    @classmethod
    def normalize(cls, value: object) -> str | None:
        """Map connector / information_schema values to a canonical table type."""
        if value is None:
            return None
        try:
            import pandas as pd

            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        text = str(value).strip().lower()
        if text in cls.LIST_OF_ALL:
            return text
        return cls._ALIASES.get(text, cls.BASE_TABLE)

    @classmethod
    def sql_case(cls, column_expr: str = "table_type") -> str:
        """SQL ``CASE`` mapping ``information_schema``-style types to canonical values."""
        return f"""CASE lower(trim({column_expr}))
            WHEN '{cls.VIEW}' THEN '{cls.VIEW}'
            WHEN 'materialized view' THEN '{cls.MATERIALIZED_VIEW}'
            WHEN '{cls.MATERIALIZED_VIEW}' THEN '{cls.MATERIALIZED_VIEW}'
            WHEN '{cls.BASE_TABLE}' THEN '{cls.BASE_TABLE}'
            ELSE '{cls.BASE_TABLE}'
        END"""


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
