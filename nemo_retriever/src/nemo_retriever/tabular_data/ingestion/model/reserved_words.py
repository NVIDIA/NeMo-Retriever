# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class Labels:
    SQL = "Sql"
    COLUMN = "Column"
    TABLE = "Table"
    SCHEMA = "Schema"
    DB = "Database"
    ALIAS = "Alias"
    SET_OP_COLUMN = "SetOpColumn"
    OPERATOR = "Operator"
    COMMAND = "Joins"
    FUNCTION = "Function"
    LIST_OF_ALL = [
        DB,
        SCHEMA,
        TABLE,
        COLUMN,
        SQL,
    ]


class Views:
    VIEW = "view"
    NON_BINDING_VIEW = "non_binding_view"
    MATERIALIZED_VIEW = "materialized_view"


class Edges:
    CONTAINS = "CONTAINS"
    FOREIGN_KEY = "FOREIGN_KEY"
    JOIN = "JOIN"
    UNION = "UNION"
    SQL = "SQL"


class Props:
    """Edge/node property keys (used by utils_dal, node)."""

    JOIN = "join"
    UNION = "union"
    SQL_ID = "sql_id"


class SQLType:
    """SQL statement type identifiers (lowercase, matching sqloxide top-level keys)."""

    QUERY = "query"
    SEMANTIC = "semantic"
    INSERT = "insert"
    CREATE_TABLE = "createtable"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"
    VIEW = "view"


class SQL:
    """SQL clause section names used to bucket edges on a Query object."""

    SELECT = "Select"
    FROM = "From"
    WHERE = "Where"
    ORDER_BY = "OrderBy"
    LIMIT = "Limit"
    TOP = "Top"
    DISTINCT = "Distinct"
    GROUP_BY = "GroupBy"
    OVER = "Over"
    WITH = "With"


class Parser:
    SUBSELECT = "Subselect"


class RelTypes(Edges):
    """Alias for Edges – kept for backward compatibility."""


data_relationships = [Edges.CONTAINS]
