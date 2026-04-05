# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class Labels:
    SQL = "Sql"
    COLUMN = "Column"
    TABLE = "Table"
    SCHEMA = "Schema"
    DB = "Db"
    ALIAS = "Alias"
    SET_OP_COLUMN = "SetOpColumn"
    OPERATOR = "Operator"

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


class Props:
    """Edge/node property keys (used by utils_dal, node)."""

    JOIN = "join"
    SOURCE_SQL_ID = "source_sql_id"
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


# Labels that have no parent owner in the graph (used by get_entity_before_update).
entities_without_owners = []


class RelTypes(Edges):
    """Alias for Edges – kept for backward compatibility."""


# Relationship types for owner traversal (used by get_node_parent_owner_by_id).
data_relationships = [Edges.CONTAINS]
