# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class Labels:
    SQL = "Sql"
    COLUMN = "Column"
    TABLE = "Table"
    SCHEMA = "Schema"
    DB = "Db"

    LIST_OF_ALL = [
        DB,
        SCHEMA,
        TABLE,
        COLUMN,
        SQL,
    ]


class Edges:
    CONTAINS = "CONTAINS"
    FOREIGN_KEY = "FOREIGN_KEY"


class Props:
    """Edge/node property keys (used by utils_dal, node)."""

    JOIN = "join"
    SOURCE_SQL_ID = "source_sql_id"
    UNION = "union"
    SQL_ID = "sql_id"


# Labels that have no parent owner in the graph (used by get_entity_before_update).
entities_without_owners = []


class RelTypes(Edges):
    """Alias for Edges – kept for backward compatibility."""


# Relationship types for owner traversal (used by get_node_parent_owner_by_id).
data_relationships = [Edges.CONTAINS]
