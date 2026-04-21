# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class Labels:
    SQL = "Sql"
    COLUMN = "Column"
    TABLE = "Table"
    SCHEMA = "Schema"
    DB = "Database"
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
    JOIN = "JOIN"
    UNION = "UNION"
    SQL = "SQL"


class Props:
    """Edge/node property keys (used by utils_dal, node)."""

    JOIN = "join"
    UNION = "union"
    SQL_ID = "sql_id"
