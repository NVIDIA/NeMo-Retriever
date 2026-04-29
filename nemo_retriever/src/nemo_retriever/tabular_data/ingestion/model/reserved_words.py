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


class Edges:
    CONTAINS = "CONTAINS"
    FOREIGN_KEY = "FOREIGN_KEY"
    JOIN = "JOIN"
    UNION = "UNION"
    SQL = "SQL"
    ANALYSIS_OF = "ANALYSIS_OF"


class Props:
    """Edge/node property keys (used by utils_dal, node)."""

    JOIN = "join"
    UNION = "union"
    SQL_ID = "sql_id"
