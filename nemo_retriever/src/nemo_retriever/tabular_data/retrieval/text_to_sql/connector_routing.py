# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connector-routing helpers for the text-to-SQL agents.

Given a list of injected ``SQLDatabase`` connectors and the
``relevant_tables`` carried on ``path_state``, these helpers pick the
connector that owns the tables the SQL was generated against. Used for
prompt-dialect resolution, SQL execution, and (in future) federated-query
detection.
"""

from __future__ import annotations

from typing import Iterable

from nemo_retriever.tabular_data.sql_database import SQLDatabase


def resolve_connector_from_tables(
    tables: Iterable[dict],
    connectors: list[SQLDatabase],
) -> SQLDatabase | None:
    """Return the first connector whose ``database_name`` matches a relevant table's ``database_name``.

    Falls back to ``connectors[0]`` when no table provides a usable
    ``database_name`` or when no connector matches.
    """
    if not connectors:
        return None

    db_to_connector: dict[str, SQLDatabase] = {
        str(getattr(c, "database_name", "")): c for c in connectors if getattr(c, "database_name", None)
    }

    for table in tables or []:
        if not isinstance(table, dict):
            continue
        connector = db_to_connector.get(table.get("database_name"))
        if connector is not None:
            return connector

    return connectors[0]


__all__ = ["resolve_connector_from_tables"]
