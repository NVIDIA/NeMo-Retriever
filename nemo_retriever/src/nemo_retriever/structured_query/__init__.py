# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured query module: SQL over local data files via DuckDB.

This module provides:
- ``DuckDBEngine``: lightweight wrapper around an in-process DuckDB connection.
- A Typer CLI entry-point (``stage.py``) accessible via ``retriever structured-query``.
"""

from nemo_retriever.structured_query.duckdb_engine import DuckDBEngine

__all__ = ["DuckDBEngine"]
