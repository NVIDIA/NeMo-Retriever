# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data module: SQL over local data files via DuckDB."""

from nemo_retriever.relational_db.db_setup.duckdb_engine import DuckDBEngine

__all__ = ["DuckDBEngine"]
