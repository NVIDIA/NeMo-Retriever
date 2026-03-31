# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL database connectors.

``SQLDatabase`` is the abstract base class all connectors must implement.
``DuckDB`` is the bundled reference implementation.
"""

from nemo_retriever.tabular_data.connectors.sql_database import SQLDatabase
from nemo_retriever.tabular_data.connectors.duckdb import DuckDB

__all__ = ["SQLDatabase", "DuckDB"]
