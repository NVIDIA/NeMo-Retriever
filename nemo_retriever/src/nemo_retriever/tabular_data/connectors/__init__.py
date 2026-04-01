# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL database connectors.

``SQLDatabase`` is the abstract base class all connectors must implement.
``DuckDB`` is a reference implementation available in ``tabular-dev-tools/duckdb.py``.
"""

from nemo_retriever.tabular_data.connectors.sql_database import SQLDatabase

__all__ = ["SQLDatabase"]
