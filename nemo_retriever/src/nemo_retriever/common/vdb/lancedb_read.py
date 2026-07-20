# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only LanceDB row-count helper."""

from __future__ import annotations


def lancedb_row_count(uri: str, table_name: str) -> int:
    """Return ``table.count_rows()`` or 0 on failure."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    return int(table.count_rows())
