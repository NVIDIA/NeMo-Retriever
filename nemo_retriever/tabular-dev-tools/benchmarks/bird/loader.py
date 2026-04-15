# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD benchmark bulk loader for DuckDB.

Provides ``load_bird``: loads BIRD SQLite databases into DuckDB,
one schema per database, with full data (not samples).

Each database in ``<bird_dir>/dev_databases/<db_name>/<db_name>.sqlite``
becomes a DuckDB schema, so you can query:

    conn.execute("SELECT * FROM california_schools.schools LIMIT 5")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import duckdb

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return ("s_" + s) if s and s[0].isdigit() else s or "unnamed"


def load_bird(
    db_path: str,
    bird_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Load BIRD SQLite databases into DuckDB using one schema per database.

    BIRD stores each database as a ``.sqlite`` file under:
      ``<bird_dir>/dev_databases/<db_name>/<db_name>.sqlite``

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file to create or update.
    bird_dir:
        Root of the BIRD data directory (e.g. ``~/bird/mini_dev_data``).
    overwrite:
        Drop and recreate schemas that already exist (default: False).
    """
    conn = duckdb.connect(database=str(db_path))

    # Ensure the sqlite scanner extension is available.
    conn.execute("INSTALL sqlite; LOAD sqlite;")

    root = Path(bird_dir).expanduser().resolve()
    dev_db_dir = root / "dev_databases"

    if not dev_db_dir.is_dir():
        raise ValueError(
            f"BIRD dev_databases directory not found: {dev_db_dir}\n"
            "Expected layout: <bird_dir>/dev_databases/<db_name>/<db_name>.sqlite"
        )

    sqlite_files = sorted(dev_db_dir.rglob("*.sqlite"))

    if not sqlite_files:
        raise ValueError(f"No .sqlite files found under {dev_db_dir}")

    loaded_schemas: List[str] = []
    skipped_schemas: List[str] = []
    failed: List[Dict[str, str]] = []

    existing_schemas = set(
        conn.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name NOT IN ('main', 'information_schema', 'pg_catalog')"
        )
        .df()["schema_name"]
        .tolist()
    )

    for sqlite_path in sqlite_files:
        schema = _sanitize(sqlite_path.stem)

        if schema in existing_schemas and not overwrite:
            logger.debug("Skipping schema '%s' — already exists.", schema)
            skipped_schemas.append(schema)
            continue

        try:
            if overwrite and schema in existing_schemas:
                conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

            # Attach the SQLite file, copy every table, then detach.
            conn.execute(f"ATTACH '{sqlite_path}' AS _bird_src (TYPE sqlite, READ_ONLY)")

            # SHOW ALL TABLES returns (database, schema, name, ...) for attached DBs.
            tables = [row[2] for row in conn.execute("SHOW ALL TABLES").fetchall() if row[0] == "_bird_src"]

            for table in tables:
                sanitized_table = _sanitize(table)
                conn.execute(
                    f'CREATE OR REPLACE TABLE "{schema}"."{sanitized_table}" '
                    f'AS SELECT * FROM _bird_src.main."{table}"'
                )
                logger.debug("Loaded %s → %s.%s", table, schema, sanitized_table)

            conn.execute("DETACH _bird_src")

            loaded_schemas.append(schema)
            existing_schemas.add(schema)
            logger.info("Schema '%s' loaded (%d tables).", schema, len(tables))

        except Exception as exc:
            try:
                conn.execute("DETACH _bird_src")
            except Exception:
                pass
            logger.error("Failed loading schema '%s': %s", schema, exc)
            failed.append({"database": sqlite_path.stem, "schema": schema, "error": str(exc)})

    conn.close()

    return {
        "dev_db_dir": str(dev_db_dir),
        "databases_found": len(sqlite_files),
        "loaded": len(loaded_schemas),
        "skipped": len(skipped_schemas),
        "failed": len(failed),
        "schemas": loaded_schemas,
        "failures": failed,
    }
