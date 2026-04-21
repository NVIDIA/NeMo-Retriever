# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spider2 bulk loaders for DuckDB.

Provides two functions:
- ``load_spider2_lite``: loads Spider2-lite JSON databases (one schema per DB).
- ``load_spider2``: loads flat Spider2 data files (Parquet / CSV / JSON).
"""

from __future__ import annotations

import json as _json
import logging
import re
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
import duckdb

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return ("s_" + s) if s and s[0].isdigit() else s or "unnamed"


def _load_spider2_lite_json(conn, schema: str, table: str, json_path: Path) -> None:
    """Load a single Spider2-lite JSON file into a DuckDB table."""
    try:
        raw = _json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        raise ValueError(f"Cannot read {json_path}: {exc}") from exc

    if isinstance(raw, dict) and "sample_rows" in raw:
        rows = raw["sample_rows"]
        if not rows:
            col_names = raw.get("column_names", [])
            if col_names:
                cols_ddl = ", ".join(f'"{n}" VARCHAR' for n in col_names)
                conn.execute(f"CREATE OR REPLACE TABLE {schema}.{table} ({cols_ddl})")
            return
        df = pd.DataFrame(rows)
    elif isinstance(raw, list):
        df = pd.DataFrame(raw)
    else:
        df = pd.DataFrame([raw])

    view = f"_tmp_{schema}_{table}"
    conn.register(view, df)
    conn.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM {view}")
    conn.unregister(view)


def load_spider2_lite(
    db_path: str,
    spider2_lite_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Load Spider2-lite databases into DuckDB using one schema per database.

    Spider2-lite stores each database as a folder of JSON files under:
      ``<spider2_lite_dir>/resource/databases/sqlite/<DbName>/<table>.json``

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file to create or update.
    spider2_lite_dir:
        Root of the ``spider2-lite`` directory (e.g. ``~/spider2/spider2-lite``).
    overwrite:
        Drop and recreate schemas that already exist (default: False).
    """
    conn = duckdb.connect(database=db_path)
    root = Path(spider2_lite_dir).expanduser().resolve()
    sqlite_dir = root / "resource" / "databases" / "sqlite"

    if not sqlite_dir.is_dir():
        raise ValueError(
            f"spider2-lite sqlite directory not found: {sqlite_dir}\n"
            "Expected layout: <spider2_lite_dir>/resource/databases/sqlite/<DbName>/<table>.json"
        )

    db_dirs = sorted(p for p in sqlite_dir.iterdir() if p.is_dir())

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

    for db_dir in db_dirs:
        schema = _sanitize(db_dir.name)

        if schema in existing_schemas and not overwrite:
            logger.debug("Skipping schema '%s' — already exists.", schema)
            skipped_schemas.append(schema)
            continue

        try:
            if overwrite and schema in existing_schemas:
                conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

            json_files = sorted(db_dir.glob("*.json"))
            for jf in json_files:
                table = _sanitize(jf.stem)
                _load_spider2_lite_json(conn, schema, table, jf)
                logger.debug("Loaded %s → %s.%s", jf.name, schema, table)

            loaded_schemas.append(schema)
            existing_schemas.add(schema)
            logger.info("Schema '%s' loaded (%d tables).", schema, len(json_files))
        except Exception as exc:
            logger.error("Failed loading schema '%s': %s", schema, exc)
            failed.append({"database": db_dir.name, "schema": schema, "error": str(exc)})

    return {
        "sqlite_dir": str(sqlite_dir),
        "databases_found": len(db_dirs),
        "loaded": len(loaded_schemas),
        "skipped": len(skipped_schemas),
        "failed": len(failed),
        "schemas": loaded_schemas,
        "failures": failed,
    }
