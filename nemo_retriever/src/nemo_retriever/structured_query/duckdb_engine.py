# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB engine wrapper for in-process SQL execution.

Wraps ``duckdb.connect()`` with helpers to register pandas DataFrames or
scan CSV/Parquet/JSON files directly from the filesystem.  No server or Docker
service is required — DuckDB runs fully in-process.

Example
-------
::

    from nemo_retriever.structured_query.duckdb_engine import DuckDBEngine

    engine = DuckDBEngine()
    engine.register_file("orders", "/data/orders.parquet")
    rows = engine.execute("SELECT COUNT(*) AS n FROM orders")
    # rows -> [{"n": 42}]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBEngine:
    """In-process DuckDB connection with convenience helpers.

    Parameters
    ----------
    database:
        Path to a persistent DuckDB database file, or ``None`` / ``":memory:"``
        for an ephemeral in-memory database (default: in-memory).
    read_only:
        Open the database in read-only mode (default: False).
    """

    def __init__(
        self,
        database: Optional[str] = None,
        read_only: bool = False,
    ) -> None:
        try:
            import duckdb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "DuckDB is required. Install it via: pip install 'duckdb>=1.2.0'"
            ) from exc

        db_path = database or ":memory:"
        self._conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.debug("DuckDB connected (database=%r, read_only=%s).", db_path, read_only)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Register a pandas DataFrame as a virtual DuckDB table.

        Parameters
        ----------
        table_name:
            Name to use in SQL queries.
        df:
            DataFrame to register.
        """
        self._conn.register(table_name, df)
        logger.debug("Registered DataFrame as table '%s' (%d rows).", table_name, len(df))

    def register_file(
        self,
        table_name: str,
        file_path: Union[str, Path],
    ) -> None:
        """Register a CSV, Parquet, or JSON file as a virtual DuckDB view.

        The file format is inferred from the file extension:
        - ``.csv`` → ``read_csv_auto``
        - ``.parquet`` or ``.pq`` → ``read_parquet``
        - ``.json`` or ``.ndjson`` → ``read_json_auto``

        Parameters
        ----------
        table_name:
            Name to use in SQL queries.
        file_path:
            Path to the data file.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        escaped = str(path).replace("'", "''")

        if ext == ".csv":
            reader = f"read_csv_auto('{escaped}')"
        elif ext in (".parquet", ".pq"):
            reader = f"read_parquet('{escaped}')"
        elif ext in (".json", ".ndjson", ".jsonl"):
            reader = f"read_json_auto('{escaped}')"
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                "Supported: .csv, .parquet, .pq, .json, .ndjson, .jsonl"
            )

        self._conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM {reader}")
        logger.debug("Registered file '%s' as view '%s'.", file_path, table_name)

    def register_directory(
        self,
        table_name: str,
        directory: Union[str, Path],
        pattern: str = "*.parquet",
    ) -> None:
        """Register all files matching *pattern* in *directory* as a single view.

        Parameters
        ----------
        table_name:
            Name to use in SQL queries.
        directory:
            Directory path to scan.
        pattern:
            Glob pattern for files to include (default: ``*.parquet``).
        """
        dir_path = Path(directory)
        escaped = str(dir_path / pattern).replace("'", "''")
        self._conn.execute(
            f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{escaped}')"
        )
        logger.debug(
            "Registered directory '%s' (pattern=%r) as view '%s'.",
            directory,
            pattern,
            table_name,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> List[Dict[str, Any]]:
        """Execute a SQL statement and return results as a list of row dicts.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters for parameterised queries.

        Returns
        -------
        list[dict]
            Each element is a row with column names as keys.
        """
        logger.debug("DuckDB executing: %s", sql[:200])
        if parameters:
            rel = self._conn.execute(sql, parameters)
        else:
            rel = self._conn.execute(sql)
        df = rel.df()
        return df.to_dict(orient="records")

    def execute_df(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        """Execute a SQL statement and return a pandas DataFrame.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters.
        """
        logger.debug("DuckDB executing (→ DataFrame): %s", sql[:200])
        if parameters:
            rel = self._conn.execute(sql, parameters)
        else:
            rel = self._conn.execute(sql)
        return rel.df()

    def list_tables(self) -> List[str]:
        """Return names of all tables and views registered in this connection."""
        rows = self.execute("SHOW TABLES")
        return [r.get("name", "") for r in rows]

    def schema(self, table_name: str) -> List[Dict[str, str]]:
        """Return column names and types for *table_name*.

        Returns
        -------
        list[dict]
            Each element has keys ``column_name`` and ``column_type``.
        """
        rows = self.execute(f"DESCRIBE {table_name}")
        return [{"column_name": r.get("column_name"), "column_type": r.get("column_type")} for r in rows]

    # ------------------------------------------------------------------
    # Spider2 bulk loaders
    # ------------------------------------------------------------------

    def _load_spider2_lite_json(self, schema: str, table: str, json_path: Path) -> None:
        """Load a single Spider2-lite JSON file into a DuckDB table.

        Spider2-lite JSON files have the shape::

            {"sample_rows": [{col: val, ...}, ...], "table_name": ..., "column_names": ..., ...}

        This method extracts ``sample_rows`` into a flat DataFrame and creates
        a persistent DuckDB table from it.  Falls back to reading the JSON
        directly if the file does not follow this structure.
        """
        import json as _json

        try:
            raw = _json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            raise ValueError(f"Cannot read {json_path}: {exc}") from exc

        # Spider2-lite wrapper format
        if isinstance(raw, dict) and "sample_rows" in raw:
            rows = raw["sample_rows"]
            if not rows:
                # Empty table — create it with the declared columns if available
                col_names = raw.get("column_names", [])
                col_types = raw.get("column_types", [])
                if col_names:
                    cols_ddl = ", ".join(
                        f'"{n}" VARCHAR' for n in col_names
                    )
                    self._conn.execute(
                        f"CREATE OR REPLACE TABLE {schema}.{table} ({cols_ddl})"
                    )
                return
            df = pd.DataFrame(rows)
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
        else:
            df = pd.DataFrame([raw])

        view = f"_tmp_{schema}_{table}"
        self._conn.register(view, df)
        self._conn.execute(
            f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM {view}"
        )
        self._conn.unregister(view)

    def load_spider2_lite(
        self,
        spider2_lite_dir: Union[str, Path],
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Load Spider2-lite SQLite databases into DuckDB using one schema per database.

        Spider2-lite stores each database as a folder of JSON files under:
          ``<spider2_lite_dir>/resource/databases/sqlite/<DbName>/<table>.json``

        Each database folder becomes a DuckDB schema named after the folder.
        Each JSON file becomes a table within that schema.

        After loading, query with::

            engine.execute("SELECT * FROM Airlines.flights LIMIT 5")

        Parameters
        ----------
        spider2_lite_dir:
            Root of the ``spider2-lite`` directory
            (e.g. ``~/spider2/spider2-lite``).
        overwrite:
            Drop and recreate schemas/tables that already exist
            (default: False — skip existing).

        Returns
        -------
        dict
            Summary with keys ``databases_found``, ``loaded``, ``skipped``,
            ``failed``, ``schemas``, and ``failures``.
        """
        import re

        root = Path(spider2_lite_dir).expanduser().resolve()
        sqlite_dir = root / "resource" / "databases" / "sqlite"

        if not sqlite_dir.is_dir():
            raise ValueError(
                f"spider2-lite sqlite directory not found: {sqlite_dir}\n"
                "Expected layout: <spider2_lite_dir>/resource/databases/sqlite/<DbName>/<table>.json"
            )

        def _sanitize(name: str) -> str:
            s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
            return ("s_" + s) if s and s[0].isdigit() else s or "unnamed"

        db_dirs = sorted(p for p in sqlite_dir.iterdir() if p.is_dir())

        loaded_schemas: List[str] = []
        skipped_schemas: List[str] = []
        failed: List[Dict[str, str]] = []

        existing_schemas = {
            r["schema_name"]
            for r in self.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('main', 'information_schema', 'pg_catalog')"
            )
        }

        for db_dir in db_dirs:
            schema = _sanitize(db_dir.name)

            if schema in existing_schemas and not overwrite:
                logger.debug("Skipping schema '%s' — already exists.", schema)
                skipped_schemas.append(schema)
                continue

            try:
                if overwrite and schema in existing_schemas:
                    self._conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                self._conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

                json_files = sorted(db_dir.glob("*.json"))
                for jf in json_files:
                    table = _sanitize(jf.stem)
                    self._load_spider2_lite_json(schema, table, jf)
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

    def list_schemas(self) -> List[str]:
        """Return all user-created schema names (excludes DuckDB system schemas)."""
        rows = self.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name NOT IN ('main', 'information_schema', 'pg_catalog') "
            "ORDER BY schema_name"
        )
        return [r["schema_name"] for r in rows]

    def schema_tables(self, schema: str) -> List[str]:
        """Return table names within a specific schema."""
        rows = self.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = ? ORDER BY table_name",
            parameters=[schema],
        )
        return [r["table_name"] for r in rows]

    def load_spider2(
        self,
        spider2_data_dir: Union[str, Path],
        *,
        recursive: bool = True,
        overwrite: bool = False,
        extensions: tuple = (".parquet", ".pq", ".csv", ".json", ".ndjson", ".jsonl"),
    ) -> Dict[str, Any]:
        """Discover and persistently load all Spider2 data files into DuckDB tables.

        Each file becomes its own ``CREATE TABLE`` so the data is stored inside
        the DuckDB database file and can be queried without the original files
        present.  Table names are derived from the file stem, sanitised to be
        valid SQL identifiers (spaces and hyphens replaced with underscores,
        leading digits prefixed with ``t_``).

        Parameters
        ----------
        spider2_data_dir:
            Root directory of the Spider2 data (e.g. ``~/spider2/spider2-duckdb/data``).
        recursive:
            Scan subdirectories recursively (default: True).
        overwrite:
            If True, drop and recreate tables that already exist.
            If False (default), skip files whose table already exists.
        extensions:
            Tuple of file extensions to include.

        Returns
        -------
        dict
            Summary with keys ``loaded``, ``skipped``, ``failed``, and
            ``tables`` (list of table names that were created/updated).
        """
        import re

        data_dir = Path(spider2_data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            raise ValueError(f"spider2_data_dir does not exist or is not a directory: {data_dir}")

        def _sanitize(stem: str) -> str:
            name = re.sub(r"[^0-9a-zA-Z_]", "_", stem)
            if name and name[0].isdigit():
                name = "t_" + name
            return name or "unnamed"

        def _reader_expr(path: Path) -> str:
            ext = path.suffix.lower()
            escaped = str(path).replace("'", "''")
            if ext in (".parquet", ".pq"):
                return f"read_parquet('{escaped}')"
            elif ext == ".csv":
                return f"read_csv_auto('{escaped}')"
            else:
                return f"read_json_auto('{escaped}')"

        pattern = "**/*" if recursive else "*"
        all_files = [
            p for p in data_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in extensions
        ]
        all_files = sorted(all_files)

        existing_tables = set(self.list_tables())

        loaded: List[str] = []
        skipped: List[str] = []
        failed: List[Dict[str, str]] = []

        for file_path in all_files:
            table_name = _sanitize(file_path.stem)

            if table_name in existing_tables and not overwrite:
                logger.debug("Skipping '%s' — table '%s' already exists.", file_path.name, table_name)
                skipped.append(table_name)
                continue

            try:
                reader = _reader_expr(file_path)
                if overwrite and table_name in existing_tables:
                    self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                self._conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM {reader}"
                )
                existing_tables.add(table_name)
                loaded.append(table_name)
                logger.info("Loaded '%s' → table '%s'.", file_path.name, table_name)
            except Exception as exc:
                logger.error("Failed loading '%s': %s", file_path, exc)
                failed.append({"file": str(file_path), "error": str(exc)})

        return {
            "data_dir": str(data_dir),
            "files_found": len(all_files),
            "loaded": len(loaded),
            "skipped": len(skipped),
            "failed": len(failed),
            "tables": loaded,
            "failures": failed,
        }

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
