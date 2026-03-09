# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer CLI entry-point for the structured-query stage.

Usage example::

    # Load Spider2 data into DuckDB (one-time setup)
    retriever structured-query load-spider2 ~/spider2/spider2-duckdb/data \\
        --database ./spider2.duckdb

    # Inspect tables
    retriever structured-query list-tables --database ./spider2.duckdb --schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from nemo_retriever.structured_query.duckdb_engine import DuckDBEngine

console = Console()
app = typer.Typer(
    help="Structured query stage: natural-language to SQL over local data files via DuckDB."
)



@app.command("load-spider2")
def load_spider2(
    spider2_data_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Root data directory of the Spider2 benchmark "
             "(e.g. ~/spider2/spider2-duckdb/data).",
    ),
    database: str = typer.Option(
        "spider2.duckdb",
        "--database",
        "-d",
        help="Path to the DuckDB database file to create or update (default: spider2.duckdb).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--skip-existing",
        help="Overwrite tables that already exist (default: skip existing).",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Scan subdirectories recursively (default: True).",
    ),
    summary_json: Optional[Path] = typer.Option(
        None,
        "--summary-json",
        help="Write a load summary to this JSON file.",
    ),
) -> None:
    """Phase 1: Load all Spider2 data files into a persistent DuckDB database.

    Discovers every CSV / Parquet / JSON file under SPIDER2_DATA_DIR and runs
    ``CREATE TABLE <stem> AS SELECT * FROM <file>`` for each one.  The data is
    stored inside DATABASE so no original files are needed afterwards.

    Example::

        retriever structured-query load-spider2 ~/spider2/spider2-duckdb/data \\
            --database ./spider2.duckdb

    After loading, list available tables::

        retriever structured-query list-tables --database ./spider2.duckdb
    """
    db_path = str(Path(database).expanduser().resolve())
    console.print(f"[bold]Opening DuckDB:[/bold] {db_path}")

    engine = DuckDBEngine(database=db_path)
    try:
        summary = engine.load_spider2(
            spider2_data_dir,
            recursive=recursive,
            overwrite=overwrite,
        )
    finally:
        engine.close()

    if summary_json:
        summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Summary written to[/green] {summary_json}")

    console.print(
        f"\n[green]Done.[/green] "
        f"files_found={summary['files_found']}  "
        f"loaded={summary['loaded']}  "
        f"skipped={summary['skipped']}  "
        f"failed={summary['failed']}"
    )

    if summary["tables"]:
        tbl = Table(title="Tables loaded into DuckDB", show_header=True, header_style="bold cyan")
        tbl.add_column("Table name")
        for t in summary["tables"]:
            tbl.add_row(t)
        console.print(tbl)

    if summary["failures"]:
        console.print("[red]Failures:[/red]")
        for f in summary["failures"]:
            console.print(f"  {f['file']} → {f['error']}")
        raise typer.Exit(code=1)


@app.command("list-tables")
def list_tables(
    database: str = typer.Option(
        "spider2.duckdb",
        "--database",
        "-d",
        help="Path to the DuckDB database file.",
    ),
    show_schema: bool = typer.Option(
        False,
        "--schema/--no-schema",
        help="Also show column names and types for each table.",
    ),
) -> None:
    """List all tables in a DuckDB database file.

    Example::

        retriever structured-query list-tables --database ./spider2.duckdb --schema
    """
    db_path = str(Path(database).expanduser().resolve())
    engine = DuckDBEngine(database=db_path, read_only=True)
    try:
        tables = engine.list_tables()
    finally:
        engine.close()

    if not tables:
        console.print("[yellow]No tables found in database.[/yellow]")
        return

    if show_schema:
        engine = DuckDBEngine(database=db_path, read_only=True)
        try:
            for t in tables:
                cols = engine.schema(t)
                tbl = Table(title=t, show_header=True, header_style="bold magenta")
                tbl.add_column("Column")
                tbl.add_column("Type")
                for c in cols:
                    tbl.add_row(c["column_name"], c["column_type"])
                console.print(tbl)
        finally:
            engine.close()
    else:
        tbl = Table(title=f"Tables in {database}", show_header=True, header_style="bold cyan")
        tbl.add_column("Table name")
        for t in tables:
            tbl.add_row(t)
        console.print(tbl)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
