# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console

from nemo_retriever.ingest_config import load_ingest_config_section
from nemo_retriever.io.dataframe import read_dataframe, write_dataframe

from nemo_retriever.chart.config import load_chart_extractor_schema_from_dict
from nemo_retriever.chart.processor import extract_chart_data_from_primitives_df

app = typer.Typer(help="Chart Extraction Stage")
console = Console()


@app.command()
def run(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Input primitives as a DataFrame file (.parquet, .jsonl, or .json). Must include a 'metadata' column.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        file_okay=True,
        help="Output path (.parquet, .jsonl, or .json). Defaults to <input>.+chart<ext>.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help=(
            "Optional ingest YAML config file. If omitted, we auto-discover ./ingest-config.yaml then "
            "$HOME/.ingest-config.yaml. Uses section: chart."
        ),
    ),
) -> None:
    """
    Load a primitives DataFrame, run chart enrichment, and write the enriched DataFrame.
    """
    try:
        df = read_dataframe(input_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    cfg_dict: Dict[str, Any] = load_ingest_config_section(config, section="chart")
    schema = load_chart_extractor_schema_from_dict(cfg_dict)
    out_df, _info = extract_chart_data_from_primitives_df(df, extractor_config=schema, task_config={})

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".chart" + input_path.suffix)

    try:
        write_dataframe(out_df, output_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    console.print(f"[green]Done[/green] wrote={output_path} rows={len(out_df)}")


def main() -> None:
    app()
