# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a pipeline extraction Parquet into LanceDB (same path as graph_pipeline).

Use this to **resume** after ingestion succeeded and ``--save-intermediate`` wrote
``extraction.parquet``, but LanceDB write or index creation failed (e.g. disk full on
``/tmp``). Set ``TMPDIR`` to a large filesystem before running::

    export TMPDIR=/raid/$USER/tmp
    mkdir -p "$TMPDIR"
    python -m nemo_retriever.examples.parquet_to_lancedb path/to/extraction.parquet \\
        --lancedb-uri lancedb
"""

from __future__ import annotations

import logging
from pathlib import Path
import typer

from nemo_retriever.io.dataframe import read_extraction_parquet
from nemo_retriever.vector_store.lancedb_store import handle_lancedb

logger = logging.getLogger(__name__)
app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


@app.command()
def main(
    parquet_path: Path = typer.Argument(
        ...,
        help="Path to extraction.parquet (from graph_pipeline --save-intermediate).",
        path_type=Path,
        exists=True,
        dir_okay=False,
    ),
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri"),
    table_name: str = typer.Option(LANCEDB_TABLE, "--table-name"),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid"),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    uri = str(Path(lancedb_uri).expanduser().resolve())
    parquet_path = Path(parquet_path).expanduser().resolve()

    logger.info("Reading %s ...", parquet_path)
    df = read_extraction_parquet(parquet_path)
    records = df.to_dict("records")
    logger.info("Loaded %s rows; writing LanceDB at uri=%s table=%s ...", len(records), uri, table_name)

    handle_lancedb(records, uri, table_name, hybrid=hybrid, mode="overwrite")
    logger.info("Done.")


if __name__ == "__main__":
    app()
