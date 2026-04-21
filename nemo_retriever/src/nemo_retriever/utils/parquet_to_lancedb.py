# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reload extraction Parquet into LanceDB (recovery helper).

Use this after ingestion succeeded and ``--save-intermediate`` wrote
``extraction.parquet``, but LanceDB write or index creation failed
(e.g. disk full on ``/tmp``).  Set ``TMPDIR`` to a large filesystem
before calling::

    export TMPDIR=/raid/$USER/tmp
    mkdir -p "$TMPDIR"
"""

from __future__ import annotations

import logging
from pathlib import Path

from nemo_retriever.io.dataframe import read_extraction_parquet
from nemo_retriever.vector_store.lancedb_store import handle_lancedb

logger = logging.getLogger(__name__)

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


def reload_parquet_to_lancedb(
    parquet_path: str | Path,
    lancedb_uri: str = LANCEDB_URI,
    table_name: str = LANCEDB_TABLE,
    hybrid: bool = False,
) -> int:
    """Read an extraction Parquet and write its records into LanceDB.

    Parameters
    ----------
    parquet_path : str or Path
        Path to ``extraction.parquet`` (from ``graph_pipeline --save-intermediate``).
    lancedb_uri : str
        LanceDB directory path.
    table_name : str
        Target table name inside LanceDB.
    hybrid : bool
        Enable hybrid (dense + sparse) indexing.

    Returns
    -------
    int
        Number of rows written.
    """
    uri = str(Path(lancedb_uri).expanduser().resolve())
    parquet_path = Path(parquet_path).expanduser().resolve()

    logger.info("Reading %s ...", parquet_path)
    df = read_extraction_parquet(parquet_path)
    records = df.to_dict("records")
    logger.info(
        "Loaded %s rows; writing LanceDB at uri=%s table=%s ...",
        len(records),
        uri,
        table_name,
    )

    handle_lancedb(records, uri, table_name, hybrid=hybrid, mode="overwrite")
    logger.info("Done.")
    return len(records)
