"""Stamp table/column metadata onto the Neo4j graph.

This module reads ``<database_name>.json`` (sitting next to it — e.g.
``dor_prod.json`` for the ``dor_prod`` database) and writes descriptions and
sample values onto the ``Table`` and ``Column`` nodes that the tabular ingest
pipeline created in Neo4j. It is intentionally a small, dev-tools-only helper
and is meant to be invoked at the end of an ingest run.

JSON shape (per table)::

    {
        "<table_name>": {
            "description": "...",
            "columns": [
                {
                    "name": "...",
                    "description": "...",
                    "value_examples": ["...", ...] | null,
                    ...
                },
                ...
            ]
        },
        ...
    }

Custom analyses (optional, separate file ``<database_name>_custom_analyses.json``)::

    [
        {
            "name": "...",
            "description": "...",
            "sql": "SELECT ..."
        },
        ...
    ]
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.params import EmbedParams, VdbUploadParams

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parent


def apply_metadata(database_name: str) -> None:
    """Stamp table/column metadata onto the Neo4j graph.

    Reads ``<this dir>/<database_name>.json`` (keyed by table name) and
    updates the following properties for every table/column belonging to
    *database_name*:

    * ``Table.description``
    * ``Column.description``
    * ``Column.sample_values`` (from the JSON's ``value_examples`` field, when
      present and non-empty)

    Tables/columns that aren't present in the graph are silently skipped
    (the MATCH simply finds nothing). Properties for which the JSON has no
    value are left untouched (``coalesce`` preserves the existing value).
    """
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

    metadata_path = DEFAULT_DIR / f"{database_name}.json"

    if not metadata_path.exists():
        logger.warning("metadata file not found at %s; skipping", metadata_path)
        return

    with metadata_path.open() as f:
        raw = json.load(f)

    table_rows: list[dict[str, str]] = []
    column_rows: list[dict[str, str | list[str] | None]] = []
    samples_count = 0
    for table_name, table_meta in raw.items():
        table_desc = table_meta.get("description")
        if table_desc:
            table_rows.append({"table_name": table_name, "description": table_desc})

        for col in table_meta.get("columns", []) or []:
            col_desc = col.get("description")
            value_examples = col.get("value_examples")
            sample_values: list[str] | None = (
                [str(v) for v in value_examples] if isinstance(value_examples, list) and value_examples else None
            )
            if not col_desc and sample_values is None:
                continue
            if sample_values is not None:
                samples_count += 1
            column_rows.append(
                {
                    "table_name": table_name,
                    "column_name": col["name"],
                    "description": col_desc or None,
                    "sample_values": sample_values,
                }
            )

    conn = get_neo4j_conn()

    if table_rows:
        conn.query_write(
            query=(
                "UNWIND $rows AS row "
                "MATCH (d:Database {name: $db_name})-[:CONTAINS]->"
                "(:Schema)-[:CONTAINS]->(t:Table {name: row.table_name}) "
                "SET t.description = coalesce(row.description, t.description)"
            ),
            parameters={"rows": table_rows, "db_name": database_name},
        )

    if column_rows:
        conn.query_write(
            query=(
                "UNWIND $rows AS row "
                "MATCH (d:Database {name: $db_name})-[:CONTAINS]->"
                "(:Schema)-[:CONTAINS]->(t:Table {name: row.table_name})"
                "-[:CONTAINS]->(c:Column {name: row.column_name}) "
                "SET c.description = coalesce(row.description, c.description), "
                "    c.sample_values = coalesce(row.sample_values, c.sample_values)"
            ),
            parameters={"rows": column_rows, "db_name": database_name},
        )

    logger.info(
        "Applied metadata: %d table description(s), %d column description(s), " "%d column sample_values from %s",
        len(table_rows),
        sum(1 for r in column_rows if r.get("description")),
        samples_count,
        metadata_path,
    )


def add_custom_analyses(
    database_name: str,
    dialect: str,
    embed_params: "EmbedParams | None" = None,
    vdb_params: "VdbUploadParams | None" = None,
) -> None:
    """Ingest custom analyses for *database_name* into the Neo4j graph and the VDB.

    Reads ``<this dir>/<database_name>_custom_analyses.json`` — a list of
    ``{"name", "description", "sql"}`` entries — and, for each entry:

    * parses the SQL against the schemas already in the graph (via
      :func:`parse_query_single`), which produces a :class:`Sql` node and the
      corresponding ``Sql -> Table/Column`` edges;
    * creates a :class:`CustomAnalysis` node with ``name`` and ``description``;
    * connects ``CustomAnalysis -[:HAS_SQL]-> Sql``.

    When *embed_params* and *vdb_params* are provided, the function then
    embeds each newly-ingested analysis (name + description + SQL) and
    **appends** the rows to the configured LanceDB table — so they live
    alongside the rows the main embed pipeline writes for ``Table`` and
    ``Column`` nodes. The append mode means the main pipeline (which uses
    ``overwrite=True``) must run *before* this function.

    Entries with no SQL, or whose SQL doesn't resolve to any known table, are
    skipped with a warning. Must be called *after* schema ingestion so the
    parser can resolve table/column references.
    """
    from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
    from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
    from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels, Props
    from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single
    from nemo_retriever.tabular_data.retrieval.utils import get_all_schemas_ids, get_schemas_by_ids

    analyses_path = DEFAULT_DIR / f"{database_name}_custom_analyses.json"

    if not analyses_path.exists():
        logger.warning("custom analyses file not found at %s; skipping", analyses_path)
        return

    with analyses_path.open() as f:
        analyses = json.load(f)

    if not isinstance(analyses, list) or not analyses:
        logger.info("No custom analyses to ingest from %s.", analyses_path)
        return

    schemas_ids = get_all_schemas_ids()
    schemas = get_schemas_by_ids(schemas_ids)

    before = time.time()
    logger.info("Starting to ingest %d custom analyses from %s.", len(analyses), analyses_path)

    ingested = 0
    for entry in analyses:
        name = entry.get("name", "")
        sql = (entry.get("sql") or "").strip()
        if not sql:
            logger.warning("Skipping custom analysis %r — no SQL provided.", name)
            continue

        query_obj = parse_query_single(sql=sql, dialect=dialect, schemas=schemas)
        if query_obj is None:
            logger.warning(
                "Could not resolve any tables for custom analysis %r — skipping.",
                name,
            )
            continue

        # Match the Sql node by its full text so re-runs reuse the existing
        # node instead of creating a fresh one (which would cause duplicate
        # HAS_SQL edges from the merged CustomAnalysis node).
        query_obj.sql_node.match_props = {"sql_full_query": sql}

        # Match the CustomAnalysis node by name so re-running the script is
        # idempotent (Tables/Columns merge by id derived from their fully
        # qualified path; CustomAnalysis has no such id, so name is the
        # natural key from the JSON spec).
        analysis_node = Neo4jNode(
            name=name,
            label=Labels.CUSTOM_ANALYSIS,
            props={
                "name": name,
                "description": entry.get("description", ""),
            },
            match_props={"name": name},
        )

        edge_props = {Props.ANALYSIS_ID: analysis_node.get_id()}
        query_obj.edges.append((analysis_node, query_obj.sql_node, edge_props))

        add_query(query_obj.get_edges())
        ingested += 1

    logger.info(
        "Ingested %d/%d custom analyses in %.2fs.",
        ingested,
        len(analyses),
        time.time() - before,
    )

    if ingested == 0:
        return

    if embed_params is None or vdb_params is None:
        logger.info("Skipping custom-analysis embedding: embed_params/vdb_params not provided.")
        return

    _embed_custom_analyses(database_name, embed_params, vdb_params)


def _embed_custom_analyses(
    database_name: str,
    embed_params: "EmbedParams",
    vdb_params: "VdbUploadParams",
) -> None:
    """Fetch ``CustomAnalysis`` docs from Neo4j, embed them, and append to LanceDB.

    Filters to analyses whose SQL references at least one table belonging to
    *database_name* via the path
    ``CustomAnalysis -[:HAS_SQL]-> Sql -[:SQL]-> Table <-[:CONTAINS]- Schema <-[:CONTAINS]- Database``,
    shapes the result into the same 5-column DataFrame the main pipeline
    produces, then uses the same embedder
    (:func:`nemo_retriever.text_embed.runtime.embed_text_main_text_embed`) and
    writer (:class:`IngestVdbOperator`) as the main pipeline — but forces
    ``overwrite=False`` so existing Table/Column rows are preserved.
    """
    import pandas as pd

    from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
    from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
    from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
    from nemo_retriever.vdb import IngestVdbOperator

    query = f"""
        MATCH (ca:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
              -[:{Edges.SQL}]->(t:{Labels.TABLE})<-[:{Edges.CONTAINS}]-(s:{Labels.SCHEMA})
              <-[:{Edges.CONTAINS}]-(d:{Labels.DB}{{name: $database_name}})
        WITH DISTINCT ca, sql,
             CASE
                 WHEN ca.description IS NOT NULL AND trim(toString(ca.description)) <> ''
                 THEN ca.description
                 ELSE ''
             END AS desc,
             CASE
                 WHEN sql.sql_full_query IS NOT NULL
                 THEN ', sql: ' + sql.sql_full_query
                 ELSE ''
             END AS sql_text
        RETURN collect({{
            text: 'custom_analysis: ' + ca.name +
                  CASE WHEN desc <> '' THEN ', description: ' + desc ELSE '' END +
                  sql_text,
            name: ca.name,
            label: labels(ca)[0],
            id: ca.id
        }}) AS docs
    """
    result = get_neo4j_conn().query_read(query, parameters={"database_name": database_name})
    docs = result[0].get("docs") if result else None
    if not docs:
        logger.info("No CustomAnalysis rows found for %r; skipping VDB upsert.", database_name)
        return

    rows = []
    for item in docs:
        node_id = item.get("id")
        path = f"neo4j:{node_id}" if node_id is not None else "neo4j:unknown"
        tabular_fields = {
            "id": node_id,
            "label": item.get("label", ""),
            "name": item.get("name", ""),
            "source_path": path,
            "database_name": database_name,
        }
        rows.append(
            {
                "text": (item.get("text") or "").strip(),
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {**tabular_fields, "content_metadata": dict(tabular_fields)},
            }
        )
    df = pd.DataFrame(rows)

    before = time.time()
    embedded = embed_text_main_text_embed(
        df,
        model_name=embed_params.model_name,
        embed_invoke_url=embed_params.embed_invoke_url,
        api_key=embed_params.api_key,
        embed_modality=embed_params.embed_modality,
    )

    append_kwargs = {**vdb_params.vdb_kwargs, "mode": "overwrite"}
    IngestVdbOperator(vdb_op=vdb_params.vdb_op, vdb_kwargs=append_kwargs)(embedded.to_dict(orient="records"))
    logger.info(
        "Embedded and appended %d CustomAnalysis row(s) to LanceDB in %.2fs.",
        len(embedded),
        time.time() - before,
    )
