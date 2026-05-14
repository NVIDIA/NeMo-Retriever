# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the embedding-ready DataFrame for tabular entities stored in Neo4j.

Two modes:

* **Full** — ``node_ids=None`` returns every ``Table``/``Column`` under
  ``database_name``. Used by the one-shot ingest pipeline that overwrites
  the LanceDB table.
* **Targeted** — ``node_ids=[...]`` returns only the listed
  ``Table``/``Column`` rows. The PATCH-driven incremental flow uses this:
  the API handler already knows exactly which nodes changed, so we
  re-embed only those instead of marking + scanning a Neo4j-side dirty
  flag.
"""

from typing import Iterable, List

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn


def _node_id_filter(
    alias: str, node_ids: Iterable[str] | None
) -> tuple[str, list[str]]:
    """Return ``(cypher_clause, params_list)`` filtering ``alias.id`` by ids.

    Empty / ``None`` input yields ``("", [])`` so callers can splice the
    fragment unconditionally. Stringification mirrors what
    :func:`server.datasources.vector_sync.sync_node_vectors` passes through
    so UUIDs and other key types both behave.
    """
    if not node_ids:
        return "", []
    cleaned = [str(n) for n in node_ids if n is not None and str(n)]
    if not cleaned:
        return "", []
    return f"WHERE {alias}.id IN $node_ids", cleaned


def query_neo4j_tables_for_embedding(
    database_name: str,
    *,
    node_ids: Iterable[str] | None = None,
) -> List[dict]:
    """Return one doc per ``Table`` node under ``database_name``.

    When ``node_ids`` is provided, only tables whose ``id`` is in the list
    are returned. ``None`` (the default) returns every table.
    """
    neo4j_conn = get_neo4j_conn()
    id_filter, ids_param = _node_id_filter("t", node_ids)
    query = f"""MATCH (d:{Labels.DB}{{name: $database_name}})-[:{Edges.CONTAINS}]->
      (s:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
               {id_filter}
               MATCH (t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
               WITH d, s, t, collect(
                 "{{name: " + c.name + ", data_type: " + c.data_type +
                 CASE WHEN c.description IS NOT NULL AND trim(c.description) <> ''
                   THEN ", description: " + c.description ELSE "" END +
                 "}}") as columns
               RETURN collect({{
                 text: "schema_name: " + s.name +
                   ", table_name: " + t.name +
                   CASE WHEN t.description IS NOT NULL AND trim(t.description) <> ''
                     THEN ", table_description: " + t.description ELSE "" END +
                   ", columns: " + apoc.text.join(columns, ' '),
                 name: t.name, label: labels(t)[0], id: t.id
               }}) as docs
            """
    params: dict = {"database_name": database_name}
    if ids_param:
        params["node_ids"] = ids_param
    result = neo4j_conn.query_read(query, parameters=params)
    if not result:
        return []
    return result[0].get("docs") or []


def query_neo4j_columns_for_embedding(
    database_name: str,
    *,
    node_ids: Iterable[str] | None = None,
) -> List[dict]:
    """Return one doc per ``Column`` node under ``database_name``.

    When ``node_ids`` is provided, only columns whose ``id`` is in the list
    are returned.
    """
    neo4j_conn = get_neo4j_conn()
    id_filter, ids_param = _node_id_filter("c", node_ids)
    query = f"""
        MATCH (d:{Labels.DB}{{name: $database_name}})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
              -[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        {id_filter}

        WITH d, s, t, c,
             CASE
                 WHEN c.description IS NOT NULL AND trim(toString(c.description)) <> ''
                 THEN ', column_description: ' + toString(c.description)
                 ELSE ''
             END AS column_desc,
             CASE
                 WHEN c.sample_values IS NOT NULL AND size(c.sample_values) > 0
                 THEN ', sample_values: ' + apoc.text.join([x IN c.sample_values[..5] | toString(x)], ', ')
                 ELSE ''
             END AS sample_vals

        RETURN collect({{
            text:  'db_name: ' + d.name +
                   ', schema_name: ' + s.name +
                   ', table_name: ' + t.name +
                   ', column_name: ' + c.name +
                   ', data_type: ' + coalesce(toString(c.data_type), '') +
                   column_desc +
                   sample_vals,
            name: c.name,
            label: labels(c)[0],
            id: c.id
        }}) AS docs
    """
    params: dict = {"database_name": database_name}
    if ids_param:
        params["node_ids"] = ids_param
    result = neo4j_conn.query_read(query, parameters=params)
    if not result:
        return []
    return result[0].get("docs") or []


def fetch_tabular_embedding_dataframe(
    database_name: str,
    *,
    node_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Fetch tabular entity docs from Neo4j as a DataFrame ready for embedding.

    Each row has: text, _embed_modality, path, page_number, metadata
    (id, label, name, source_path) — matching the format produced by the
    unstructured pipeline so run_pipeline_tasks_on_df works without changes.

    When ``node_ids`` is provided, only those table/column ids are
    returned. The default (``None``) returns every table and column under
    ``database_name`` — the full-reindex case.
    """
    _empty = pd.DataFrame(
        columns=["text", "_embed_modality", "path", "page_number", "metadata"]
    )
    table_docs = query_neo4j_tables_for_embedding(
        database_name=database_name, node_ids=node_ids
    )
    column_docs = query_neo4j_columns_for_embedding(
        database_name=database_name, node_ids=node_ids
    )
    docs = list(table_docs) + list(column_docs)
    if not docs:
        return _empty

    rows = []
    for item in docs:
        text = (item.get("text") or "").strip()
        node_id = item.get("id")
        label = item.get("label", "")
        name = item.get("name", "")
        path = f"neo4j:{node_id}" if node_id is not None else "neo4j:unknown"
        # Nest tabular identifiers under content_metadata so they survive the
        # IngestVdbOperator → LanceDB write path (which only persists
        # content_metadata + source_metadata into the table's metadata column).
        # Top-level copies are kept for any in-memory consumer of this DataFrame.
        tabular_fields = {
            "id": node_id,
            "label": label,
            "name": name,
            "source_path": path,
            "database_name": database_name,
        }
        rows.append(
            {
                "text": text,
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {
                    **tabular_fields,
                    "content_metadata": dict(tabular_fields),
                },
            }
        )
    return pd.DataFrame(rows)
