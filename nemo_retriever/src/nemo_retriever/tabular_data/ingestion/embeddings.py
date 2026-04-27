# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pandas as pd

from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels


def query_neo4j_tables_for_embedding() -> List[dict]:
    """Run the Neo4j query for tables not yet info_embedded; return list of doc dicts."""
    neo4j_conn = get_neo4j_conn()
    query = f"""
        MATCH (d:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
              -[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})

        WITH d, s, t, c,
             CASE
                 WHEN c.description IS NOT NULL AND trim(c.description) <> ''
                 THEN ', description: ' + c.description
                 ELSE ''
             END AS column_desc

        WITH d, s, t,
             collect('{{name: ' + c.name +
                     ', data_type: ' + c.data_type +
                     column_desc + '}}') AS columns

        WITH d, s, t, columns,
             CASE
                 WHEN t.description IS NOT NULL AND trim(t.description) <> ''
                 THEN ', table_description: ' + t.description
                 ELSE ''
             END AS table_desc

        RETURN collect({{
            text:  'db_name: ' + d.name +
                   ', schema_name: ' + s.name +
                   ', table_name: ' + t.name +
                   table_desc +
                   ', columns: ' + apoc.text.join(columns, ' '),
            name: t.name,
            label: labels(t)[0],
            id: t.id
        }}) AS docs
    """
    result = neo4j_conn.query_read(query, parameters={})
    if not result:
        return []
    return result[0].get("docs") or []


def query_neo4j_columns_for_embedding() -> List[dict]:
    """Return one doc per ``Column`` node for embedding (distinct from table-level rows)."""
    neo4j_conn = get_neo4j_conn()
    query = f"""
        MATCH (d:{Labels.DB})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
              -[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
              -[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})

        WITH d, s, t, c,
             CASE
                 WHEN c.description IS NOT NULL AND trim(toString(c.description)) <> ''
                 THEN ', column_description: ' + toString(c.description)
                 ELSE ''
             END AS column_desc

        RETURN collect({{
            text:  'db_name: ' + d.name +
                   ', schema_name: ' + s.name +
                   ', table_name: ' + t.name +
                   ', column_name: ' + c.name +
                   ', data_type: ' + coalesce(toString(c.data_type), '') +
                   column_desc,
            name: c.name,
            label: labels(c)[0],
            id: c.id
        }}) AS docs
    """
    result = neo4j_conn.query_read(query, parameters={})
    if not result:
        return []
    return result[0].get("docs") or []


def fetch_tabular_embedding_dataframe() -> pd.DataFrame:
    """Fetch all tabular entity docs from Neo4j and return a DataFrame ready for embedding.

    Each row has: text, _embed_modality, path, page_number, metadata
    (id, label, name, source_path) — matching the format produced by the
    unstructured pipeline so run_pipeline_tasks_on_df works without changes.
    """
    _empty = pd.DataFrame(columns=["text", "_embed_modality", "path", "page_number", "metadata"])
    table_docs = query_neo4j_tables_for_embedding()
    column_docs = query_neo4j_columns_for_embedding()
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
        rows.append(
            {
                "text": text,
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {
                    "id": node_id,
                    "label": label,
                    "name": name,
                    "source_path": path,
                },
            }
        )
    return pd.DataFrame(rows)
