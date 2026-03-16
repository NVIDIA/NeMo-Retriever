from typing import List

import pandas as pd

from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn


def query_neo4j_tables_for_embedding() -> List[dict]:
    """Run the Neo4j query for tables not yet info_embedded; return list of doc dicts."""
    neo4j_conn = get_neo4j_conn()
    query = """MATCH (d:Db)-[:CONTAINS]->(s:Schema)-[:CONTAINS]->(t:Table)
               MATCH (t)-[:CONTAINS]->(c:Column)
               WITH d, s, t, collect("{name: " + c.name + ", data_type: " + c.data_type +
                 CASE WHEN c.description IS NOT NULL AND trim(c.description) <> '' THEN ", description: " + c.description ELSE "" END + "}") as columns
               RETURN collect({text: "db_name: " + d.name + ", schema_name: " + s.name + ", table_name: " + t.name +
                 CASE WHEN t.description IS NOT NULL AND trim(t.description) <> '' THEN ", table_description: " + t.description ELSE "" END +
                 ", columns: " + apoc.text.join(columns, ' '),
               name: t.name, label: labels(t)[0], id: t.id}) as docs
            """
    result = neo4j_conn.query_write(query, parameters={})
    if not result:
        return []
    return result[0].get("docs") or []


def neo4j_tables_result_to_embedding_dataframe(
    neo4j_docs: List[dict],
    *,
    text_key: str = "text",
    id_key: str = "id",
    label_key: str = "label",
    name_key: str = "name",
    embed_modality: str = "text",
) -> pd.DataFrame:
    """
    Build a DataFrame from Neo4j table/column query results for use with
    nemo_retriever's embed_text_main_text_embed (same as InProcessIngestor.embed()).

    Each row has: text, _embed_modality, path, page_number, metadata
    (id, label, name, source_path) — matching the format produced by the
    unstructured pipeline so run_pipeline_tasks_on_df works without changes.
    """
    if not neo4j_docs:
        return pd.DataFrame(columns=["text", "_embed_modality", "path", "page_number", "metadata"])

    rows = []
    for item in neo4j_docs:
        text = (item.get(text_key) or "").strip()
        node_id = item.get(id_key)
        label = item.get(label_key, "")
        name = item.get(name_key, "")
        path = f"neo4j:{node_id}" if node_id is not None else "neo4j:unknown"
        rows.append({
            "text": text,
            "_embed_modality": embed_modality,
            "path": path,
            "page_number": -1,
            "metadata": {
                "id": node_id,
                "label": label,
                "name": name,
                "source_path": path,
            },
        })
    return pd.DataFrame(rows)


def fetch_relational_db_for_embedding() -> List[dict]:
    """Collect all docs to embed from the relational DB graph.

    Each source function returns a list of doc dicts; results are unioned here.
    Add future fetch calls below and extend `all_docs` accordingly.
    """
    all_docs: List[dict] = []
    all_docs.extend(query_neo4j_tables_for_embedding())
    return all_docs
