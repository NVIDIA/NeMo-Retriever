import os
from typing import Any, List, Optional

import pandas as pd

from nemo_retriever.relational_db.population.graph.model.reserved_words import Labels
from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn


# Keep backward compatibility
# azure_embeddings = get_azure_embeddings_client()




def get_conn_credentials():
    conn_creds = get_neo4j_conn().get_connection()
    url = conn_creds._Neo4jConnection__uri
    username = conn_creds._Neo4jConnection__username
    password = conn_creds._Neo4jConnection__password
    return {"url": url, "username": username, "password": password}




def generate_embeddings_for_node_names_and_descriptions(list_of_labels):
    neo4j_conn = get_neo4j_conn()
    embeddings = get_embeddings()
    if embeddings is None:
        return

    docs = []
    for label in list_of_labels:
        if label == Labels.ATTR:
            query = """MATCH (term:term)-[:term_of]->(attr:attribute WHERE coalesce(attr.embedded, FALSE)=FALSE)
                       WITH term, attr, CASE WHEN attr.description is null THEN "term_description: " + term.description ELSE attr.description END as description
                       SET attr.embedded = TRUE
                       RETURN collect({text: "term_name: " + term.name + ", attribute_name: " + attr.name + ", description: " + coalesce(description, 'null'),
                       name: attr.name, label: labels(attr)[0], id: attr.id}) AS docs
                    """
        else:
            query = f"""MATCH (n:{label} WHERE coalesce(n.recommended, FALSE)=FALSE AND coalesce(n.embedded, FALSE)=FALSE)
                        SET n.embedded = TRUE
                        RETURN collect({{text: "name: " + n.name + ", description: " + coalesce(n.description, 'null'), name: n.name, label: labels(n)[0], id: n.id}}) AS docs
                    """
        result = neo4j_conn.query_write(query, parameters={})
        if len(result) > 0:
            result = result[0]["docs"]
            docs.extend(
                [
                    Document(
                        page_content=item["text"] or "",
                        metadata={
                            "id": item["id"],
                            "label": item["label"],
                            "name": item["name"],
                        },
                    )
                    for item in result
                ]
            )



def generate_embeddings_for_tables_and_columns(
    *,
    return_dataframe_for_embed: bool = False,
):
    """
    Query Neo4j for tables not yet info_embedded, set info_embedded=TRUE, and return
    either LangChain Document list (default) or a DataFrame ready for nemo_retriever
    .embed() when return_dataframe_for_embed=True.

    When return_dataframe_for_embed=False: requires get_embeddings();
    returns list of Document or None (if no embeddings client).
    When return_dataframe_for_embed=True: does not use get_embeddings; returns
    pd.DataFrame with columns text, path, page_number, metadata for feeding into
    embed_neo4j_tables_dataframe or nemo_retriever's embed pipeline.
    """
    neo4j_conn = get_neo4j_conn()
    if not return_dataframe_for_embed:
        embeddings = get_embeddings()
        if embeddings is None:
            return None

    query = """MATCH (d:db)-[:schema]->(s:schema)-[:schema]->(t:table WHERE coalesce(t.info_embedded, FALSE)=FALSE)-[:schema]->(c:column)
               WITH d, s, t, collect("{name: " + c.name +", data_type: " + c.data_type + ", description: " + coalesce(c.description, 'null') +"}") as columns
               SET t.info_embedded = TRUE
               RETURN collect({text: "db_name: " + d.name + ", schema_name: " + s.name + ", table_name: " + t.name +
               ", table_description: " + coalesce(t.description, 'null') + ", columns: " + apoc.text.join(columns, ' '),
               name: t.name, label: labels(t)[0], id: t.id}) as docs
            """
    result = neo4j_conn.query_write(query, parameters={})
    if len(result) == 0:
        items: List[dict] = []
    else:
        items = result[0].get("docs") or []

    if return_dataframe_for_embed:
        return neo4j_tables_result_to_embedding_dataframe(items)

    docs = [
        Document(
            page_content=item["text"] or "",
            metadata={
                "id": item["id"],
                "label": item["label"],
                "name": item["name"],
            },
        )
        for item in items
    ]
    return docs





def query_neo4j_tables_for_embedding() -> List[dict]:
    """Run the Neo4j query for tables not yet info_embedded; return list of doc dicts."""
    neo4j_conn = get_neo4j_conn()
    query = """MATCH (d:db)-[:schema]->(s:schema)-[:schema]->(t:table)
               MATCH (t)-[:schema]->(c:column)
               WITH d, s, t, collect("{name: " + c.name +", data_type: " + c.data_type + ", description: " + coalesce(c.description, 'null') +"}") as columns
               RETURN collect({text: "db_name: " + d.name + ", schema_name: " + s.name + ", table_name: " + t.name +
               ", table_description: " + coalesce(t.description, 'null') + ", columns: " + apoc.text.join(columns, ' '),
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


def build_embed_kwargs_for_neo4j(
    params: Optional[Any] = None,
    **kwargs: Any,
) -> dict:
    """
    Build the exact kwargs passed to embed_text_main_text_embed by
    InProcessIngestor.embed(), so Neo4j-sourced DataFrames use the same
    embedding config as the main pipeline.

    Use EmbedParams (or kwargs matching EmbedParams fields). Examples:

        from nemo_retriever.params import EmbedParams

        # Remote NIM
        kwargs = build_embed_kwargs_for_neo4j(
            EmbedParams(embedding_endpoint="http://embedding:8000/v1")
        )

        # Local model
        kwargs = build_embed_kwargs_for_neo4j(
            EmbedParams(model_name="nvidia/nemotron-embed-1b-v2")
        )

        # Then: embed_text_main_text_embed(df, **kwargs)
    """
    from nemo_retriever.params import EmbedParams
    from nemo_retriever.ingest_modes.inprocess import _coerce_params

    resolved = _coerce_params(params, EmbedParams, kwargs)
    embed_modality = resolved.embed_modality

    embed_kwargs = {
        **resolved.model_dump(
            mode="python",
            exclude={"runtime", "batch_tuning", "fused_tuning"},
            exclude_none=True,
        ),
        **resolved.runtime.model_dump(mode="python", exclude_none=True),
    }
    if "embedding_endpoint" not in embed_kwargs and embed_kwargs.get("embed_invoke_url"):
        embed_kwargs["embedding_endpoint"] = embed_kwargs.get("embed_invoke_url")
    embed_kwargs["embed_modality"] = embed_modality

    endpoint = (
        (embed_kwargs.get("embedding_endpoint") or embed_kwargs.get("embed_invoke_url") or "").strip()
        or None
    )

    if endpoint:
        embed_kwargs.setdefault("input_type", "passage")
        return embed_kwargs

    # Local HF path: create model and inject into kwargs (same as InProcessIngestor.embed)
    from nemo_retriever.model import create_local_embedder

    device = embed_kwargs.pop("device", None)
    hf_cache_dir = embed_kwargs.pop("hf_cache_dir", None)
    normalize = bool(embed_kwargs.pop("normalize", True))
    max_length = int(embed_kwargs.pop("max_length", 8192))
    model_name_raw = embed_kwargs.pop("model_name", None)
    embed_kwargs.setdefault("input_type", "passage")
    embed_kwargs["model"] = create_local_embedder(
        model_name_raw,
        device=str(device) if device is not None else None,
        hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        normalize=normalize,
        max_length=max_length,
    )
    return embed_kwargs


def embed_neo4j_tables_with_retriever(
    df: pd.DataFrame,
    params: Optional[Any] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Run nemo_retriever's embed stage on a DataFrame built from Neo4j table/column
    data (e.g. from neo4j_tables_result_to_embedding_dataframe or
    generate_embeddings_for_tables_and_columns(..., return_dataframe_for_embed=True)).

    Uses the same kwargs as InProcessIngestor.embed() via build_embed_kwargs_for_neo4j.

    Example:
        from nemo_retriever.params import EmbedParams
        df = generate_embeddings_for_tables_and_columns(return_dataframe_for_embed=True)
        df_embedded = embed_neo4j_tables_with_retriever(
            df, EmbedParams(embedding_endpoint="http://localhost:8012/v1")
        )
    """
    from nemo_retriever.ingest_modes.inprocess import embed_text_main_text_embed

    if df.empty or "text" not in df.columns:
        return df

    embed_kwargs = build_embed_kwargs_for_neo4j(params, **kwargs)
    return embed_text_main_text_embed(df, **embed_kwargs)


def fetch_relational_db_for_embedding() -> List[dict]:
    """Collect all docs to embed from the relational DB graph.

    Each source function returns a list of doc dicts; results are unioned here.
    Add future fetch calls below and extend `all_docs` accordingly.
    """
    all_docs: List[dict] = []
    all_docs.extend(query_neo4j_tables_for_embedding())
    return all_docs
