import os
from typing import Any, List, Optional

import pandas as pd

from nemo_retriever.relational_db.population.graph.model.reserved_words import Labels
from nemo_retriever.relational_db.infra.Neo4jConnection import get_neo4j_conn
from nemo_retriever.relational_db.infra.PostgresConnection import get_postgres_conn
from nemo_retriever.relational_db.ai_services.config import (
    get_embeddings_client,
    get_azure_embeddings_client,
)
from nemo_retriever.relational_db.features import Feature, is_feature_enabled
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document


# Keep backward compatibility
# azure_embeddings = get_azure_embeddings_client()

conn = get_neo4j_conn()
# pg_conn = get_postgres_conn()


# def get_embeddings(account_id: str, is_embeddings: bool = False, provider: str = None):
#     """
#     Get embeddings client based on account configuration and provider.

#     Args:
#         account_id: Account identifier
#         is_embeddings: Force enable embeddings
#         provider: Embedding provider ('azure'). If None, uses environment variable.

#     Returns:
#         Embeddings client or None
#     """
#     is_omni_enabled = is_feature_enabled(account_id, Feature.OMNI)

#     if is_embeddings or is_omni_enabled:
#         # Determine provider
#         if provider is None:
#             provider = os.environ.get("LMX_AI_PROVIDER", "azure")

#         embeddings_client = get_embeddings_client(provider)

#         # Fallback to Azure if requested provider fails
#         if embeddings_client is None and provider != "azure":
#             embeddings_client = get_embeddings_client("azure")

#         return embeddings_client

#     return None


def get_conn_credentials(account_id):
    conn_creds = conn._get_connection_for_account(parameters={"account_id": account_id})
    url = conn_creds._Neo4jConnection__uri
    username = conn_creds._Neo4jConnection__username
    password = conn_creds._Neo4jConnection__password
    return {"url": url, "username": username, "password": password}


# def update_single_item_embedding(account_id, item_id, label):
#     embeddings = get_embeddings(account_id)
#     if embeddings is None:
#         return

#     account_simple_str = account_id.replace("-", "_")
#     collection_name = f"{account_simple_str}_nodes_vector_store"

#     # first, update general nodes' embeddings collection
#     if label in [Labels.ATTR, Labels.BT]:
#         if label == Labels.BT:
#             query = """ MATCH (term:term {account_id:$account_id, id: $id})-[:term_of]->(attr:attribute)
#                         WITH term, collect({text: "term_name: " + term.name + ", attribute_name: " + attr.name + ", description: " + coalesce(attr.description, 'null'), 
#                         name: attr.name, label: labels(attr)[0], id: attr.id, account_id: $account_id}) as attr_docs 
#                         RETURN [{text: "name: " + term.name + ", description: " + coalesce(term.description, 'null'), 
#                         name: term.name, label: labels(term)[0], id: term.id, account_id: $account_id}] + attr_docs as docs
#                     """
#         if label == Labels.ATTR:
#             query = """MATCH (term:term {account_id:$account_id})-[:term_of]->(attr:attribute {id: $id})
#                        RETURN [{text: "term_name: " + term.name + ", attribute_name: " + attr.name + ", description: " + coalesce(attr.description, 'null'), 
#                        name: attr.name, label: labels(attr)[0], id: attr.id, account_id: $account_id}] as docs
#                     """
#     else:
#         query = f"""MATCH (n:{label} {{account_id:$account_id, id: $id}})
#                     // WHERE coalesce(n.recommended,false)=false
#                     RETURN [{{text: "name: " + n.name + ", description: " + coalesce(n.description, 'null'),
#                     name: n.name, label: labels(n)[0], id: n.id, account_id: $account_id}}] as docs
#                 """
#     result = conn.query_read_only(
#         query, parameters={"account_id": account_id, "id": item_id}
#     )
#     if len(result) > 0:
#         result = result[0]["docs"]

#     docs = [
#         Document(
#             page_content=item["text"] or "",
#             metadata={
#                 "id": item["id"],
#                 "label": item["label"],
#                 "account_id": item["account_id"],
#                 "name": item["name"],
#             },
#         )
#         for item in result
#     ]

#     # For single-item updates we want to REPLACE existing vectors for the same ids,
#     # not just append new rows. We do this by directly deleting from the underlying
#     # PGVector table (public.langchain_pg_embedding) using the metadata id, and then
#     # inserting fresh embeddings via from_documents.
#     ids_to_delete = [doc.metadata["id"] for doc in docs]
#     if ids_to_delete:
#         placeholders = ", ".join(["%s"] * len(ids_to_delete))
#         delete_query = f"""
#             DELETE FROM public.langchain_pg_embedding
#             WHERE collection_id = (
#                 SELECT uuid FROM public.langchain_pg_collection WHERE name = %s
#             )
#               AND cmetadata->>'id' IN ({placeholders})
#         """
#         pg_conn.delete_rows(delete_query, [collection_name, *ids_to_delete])

#     PGVector.from_documents(
#         documents=docs,
#         embedding=embeddings,
#         collection_name=collection_name,
#         connection_string=pg_conn.connection_string,
#     )

#     # secondly, if a column or a table have been updated, update table_info embeddings collection
#     if label in [Labels.COLUMN, Labels.TABLE]:
#         if label == Labels.COLUMN:
#             query = """MATCH (d:db{account_id:$account_id})-[:schema]->(s:schema)-[:schema]->(t:table)-[:schema]->(c:column {id: $id}) 
#                     """
#         else:
#             query = """MATCH (d:db{account_id:$account_id})-[:schema]->(s:schema)-[:schema]->(t:table {id: $id})-[:schema]->(c:column) """
#         query += """
#                    WITH d, s, t, collect("{name: " + c.name +", data_type: " + c.data_type + ", description: " + coalesce(c.description, 'null') +"}") as columns
#                    RETURN collect({text: "db_name: " + d.name + "schema_name: " + s.name + ", table_name: " + t.name + 
#                    ", table_description: " + coalesce(t.description, 'null') + ", columns: " + apoc.text.join(columns, ' '),
#                    name: t.name, label: labels(t)[0], id: t.id, account_id: $account_id}) as docs
#                 """
#         result = conn.query_read_only(
#             query, parameters={"account_id": account_id, "id": item_id}
#         )
#         if len(result) > 0:
#             result = result[0]["docs"]

#         docs = [
#             Document(
#                 page_content=item["text"] or "",
#                 metadata={
#                     "id": item["id"],
#                     "label": item["label"],
#                     "account_id": item["account_id"],
#                     "name": item["name"],
#                 },
#             )
#             for item in result
#         ]

#         tables_collection_name = f"{account_simple_str}_tables_info_vector_store"

#         # Same replacement logic for table/column info embeddings.
#         ids_to_delete = [doc.metadata["id"] for doc in docs]
#         if ids_to_delete:
#             placeholders = ", ".join(["%s"] * len(ids_to_delete))
#             delete_query = f"""
#                 DELETE FROM public.langchain_pg_embedding
#                 WHERE collection_id = (
#                     SELECT uuid FROM public.langchain_pg_collection WHERE name = %s
#                 )
#                   AND cmetadata->>'id' IN ({placeholders})
#             """
#             pg_conn.delete_rows(delete_query, [tables_collection_name, *ids_to_delete])

#         PGVector.from_documents(
#             documents=docs,
#             embedding=embeddings,
#             collection_name=tables_collection_name,
#             connection_string=pg_conn.connection_string,
#         )


def generate_embeddings_for_node_names_and_descriptions(account_id, list_of_labels):
    embeddings = get_embeddings(account_id)
    if embeddings is None:
        return

    account_simple_str = account_id.replace("-", "_")
    collection_name = f"{account_simple_str}_nodes_vector_store"
    docs = []
    for label in list_of_labels:
        if label == Labels.ATTR:
            query = """MATCH (term:term {account_id:$account_id})-[:term_of]->(attr:attribute {account_id:$account_id} WHERE coalesce(attr.embedded, FALSE)=FALSE)
                       WITH term, attr, CASE WHEN attr.description is null THEN "term_description: " + term.description ELSE attr.description END as description  
                       SET attr.embedded = TRUE
                       RETURN collect({text: "term_name: " + term.name + ", attribute_name: " + attr.name + ", description: " + coalesce(description, 'null'), 
                       name: attr.name, label: labels(attr)[0], id: attr.id, account_id: $account_id}) AS docs
                    """
        else:
            query = f"""MATCH (n:{label} {{account_id:$account_id}} WHERE coalesce(n.recommended, FALSE)=FALSE AND coalesce(n.embedded, FALSE)=FALSE)
                        SET n.embedded = TRUE
                        RETURN collect({{text: "name: " + n.name + ", description: " + coalesce(n.description, 'null'), name: n.name, label: labels(n)[0], id: n.id, account_id: $account_id}}) AS docs
                    """
        result = conn.query_write(query, parameters={"account_id": account_id})
        if len(result) > 0:
            result = result[0]["docs"]
            docs.extend(
                [
                    Document(
                        page_content=item["text"] or "",
                        metadata={
                            "id": item["id"],
                            "label": item["label"],
                            "account_id": item["account_id"],
                            "name": item["name"],
                        },
                    )
                    for item in result
                ]
            )



def generate_embeddings_for_tables_and_columns(
    account_id: str,
    *,
    return_dataframe_for_embed: bool = False,
):
    """
    Query Neo4j for tables not yet info_embedded, set info_embedded=TRUE, and return
    either LangChain Document list (default) or a DataFrame ready for nemo_retriever
    .embed() when return_dataframe_for_embed=True.

    When return_dataframe_for_embed=False: requires get_embeddings(account_id);
    returns list of Document or None (if no embeddings client).
    When return_dataframe_for_embed=True: does not use get_embeddings; returns
    pd.DataFrame with columns text, path, page_number, metadata for feeding into
    embed_neo4j_tables_dataframe or nemo_retriever's embed pipeline.
    """
    if not return_dataframe_for_embed:
        embeddings = get_embeddings(account_id)
        if embeddings is None:
            return None

    account_simple_str = account_id.replace("-", "_")
    collection_name = f"{account_simple_str}_tables_info_vector_store"

    query = """MATCH (d:db{account_id:$account_id})-[:schema]->(s:schema {account_id:$account_id})-[:schema]->(t:table {account_id:$account_id} WHERE coalesce(t.info_embedded, FALSE)=FALSE)-[:schema]->(c:column)
               WITH d, s, t, collect("{name: " + c.name +", data_type: " + c.data_type + ", description: " + coalesce(c.description, 'null') +"}") as columns
               SET t.info_embedded = TRUE
               RETURN collect({text: "db_name: " + d.name + ", schema_name: " + s.name + ", table_name: " + t.name +
               ", table_description: " + coalesce(t.description, 'null') + ", columns: " + apoc.text.join(columns, ' '),
               name: t.name, label: labels(t)[0], id: t.id, account_id: $account_id}) as docs
            """
    result = conn.query_write(query, parameters={"account_id": account_id})
    if len(result) == 0:
        items: List[dict] = []
    else:
        items = result[0].get("docs") or []

    if return_dataframe_for_embed:
        return neo4j_tables_result_to_embedding_dataframe(items, account_id)

    docs = [
        Document(
            page_content=item["text"] or "",
            metadata={
                "id": item["id"],
                "label": item["label"],
                "account_id": item["account_id"],
                "name": item["name"],
            },
        )
        for item in items
    ]
    return docs





def query_neo4j_tables_for_embedding(account_id: str) -> List[dict]:
    """Run the Neo4j query for tables not yet info_embedded; return list of doc dicts."""
    query = """MATCH (d:db{account_id:$account_id})-[:schema]->(s:schema {account_id:$account_id})-[:schema]->(t:table {account_id:$account_id}) WHERE coalesce(t.info_embedded, FALSE)=FALSE
               MATCH (t)-[:schema]->(c:column)
               WITH d, s, t, collect("{name: " + c.name +", data_type: " + c.data_type + ", description: " + coalesce(c.description, 'null') +"}") as columns
               SET t.info_embedded = TRUE
               RETURN collect({text: "db_name: " + d.name + ", schema_name: " + s.name + ", table_name: " + t.name +
               ", table_description: " + coalesce(t.description, 'null') + ", columns: " + apoc.text.join(columns, ' '),
               name: t.name, label: labels(t)[0], id: t.id, account_id: $account_id}) as docs
            """
    result = conn.query_write(query, parameters={"account_id": account_id})
    if not result:
        return []
    return result[0].get("docs") or []


def neo4j_tables_result_to_embedding_dataframe(
    neo4j_docs: List[dict],
    account_id: str,
    *,
    text_key: str = "text",
    id_key: str = "id",
    label_key: str = "label",
    name_key: str = "name",
) -> pd.DataFrame:
    """
    Build a DataFrame from Neo4j table/column query results for use with
    nemo_retriever's embed_text_main_text_embed (same as InProcessIngestor.embed()).

    Each row has: text, path, page_number, metadata (id, label, account_id, name, source_path).
    No _embed_modality column: embed will use embed_modality from kwargs (default "text").
    """
    if not neo4j_docs:
        return pd.DataFrame(columns=["text", "path", "page_number", "metadata"])

    rows = []
    for item in neo4j_docs:
        text = (item.get(text_key) or "").strip()
        node_id = item.get(id_key)
        label = item.get(label_key, "")
        name = item.get(name_key, "")
        path = f"neo4j:{account_id}:{node_id}" if node_id is not None else f"neo4j:{account_id}"
        rows.append({
            "text": text,
            "path": path,
            "page_number": -1,
            "metadata": {
                "id": node_id,
                "label": label,
                "account_id": item.get("account_id", account_id),
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
        df = generate_embeddings_for_tables_and_columns(account_id, return_dataframe_for_embed=True)
        df_embedded = embed_neo4j_tables_with_retriever(
            df, EmbedParams(embedding_endpoint="http://localhost:8012/v1")
        )
    """
    from nemo_retriever.ingest_modes.inprocess import embed_text_main_text_embed

    if df.empty or "text" not in df.columns:
        return df

    embed_kwargs = build_embed_kwargs_for_neo4j(params, **kwargs)
    return embed_text_main_text_embed(df, **embed_kwargs)



