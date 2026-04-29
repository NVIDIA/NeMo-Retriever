from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from itertools import groupby
from typing import TYPE_CHECKING, Union
import pandas as pd
import numpy as np
from datetime import date, timedelta
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels

if TYPE_CHECKING:
    from nemo_retriever.retriever import Retriever
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.schema import Schema
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================

# Hard ceiling on how many candidate snippets we want to reason over for a single question.
# Larger numbers tend to confuse the LLM and increase latency.
MAX_CALCULATION_CANDIDATES = 15

# k for each get_candidates_information call (qnv, qwv, per-entity).
CANDIDATES_K = 10

# Per-attempt batch size when pulling semantic candidates. We keep this modest
# to balance recall with response time and to leave room for deduping.
ATTR_CANDIDATE_BATCH_SIZE = 6

# Maximum number of batch pulls before we give up looking for fresh attributes;
# empirically, three rounds either fill the quota or show the backend is stuck on duplicates.
ATTR_CANDIDATE_MAX_ATTEMPTS = 3

# Budget for entity-specific candidate searches. This limits how many candidates we fetch
# per entity when ensuring entity coverage. Distributed evenly across entities (at least 1 per entity).
# This prevents over-fetching when there are many entities while ensuring each gets coverage.
ENTITY_CANDIDATE_BUDGET = 4


QUERIES_USAGE_PERCENTILE = "queries_usage_percentile"
TABLES_USAGE_PERCENTILE = "tables_usage_percentile"
COLUMNS_USAGE_PERCENTILE = "columns_usage_percentile"


def get_llm_client() -> ChatNVIDIA:
    return ChatNVIDIA(
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get("NVIDIA_API_KEY"),
        model=os.environ.get("MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b"),
    )


def store_usage_percentiles(
    percentiles_type_name: str,
    usage_percentile_25: int,
    usage_percentile_75: int,
):
    query = f"""
            MATCH (d:{Labels.DB})
            WITH d
            CALL apoc.create.setProperties(d,
                [$percentiles_type_name_25, $percentiles_type_name_75],
                [$usage_percentile_25, $usage_percentile_75])
            YIELD node
            RETURN d
            """
    get_neo4j_conn().query_write(
        query=query,
        parameters={
            "percentiles_type_name_25": f"{percentiles_type_name}_25",
            "percentiles_type_name_75": f"{percentiles_type_name}_75",
            "usage_percentile_25": usage_percentile_25,
            "usage_percentile_75": usage_percentile_75,
        },
    )


def get_stored_usage_percentiles(percentiles_type_name: str):
    query = f"""
                MATCH (n:{Labels.DB})
                RETURN n.{f"{percentiles_type_name}_25"} as usage_percentile_25,
                       n.{f"{percentiles_type_name}_75"} as usage_percentile_75
                """
    results = get_neo4j_conn().query_read(
        query=query,
        parameters={},
    )
    return results


def get_count_str_by_month(alias: str):
    current = date.today().replace(day=1)
    count_3_month = []

    for i in range(0, 3):
        count_3_month.append(f"coalesce({alias}.cnt_{current.month}_{current.year}, 0)")
        prev = current - timedelta(days=1)
        current = prev.replace(day=1)

    count_str = "+".join(count_3_month)
    return count_str


def init_queries_usage_percentiles():
    count_string = get_count_str_by_month("n")
    query_all = f"""match(n:{Labels.SQL}{{is_sub_select:FALSE}}) return collect({count_string}) as usages"""
    usages_result = get_neo4j_conn().query_read(query=query_all, parameters={})
    usages = usages_result[0]["usages"]
    if len(usages) == 0:
        return 0, 0
    usage_percentile_25 = np.percentile(usages, 25)
    usage_percentile_75 = np.percentile(usages, 75)
    store_usage_percentiles(QUERIES_USAGE_PERCENTILE, usage_percentile_25, usage_percentile_75)
    return usage_percentile_25, usage_percentile_75


def get_usage_percentiles():
    stored_percentiles = get_stored_usage_percentiles(QUERIES_USAGE_PERCENTILE)
    if len(stored_percentiles) == 0 or (stored_percentiles[0]["usage_percentile_25"] is None):
        usage_percentile_25, usage_percentile_75 = init_queries_usage_percentiles()
    else:
        usage_percentile_25 = stored_percentiles[0]["usage_percentile_25"]
        usage_percentile_75 = stored_percentiles[0]["usage_percentile_75"]

    return usage_percentile_25, usage_percentile_75


def get_queries_usage_percentiles(node_str="node"):
    count_str = get_count_str_by_month(node_str)
    usage_percentile_25, usage_percentile_75 = get_usage_percentiles()
    return usage_percentile_25, usage_percentile_75, count_str


def clean_results(raw_candidates: list[dict]) -> list[dict]:
    """
    Normalize raw semantic hits: require id, dedupe by (label, id), preserve order.
    """
    out: list[dict] = []
    seen: set[tuple[str | None, str]] = set()
    for c in raw_candidates or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


queries_for_columns_params = {
    "wildcard_names": ["Wildcard", "QualifiedWildcard"],
    "sql_subgraph_rel": "<SQL",
    "sql_subgraph_labels": ">sql|-table",
    "sql_type": Labels.SQL,
}
queries_for_columns_params_keys = ", ".join([f"{key}:${key}" for key in queries_for_columns_params.keys()])


def expand_info(ids_and_labels):
    """Fetch Neo4j properties per (label, id). Column nodes merge parent table into ``relevant_tables``."""
    items: list[dict] = []
    for x in ids_and_labels or []:
        if not isinstance(x, dict):
            continue
        if x.get("id") is None:
            continue
        if str(x.get("label") or "").strip() == "":
            continue
        items.append({"id": x["id"], "label": x["label"]})

    results = {}

    (
        queries_percentile_25,
        queries_percentile_75,
        _cnt_str,
    ) = get_queries_usage_percentiles("sql_node")

    allowed_labels = set(Labels.LIST_OF_ALL)
    for label, ids in groupby(
        sorted(items, key=lambda d: str(d.get("label") or "").strip()),
        key=lambda d: str(d.get("label") or "").strip(),
    ):
        label_id_pairs_for_current_label = list(ids)
        if not label:
            continue
        if label not in allowed_labels:
            logger.warning("Skipping unknown label %r in expand_info", label)
            continue
        query = f"""UNWIND $label_id_pairs as label_id
                    MATCH (n:{label} {{id: label_id.id}})
                    CALL apoc.case([
                        n:{Labels.CUSTOM_ANALYSIS},
                            'MATCH(n)-[:analysis_of]->(sql:{Labels.SQL})
                            WITH n, collect(distinct {{sql_code: sql.sql_full_query}}) as sql
                            RETURN apoc.map.setKey(properties(n), "sql", sql) as item',
                        n:{Labels.COLUMN},
                            'MATCH(n)<-[:CONTAINS]-(parent)
                            WITH n, parent,
                                 [(parent)-[:CONTAINS]->(c:{Labels.COLUMN}) |
                                  {{name: c.name, data_type: toString(coalesce(c.data_type, ""))}}] AS column_list
                            WITH n, parent, column_list,
                                 apoc.map.merge(
                                     properties(parent),
                                     {{label: coalesce(parent.label,
                                      toLower(head(labels(parent))), "{Labels.TABLE}"),
                                      columns: column_list}}
                                 ) AS t0
                            RETURN apoc.map.merge(
                                     apoc.map.setPairs(properties(n),[
                                         ["table_name", parent.name],
                                         ["table_type", parent.type],
                                         ["parent_id", parent.id]
                                     ]),
                                     {{relevant_tables: [t0]}}
                                 ) as item'
                        ],
                        'with n RETURN n{{ .*}} as item ',
                        {{n:n, sql_type: $sql_type, usage_percentile_25: $usage_percentile_25,
                        usage_percentile_75: $usage_percentile_75, {queries_for_columns_params_keys} }}
                        )
                    YIELD value as response
                    WITH collect(response.item) as all_items
                    RETURN apoc.map.groupBy(all_items,'id') as ids_to_props
                    """
        params = {
            "sql_type": Labels.SQL,
            "label_id_pairs": label_id_pairs_for_current_label,
            "usage_percentile_25": queries_percentile_25,
            "usage_percentile_75": queries_percentile_75,
        }
        params.update(queries_for_columns_params)
        result = get_neo4j_conn().query_read(
            query=query,
            parameters=params,
        )
        if len(result) > 0:
            results = results | result[0]["ids_to_props"]

    return results


def _parse_lancedb_row_metadata(hit: dict) -> dict:
    """Normalize LanceDB hit ``metadata`` (dict or JSON string) to a flat dict."""
    raw = hit.get("metadata")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Ingestion sometimes stores Python repr (single-quoted keys) — not valid JSON.
            try:
                ev = ast.literal_eval(raw)
                if isinstance(ev, dict):
                    return ev
            except (ValueError, SyntaxError, TypeError):
                pass
            return {}
    return {}


def _vector_distance_value(distance: object | None) -> float:
    """LanceDB dense search ``_distance`` (L2); lower is better. Missing → +inf for sorting."""
    if distance is None:
        return float("inf")
    try:
        return float(distance)
    except (TypeError, ValueError):
        return float("inf")


def _hits_to_semantic_rows(hits: list[dict], _allowed_labels: set[str], k: int) -> list[dict]:
    """Turn raw LanceDB hits into candidate dicts.

    Label filtering is already applied in LanceDB via ``label_in``; each row uses ``text`` as
    the candidate string. ``id`` / ``label`` are taken from metadata (minimal parse) for
    ``expand_info`` / Neo4j enrichment only.

    ``score`` is the raw vector ``_distance`` from Lance (lower is better; see sorts below).
    """
    rows: list[dict] = []
    for hit in hits:
        meta = _parse_lancedb_row_metadata(hit)
        cid = meta.get("id")
        if cid is None:
            continue
        lab = meta.get("label") if meta.get("label") is not None else hit.get("label")
        score = _vector_distance_value(hit.get("_distance"))
        rows.append(
            {
                "text": (hit.get("text") or "").strip(),
                "id": cid,
                "label": lab,
                "score": score,
            }
        )
        if len(rows) >= int(k):
            break
    return rows[: int(k)]


def search_lancedb_semantic_index(
    retriever: "Retriever",
    entity: str,
    k: int = 30,
    label_filter: list[str] | None = None,
) -> list[dict]:
    """
    Vector search over LanceDB via the injected :class:`~nemo_retriever.retriever.Retriever`
    (same stack as ``generate_sql.get_sql_tool_response_top_k``).

    ``Retriever`` applies ``label_filter`` in LanceDB with ``(label IN (...)) OR
    (metadata LIKE …)`` when those columns exist (substring patterns include Neo4j-style
    ``Column`` vs ``column``). ``_hits_to_semantic_rows`` maps hits to ``text`` + ``id``/``label``
    for downstream enrichment (no second label filter).

    The retriever's ``lancedb_uri`` / ``lancedb_table`` / ``embedder`` / embedding
    endpoint and credentials are fully decided at retriever construction time;
    this function does not read any environment variables.
    """
    allowed_labels = {str(x) for x in (label_filter or []) if x is not None}
    limit = max(1, int(k))

    retriever.top_k = limit

    hits = retriever.query(
        entity,
        label_in=sorted(allowed_labels) if allowed_labels else None,
    )

    return _hits_to_semantic_rows(hits, allowed_labels, limit)


def get_candidates_information(
    retriever: "Retriever",
    entity: str,
    k: int = 5,
    list_of_semantic: list | None = None,
):
    """
    Vector search over LanceDB, then merge graph properties from ``expand_info``.

    Column nodes get ``relevant_tables``: one normalized table dict (parent ``:table`` via
    ``<-[:schema]-``), same shape as :func:`get_relevant_tables` / ``_normalize_table_to_relevant_shape``.

    Matches the call shape used by ``extract_candidates``:
    ``get_candidates_information(retriever, text, k=..., list_of_semantic=[...])``.
    """
    if list_of_semantic is None:
        list_of_semantic = [Labels.CUSTOM_ANALYSIS, Labels.COLUMN]
    labels = list_of_semantic
    results: list[dict] = []

    nodes_results = search_lancedb_semantic_index(
        retriever,
        entity,
        k=k,
        label_filter=labels,
    )
    results.extend(nodes_results)

    ids_and_labels = [{"label": x["label"], "id": x["id"]} for x in results]
    props_by_id = expand_info(ids_and_labels)
    for c in results:
        cid = c.get("id")
        if cid is None:
            continue
        extra = props_by_id.get(cid) or props_by_id.get(str(cid))
        if isinstance(extra, dict):
            c.update(extra)
            rel_tabs = c.get("relevant_tables")
            if isinstance(rel_tabs, list):
                c["relevant_tables"] = [_normalize_table_to_relevant_shape(t) for t in rel_tabs if isinstance(t, dict)]

    results.sort(key=lambda item: float(item.get("score") if item.get("score") is not None else float("inf")))
    return results


def _dedupe_best_score_sort_cap(combined: list[dict]) -> list[dict]:
    """Deduplicate by (label, id), keep lowest ``score`` (L2 distance), sort ascending, cap."""
    best_by_key: dict[tuple[str | None, str], dict] = {}
    for c in combined:
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        dist = c.get("score")
        score = float(dist) if dist is not None else float("inf")
        prev = best_by_key.get(key)
        prev_d = prev.get("score") if prev is not None else None
        prev_score = float(prev_d) if prev_d is not None else float("inf")
        if prev is None or score < prev_score:
            best_by_key[key] = c

    unique = list(best_by_key.values())
    unique.sort(key=lambda x: float(x.get("score")) if x.get("score") is not None else float("inf"))
    return unique[:MAX_CALCULATION_CANDIDATES]


def extract_candidates(
    retriever: "Retriever",
    entities: list[str],
    query_no_values: str,
    query_with_values: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    One semantic search per string: ``query_no_values``, ``query_with_values`` (if distinct),
    and each entity name. For each string, pull custom analyses and columns via
    ``get_candidates_information``. Merge streams, dedupe by (label, id) keeping the
    lowest vector distance (``score``), sort ascending by distance, cap at ``MAX_CALCULATION_CANDIDATES`` per stream.

    Returns:
        ``(custom_analysis_candidates, column_candidates)``
    """
    qnv = (query_no_values or "").strip()
    pulls: list[str] = []
    if qnv:
        pulls.append(qnv)
    if (qwv := (query_with_values or "").strip()) and qwv != qnv:
        pulls.append(qwv)
    for ent in entities or []:
        if t := (ent or "").strip():
            pulls.append(t)

    combined_custom: list[dict] = []
    combined_columns: list[dict] = []
    for text in pulls:
        combined_custom.extend(
            get_candidates_information(
                retriever,
                text,
                k=CANDIDATES_K,
                list_of_semantic=[Labels.CUSTOM_ANALYSIS],
            )
            or []
        )
        combined_columns.extend(
            get_candidates_information(
                retriever,
                text,
                k=CANDIDATES_K,
                list_of_semantic=[Labels.COLUMN],
            )
            or []
        )

    out_custom = _dedupe_best_score_sort_cap(combined_custom)
    out_columns = _dedupe_best_score_sort_cap(combined_columns)

    logger.info(
        f"extract_candidates: {len(out_custom)} custom_analysis, {len(out_columns)} column "
        f"(max {MAX_CALCULATION_CANDIDATES} each), {len(pulls)} pulls"
    )

    return out_custom, out_columns


def get_custom_analyses_ids(items):
    """Filter custom analyses by classification flag and return their IDs."""
    if not items:
        return []

    def _get(obj, key, default=None):
        """Safe getter for both Pydantic-style objects and plain dicts."""
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    classified_ids_and_labels = []
    for item in items:
        is_relevant = bool(_get(item, "classification", False))
        if not is_relevant:
            continue
        item_id = _get(item, "id")
        item_label = _get(item, "label")
        if item_id and item_label:
            classified_ids_and_labels.append({"id": item_id, "label": item_label})

    return classified_ids_and_labels


def extract_entities_with_id_name_label(data):
    result = {}

    def recurse(obj):
        if isinstance(obj, dict):
            # Main entity case: id + name + (type or label)
            if "id" in obj and "name" in obj and ("type" in obj or "label" in obj):

                final_label = obj["label"]

                result[obj["id"]] = (
                    obj["name"],
                    final_label,
                    obj.get("parent_id"),
                )

            # explicitly capture table inside column
            if "table" in obj and isinstance(obj["table"], dict):
                table = obj["table"]
                if "id" in table and "name" in table:
                    result[table["id"]] = (table["name"], "table", None)

            # Continue recursion
            for value in obj.values():
                recurse(value)

        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(data)
    return result


_ALLOWED_NODE_LABELS = frozenset(Labels.LIST_OF_ALL)


def get_node_properties_by_id(id, label: str | list[str]):
    labels_list = label if isinstance(label, list) else [label]
    for lbl in labels_list:
        if lbl not in _ALLOWED_NODE_LABELS:
            logger.warning("Rejecting unknown label %r in get_node_properties_by_id", lbl)
            return None
    label_filter = "|".join(labels_list)
    query = f"""
        MATCH(n:{label_filter}{{id:$id}})
        RETURN apoc.map.setKey(properties(n),"label", labels(n)[0]) as props
    """

    props = get_neo4j_conn().query_read_only(query, parameters={"id": id})
    if len(props) == 0:
        return None
    else:
        return props[0]["props"]


def get_item_by_id(item_id, label):
    result = get_node_properties_by_id(item_id, label)
    if result:
        return result
    else:
        logger.error(f"The required item with id : {item_id} is not found in graph. ERROR.")
        return None


def highlight_entity(items_present: dict, text: str) -> str:
    """
    Processes [[[entity]]] patterns in the text.

    Supported formats:
    - [[[name/id]]]
    - [[[label/id|display_name]]]

    Replaces valid ones with hyperlinks using `prepare_link`.
    Falls back to bolding just the entity name if invalid or not found.
    """

    def replace_entity(match):
        raw = match.group(1)
        cleaned = re.sub(r"\s*/\s*", "/", raw.strip())  # Remove whitespace around slash

        # Handle display name (e.g., [[[label/id|display_name]]])
        if "|" in cleaned:
            entity_part, display_name = cleaned.split("|", 1)
            display_name = display_name.strip()
        else:
            entity_part, display_name = cleaned, None

        # Now parse entity part → name/ID
        if "/" in entity_part:
            name_or_label, eid = entity_part.split("/", 1)
            name_or_label = name_or_label.strip()
            eid = eid.strip()

            # Lookup by ID
            entity = items_present.get(eid)
            if entity and (
                entity[0].lower() == name_or_label.lower()
                or (display_name and entity[0].lower() == display_name.lower())
            ):
                shown_name = display_name or name_or_label
                return f"<{prepare_link(shown_name, eid, entity[1], entity[2])}>"
            elif entity:
                logger.warning(
                    "ASSUMPTION: the name in link is not found correctly by llm, take it by id from candidates"
                )
                return f"<{prepare_link(entity[0], eid, entity[1], entity[2])}>"

            # no entity in candidates
            try:
                item = get_item_by_id(eid, name_or_label)
            except Exception:
                logger.error("Something not ok with id, error raised")
                return f"*{display_name or name_or_label}*"

            if item:
                return f"<{prepare_link(item['name'], eid, name_or_label)}>"
            else:
                logger.warning(f"Entity ID mismatch or not found: {name_or_label}/{eid}")
                return f"*{display_name or name_or_label}*"
        else:
            logger.warning(f"No ID found in entity: {cleaned}")
            return f"*{cleaned}*"

    return re.sub(r"\[\[\[(.*?)\]\]\]", replace_entity, text)


def format_response(candidates, response):
    final_response_formatted = response.replace("%%%", "```").replace("**", "*")
    final_response_formatted = re.sub(r"(\\+n|\n)", "\n ", final_response_formatted)
    all_entities_present = extract_entities_with_id_name_label(candidates)

    try:
        final_response_highlighted = highlight_entity(all_entities_present, final_response_formatted)
    except Exception:
        return final_response_formatted
    return final_response_highlighted


def get_relevant_fks(tables_ids):
    # Build a connected graph by expanding from target tables through FK relationships
    query = f"""
    // Start with target tables and expand outward to find connected tables
    WITH $tables_ids as current_ids

    // Level 1: Find tables connected via FK
    OPTIONAL MATCH (t0:{Labels.TABLE} WHERE t0.id IN current_ids)
          -[:schema]->(:{Labels.COLUMN})-[:fk]-(:{Labels.COLUMN})<-[:schema]-(t1:{Labels.TABLE})
    WITH current_ids, collect(DISTINCT t1.id) as new_ids_1
    WITH current_ids + new_ids_1 as level_1_ids

    // Level 2
    OPTIONAL MATCH (t1:{Labels.TABLE} WHERE t1.id IN level_1_ids)
          -[:schema]->(:{Labels.COLUMN})-[:fk]-(:{Labels.COLUMN})<-[:schema]-(t2:{Labels.TABLE})
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids

    // Level 3
    OPTIONAL MATCH (t2:{Labels.TABLE} WHERE t2.id IN level_2_ids)
          -[:schema]->(:{Labels.COLUMN})-[:fk]-(:{Labels.COLUMN})<-[:schema]-(t3:{Labels.TABLE})
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids

    // Get all FK relationships between these tables
    MATCH (t1:{Labels.TABLE})-[:schema]->(col1:{Labels.COLUMN})-[:fk]-(col2:{Labels.COLUMN})
        <-[:schema]-(t2:{Labels.TABLE})
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id  // Avoid duplicates by keeping only one direction

    RETURN collect(DISTINCT {{
        table1: t1.schema_name + '.' + t1.name,
        column1: col1.name,
        column1_datatype: coalesce(col1.data_type, 'None'),
        table2: t2.schema_name + '.' + t2.name,
        column2: col2.name,
        column2_datatype: coalesce(col2.data_type, 'None')
    }}) as list_of_foreign_keys
    """
    results = get_neo4j_conn().query_read(query, {"tables_ids": tables_ids})
    if len(results) > 0:
        result_fks = results[0]["list_of_foreign_keys"]
    else:
        result_fks = []

    # Build a connected graph by expanding from target tables through FK relationships
    query = f"""
    // Start with target tables and expand outward to find connected tables

    // Level 1: Find tables connected via FK
    OPTIONAL MATCH (t0:{Labels.TABLE} WHERE t0.id IN $tables_ids)-[:join]-(t1:{Labels.TABLE})
    WITH collect(DISTINCT t1.id) as new_ids_1
    WITH $tables_ids + new_ids_1 as level_1_ids

    // Level 2
    OPTIONAL MATCH (t1:{Labels.TABLE} WHERE t1.id IN level_1_ids)-[:join]-(t2:{Labels.TABLE})
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids

    // Level 3
    OPTIONAL MATCH (t2:{Labels.TABLE} WHERE t2.id IN level_2_ids)-[:join]-(t3:{Labels.TABLE})
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids

    // Get all join relationships between these tables and parse the join property
    MATCH (t1:{Labels.TABLE})-[rel:join]-(t2:{Labels.TABLE})
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id  // Avoid duplicates by keeping only one direction
      AND rel.join IS NOT NULL

    // Parse the join property: split by operators and extract left/right sides
    WITH t1, t2, rel,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[0]) as left_side,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[1]) as right_side

    // Parse left side: SCHEMA.TABLE.COLUMN (handle potential whitespace)
    WITH t1, t2, rel, left_side, right_side,
         trim(split(left_side, '.')[0]) as left_schema,
         trim(split(left_side, '.')[1]) as left_table,
         trim(split(left_side, '.')[2]) as left_column,
         trim(split(right_side, '.')[0]) as right_schema,
         trim(split(right_side, '.')[1]) as right_table,
         trim(split(right_side, '.')[2]) as right_column
    WHERE left_schema IS NOT NULL AND left_table IS NOT NULL AND left_column IS NOT NULL
      AND right_schema IS NOT NULL AND right_table IS NOT NULL AND right_column IS NOT NULL

    // Match the actual column nodes for left side
    OPTIONAL MATCH (s1:{Labels.SCHEMA} {{name: left_schema}})
        -[:schema]->(tbl1:{Labels.TABLE} {{name: left_table}})
        -[:schema]->(col1:{Labels.COLUMN} {{name: left_column}})

    // Match the actual column nodes for right side
    OPTIONAL MATCH (s2:{Labels.SCHEMA} {{name: right_schema}})
        -[:schema]->(tbl2:{Labels.TABLE} {{name: right_table}})
        -[:schema]->(col2:{Labels.COLUMN} {{name: right_column}})

    // Return the structured format
    RETURN collect(DISTINCT {{
        table1: t1.schema_name + '.' + t1.name,
        column1: coalesce(col1.name, left_column),
        column1_datatype: coalesce(col1.data_type, 'None'),
        table2: t2.schema_name + '.' + t2.name,
        column2: coalesce(col2.name, right_column),
        column2_datatype: coalesce(col2.data_type, 'None')
    }}) as list_of_foreign_keys
    """
    results = get_neo4j_conn().query_read(query, {"tables_ids": tables_ids})
    if len(results) > 0:
        result_joins = results[0]["list_of_foreign_keys"]
    else:
        result_joins = []
    results = result_fks + result_joins

    # Convert to JSON strings, use set to remove duplicates, then convert back
    unique_strings = set(json.dumps(d, sort_keys=True) for d in results)
    unique_results = [json.loads(s) for s in unique_strings]

    key_order = [
        "table1",
        "column1",
        "column1_datatype",
        "table2",
        "column2",
        "column2_datatype",
    ]
    sorted_results = [{key: d[key] for key in key_order} for d in unique_results]
    return sorted_results


def _parse_table_text(text: str) -> dict:
    """Parse db_name, schema_name, table_name, and columns from LanceDB-style table text."""
    parsed: dict = {}
    try:
        if not isinstance(text, str):
            return parsed

        db_match = re.search(r"db_name:\s*([^,]+)", text)
        if db_match:
            parsed["db_name"] = db_match.group(1).strip()

        schema_match = re.search(r"schema_name:\s*([^,]+)", text)
        if schema_match:
            parsed["schema_name"] = schema_match.group(1).strip()

        table_match = re.search(r"table_name:\s*([^,]+)", text)
        if table_match:
            parsed["table_name"] = table_match.group(1).strip()

        columns_match = re.search(r"columns:\s*(.+)$", text)
        if columns_match:
            columns_str = columns_match.group(1).strip()
            column_pattern = r"\{name:\s*([^,}]+)(?:,\s*data_type:\s*([^,}]+))?(?:,\s*description:\s*([^}]+))?\}"
            columns = []
            for match in re.finditer(column_pattern, columns_str):
                column = {
                    "name": match.group(1).strip(),
                }
                if match.group(2):
                    column["data_type"] = match.group(2).strip()
                if match.group(3):
                    desc = match.group(3).strip()
                    if desc != "null":
                        column["description"] = desc
                columns.append(column)
            if columns:
                parsed["columns"] = columns
    except Exception:
        pass

    return parsed


def get_schemas_from_graph_by_ids(
    relevant_schemas_ids: list | None = None,
) -> list[dict[str, str]]:
    schema_ids = relevant_schemas_ids or []
    query = f"""
    MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(schema:{Labels.SCHEMA})
          -[:{Edges.CONTAINS}]->(table:{Labels.TABLE})
          -[:{Edges.CONTAINS}]->(column:{Labels.COLUMN})
    WHERE size($relevant_schemas_ids) = 0
       OR schema.id IN $relevant_schemas_ids
    RETURN collect({{
        column_name:  column.name,
        table_name:   table.name,
        db_name:      db.name,
        table_schema: schema.name,
        data_type:    column.data_type
    }}) AS data
    """
    result = get_neo4j_conn().query_read(query, {"relevant_schemas_ids": schema_ids})
    if len(result) > 0:
        return result[0]["data"]
    return []


def get_all_schemas_ids():
    query = f"""MATCH(s:{Labels.SCHEMA}) RETURN s.id as schema_id"""
    result = pd.DataFrame(
        get_neo4j_conn().query_read(
            query=query,
            parameters=None,
        )
    )
    return result["schema_id"].tolist()


def get_schemas_by_ids(relevant_schemas_ids: list = None):
    ## This function is for sql validations in the app - (for example metrics or analyses sql validations)
    ## in these cases we get a slim version of the data so the validation is faster
    before_get_all = time.time()
    data_array = get_schemas_from_graph_by_ids(relevant_schemas_ids)
    logger.info(f"time took to get all data from graph: {time.time() - before_get_all}")
    data_df = pd.DataFrame(data_array)
    dbs = list(data_df["db_name"].unique())

    schemas = data_df[["db_name", "table_schema"]]
    schemas = schemas.drop_duplicates().to_dict(orient="records")

    all_schemas = {}
    schema_dfs = {}
    dbs_nodes = {}
    for db_name in dbs:
        db_node = Neo4jNode(name=db_name, label=Labels.DB, props={"name": db_name})
        dbs_nodes[db_name] = db_node

    tables_df = data_df[["db_name", "table_schema", "table_name"]]
    tables_df = tables_df.drop_duplicates()

    unique_schemas = data_df.table_schema.unique()
    for table_schema in unique_schemas:
        schema_tables_df = tables_df.loc[tables_df["table_schema"] == table_schema]
        schema_dfs[table_schema] = {"tables": schema_tables_df.to_dict(orient="records")}

    for table_schema in unique_schemas:
        columns_df = data_df.loc[data_df["table_schema"] == table_schema]
        schema_dfs[table_schema]["columns"] = columns_df.to_dict(orient="records")

    before_modify_all = time.time()
    for schema in schemas:
        table_schema: str = schema.get("table_schema")
        if not table_schema:
            continue

        schema_db_name: str = schema["db_name"]
        schema_db_node = dbs_nodes[schema_db_name]
        tables_df = pd.DataFrame(schema_dfs[table_schema]["tables"])
        columns_df = pd.DataFrame(schema_dfs[table_schema]["columns"])

        all_schemas[table_schema.lower()] = Schema(
            schema_db_node,
            tables_df,
            columns_df,
            table_schema,
            is_creation_mode=False,
        )
    logger.info(f"total time it took to create all schemas nodes: {time.time() - before_modify_all}")
    logger.info(f"total time for get_schemas_by_ids(): {time.time() - before_get_all}")
    return all_schemas


def build_custom_analyses_section(items, candidates):
    """Build a markdown section listing custom analyses that were used."""
    if not items:
        return ""

    # Normalize to attribute access via getattr (fallback to dict.get)
    def _get(obj, key, default=None):
        return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)

    # Map candidate id -> candidate object
    by_id = {_get(c, "id"): c for c in candidates if _get(c, "id")}

    matched_lines = []
    for item in items:
        cid = _get(item, "id")
        candidate = by_id.get(cid)
        if not candidate:  # skip if candidate not found
            continue

        name = _get(candidate, "name", "<unknown name>")
        relevant = _get(item, "classification", False)
        if relevant:
            matched_lines.append(f"- [[[{name}/{cid}]]]")

    # Only add header if there are matched items
    if not matched_lines:
        return ""

    return "\n\n**Semantic items used**:\n" + "\n".join(matched_lines)


def _normalize_table_to_relevant_shape(table: dict) -> dict:
    """Build the same per-table dict shape as :func:`get_relevant_tables` returns."""
    text = str(table.get("table_info") or table.get("text") or "")
    parsed = _parse_table_text(text)
    name = str(table.get("name") or "").strip()
    if not name:
        name = str(parsed.get("table_name") or "").strip()
    entry: dict = {
        "name": name,
        "label": str(table.get("label") or Labels.TABLE),
        "id": str(table.get("id") or ""),
        "table_info": text,
        **parsed,
    }
    if table.get("db_name") and not entry.get("db_name"):
        entry["db_name"] = table["db_name"]
    if table.get("schema_name") and not entry.get("schema_name"):
        entry["schema_name"] = table["schema_name"]
    if table.get("columns") and not entry.get("columns"):
        entry["columns"] = table["columns"]
    if table.get("pk") is not None:
        entry["primary_key"] = table["pk"]
    if not isinstance(entry.get("columns"), list):
        entry["columns"] = []
    return entry


def _merge_two_relevant_table_dicts(a: dict, b: dict) -> dict:
    """Merge two table dicts with the same ``id`` (e.g. Neo4j vs Lance); prefer non-empty / richer fields."""
    out = dict(a)
    for k, v in b.items():
        if v is None:
            continue
        if k == "columns":
            ca = out.get("columns") if isinstance(out.get("columns"), list) else []
            cb = v if isinstance(v, list) else []
            if len(cb) > len(ca):
                out["columns"] = cb
            elif not ca and cb:
                out["columns"] = cb
            continue
        if k in ("table_info", "text"):
            sa = str(out.get(k) or "").strip()
            sb = str(v).strip()
            if len(sb) > len(sa):
                out[k] = v
            elif not sa and sb:
                out[k] = v
            continue
        if k in ("foreign_key", "primary_key"):
            if not out.get(k) and v:
                out[k] = v
            continue
        cur = out.get(k)
        if cur in (None, "") or (isinstance(cur, list) and len(cur) == 0):
            if v not in (None, ""):
                out[k] = v
    return out


def dedupe_merge_relevant_tables(tables: list[dict]) -> list[dict]:
    """Return one dict per table ``id``, merging sparse and rich rows so ``table_info`` / ``columns`` are filled."""
    by_id: dict[str, list[dict]] = {}
    for t in tables:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "").strip()
        if not tid:
            continue
        by_id.setdefault(tid, []).append(t)

    merged: list[dict] = []
    for tid in sorted(by_id.keys()):
        group = by_id[tid]
        acc = dict(group[0])
        for other in group[1:]:
            acc = _merge_two_relevant_table_dicts(acc, other)
        merged.append(_normalize_table_to_relevant_shape(acc))
    return merged


def apply_foreign_key_hints(tables: list[dict], relevant_fks: list) -> None:
    """Set ``foreign_key`` on tables when name matches FK side (same as ``get_relevant_tables``)."""
    for table in tables:
        for fk in relevant_fks:
            if table["name"] == fk["table1"]:
                table["foreign_key"] = f"'{table['name']}.{fk['column1']}' = '{fk['table2']}.{fk['column2']}'"


def get_relevant_fks_from_candidates_tables(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Extract tables and foreign keys from flat candidate dicts.

    Reads ``relevant_tables`` on each candidate (when present), deduplicates by table id,
    calls :func:`get_relevant_fks` for those table ids, then removes
    ``relevant_tables`` from each candidate in place.

    Returns:
        ``(relevant_tables, relevant_fks)`` — same table dict shape as :func:`get_relevant_tables`
        (``name``, ``label``, ``id``, ``table_info``, parsed fields, optional ``primary_key``,
        optional ``foreign_key``), and ``relevant_fks`` as returned by :func:`get_relevant_fks`.
    """
    table_by_id: dict[str, dict] = {}

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        rel = cand.get("relevant_tables")
        if not rel:
            continue
        for table in rel:
            if not isinstance(table, dict):
                continue
            tid = table.get("id")
            if tid is None:
                continue
            tid_s = str(tid)
            if tid_s not in table_by_id:
                table_by_id[tid_s] = table

    def _strip_relevant_tables() -> None:
        for cand in candidates:
            if isinstance(cand, dict) and "relevant_tables" in cand:
                cand.pop("relevant_tables", None)

    if not table_by_id:
        _strip_relevant_tables()
        return [], []

    relevant_tables = [_normalize_table_to_relevant_shape(table_by_id[tid]) for tid in table_by_id]
    relevant_fks: list[dict] = []
    try:
        relevant_fks = get_relevant_fks([x["id"] for x in relevant_tables])
    except Exception:
        logger.exception("get_relevant_fks failed for candidate tables")
        relevant_fks = []

    apply_foreign_key_hints(relevant_tables, relevant_fks)
    _strip_relevant_tables()
    return relevant_tables, relevant_fks


def get_relevant_tables(
    retriever: "Retriever",
    initial_question,
    k=15,
):
    """Semantic search over the same LanceDB index as candidate retrieval, label ``table`` only."""
    try:
        raw_rows = search_lancedb_semantic_index(
            retriever,
            initial_question,
            k=k,
            label_filter=[Labels.TABLE],
        )
    except Exception:
        logger.exception("get_relevant_tables: LanceDB search failed")
        raw_rows = []

    relevant_tables_list = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "")
        name = row.get("name")
        tid = row.get("id")
        lab = row.get("label") or Labels.TABLE
        if name is None and tid is None:
            continue
        entry = _normalize_table_to_relevant_shape(
            {
                "name": name,
                "label": lab,
                "id": tid,
                "text": text,
                "pk": row.get("pk"),
            }
        )
        relevant_tables_list.append(entry)

    relevant_fks: list = []
    if relevant_tables_list:
        try:
            relevant_fks = get_relevant_fks([x["id"] for x in relevant_tables_list])
        except Exception:
            logger.exception("get_relevant_fks failed in get_relevant_tables")
            relevant_fks = []
    apply_foreign_key_hints(relevant_tables_list, relevant_fks)

    return relevant_tables_list, relevant_fks


def prepare_link(name: str, id: str, label: Labels, parent_id: str = None) -> str:
    match label:
        case label if label in [Labels.CUSTOM_ANALYSIS]:
            return f"{label}/{id}|{name}"
        case Labels.COLUMN:
            return f"data/{parent_id}?searchId={id}|{name}"
        case _:
            return f"data/{id}|{name}"


_INFRA_AUTH_ERROR_PATTERNS = [
    # Snowflake
    "insufficient privileges",
    "incorrect username or password",
    "user temporarily locked",
    "user is not found",
    "saml response is invalid",
    "failed to connect",
    "connection refused",
    "connection reset",
    "network is unreachable",
    "no trusted certificate found",
    "ssl peer certificate",
    "ssh remote key",
    "failed to find the root",
    "broken pipe",
    "remote host terminated",
    "target server failed",
    "communication error",
    # MSSQL
    "login failed",
    "cannot open database",
    "user does not have permission",
    "the server was not found or was not accessible",
    "error locating server/instance",
    "could not open a connection",
    "timeout expired",
    "cannot generate sspi context",
    "insufficient system memory",
    "filegroup is full",
    "transaction log for database is full",
    "access denied",
    # Databricks
    "permission_denied",
    "does not have permission",
    "not authorized",
    "user not authorized",
    "invalid access token",
    "unauthorized",
    "forbidden",
    "authentication failed",
    "connection timed out",
    "host not found",
    "no route to host",
    "out of memory",
    "memory limit exceeded",
    "no space left on device",
    "insufficient capacity",
    "quota exceeded",
]

_INFRA_AUTH_RE = re.compile(
    "|".join(re.escape(p) for p in _INFRA_AUTH_ERROR_PATTERNS),
    re.IGNORECASE,
)


def is_infra_or_auth_error(error: Union[Exception, str]) -> bool:
    """Return True if *error* looks like an infrastructure / auth problem
    rather than a SQL logic mistake the agent could fix."""
    msg = str(error)
    logger.error("Error from query execution: %s", msg)

    if re.search(r"^'?Connection [\w-]+ not found'?$", msg, re.IGNORECASE):
        return True

    return bool(_INFRA_AUTH_RE.search(msg))
