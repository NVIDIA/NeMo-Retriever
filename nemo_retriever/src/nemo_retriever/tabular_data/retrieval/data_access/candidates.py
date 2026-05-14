# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate retrieval: vector hits + Neo4j graph enrichment.

* :func:`_expand_info` pulls graph properties for the (label, id) pairs
  returned by :func:`semantic_search.search_semantic_index`.
* :func:`_get_candidates_information` glues the two together for a single
  question string.
* :func:`extract_candidates` runs the per-entity / per-query-with-values
  fan-out, deduplicates by (label, id) keeping the lowest vector distance,
  and splits into ``(custom_analysis, column)`` streams.
"""

from __future__ import annotations

import logging
from itertools import groupby
from typing import TYPE_CHECKING

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.retrieval.data_access.relevant_tables import (
    _normalize_table_to_relevant_shape,
)
from nemo_retriever.tabular_data.retrieval.data_access.semantic_search import (
    MAX_CALCULATION_CANDIDATES,
    PER_LABEL_LIMIT,
    PER_LABEL_LIMITS,
    search_semantic_index,
)

if TYPE_CHECKING:
    from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)


def _expand_info(ids_and_labels):
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

    allowed_labels = set(Labels.LIST_OF_ALL)
    for label, ids in groupby(
        sorted(items, key=lambda d: str(d.get("label") or "").strip()),
        key=lambda d: str(d.get("label") or "").strip(),
    ):
        label_id_pairs_for_current_label = list(ids)
        if not label:
            continue
        if label not in allowed_labels:
            logger.warning("Skipping unknown label %r in _expand_info", label)
            continue
        query = f"""UNWIND $label_id_pairs as label_id
                    MATCH (n:{label} {{id: label_id.id}})
                    CALL apoc.case([
                        n:{Labels.CUSTOM_ANALYSIS},
                            'OPTIONAL MATCH(n)-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
                            WITH n, head(collect(sql.sql_full_query)) as sql_code, head(collect(sql)) as sql_node
                            OPTIONAL MATCH (sql_node)-[:{Edges.SQL}]->(t:{Labels.TABLE})
                                <-[:{Edges.CONTAINS}]-(schema:{Labels.SCHEMA})
                            WITH n, sql_code,
                                 [x IN collect(
                                     CASE WHEN t IS NOT NULL THEN
                                         apoc.map.merge(
                                             properties(t),
                                             {{label: "{Labels.TABLE}",
                                              schema_name: schema.name,
                                              columns: [(t)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN}) |
                                                  {{name: c.name,
                                                    data_type: toString(coalesce(c.data_type, "")),
                                                    description: CASE
                                                        WHEN c.description IS NOT NULL AND trim(c.description) <> ""
                                                        THEN c.description ELSE null END,
                                                    sample_values: CASE
                                                        WHEN c.sample_values IS NOT NULL AND size(c.sample_values) > 0
                                                        THEN c.sample_values ELSE null END
                                                  }}]
                                             }}
                                         )
                                     ELSE null END
                                 ) WHERE x IS NOT NULL] AS tables
                            RETURN apoc.map.merge(
                                apoc.map.setKey(properties(n), "sql", coalesce(sql_code, "")),
                                {{relevant_tables: tables}}
                            ) as item',
                        n:{Labels.COLUMN},
                            'MATCH(n)<-[:{Edges.CONTAINS}]-(parent)<-[:{Edges.CONTAINS}]-(schema:{Labels.SCHEMA})
                            WITH n, parent, schema,
                                 [(parent)-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN}) |
                                  {{name: c.name,
                                    data_type: toString(coalesce(c.data_type, "")),
                                    description: CASE WHEN c.description IS NOT NULL AND trim(c.description) <> ""
                                                      THEN c.description ELSE null END,
                                    sample_values: CASE WHEN c.sample_values IS NOT NULL AND size(c.sample_values) > 0
                                                        THEN c.sample_values ELSE null END
                                  }}] AS column_list
                            WITH n, parent, schema, column_list,
                                 apoc.map.merge(
                                     properties(parent),
                                     {{label: coalesce(parent.label,
                                      toLower(head(labels(parent))), "{Labels.TABLE}"),
                                      columns: column_list,
                                      schema_name: schema.name}}
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
                        {{n:n, sql_type: $sql_type }}
                        )
                    YIELD value as response
                    WITH collect(response.item) as all_items
                    RETURN apoc.map.groupBy(all_items,'id') as ids_to_props
                    """
        params = {
            "sql_type": Labels.SQL,
            "label_id_pairs": label_id_pairs_for_current_label,
        }
        result = get_neo4j_conn().query_read(
            query=query,
            parameters=params,
        )
        if len(result) > 0:
            results = results | result[0]["ids_to_props"]

    return results


def _get_candidates_information(
    retriever: "Retriever",
    entity: str,
    list_of_semantic: list | None = None,
    database_name: str | None = None,
    per_label_k: "int | dict[str, int]" = PER_LABEL_LIMIT,
):
    """Vector search, then merge graph properties from :func:`_expand_info`.

    Runs one query per label with a server-side ``where`` predicate
    (label + *database_name*) keeping at most *per_label_k* per label,
    then enriches each hit with Neo4j graph properties.
    """
    results: list[dict] = list(
        search_semantic_index(
            retriever,
            entity,
            label_filter=list_of_semantic,
            database_name=database_name,
            per_label_k=per_label_k,
        )
    )

    ids_and_labels = [{"label": x["label"], "id": x["id"]} for x in results]
    props_by_id = _expand_info(ids_and_labels)
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
    query_with_values: str = "",
    database_name: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """One semantic search per pull string (``query_with_values`` and each entity name).

    Each search fetches both custom-analysis and column candidates in a single
    vector-store call, then splits by label in Python.

    Merge streams, dedupe by (label, id) keeping the lowest vector distance
    (``score``), sort ascending by distance, cap at ``MAX_CALCULATION_CANDIDATES``
    per stream.

    Returns:
        ``(custom_analysis_candidates, column_candidates)``
    """
    target_labels = [Labels.CUSTOM_ANALYSIS, Labels.COLUMN]

    pulls: list[str] = []
    if qwv := (query_with_values or "").strip():
        pulls.append(qwv)
    for ent in entities or []:
        if t := (ent or "").strip():
            pulls.append(t)

    combined_custom: list[dict] = []
    combined_columns: list[dict] = []

    for text in pulls:
        hits = (
            _get_candidates_information(
                retriever,
                text,
                list_of_semantic=target_labels,
                database_name=database_name,
                per_label_k=PER_LABEL_LIMITS,
            )
            or []
        )
        for hit in hits:
            lab = str(hit.get("label") or "")
            if lab == Labels.CUSTOM_ANALYSIS:
                combined_custom.append(hit)
            elif lab == Labels.COLUMN:
                combined_columns.append(hit)

    out_custom = _dedupe_best_score_sort_cap(combined_custom)
    out_columns = _dedupe_best_score_sort_cap(combined_columns)

    logger.info(
        f"extract_candidates: {len(out_custom)} custom_analysis, {len(out_columns)} column "
        f"(max {MAX_CALCULATION_CANDIDATES} each), {len(pulls)} pulls"
    )

    return out_custom, out_columns
