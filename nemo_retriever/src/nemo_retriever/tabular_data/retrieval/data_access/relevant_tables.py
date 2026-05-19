# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relevant-table dict shaping and lookup helpers.

The text-to-SQL prompts expect each table to be a flat dict with
``name``, ``label``, ``id``, ``table_info``, ``columns``, optional
``primary_key`` / ``foreign_key`` etc. This module:

* parses the vector hit ``text`` into structured fields,
* normalises a single table dict into the prompt shape, merges duplicates
  with different richness (Neo4j vs vector hits), and
* runs the semantic search restricted to ``Table`` rows.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.retrieval.data_access.semantic_search import (
    search_semantic_index,
)

if TYPE_CHECKING:
    from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)


def _parse_table_text(text: str) -> dict:
    """Parse db_name, schema_name, table_name, and columns from the table-row text."""
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

        desc_match = re.search(r"table_description:\s*([^,]+)", text)
        if desc_match:
            parsed["description"] = desc_match.group(1).strip()

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


def _normalize_table_to_relevant_shape(table: dict) -> dict:
    """Build the same per-table dict shape as :func:`get_relevant_tables` returns."""
    text = str(table.get("table_info") or table.get("text") or "")
    parsed = _parse_table_text(text)
    name = str(table.get("name") or "").strip()
    if not name:
        name = str(parsed.get("table_name") or "").strip()
    entry: dict = {
        "name": name,
        "description": table.get("description") if table.get("description") else "",
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
    """Merge two table dicts with the same ``id`` (e.g. Neo4j vs vector); prefer non-empty / richer fields."""
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


def get_relevant_tables_from_candidates(
    candidates: list[dict],
) -> list[dict]:
    """Extract relevant tables from flat candidate dicts.

    Reads ``relevant_tables`` on each candidate (when present), deduplicates by table id,
    then removes ``relevant_tables`` from each candidate in place.

    Returns:
        List of normalized table dicts — same shape as :func:`get_relevant_tables`
        (``name``, ``label``, ``id``, ``table_info``, parsed fields, optional ``primary_key``).
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

    for cand in candidates:
        if isinstance(cand, dict) and "relevant_tables" in cand:
            cand.pop("relevant_tables", None)

    if not table_by_id:
        return []

    return [_normalize_table_to_relevant_shape(table_by_id[tid]) for tid in table_by_id]


def get_relevant_tables(
    retriever: "Retriever",
    initial_question,
    k: int | None = None,
    database_name: str | None = None,
) -> list[dict]:
    """Semantic search over the same vector index as candidate retrieval, label ``table`` only."""
    try:
        raw_rows = search_semantic_index(
            retriever,
            initial_question,
            label_filter=[Labels.TABLE],
            per_label_k=k,
            database_name=database_name,
        )
    except Exception:
        logger.exception("get_relevant_tables: vector search failed")
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

    return relevant_tables_list
