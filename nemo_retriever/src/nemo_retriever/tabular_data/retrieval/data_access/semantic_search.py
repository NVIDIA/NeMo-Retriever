# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector semantic search primitives.

The functions here build ``where`` predicates over the JSON ``metadata``
column, run per-label vector queries through the injected
:class:`~nemo_retriever.retriever.Retriever`, and shape the raw hits into a
common ``{text, id, label, score}`` candidate dict consumed by the rest of
the retrieval data-access modules.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)


# Hard ceiling on how many candidate snippets we want to reason over for a single question.
# Larger numbers tend to confuse the LLM and increase latency.
MAX_CALCULATION_CANDIDATES = 15

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels

DEFAULT_FETCH_LIMIT = 20
PER_LABEL_LIMIT = 10
PER_LABEL_LIMITS: dict[str, int] = {
    Labels.COLUMN: 10,
    Labels.CUSTOM_ANALYSIS: 3,
}


def clean_results(raw_candidates: list[dict]) -> list[dict]:
    """Normalize raw semantic hits: require id, dedupe by (label, id), preserve order."""
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


def _parse_hit_metadata(hit: dict) -> dict:
    """Normalize a vector hit's ``metadata`` (dict or JSON string) to a flat dict."""
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
    """Coerce a vector ``_distance`` score (L2) to float; lower is better. Missing → +inf."""
    if distance is None:
        return float("inf")
    try:
        return float(distance)
    except (TypeError, ValueError):
        return float("inf")


def _resolve_label_k(per_label_k: "int | dict[str, int]", label: str | None) -> int:
    """Return the top-k for *label* given a scalar or per-label dict."""
    if isinstance(per_label_k, dict):
        return per_label_k.get(label or "", PER_LABEL_LIMIT)
    return int(per_label_k)


def _escape_like(value: str) -> str:
    """Escape a literal for use inside a LIKE pattern with ``ESCAPE '\\'``."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")


def _build_metadata_where_clause(
    labels: list[str] | None = None,
    database_name: str | None = None,
) -> str | None:
    """Build a SQL ``where`` predicate for the ``metadata`` JSON column.

    Uses ``LIKE`` on the compact-JSON string (no spaces after ``:``) to match
    ``"label":"<value>"`` and ``"database_name":"<value>"`` substrings. Values
    are escaped via :func:`_escape_like` and the predicate declares
    ``ESCAPE '\\'`` so ``%`` / ``_`` / ``\\`` in inputs are treated literally.
    """
    parts: list[str] = []
    if labels:
        label_preds = [f"""metadata LIKE '%"label":"{_escape_like(lab)}"%' ESCAPE '\\'""" for lab in labels]
        parts.append("(" + " OR ".join(label_preds) + ")" if len(label_preds) > 1 else label_preds[0])
    if database_name:
        parts.append(f"""metadata LIKE '%"database_name":"{_escape_like(database_name)}"%' ESCAPE '\\'""")
    return " AND ".join(parts) if parts else None


def _hits_to_semantic_rows(
    hits: list[dict],
    label_filter: set[str] | None = None,
    per_label_k: "int | dict[str, int]" = PER_LABEL_LIMIT,
) -> list[dict]:
    """Turn raw vector hits into candidate dicts, filtering by label in Python.

    Hits are already sorted by vector distance. For each allowed label,
    at most *per_label_k* rows are kept (best-first).  *per_label_k* can
    be a single int (same cap for every label) or a ``{label: k}`` dict.

    ``score`` is the raw vector ``_distance`` (lower is better).
    """
    label_counts: dict[str, int] = {}
    rows: list[dict] = []
    for hit in hits:
        meta = _parse_hit_metadata(hit)
        cid = meta.get("id")
        if cid is None:
            continue
        lab = meta.get("label") if meta.get("label") is not None else hit.get("label")
        lab_str = str(lab) if lab is not None else ""
        if label_filter and lab_str not in label_filter:
            continue
        cnt = label_counts.get(lab_str, 0)
        if cnt >= _resolve_label_k(per_label_k, lab_str):
            continue
        label_counts[lab_str] = cnt + 1
        score = _vector_distance_value(hit.get("_distance"))
        rows.append(
            {
                "text": (hit.get("text") or "").strip(),
                "id": cid,
                "label": lab,
                "score": score,
            }
        )
    return rows


def search_semantic_index(
    retriever: "Retriever",
    entity: str,
    label_filter: list[str] | None = None,
    per_label_k: "int | dict[str, int]" = PER_LABEL_LIMIT,
    database_name: str | None = None,
) -> list[dict]:
    """Vector search via the injected :class:`~nemo_retriever.retriever.Retriever`.

    Runs one query **per label** with a server-side ``where`` predicate on the
    ``metadata`` JSON column (label + database_name), requesting exactly
    the label-specific *k* rows.  *per_label_k* can be a single int or a
    ``{label: k}`` dict (e.g. ``{"Column": 10, "CustomAnalysis": 3}``).
    When no *label_filter* is given, falls back to a single query with
    ``DEFAULT_FETCH_LIMIT``.
    """
    allowed_labels = {str(x) for x in (label_filter or []) if x is not None} or None
    labels_to_query = list(allowed_labels) if allowed_labels else [None]

    all_hits: list[dict] = []
    for label in labels_to_query:
        where_clause = _build_metadata_where_clause(
            labels=[label] if label else None,
            database_name=database_name,
        )
        vdb_kwargs = {"where": where_clause} if where_clause else None
        top_k = _resolve_label_k(per_label_k, label) if where_clause else DEFAULT_FETCH_LIMIT
        hits = retriever.query(entity, top_k=top_k, vdb_kwargs=vdb_kwargs)
        all_hits.extend(hits)

    return _hits_to_semantic_rows(all_hits, label_filter=allowed_labels, per_label_k=per_label_k)
