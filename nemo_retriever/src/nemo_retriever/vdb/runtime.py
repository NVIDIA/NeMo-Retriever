# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime observability helpers for VDB-backed retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "bearer",
        "credential",
        "password",
        "secret",
        "token",
    }
)
_EXECUTION_ONLY_CONFIG_KEYS = frozenset({"query_texts"})


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _sanitize_config_value(k, v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in value]
    return str(value)


def _sanitize_config_value(key: object, value: Any) -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    return _json_safe(value)


def _sanitize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _sanitize_config_value(key, value)
        for key, value in config.items()
        if str(key).strip().lower() not in _EXECUTION_ONLY_CONFIG_KEYS
    }


def _first_present(config: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return None


def _target_summary(vdb_op: str, config: Mapping[str, Any]) -> dict[str, Any]:
    target: dict[str, Any] = {}

    uri = _first_present(config, "uri", "lancedb_uri")
    table_name = _first_present(config, "table_name", "lancedb_table")
    collection_name = _first_present(config, "collection_name", "index_name")

    if uri is None and vdb_op == "lancedb":
        uri = "lancedb"
    if table_name is None and vdb_op == "lancedb":
        table_name = "nv-ingest"

    if uri is not None:
        target["uri"] = _json_safe(uri)
    if table_name is not None:
        target["table_name"] = _json_safe(table_name)
    if collection_name is not None and collection_name != table_name:
        target["collection_name"] = _json_safe(collection_name)

    return target


def _retrieval_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    hybrid = bool(config.get("hybrid", False))
    summary: dict[str, Any] = {
        "mode": "hybrid" if hybrid else "dense",
        "signals": ["dense_vector", "lexical_text"] if hybrid else ["dense_vector"],
        "uses_query_texts": hybrid,
    }

    top_k = _first_present(config, "top_k")
    refine_factor = _first_present(config, "refine_factor")
    nprobes = _first_present(config, "n_probe", "nprobes")

    if top_k is not None:
        summary["top_k"] = _json_safe(top_k)
    if refine_factor is not None:
        summary["refine_factor"] = _json_safe(refine_factor)
    if nprobes is not None:
        summary["nprobes"] = _json_safe(nprobes)

    search_kwargs = config.get("search_kwargs")
    if isinstance(search_kwargs, Mapping):
        summary["search_kwargs"] = _sanitize_config(search_kwargs)

    return summary


def describe_vdb_runtime(vdb_op: str, vdb_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a JSON-safe, VDB-neutral runtime summary.

    The summary records the selected ADT VDB backend, target, normalized
    retrieval mode, and sanitized config. It deliberately does not inspect a
    live database; backend-specific health/index details can be layered in by
    future helpers without changing this base contract.
    """
    op = str(vdb_op or "").strip().lower() or "unknown"
    config = dict(vdb_kwargs or {})

    return {
        "op": op,
        "target": _target_summary(op, config),
        "retrieval": _retrieval_summary(config),
        "config": _sanitize_config(config),
    }
