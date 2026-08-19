# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy resolution of concrete VDB operator classes (avoids import cycles).

Built-in backends are registered in :data:`_BUILTIN_VDB_OPS`. Third-party
packages register additional backends via the ``nemo_retriever.vdb_operators``
setuptools entry-point group::

    [project.entry-points."nemo_retriever.vdb_operators"]
    opensearch = "my_company.vdb.opensearch:OpenSearchVDB"

The worker pod must have the registering package installed; service-mode
clients select a backend with ``.vdb_upload(vdb_op="opensearch", ...)`` when
the operator has allowlisted that key in ``retriever-service.yaml``.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from nemo_retriever.common.vdb.adt_vdb import VDB

_ENTRY_POINT_GROUP = "nemo_retriever.vdb_operators"

# Keys checked (in order) when validating a remote connection URI for
# service-mode ``vdb_upload_params.vdb_kwargs``.
VDB_CONNECTION_URI_KEYS: tuple[str, ...] = (
    "lancedb_uri",
    "uri",
    "endpoint_url",
    "host",
    "url",
)

_REGISTRY_CACHE: dict[str, type[VDB]] | None = None


def extract_vdb_connection_uri(vdb_kwargs: dict[str, Any] | None) -> str | None:
    """Return the first non-empty connection URI/host field from *vdb_kwargs*."""
    if not vdb_kwargs:
        return None
    for key in VDB_CONNECTION_URI_KEYS:
        value = vdb_kwargs.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _load_builtin_vdb_op(name: str) -> type[VDB]:
    if name == "lancedb":
        from nemo_retriever.common.vdb.lancedb import LanceDB

        return LanceDB
    raise KeyError(name)


def _load_vdb_op_registry() -> dict[str, type[VDB]]:
    """Build the merged built-in + entry-point registry (cached)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE

    registry: dict[str, type[VDB]] = {}
    registry["lancedb"] = _load_builtin_vdb_op("lancedb")

    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        if ep.name in registry:
            raise ValueError(
                f"Duplicate vdb_op entry point {ep.name!r}: built-in backends "
                f"and entry points must use distinct names."
            )
        cls = ep.load()
        if not isinstance(cls, type) or not issubclass(cls, VDB):
            raise TypeError(
                f"Entry point {ep.name!r} in group {_ENTRY_POINT_GROUP!r} must "
                f"resolve to a VDB subclass; got {cls!r}."
            )
        registry[ep.name] = cls

    _REGISTRY_CACHE = registry
    return registry


def clear_vdb_op_registry_cache() -> None:
    """Drop the cached registry (for tests and hot reload)."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None


def list_vdb_ops() -> list[str]:
    """Return sorted registered ``vdb_op`` keys (built-ins + entry points)."""
    return sorted(_load_vdb_op_registry())


def get_vdb_op_cls(vdb_op: str) -> type[VDB]:
    """Return the concrete ``VDB`` subclass for *vdb_op* or raise ``ValueError``."""
    registry = _load_vdb_op_registry()
    try:
        return registry[vdb_op]
    except KeyError:
        available = list_vdb_ops()
        raise ValueError(f"Invalid vdb_op: {vdb_op}. Available vdb_ops - {available}.")
