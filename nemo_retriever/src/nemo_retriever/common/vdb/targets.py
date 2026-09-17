# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral description of the index a CLI or SDK workflow reads and writes."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, get_args

from nemo_retriever.common.vdb.adt_vdb import VDB
from nemo_retriever.common.vdb.factory import get_vdb_op_cls

VdbOpValue = Literal["lancedb", "qdrant"]
SUPPORTED_VDB_OPS: tuple[VdbOpValue, ...] = get_args(VdbOpValue)
DEFAULT_QDRANT_URL = "http://localhost:6333"


def qdrant_api_key_from_env() -> str | None:
    """Read the Qdrant API key from ``QDRANT_API_KEY`` or the file named by ``QDRANT_API_KEY_FILE``."""
    key = os.environ.get("QDRANT_API_KEY")
    if not key and (key_file := os.environ.get("QDRANT_API_KEY_FILE")):
        key = Path(key_file).read_text(encoding="utf-8")
    return (key or "").strip() or None


@dataclass(frozen=True)
class VdbTarget:
    """A LanceDB table or Qdrant collection. ``table_name`` names either."""

    vdb_op: VdbOpValue = "lancedb"
    lancedb_uri: str = "lancedb"
    table_name: str = "nemo-retriever"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    def __post_init__(self) -> None:
        vdb_op = str(self.vdb_op).strip().lower()
        if vdb_op not in SUPPORTED_VDB_OPS:
            raise ValueError(f"vdb_op must be one of {', '.join(SUPPORTED_VDB_OPS)}, got {self.vdb_op!r}.")
        object.__setattr__(self, "vdb_op", vdb_op)

    @classmethod
    def from_options(cls, options: Any) -> VdbTarget:
        """Build a target from any object with the same field names, such as CLI storage options."""
        return cls(**{field.name: getattr(options, field.name) for field in fields(cls)})

    @property
    def backend_name(self) -> str:
        return "Qdrant" if self.vdb_op == "qdrant" else "LanceDB"

    @property
    def index_noun(self) -> str:
        return "collection" if self.vdb_op == "qdrant" else "table"

    @property
    def location(self) -> str:
        return self.lancedb_uri if self.vdb_op == "lancedb" else (self.qdrant_url or DEFAULT_QDRANT_URL)

    def vdb_kwargs(self) -> dict[str, Any]:
        if self.vdb_op == "lancedb":
            return {"uri": self.lancedb_uri, "table_name": self.table_name}
        kwargs: dict[str, Any] = {"collection_name": self.table_name}
        if self.qdrant_url:
            kwargs["url"] = self.qdrant_url
        if self.qdrant_api_key:
            kwargs["api_key"] = self.qdrant_api_key
        return kwargs

    def describe(self) -> str:
        """Describe the target without credentials, e.g. ``LanceDB lancedb/nemo-retriever``."""
        return f"{self.backend_name} {self.location}/{self.table_name}"

    def backend(self, **kwargs: Any) -> VDB:
        return get_vdb_op_cls(self.vdb_op)(**{**self.vdb_kwargs(), **kwargs})
