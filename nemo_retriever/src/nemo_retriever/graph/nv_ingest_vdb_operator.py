# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin graph adapter for the nv-ingest-client VDB abstraction."""

from __future__ import annotations

from typing import Any

from nv_ingest_client.util.vdb import VDB, get_vdb_op_cls

from nemo_retriever.graph.abstract_operator import AbstractOperator


class NvIngestVdbOperator(AbstractOperator):
    """Delegate graph batches to an existing nv-ingest-client VDB implementation."""

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
        phase: str = "ingest",
    ) -> None:
        if vdb is not None and vdb_op is not None:
            raise ValueError("Pass either vdb or vdb_op, not both.")
        if vdb is None and vdb_op is None:
            raise ValueError("Either vdb or vdb_op is required.")

        vdb_kwargs = dict(vdb_kwargs or {})
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=vdb_kwargs, phase=phase)

        self._vdb = vdb if vdb is not None else get_vdb_op_cls(vdb_op)(**vdb_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if self.phase == "ingest":
            return self._vdb.run(data)
        if self.phase == "retrieve":
            return self._vdb.retrieval(data, **kwargs)
        raise ValueError(f"Unsupported VDB phase: {self.phase!r}.")

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


__all__ = ["NvIngestVdbOperator"]
