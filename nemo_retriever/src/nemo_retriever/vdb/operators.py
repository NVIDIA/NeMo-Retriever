# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin operators around the nv-ingest-client VDB abstraction."""

from __future__ import annotations

from typing import Any

from nv_ingest_client.util.vdb import VDB, get_vdb_op_cls

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.vdb.records import normalize_retrieval_results, to_client_vdb_records


def _construct_vdb(
    *,
    vdb: VDB | None = None,
    vdb_op: str | None = None,
    vdb_kwargs: dict[str, Any] | None = None,
) -> VDB:
    if vdb is not None and vdb_op is not None:
        raise ValueError("Pass either vdb or vdb_op, not both.")
    if vdb is None and vdb_op is None:
        raise ValueError("Either vdb or vdb_op is required.")

    return vdb if vdb is not None else get_vdb_op_cls(str(vdb_op))(**dict(vdb_kwargs or {}))


class IngestVdbOperator(AbstractOperator):
    """Upload already-embedded graph output through an nv-ingest-client VDB."""

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        vdb_kwargs = dict(vdb_kwargs or {})
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=vdb_kwargs)
        self._vdb_kwargs = vdb_kwargs
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=vdb_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        records = to_client_vdb_records(data)
        self._vdb.run(records)
        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class RetrieveVdbOperator(AbstractOperator):
    """Retrieve hits from an nv-ingest-client VDB using precomputed query vectors."""

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        vdb_kwargs = dict(vdb_kwargs or {})
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=vdb_kwargs)
        self._vdb_kwargs = vdb_kwargs
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=vdb_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        retrieval_kwargs = {**self._vdb_kwargs, **kwargs}
        return normalize_retrieval_results(self._vdb.retrieval(data, **retrieval_kwargs))

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
