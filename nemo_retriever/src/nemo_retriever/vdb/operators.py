# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin operators around the nv-ingest-client VDB abstraction."""

from __future__ import annotations

from typing import Any

from nemo_retriever.vdb.adt_vdb import VDB
from nemo_retriever.vdb.factory import get_vdb_op_cls

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.vdb.records import normalize_retrieval_results, to_client_vdb_records
from nemo_retriever.vdb.sidecar_metadata import (
    apply_sidecar_metadata_to_client_batches,
    materialize_sidecar_dataframe,
    split_sidecar_from_vdb_kwargs,
)


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
        merged = dict(vdb_kwargs or {})
        clean_kwargs, sidecar = split_sidecar_from_vdb_kwargs(merged)
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=merged)
        self._vdb_kwargs = merged
        self._sidecar_spec = sidecar
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        # Compatibility shim: graph_pipeline emits flat embedded rows, while
        # nv-ingest-client VDB.run still expects nested NV-Ingest records.
        records = to_client_vdb_records(data)
        if self._sidecar_spec is not None:
            meta_df = materialize_sidecar_dataframe(self._sidecar_spec)
            records = apply_sidecar_metadata_to_client_batches(
                records,
                meta_df=meta_df,
                meta_source_field=self._sidecar_spec["meta_source_field"],
                meta_fields=self._sidecar_spec["meta_fields"],
                join_key=self._sidecar_spec["meta_join_key"],
            )
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
        merged = dict(vdb_kwargs or {})
        clean_kwargs, _sidecar = split_sidecar_from_vdb_kwargs(merged)
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=merged)
        self._vdb_kwargs = merged
        self._retrieval_vdb_kwargs = clean_kwargs
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        retrieval_kwargs = {**self._retrieval_vdb_kwargs, **kwargs}
        return normalize_retrieval_results(self._vdb.retrieval(data, **retrieval_kwargs))

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
