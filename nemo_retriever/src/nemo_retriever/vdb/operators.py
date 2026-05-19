# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin operators around the nv-ingest-client VDB abstraction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.vdb.adt_vdb import VDB
from nemo_retriever.vdb.factory import get_vdb_op_cls

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.vdb.records import normalize_retrieval_results, to_client_vdb_records
from nemo_retriever.vdb.sidecar_metadata import (
    apply_sidecar_metadata_to_client_batches,
    build_sidecar_lookup,
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


def _coerce_embedding_vector(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        value = value.get("embedding")
    if not isinstance(value, list):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
    if isinstance(value, list) and value:
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    return None


def _is_direct_embedding_column(column_name: object) -> bool:
    name = str(column_name).strip().lower()
    return "embedding" in name or name == "vector" or name.endswith("_vector")


def query_vectors_from_embedded_dataframe(df: pd.DataFrame) -> list[list[float]]:
    """Extract one query vector per row from batch-embed output (metadata or payload columns)."""
    vectors: list[list[float]] = []
    for _, row in df.iterrows():
        vec: list[float] | None = None
        md = row.get("metadata")
        if isinstance(md, dict):
            vec = _coerce_embedding_vector(md)
        if vec is None:
            for col in df.columns:
                if col == "metadata":
                    continue
                val = row.get(col)
                if isinstance(val, dict) or _is_direct_embedding_column(col):
                    vec = _coerce_embedding_vector(val)
                if vec is not None:
                    break
        if vec is None:
            raise ValueError(
                "Expected query embeddings in each row's metadata['embedding'] or a payload column "
                f"with key 'embedding'; columns={list(df.columns)}"
            )
        vectors.append(vec)
    return vectors


class IngestVdbOperator(AbstractOperator):
    """Upload already-embedded graph output through an nv-ingest-client VDB."""

    #: Ray batch mode: repartition to one block and one ``map_batches`` call so
    #: ``VDB.run`` sees the full dataset once (matches historical post-graph upload).
    REQUIRES_GLOBAL_BATCH: bool = True

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(vdb_kwargs or {})
        clean_kwargs, sidecar = split_sidecar_from_vdb_kwargs(merged)
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)
        self._vdb_kwargs = clean_kwargs
        self._sidecar_spec = sidecar
        self._sidecar_lookup: dict[str, dict[str, Any]] | None = None
        if sidecar is not None:
            _df = materialize_sidecar_dataframe(sidecar)
            self._sidecar_lookup = build_sidecar_lookup(
                _df,
                sidecar["meta_source_field"],
                sidecar["meta_fields"],
            )
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        # Compatibility shim: graph_pipeline emits flat embedded rows, while
        # nv-ingest-client VDB.run still expects nested NV-Ingest records.
        records = to_client_vdb_records(data)
        if self._sidecar_spec is not None and self._sidecar_lookup is not None:
            records = apply_sidecar_metadata_to_client_batches(
                records,
                lookup=self._sidecar_lookup,
                meta_fields=self._sidecar_spec["meta_fields"],
                join_key=self._sidecar_spec["meta_join_key"],
            )
        if records and any(batch for batch in records):
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
        explode_for_rerank: bool = False,
    ) -> None:
        merged = dict(vdb_kwargs or {})
        clean_kwargs, _sidecar = split_sidecar_from_vdb_kwargs(merged)
        clean_kwargs.pop("query_texts", None)
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs, explode_for_rerank=explode_for_rerank)
        self._vdb_kwargs = clean_kwargs
        self._retrieval_vdb_kwargs = clean_kwargs
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)
        self._explode_for_rerank = bool(explode_for_rerank)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, pd.DataFrame):
            return query_vectors_from_embedded_dataframe(data)
        return data

    def process(self, data: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        from nemo_retriever.retriever_graph_utils import filter_retrieval_kwargs

        retrieval_kwargs = {**self._retrieval_vdb_kwargs, **filter_retrieval_kwargs(kwargs)}
        if retrieval_kwargs.get("hybrid") and "query_texts" in kwargs:
            retrieval_kwargs["query_texts"] = kwargs["query_texts"]
        return normalize_retrieval_results(self._vdb.retrieval(data, **retrieval_kwargs))

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        if not self._explode_for_rerank:
            return data
        query_texts = kwargs.get("query_texts")
        if not query_texts:
            return data
        from nemo_retriever.retriever_graph_utils import hits_lists_to_rerank_dataframe

        if not isinstance(data, list):
            return data
        return hits_lists_to_rerank_dataframe(list(query_texts), data)
