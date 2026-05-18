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
from nemo_retriever.vdb.records import (
    _client_record_from_graph_row,
    normalize_retrieval_results,
    to_client_vdb_records,
)
from nemo_retriever.vdb.sidecar_metadata import (
    apply_sidecar_metadata_to_client_batches,
    build_sidecar_lookup,
    materialize_sidecar_dataframe,
    split_sidecar_from_vdb_kwargs,
)


#: Per-row boolean column emitted by ``IngestVdbOperator`` indicating whether
#: this row produced a client-VDB record. Read by
#: ``IngestResult.count_uploadable_vdb_records`` so the count survives the
#: projection (which drops the embedding/text that the validity check needed).
VDB_UPLOADABLE_COLUMN = "_vdb_uploadable"


#: Whitelist of accounting columns kept in the IngestVdbOperator output.
#: Every other column (embedding floats, nested metadata, full chunk text,
#: raw PDF bytes, page images, …) is dropped before the row travels through
#: the downstream global-batch barrier — a whitelist beats a blacklist here
#: because any future schema addition is heavy-by-default and would otherwise
#: silently re-fill plasma + driver memory at the end of the run.
#:
#: - ``source_id`` / ``source_path`` / ``path``: driver-side ``unique_source_count``
#:   and detection-summary key.
#: - ``page_number``: detection-summary key.
#: - ``VDB_UPLOADABLE_COLUMN``: per-row uploadable flag (see above).
_ACCOUNTING_COLUMNS = frozenset({"source_id", "source_path", "path", "page_number", VDB_UPLOADABLE_COLUMN})


def _project_to_accounting_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    keep = [col for col in df.columns if col in _ACCOUNTING_COLUMNS]
    return df[keep] if len(keep) != len(df.columns) else df


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
    """Stream already-embedded graph output into a VDB.

    Each call to `process` writes a single Ray Data block (in batch mode) or
    the full DataFrame (inprocess mode) via `VDB.append`. The first call uses
    ``overwrite=True`` to drop any pre-existing table; subsequent calls append.
    Index construction is deferred to `GraphIngestor._finalize_vdb_upload`.
    """

    #: No global-batch repartition: peak memory is bounded by the batch size,
    #: not the corpus. Requires concurrency=1 (pinned in ``ingestor_runtime``).
    REQUIRES_GLOBAL_BATCH: bool = False

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
        self._wrote_first_batch = False

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        # graph_pipeline emits flat embedded rows; nv-ingest-client VDBs expect
        # the nested record shape from ``to_client_vdb_records``.
        if isinstance(data, pd.DataFrame):
            # Compute per-row validity in lock-step with record conversion so
            # we can emit a small ``_vdb_uploadable`` flag column. Without
            # this, the downstream global-batch barrier accumulates the full
            # embedded payload in Ray object store and starves the upstream
            # pipeline.
            valid_records: list[dict[str, Any]] = []
            uploadable_mask: list[bool] = []
            for row in data.to_dict("records"):
                rec = _client_record_from_graph_row(row)
                if rec is None:
                    uploadable_mask.append(False)
                else:
                    valid_records.append(rec)
                    uploadable_mask.append(True)
            records: list[list[dict[str, Any]]] = [valid_records] if valid_records else []
        else:
            records = to_client_vdb_records(data)
            uploadable_mask = []

        if self._sidecar_spec is not None and self._sidecar_lookup is not None:
            records = apply_sidecar_metadata_to_client_batches(
                records,
                lookup=self._sidecar_lookup,
                meta_fields=self._sidecar_spec["meta_fields"],
                join_key=self._sidecar_spec["meta_join_key"],
            )
        if records and any(batch for batch in records):
            # ``--append`` mode (``vdb.overwrite=False``) must preserve any table
            # left behind by an earlier run. Only the first batch of an
            # ``overwrite=True`` run is allowed to drop the existing table; every
            # other batch — and every batch in append mode — uses ``table.add``.
            vdb_overwrite = bool(getattr(self._vdb, "overwrite", True))
            overwrite_this_batch = vdb_overwrite and not self._wrote_first_batch
            # ``append`` returns False when the VDB filtered every record out
            # (e.g. wrong embedding length) and didn't actually write. In that
            # case the next batch must keep ``overwrite=True`` — otherwise we'd
            # call ``table.add`` on a table that was never created.
            if self._vdb.append(records, overwrite=overwrite_this_batch):
                self._wrote_first_batch = True

        if isinstance(data, pd.DataFrame):
            # ``.assign`` returns a shallow-copied frame, so the operator's
            # input ``data`` is not mutated and the projection result is a
            # fresh DataFrame with the uploadable flag attached.
            return _project_to_accounting_columns(data).assign(**{VDB_UPLOADABLE_COLUMN: uploadable_mask})
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
