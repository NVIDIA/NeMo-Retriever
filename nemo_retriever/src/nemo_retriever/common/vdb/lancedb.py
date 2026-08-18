# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import threading
import time

from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any, Final, FrozenSet

import lancedb
import pyarrow as pa
import pyarrow.compute as pc

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
)
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    VDB,
)
from nemo_retriever.common.vdb.lancedb_capabilities import (
    inspect_lancedb_table_object,
    wait_for_column_index,
)
from nemo_retriever.common.vdb.lancedb_schema import (
    build_lancedb_row,
    infer_vector_dim,
    lancedb_schema,
    normalize_content_type,
)

logger = logging.getLogger(__name__)


_DEFAULT_VECTOR_DIM: Final[int] = 2048
_VALID_ON_BAD_VECTORS: Final[FrozenSet[str]] = frozenset({"drop", "fill", "null", "error"})
_RETRIEVAL_MODE_METADATA_KEY: Final[bytes] = b"retrieval_mode"
_NEMO_RETRIEVER_RETRIEVAL_MODE_METADATA_KEY: Final[bytes] = b"nemo_retriever.retrieval_mode"
_EMBEDDING_MODEL_METADATA_KEY: Final[bytes] = b"nemo_retriever.embedding_model_name"
_EMBEDDING_MODEL_REVISION_METADATA_KEY: Final[bytes] = b"nemo_retriever.embedding_model_revision"


def _normalize_on_bad_vectors(value: str) -> str:
    """Validate and normalize an ``on_bad_vectors`` policy string.

    LanceDB's ``Table.create`` accepts a fixed set of policies for handling rows
    whose vector column does not match the declared fixed-size schema. We
    surface the same vocabulary on this wrapper for direct SDK configuration.

    Args:
        value: User-supplied policy name. Whitespace and case are ignored.

    Returns:
        The normalized lower-case policy string.

    Raises:
        ValueError: If ``value`` is not one of ``drop``, ``fill``, ``null``,
            or ``error``.
    """
    normalized = (value or "drop").strip().lower()
    if normalized not in _VALID_ON_BAD_VECTORS:
        raise ValueError(f"on_bad_vectors must be one of {sorted(_VALID_ON_BAD_VECTORS)}; got {value!r}")
    return normalized


def _json_str(value) -> str:
    """
    Convert Python objects (dict/list/etc.) to a compact JSON string.

    LanceDB table schema stores `metadata` and `source` as strings, so we must
    serialize nested structures before ingestion.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def _maybe_parse_json(value):
    """Best-effort parse for JSON-serialized string columns."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return {}
    # Avoid accidental parsing of plain strings that are not JSON objects/arrays.
    if not (s.startswith("{") or s.startswith("[")):
        return value
    try:
        return json.loads(s)
    except Exception:
        return value


def _is_ivf_vector_index(index_type: object) -> bool:
    """Return True if ``index_type`` names an IVF-style index (K-means partitions)."""
    s = str(index_type or "").upper()
    return s.startswith("IVF") or "IVF_" in s


def _effective_ivf_num_partitions(num_rows: int, requested: int) -> int | None:
    """Compute a valid ``num_partitions`` for Lance IVF training.

    K-means centroids must be strictly fewer than the number of training vectors
    (``num_partitions < num_rows``). For empty or single-row tables, IVF
    training is skipped (return ``None``).

    Args:
        num_rows: Row count in the table.
        requested: Caller-configured ``num_partitions``.

    Returns:
        Clamped partition count, or ``None`` if vector index build should be skipped.
    """
    if num_rows <= 0:
        return None
    if num_rows == 1:
        return None
    cap = num_rows - 1
    return min(int(requested), max(1, cap))


def _with_retrieval_mode_metadata(
    schema: pa.Schema,
    retrieval_mode: str | None,
    embedding_model_name: str | None = None,
    embedding_model_revision: str | None = None,
) -> pa.Schema:
    if retrieval_mode is None:
        return schema
    metadata = dict(schema.metadata or {})
    encoded_mode = str(retrieval_mode).encode("utf-8")
    metadata[_RETRIEVAL_MODE_METADATA_KEY] = encoded_mode
    metadata[_NEMO_RETRIEVER_RETRIEVAL_MODE_METADATA_KEY] = encoded_mode
    if embedding_model_name:
        metadata[_EMBEDDING_MODEL_METADATA_KEY] = embedding_model_name.encode("utf-8")
    if embedding_model_revision:
        metadata[_EMBEDDING_MODEL_REVISION_METADATA_KEY] = embedding_model_revision.encode("utf-8")
    return schema.with_metadata(metadata)


def _lancedb_arrow_schema(
    vector_dim: int,
    *,
    retrieval_mode: str | None = None,
    embedding_model_name: str | None = None,
    embedding_model_revision: str | None = None,
) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), int(vector_dim))),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ]
    )
    return _with_retrieval_mode_metadata(
        schema,
        retrieval_mode,
        embedding_model_name,
        embedding_model_revision,
    )


def _sparse_lancedb_arrow_schema(*, retrieval_mode: str | None = "sparse") -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ]
    )
    return _with_retrieval_mode_metadata(schema, retrieval_mode)


def _table_schema(table: Any) -> pa.Schema:
    schema = table.schema
    return schema() if callable(schema) else schema


def _schema_vector_dim(schema: pa.Schema) -> int | None:
    """Return a fixed vector width from a LanceDB table schema when present."""
    try:
        vector_type = schema.field("vector").type
    except KeyError:
        return None
    if pa.types.is_fixed_size_list(vector_type):
        return int(vector_type.list_size)
    return None


def lancedb_row_count(uri: str, table_name: str) -> int:
    """Return the number of rows in a LanceDB table."""
    table = lancedb.connect(uri).open_table(table_name)
    return int(table.count_rows())


def _validate_append_schema(table: Any, expected_schema: pa.Schema, *, table_name: str, uri: str) -> None:
    """Fail before append when an existing table cannot accept this writer's rows."""
    existing_schema = _table_schema(table)
    existing_fields = {field.name: field for field in existing_schema}

    for expected_field in expected_schema:
        existing_field = existing_fields.get(expected_field.name)
        if existing_field is None:
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} is missing required field "
                f"{expected_field.name!r}; use overwrite=True to replace the table."
            )
        if existing_field.type != expected_field.type:
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} has incompatible field "
                f"{expected_field.name!r}: got {existing_field.type}, expected {expected_field.type}; "
                "use overwrite=True to replace the table."
            )


def _validate_append_embedding_model(
    table: Any,
    embedding_model_name: str | None,
    embedding_model_revision: str | None,
    *,
    table_name: str,
    uri: str,
) -> None:
    """Reject appends that would mix known embedding models in one table."""
    if not embedding_model_name:
        return

    metadata = _table_schema(table).metadata or {}
    stored_value = metadata.get(_EMBEDDING_MODEL_METADATA_KEY)
    if stored_value is None:
        return

    stored_model = stored_value.decode("utf-8", errors="replace").strip()
    if stored_model and stored_model != embedding_model_name:
        raise ValueError(
            f"LanceDB table {table_name!r} at {uri!r} uses embedding model {stored_model!r}; "
            f"cannot append vectors from {embedding_model_name!r}. Use the table model or overwrite the table."
        )

    stored_revision_value = metadata.get(_EMBEDDING_MODEL_REVISION_METADATA_KEY)
    if stored_revision_value is None:
        return
    stored_revision = stored_revision_value.decode("utf-8", errors="replace").strip()
    if stored_revision and not embedding_model_revision:
        raise ValueError(
            f"LanceDB table {table_name!r} at {uri!r} uses embedding model revision {stored_revision!r}; "
            "cannot append vectors without a known revision. Use the table revision or overwrite the table."
        )
    if stored_revision and stored_revision != embedding_model_revision:
        raise ValueError(
            f"LanceDB table {table_name!r} at {uri!r} uses embedding model revision {stored_revision!r}; "
            f"cannot append vectors from revision {embedding_model_revision!r}. "
            "Use the table revision or overwrite the table."
        )


def _is_missing_lancedb_table_error(exc: ValueError) -> bool:
    return "was not found" in str(exc)


def _record_timing(event: str, duration_s: float, extra: dict | None = None):
    timing_path = os.getenv("NV_INGEST_LANCEDB_TIMING_PATH")
    if not timing_path:
        return
    payload = {
        "event": event,
        "duration_s": duration_s,
        "timestamp_s": time.time(),
    }
    if extra:
        payload.update(extra)
    timing_dir = os.path.dirname(timing_path)
    if timing_dir:
        os.makedirs(timing_dir, exist_ok=True)
    with open(timing_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _get_text_for_element(element):
    """
    Extract searchable text from an element based on document_type.

    This prevents base64-encoded images from being stored in the text field.
    """
    doc_type = element.get("document_type")
    metadata = element.get("metadata", {})

    if doc_type == "text":
        return metadata.get("content")
    elif doc_type == "structured":
        # Tables, charts, infographics
        table_meta = metadata.get("table_metadata", {})
        return table_meta.get("table_content")
    elif doc_type == "image":
        # Use caption/OCR text, not raw base64 image data
        image_meta = metadata.get("image_metadata", {})
        content_meta = metadata.get("content_metadata", {})
        if content_meta.get("subtype") == "page_image":
            return image_meta.get("text")
        else:
            return image_meta.get("caption")
    elif doc_type == "audio":
        audio_meta = metadata.get("audio_metadata", {})
        return audio_meta.get("audio_transcript")
    else:
        # Fallback for unknown types
        return metadata.get("content")


def _create_lancedb_results(
    results,
    *,
    expected_dim: int | None = _DEFAULT_VECTOR_DIM,
) -> tuple[list, dict[str, int]]:
    """Transform Nemo Retriever Library (NRL) pipeline results into LanceDB ingestible rows.

    Extracts the appropriate searchable text per ``document_type`` and, when
    ``expected_dim`` is set, validates that each row's embedding is shaped
    consistently with the LanceDB fixed-size-list schema before forwarding it
    to the writer. Canonical image records may use ``text=""`` when both
    ``document_type`` and ``content_metadata.type`` are ``"image"``; all other
    dense records still require text. The graph adapter owns image-backing
    validation and emits that normalized record shape.
    Rows whose embedding is missing, of the wrong type, or of the wrong length
    are dropped and counted; per-row reasons are emitted at ``DEBUG`` and a
    single structured ``WARNING`` summary is emitted at the end of the call
    when any drops occurred.

    Passing ``expected_dim=None`` disables the length check entirely. Callers
    that prefer to defer to LanceDB's ``on_bad_vectors`` policy on the writer
    side (e.g. ``LanceDB(on_bad_vectors="error")``) should use this mode so
    bad rows reach LanceDB rather than being silently dropped at the wrapper.

    Args:
        results: Iterable of pipeline output result lists, where each element
            is a per-document list of NRL record dicts.
        expected_dim: Required vector length, or ``None`` to skip the length
            check. Defaults to :data:`_DEFAULT_VECTOR_DIM`.

    Returns:
        ``(rows, counts)`` where ``rows`` is the list of dicts shaped for
        LanceDB ingestion (``vector``, ``text``, ``metadata``, ``source``)
        and ``counts`` is a dict containing ``accepted``,
        ``dropped_no_embedding``, ``dropped_bad_length``, and
        ``dropped_no_text`` keys.
        Also ``empty_embedding``, added by the incomplete-index guard.

    An empty embedding, ``[]``, means the embed stage produced no vector for that
    row. Such rows are counted as ``empty_embedding`` and make this function
    raise :class:`RuntimeError` after the loop, so no table is written.

    :meth:`LanceDB.create_index` calls this function twice when ``vector_dim`` is
    ``None``: pass 1 infers the dimension and its rows are discarded, pass 2
    filters. The check ignores ``expected_dim``, so it raises in pass 1 and pass 2
    never runs - which is why the error reports ``expected_dim=None``.
    """
    lancedb_rows: list = []
    accepted = 0
    dropped_no_embedding = 0
    empty_embedding = 0
    dropped_bad_length = 0
    dropped_no_text = 0

    enforce_length = expected_dim is not None
    expected_dim_int = int(expected_dim) if enforce_length else None

    for result in results:
        for element in result:
            metadata = element.get("metadata", {})
            doc_type = element.get("document_type")

            embedding = metadata.get("embedding")
            if embedding is None:
                dropped_no_embedding += 1
                continue

            # ``[]`` is not ``None``, so the branch above misses it. Explicit
            # isinstance/len, never ``not embedding``, which raises on an array.
            if isinstance(embedding, (list, tuple)) and len(embedding) == 0:
                empty_embedding += 1
                logger.debug("Dropping row with an empty embedding (doc_type=%s)", doc_type)
                continue

            if enforce_length and (not isinstance(embedding, (list, tuple)) or len(embedding) != expected_dim_int):
                dropped_bad_length += 1
                got_len: Any = len(embedding) if hasattr(embedding, "__len__") else "n/a"
                logger.debug(
                    "Dropping row with bad embedding (got_len=%s, expected=%d, doc_type=%s)",
                    got_len,
                    expected_dim_int,
                    doc_type,
                )
                continue

            content_meta = metadata.get("content_metadata", {})

            text = _get_text_for_element(element)

            if not isinstance(text, str) or not text.strip():
                is_canonical_image = (
                    doc_type == "image" and isinstance(content_meta, dict) and content_meta.get("type") == "image"
                )
                if not is_canonical_image:
                    dropped_no_text += 1
                    source_name = metadata.get("source_metadata", {}).get("source_name", "unknown")
                    pg_num = content_meta.get("page_number")
                    logger.debug(f"No text found for entity: {source_name} page: {pg_num} type: {doc_type}")
                    continue
                text = ""

            row_id = content_meta.get("id") if isinstance(content_meta, dict) else None
            if row_id is None and isinstance(metadata, dict):
                row_id = metadata.get("id")
            row_id_str = str(row_id) if row_id is not None else ""

            lancedb_rows.append(
                {
                    "vector": embedding,
                    "text": text,
                    "metadata": _json_str(content_meta),
                    "source": _json_str(metadata.get("source_metadata", {})),
                    "id": row_id_str,
                }
            )
            accepted += 1

    counts: dict[str, int] = {
        "accepted": accepted,
        "dropped_no_embedding": dropped_no_embedding,
        "empty_embedding": empty_embedding,
        "dropped_bad_length": dropped_bad_length,
        "dropped_no_text": dropped_no_text,
    }

    if dropped_no_embedding or dropped_bad_length or dropped_no_text:
        expected_dim_repr = expected_dim_int if enforce_length else "None"
        logger.warning(
            "_create_lancedb_results: accepted=%d dropped_no_embedding=%d "
            "dropped_bad_length=%d dropped_no_text=%d expected_dim=%s",
            accepted,
            dropped_no_embedding,
            dropped_bad_length,
            dropped_no_text,
            expected_dim_repr,
        )

    if empty_embedding:
        logger.warning(
            "_create_lancedb_results: empty_embedding=%d",
            empty_embedding,
        )
        total = accepted + dropped_no_embedding + empty_embedding + dropped_bad_length + dropped_no_text
        raise RuntimeError(
            "Refusing to build an incomplete index: "
            f"{empty_embedding} of {total} rows had no embedding. No table is written and "
            "this run fails; the alternative is an index that is silently short by those "
            "rows while the run reports success. "
            f"Counters: empty_embedding={empty_embedding}, no_embedding={dropped_no_embedding}, "
            f"bad_length={dropped_bad_length}, no_text={dropped_no_text}, "
            f"expected_dim={expected_dim_int if enforce_length else 'None'}. "
            "Only empty_embedding caused this failure - the other three are rows filtered "
            "under the configured policy. This normally means the embed stage failed for "
            "whole batches: check the embed actor logs for engine initialization or "
            "out-of-memory errors."
        )

    return lancedb_rows, counts


def _to_service_lancedb_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt canonical dense rows to the established service table schema."""
    wide_rows: list[dict[str, Any]] = []
    for row in rows:
        content_metadata = _maybe_parse_json(row.get("metadata"))
        if not isinstance(content_metadata, dict):
            content_metadata = {}
        source_metadata = _maybe_parse_json(row.get("source"))
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        source_id = next(
            (
                str(value).strip()
                for value in (
                    source_metadata.get("source_id"),
                    source_metadata.get("source_name"),
                )
                if isinstance(value, str) and value.strip()
            ),
            "",
        )
        content_type = normalize_content_type(content_metadata.get("type") or content_metadata.get("_content_type"))
        if content_type:
            content_metadata = dict(content_metadata)
            content_metadata["type"] = content_type
            content_metadata["_content_type"] = content_type
        wide_row = build_lancedb_row(
            SimpleNamespace(
                metadata={
                    "embedding": row.get("vector"),
                    "source_path": source_id,
                    "content_metadata": content_metadata,
                },
                path=source_id,
                page_number=content_metadata.get("page_number"),
                text=row.get("text") or "",
                _stored_image_uri=content_metadata.get("stored_image_uri"),
                _content_type=content_type,
                _bbox_xyxy_norm=content_metadata.get("bbox_xyxy_norm"),
            )
        )
        if wide_row is None:
            continue
        wide_row["metadata"] = _json_str(content_metadata)
        wide_row["source"] = _json_str(source_metadata)
        wide_row["content_type"] = content_type or ""
        wide_rows.append(wide_row)
    return wide_rows


def _create_sparse_lancedb_results(results) -> tuple[list, dict[str, int]]:
    """Transform NRL records into LanceDB rows for FTS-only sparse retrieval."""
    lancedb_rows: list = []
    accepted = 0
    dropped_no_text = 0

    for result in results:
        for element in result:
            metadata = element.get("metadata", {})
            content_meta = metadata.get("content_metadata", {})
            text = _get_text_for_element(element)

            if not isinstance(text, str) or not text.strip():
                dropped_no_text += 1
                source_name = metadata.get("source_metadata", {}).get("source_name", "unknown")
                pg_num = content_meta.get("page_number") if isinstance(content_meta, dict) else None
                logger.debug("No text found for sparse entity: %s page: %s", source_name, pg_num)
                continue

            row_id = content_meta.get("id") if isinstance(content_meta, dict) else None
            if row_id is None and isinstance(metadata, dict):
                row_id = metadata.get("id")
            row_id_str = str(row_id) if row_id is not None else ""

            lancedb_rows.append(
                {
                    "text": text,
                    "metadata": _json_str(content_meta),
                    "source": _json_str(metadata.get("source_metadata", {})),
                    "id": row_id_str,
                }
            )
            accepted += 1

    counts: dict[str, int] = {
        "accepted": accepted,
        "dropped_no_text": dropped_no_text,
    }
    if dropped_no_text:
        logger.warning(
            "_create_sparse_lancedb_results: accepted=%d dropped_no_text=%d",
            accepted,
            dropped_no_text,
        )
    return lancedb_rows, counts


class LanceDB(VDB):
    """LanceDB operator implementing the VDB interface."""

    def __init__(
        self,
        uri: str | None = None,
        overwrite: bool = True,
        table_name: str = "nv-ingest",
        index_type: str = "IVF_HNSW_SQ",
        metric: str = "l2",
        num_partitions: int = 16,
        num_sub_vectors: int = 256,
        hybrid: bool = False,
        sparse: bool = False,
        fts_language: str = "English",
        embedding_model_name: str | None = None,
        vector_dim: int | None = _DEFAULT_VECTOR_DIM,
        on_bad_vectors: str = "drop",
        fill_value: float = 0.0,
        validate_vector_length: bool = True,
        build_index: bool | None = None,
        expiration_cleanup_enabled: bool = True,
        embedding_model_revision: str | None = None,
        **kwargs,
    ):
        create_index = kwargs.pop("create_index", None)
        service_table_schema = bool(kwargs.pop("_service_table_schema", False))
        if build_index is None:
            build_index = True if create_index is None else bool(create_index)
        elif create_index is not None and bool(create_index) != bool(build_index):
            raise ValueError("Pass only one index toggle: build_index or create_index.")

        if vector_dim is not None and int(vector_dim) <= 0:
            raise ValueError(f"vector_dim must be positive; got {vector_dim}")
        if sparse and hybrid:
            raise ValueError("LanceDB sparse ingest cannot also be hybrid; pass only one retrieval mode.")
        self.uri = uri or "lancedb"
        self.overwrite = bool(overwrite)
        self.table_name = table_name
        self.build_index = bool(build_index)
        self.index_type = index_type
        self.metric = metric
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        self.hybrid = hybrid
        self.sparse = bool(sparse)
        self.fts_language = fts_language
        self.embedding_model_name = embedding_model_name
        self.embedding_model_revision = embedding_model_revision
        self.vector_dim = int(vector_dim) if vector_dim is not None else None
        self.on_bad_vectors = _normalize_on_bad_vectors(on_bad_vectors)
        self.fill_value = float(fill_value)
        self.validate_vector_length = bool(validate_vector_length)
        self.expiration_cleanup_enabled = bool(expiration_cleanup_enabled)
        self._service_table_schema = service_table_schema
        self._collection_store: Any | None = None
        self._collection_store_init_failed = False
        self._collection_store_lock = threading.Lock()
        # Row admission is serialized on its own short-lived lock so a caller
        # never waits on index maintenance to get its rows committed.
        self._write_lock = threading.Lock()
        # LanceDB treats competing index commits as a conflict, so only one
        # rebuild may run at a time. Rebuilds are coalesced by generation:
        # a rebuild that starts after a batch was committed also covers it.
        self._index_lock = threading.Lock()
        self._index_generation_lock = threading.Lock()
        self._index_requested_generation = 0
        self._index_completed_generation = 0
        super().__init__(**kwargs)

    def _get_collection_store(self) -> Any:
        """Lazily initialize collection catalogs only when a collection API is used."""

        store = self._collection_store
        if store is None:
            with self._collection_store_lock:
                store = self._collection_store
                if store is None:
                    from nemo_retriever.common.vdb.lancedb_collections import (
                        LanceDBCollectionStore,
                    )

                    try:
                        store = LanceDBCollectionStore(
                            self,
                            expiration_cleanup_enabled=self.expiration_cleanup_enabled,
                        )
                    except Exception:
                        self._collection_store_init_failed = True
                        raise
                    self._collection_store_init_failed = False
                    self._collection_store = store
        return store

    def create_collection(
        self,
        *,
        scope: str,
        request: CollectionCreateRequest,
    ) -> CollectionInfo:
        """Create a logical collection through the LanceDB collection store."""

        return self._get_collection_store().create_collection(scope, request)

    def get_collection(
        self,
        *,
        scope: str,
        collection_name: str,
    ) -> CollectionInfo:
        """Return a logical collection from the LanceDB collection store."""

        return self._get_collection_store().get_collection(scope, collection_name)

    def list_collections(
        self,
        *,
        scope: str,
        limit: int,
        continuation_token: str | None,
    ) -> CollectionPage:
        """List logical collections through the LanceDB collection store."""

        return self._get_collection_store().list_collections(
            scope,
            limit,
            continuation_token,
        )

    def update_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        request: CollectionUpdateRequest,
    ) -> CollectionInfo:
        """Update a logical collection through the LanceDB collection store."""

        return self._get_collection_store().update_collection(
            scope,
            collection_name,
            request,
        )

    def delete_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        if_exists: bool,
    ) -> CollectionDeleteResult:
        """Delete a logical collection through the LanceDB collection store."""

        return self._get_collection_store().delete_collection(
            scope,
            collection_name,
            if_exists,
        )

    def get_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
    ) -> DocumentInfo:
        """Return one collection document through the LanceDB collection store."""

        return self._get_collection_store().get_document(
            scope,
            collection_name,
            document_id,
        )

    def list_documents(
        self,
        *,
        scope: str,
        collection_name: str,
        limit: int,
        continuation_token: str | None,
    ) -> DocumentPage:
        """List collection documents through the LanceDB collection store."""

        return self._get_collection_store().list_documents(
            scope,
            collection_name,
            limit,
            continuation_token,
        )

    def delete_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
        if_exists: bool,
    ) -> DocumentDeleteResult:
        """Delete one collection document through the LanceDB collection store."""

        return self._get_collection_store().delete_document(
            scope,
            collection_name,
            document_id,
            if_exists,
        )

    def write_collection(
        self,
        records: list,
        *,
        context: CollectionWriteContext,
    ) -> CollectionWriteResult:
        """Write canonical records using the collection lifecycle contract."""

        return self._get_collection_store().write_collection(records, context=context)

    def retrieve_collection(
        self,
        vectors: list,
        *,
        scope: str,
        collection_name: str,
        query_texts: list[str],
        top_k: int,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        """Retrieve scoped collection hits using LanceDB's collection contract."""

        return self._get_collection_store().retrieve_collection(
            vectors,
            scope=scope,
            collection_name=collection_name,
            query_texts=query_texts,
            top_k=top_k,
            **kwargs,
        )

    def reconcile_collections(self) -> dict[str, int]:
        """Resume interrupted collection and document lifecycle operations."""

        return self._get_collection_store().reconcile_collections()

    def health(self) -> dict[str, Any]:
        """Return legacy table and optional collection-store health."""

        from nemo_retriever.common.vdb.lancedb_collections import LanceDBCollectionStore

        db = lancedb.connect(uri=self.uri)
        table_exists = self.table_name in db.list_tables().tables
        total_rows = 0
        effective_mode: str | None = None
        retrieval_strategies: list[str] = []
        if table_exists:
            try:
                total_rows = int(db.open_table(self.table_name).count_rows())
            except Exception:
                logger.warning(
                    "Failed to count rows in the default LanceDB table",
                    exc_info=True,
                )
            try:
                capabilities = inspect_lancedb_table_object(db.open_table(self.table_name))
                mode = capabilities.retrieval_mode
                if mode in {"dense", "hybrid"}:
                    effective_mode = str(mode)
                    retrieval_strategies = [str(mode)]
                else:
                    effective_mode = "unknown"
            except Exception:
                effective_mode = "unknown"
                logger.warning(
                    "Failed to resolve the default LanceDB retrieval mode",
                    exc_info=True,
                )

        if self._collection_store_init_failed:
            raise RuntimeError("Collection catalog initialization failed")
        store = self._collection_store
        collection_health = store.health() if store is not None else LanceDBCollectionStore.empty_health()
        return {
            **collection_health,
            "total_rows": total_rows,
            "table_exists": table_exists,
            "effective_retrieval_mode": effective_mode,
            "retrieval_strategies": retrieval_strategies,
        }

    def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
        """Read one NeMo Retriever metadata value from the selected table."""
        uri = str(kwargs.get("table_path") or kwargs.get("uri") or kwargs.get("lancedb_uri") or self.uri)
        table_name = str(kwargs.get("table_name") or kwargs.get("lancedb_table") or self.table_name)
        table = lancedb.connect(uri=uri).open_table(table_name)
        metadata = _table_schema(table).metadata or {}
        value = metadata.get(f"nemo_retriever.{key}".encode("utf-8"))
        if value is None and key == "retrieval_mode":
            value = metadata.get(_RETRIEVAL_MODE_METADATA_KEY)
        if value is None:
            return None
        return value.decode("utf-8", errors="replace").strip() or None

    def create_index(self, records=None, table_name: str = "nv-ingest", **kwargs):
        """Create or update a LanceDB table and populate it with transformed records.

        Validates per-row vector shape (when ``validate_vector_length`` is set
        on the instance and ``on_bad_vectors`` is not ``"error"``) and forwards
        LanceDB's ``on_bad_vectors`` policy as defense-in-depth so that any
        rows escaping the row-builder check are still handled by the LanceDB
        writer instead of aborting the run. When ``on_bad_vectors == "error"``
        the wrapper deliberately skips its own length check so that LanceDB
        itself raises on the bad row, matching the documented strict-fail
        semantics of that policy.
        """
        connect_start = time.perf_counter()
        db = lancedb.connect(uri=self.uri)
        _record_timing("lancedb.connect", time.perf_counter() - connect_start)
        record_batches = list(records or [])

        if self.sparse:
            results, counts = _create_sparse_lancedb_results(record_batches)
            schema = _sparse_lancedb_arrow_schema()
            write_kwargs: dict[str, Any] = {}
        else:
            enforce_dim = self.validate_vector_length and self.on_bad_vectors != "error"
            vector_dim = self.vector_dim
            if vector_dim is None and not self.overwrite:
                try:
                    existing_table = db.open_table(self.table_name)
                except ValueError as exc:
                    if not _is_missing_lancedb_table_error(exc):
                        raise
                else:
                    vector_dim = _schema_vector_dim(_table_schema(existing_table))

            if vector_dim is None:
                results, counts = _create_lancedb_results(record_batches, expected_dim=None)
                vector_dim = infer_vector_dim(results)
                if vector_dim <= 0:
                    raise ValueError("Cannot infer LanceDB vector_dim because no non-empty embedding was produced.")
                if enforce_dim:
                    results, counts = _create_lancedb_results(record_batches, expected_dim=vector_dim)
            else:
                results, counts = _create_lancedb_results(
                    record_batches, expected_dim=vector_dim if enforce_dim else None
                )

            if self._service_table_schema:
                results = _to_service_lancedb_rows(results)
                schema = _with_retrieval_mode_metadata(
                    lancedb_schema(vector_dim),
                    "hybrid" if self.hybrid else "dense",
                    embedding_model_name=self.embedding_model_name,
                    embedding_model_revision=self.embedding_model_revision,
                )
            else:
                schema = _lancedb_arrow_schema(
                    vector_dim,
                    retrieval_mode="hybrid" if self.hybrid else "dense",
                    embedding_model_name=self.embedding_model_name,
                    embedding_model_revision=self.embedding_model_revision,
                )

            write_kwargs = {
                "on_bad_vectors": self.on_bad_vectors,
            }
            if self.on_bad_vectors == "fill":
                write_kwargs["fill_value"] = self.fill_value

        create_kwargs: dict[str, Any] = {
            "schema": schema,
            **write_kwargs,
        }

        create_start = time.perf_counter()

        if self.overwrite:
            table = db.create_table(
                table_name,
                data=results,
                mode="overwrite",
                **create_kwargs,
            )
            event = "lancedb.create_table"
        else:
            try:
                table = db.open_table(table_name)
            except ValueError as exc:
                if not _is_missing_lancedb_table_error(exc):
                    raise
                table = db.create_table(
                    table_name,
                    data=results,
                    mode="create",
                    **create_kwargs,
                )
                event = "lancedb.create_table"
            else:
                _validate_append_schema(table, schema, table_name=table_name, uri=self.uri)
                if results:
                    _validate_append_embedding_model(
                        table,
                        self.embedding_model_name,
                        self.embedding_model_revision,
                        table_name=table_name,
                        uri=self.uri,
                    )
                    existing_rows = int(table.count_rows())
                    logger.warning(
                        "Appending %d row(s) to existing LanceDB table %r at %s "
                        "(existing_rows=%d). Append mode does not deduplicate; rerunning the same inputs "
                        "will duplicate rows.",
                        len(results),
                        table_name,
                        self.uri,
                        existing_rows,
                    )
                    table.add(
                        results,
                        mode="append",
                        **write_kwargs,
                    )
                event = "lancedb.add_rows"

        _record_timing(
            event,
            time.perf_counter() - create_start,
            {"rows": len(results), **counts},
        )
        return table

    def write_to_index(
        self,
        records,
        table=None,
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        hybrid: bool | None = None,
        sparse: bool | None = None,
        fts_language: str | None = None,
        **kwargs,
    ):
        """Create vector and optionally FTS indexes on the LanceDB table.

        For IVF index types, ``num_partitions`` is clamped so that
        ``num_partitions < row_count`` (Lance K-means requirement). Empty or
        single-row tables skip the vector index; hybrid FTS may still be built.
        """
        hybrid = hybrid if hybrid is not None else self.hybrid
        sparse = sparse if sparse is not None else self.sparse
        fts_language = fts_language or self.fts_language

        if sparse:
            fts_index_start = time.perf_counter()
            sparse_rows = int(table.count_rows())
            table.create_fts_index("text", language=fts_language, replace=True)
            wait_for_column_index(table, "text", covered_rows=sparse_rows)
            _record_timing("lancedb.fts_index_ready", time.perf_counter() - fts_index_start)
            return

        num_rows = int(table.count_rows())
        requested_partitions = int(num_partitions)
        use_ivf = _is_ivf_vector_index(index_type)
        effective_partitions: int | None
        if use_ivf:
            effective_partitions = _effective_ivf_num_partitions(num_rows, requested_partitions)
        else:
            effective_partitions = requested_partitions

        vector_index_start = time.perf_counter()
        if use_ivf and effective_partitions is None:
            if num_rows == 0:
                logger.warning(
                    "Skipping LanceDB vector index: empty table (index_type=%s).",
                    index_type,
                )
            else:
                logger.info(
                    "Skipping LanceDB vector index: IVF needs at least two rows (got %d; index_type=%s).",
                    num_rows,
                    index_type,
                )
        else:
            partitions_for_index = (
                int(effective_partitions) if effective_partitions is not None else requested_partitions
            )
            if use_ivf and partitions_for_index != requested_partitions:
                logger.info(
                    "Clamping num_partitions from %d to %d (table has %d rows; IVF requires partitions < row count).",
                    requested_partitions,
                    partitions_for_index,
                    num_rows,
                )
            table.create_index(
                index_type=index_type,
                metric=metric,
                num_partitions=partitions_for_index,
                num_sub_vectors=num_sub_vectors,
                vector_column_name="vector",
                replace=True,
            )
            wait_for_column_index(table, "vector", covered_rows=num_rows)
            _record_timing("lancedb.vector_index_ready", time.perf_counter() - vector_index_start)

        if hybrid:
            fts_index_start = time.perf_counter()
            table.create_fts_index("text", language=fts_language, replace=True)
            wait_for_column_index(table, "text", covered_rows=num_rows)
            _record_timing("lancedb.fts_index_ready", time.perf_counter() - fts_index_start)

    def run(self, records):
        """Commit rows, then bring the table indexes up to date.

        The VectorDB service can dispatch multiple ingestion callbacks at once.
        Row admission and index maintenance are two separate critical sections:

        * ``_write_lock`` serializes only the table create/append, so a
          concurrent write is never held behind another writer's index build.
          Rows are durable and queryable (Lance scans unindexed fragments)
          as soon as this section commits.
        * ``_index_lock`` serializes index commits, which LanceDB rejects when
          they conflict. Rebuilds are coalesced: a rebuild that starts after
          this batch was committed also indexes this batch, so concurrent
          writers share one rebuild instead of queueing one each.
        """
        with self._write_lock:
            table = self.create_index(records=records, table_name=self.table_name)

        if not self.build_index:
            logger.info(
                "Skipping LanceDB index creation for table %r because build_index=False.",
                self.table_name,
            )
            return records

        self._maintain_indexes(records, table)
        return records

    def _maintain_indexes(self, records, table) -> None:
        """Rebuild table indexes so they cover the rows committed by this call.

        Returns once an index build that started after this caller's rows were
        committed has finished. Callers that arrive while such a build is
        already running wait for it rather than queueing another one.
        """
        with self._index_generation_lock:
            self._index_requested_generation += 1
            required_generation = self._index_requested_generation

        while True:
            with self._index_generation_lock:
                if self._index_completed_generation >= required_generation:
                    return

            with self._index_lock:
                with self._index_generation_lock:
                    if self._index_completed_generation >= required_generation:
                        return
                    building_generation = self._index_requested_generation

                # Index the newest committed version, not the snapshot this
                # caller happened to open, so one rebuild can cover the rows
                # of every writer it has coalesced.
                self._checkout_latest(table)
                self.write_to_index(
                    records,
                    table=table,
                    index_type=self.index_type,
                    metric=self.metric,
                    num_partitions=self.num_partitions,
                    num_sub_vectors=self.num_sub_vectors,
                    hybrid=self.hybrid,
                    sparse=self.sparse,
                    fts_language=self.fts_language,
                )

                with self._index_generation_lock:
                    self._index_completed_generation = max(
                        self._index_completed_generation,
                        building_generation,
                    )

    @staticmethod
    def _checkout_latest(table) -> None:
        """Advance ``table`` to the latest committed version when supported."""
        checkout_latest = getattr(table, "checkout_latest", None)
        if not callable(checkout_latest):
            return
        try:
            checkout_latest()
        except Exception as exc:  # noqa: BLE001 - version refresh is advisory.
            logger.debug("Could not advance LanceDB table handle to the latest version: %s", exc)

    def put(
        self,
        records,
        table_name: str | None = None,
        key: str = "id",
    ) -> dict[str, int]:
        """Replace existing rows of a LanceDB table in place, keyed by ``key``.

        Strict update-only semantics:

        * Rows matching an existing row by ``key`` are **updated in place**
          (all columns, including ``vector``, are replaced).
        * Rows whose ``key`` value is missing/empty raise :class:`KeyError`
          — a put operation has no stable identity to target without a key.
        * Rows whose ``key`` value does not match any row currently in the
          table raise :class:`KeyError` — ``put`` never inserts new rows.
        * Rows already in the table that are *not* referenced are **left
          untouched** — ``put`` never deletes.

        If the target table does not exist, :class:`FileNotFoundError` is
        raised; ``put`` will not create tables on the fly.

        Vector / FTS indexes are intentionally **not** rebuilt here:
        incremental puts typically carry only a handful of rows. Indexes
        will be (re)built by the next full :meth:`run` /
        :meth:`write_to_index` call.

        Returns the row counts dict from :func:`_create_lancedb_results`
        plus: ``put``.
        """
        target_name = table_name or self.table_name
        connect_start = time.perf_counter()
        db = lancedb.connect(uri=self.uri)
        _record_timing("lancedb.connect", time.perf_counter() - connect_start)

        if self.validate_vector_length and self.on_bad_vectors != "error":
            expected_dim: int | None = self.vector_dim
        else:
            expected_dim = None

        rows, counts = _create_lancedb_results(records or [], expected_dim=expected_dim)
        counts["put"] = 0

        if not rows:
            logger.info("LanceDB.put: nothing to put into table %r.", target_name)
            return counts

        rows_missing_key = [r for r in rows if not r.get(key)]
        if rows_missing_key:
            raise KeyError(
                f"LanceDB.put: {len(rows_missing_key)} row(s) have an empty {key!r} value; "
                "put() requires a stable id for every row."
            )

        try:
            table = db.open_table(target_name)
        except (ValueError, FileNotFoundError) as exc:
            if isinstance(exc, ValueError) and not _is_missing_lancedb_table_error(exc):
                raise
            raise FileNotFoundError(
                f"LanceDB.put: table {target_name!r} not found at uri={self.uri!r}; "
                "put() only updates existing rows and will not create tables."
            ) from exc

        input_ids = [r[key] for r in rows]
        unique_input_ids = list(dict.fromkeys(input_ids))

        filter_expr = pc.field(key).isin(pa.array(unique_input_ids, type=pa.string()))
        existing_arrow = table.to_lance().to_table(columns=[key], filter=filter_expr)
        existing_ids = set(existing_arrow.column(key).to_pylist())

        missing_ids = [i for i in unique_input_ids if i not in existing_ids]
        if missing_ids:
            raise KeyError(
                f"LanceDB.put: row(s) with {key}={missing_ids!r} not found in table "
                f"{target_name!r}; put() only updates existing rows."
            )

        put_start = time.perf_counter()
        table.merge_insert(key).when_matched_update_all().execute(rows)
        _record_timing(
            "lancedb.put",
            time.perf_counter() - put_start,
            {"rows": len(rows), "table": target_name},
        )

        counts["put"] = len(rows)
        return counts

    def sparse_retrieval(self, query_texts: Iterable[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Search a LanceDB FTS-only table without query embeddings."""
        table_path = kwargs.pop("table_path", self.uri)
        table_name = kwargs.pop("table_name", self.table_name)
        result_fields = kwargs.pop("result_fields", None)
        top_k = int(kwargs.pop("top_k", 10))
        text_column_value = kwargs.pop("text_column_name", kwargs.pop("fts_columns", "text"))
        if isinstance(text_column_value, (list, tuple)):
            text_column_name = str(text_column_value[0] if text_column_value else "text")
        else:
            text_column_name = str(text_column_value)

        search_kwargs_raw = kwargs.pop("search_kwargs", None)
        if search_kwargs_raw is None:
            search_kwargs: dict[str, Any] = {}
        elif not isinstance(search_kwargs_raw, dict):
            raise TypeError(f"search_kwargs must be a dict or None; got {type(search_kwargs_raw).__name__}")
        else:
            search_kwargs = dict(search_kwargs_raw)

        query_type = search_kwargs.get("query_type")
        if query_type is not None:
            query_type_value = getattr(query_type, "value", query_type)
            if str(query_type_value).lower() != "fts":
                raise ValueError(
                    "LanceDB sparse retrieval requires search_kwargs['query_type']='fts'; " f"got {query_type!r}."
                )
        search_kwargs["query_type"] = "fts"
        search_kwargs.setdefault("fts_columns", text_column_name)

        where_clause = kwargs.pop("where", None)
        _filter_fallback = kwargs.pop("_filter", None)
        if where_clause is None:
            where_clause = _filter_fallback
        if where_clause is not None:
            where_clause = str(where_clause).strip() or None

        table = lancedb.connect(uri=table_path).open_table(table_name)

        search_results = []
        for query_text in query_texts:
            query = table.search(str(query_text), **search_kwargs)
            if where_clause is not None:
                query = query.where(where_clause)
            query = query.limit(top_k)
            if result_fields is not None:
                query = query.select(result_fields)
            search_results.append(query.to_list())

        return search_results

    def retrieval(self, vectors: Iterable[Sequence[float]], **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Search LanceDB with precomputed query vectors.

        Keyword arguments
        -----------------
        where:
            Optional SQL predicate (Lance / DataFusion) applied on the vector
            query builder via ``.where(...)`` before ``limit``. Filter against
            table columns: ``vector``, ``text``, ``metadata``, ``source``.
            Note: ``metadata`` and ``source`` are JSON strings at rest.
        _filter:
            Alias for ``where`` when ``where`` is omitted (call-site parity).
        search_kwargs:
            Optional dict of extra keyword arguments forwarded to
            ``table.search`` (e.g. ``query_type``, ``fts_columns``). Do not
            pass ``vector_column_name`` here; use the top-level
            ``vector_column_name`` retrieval argument instead.
        query_texts:
            Raw query strings aligned with ``vectors``. Required for
            ``hybrid=True`` and ignored for dense-only retrieval.
        """
        hybrid = kwargs.pop("hybrid", self.hybrid)
        query_texts = kwargs.pop("query_texts", None)
        table_path = kwargs.pop("table_path", self.uri)
        table_name = kwargs.pop("table_name", self.table_name)

        result_fields = kwargs.pop("result_fields", None)
        top_k = int(kwargs.pop("top_k", 10))
        refine_factor = int(kwargs.pop("refine_factor", 50))
        n_probe = int(kwargs.pop("n_probe", kwargs.pop("nprobes", 64)))
        vector_column_name = str(kwargs.pop("vector_column_name", "vector"))

        search_kwargs_raw = kwargs.pop("search_kwargs", None)
        if search_kwargs_raw is None:
            search_kwargs: dict[str, Any] = {}
        elif not isinstance(search_kwargs_raw, dict):
            raise TypeError(f"search_kwargs must be a dict or None; got {type(search_kwargs_raw).__name__}")
        else:
            search_kwargs = dict(search_kwargs_raw)

        if hybrid:
            if query_texts is None:
                raise ValueError(
                    "LanceDB hybrid retrieval requires query_texts. Pass query_texts=your_queries "
                    "alongside vectors when calling retrieval() with hybrid=True."
                )
            query_type = search_kwargs.get("query_type")
            if query_type is not None:
                query_type_value = getattr(query_type, "value", query_type)
                if str(query_type_value).lower() != "hybrid":
                    raise ValueError(
                        "LanceDB hybrid retrieval requires search_kwargs['query_type']='hybrid'; "
                        f"got {query_type!r}."
                    )
            search_kwargs["query_type"] = "hybrid"
            search_kwargs.setdefault("fts_columns", "text")

        where_clause = kwargs.pop("where", None)
        _filter_fallback = kwargs.pop("_filter", None)
        if where_clause is None:
            where_clause = _filter_fallback
        if where_clause is not None:
            where_clause = str(where_clause).strip() or None

        table = lancedb.connect(uri=table_path).open_table(table_name)

        if hybrid:
            vectors_for_search = list(vectors)
            query_texts_list = [query_texts] if isinstance(query_texts, str) else list(query_texts)
            if len(query_texts_list) != len(vectors_for_search):
                raise ValueError(
                    "LanceDB hybrid retrieval requires query_texts length to match vectors length; "
                    f"got query_texts={len(query_texts_list)} vectors={len(vectors_for_search)}."
                )
        else:
            vectors_for_search = vectors
            query_texts_list = []

        search_results = []
        for idx, vector in enumerate(vectors_for_search):
            if hybrid:
                query = (
                    table.search(vector_column_name=vector_column_name, **search_kwargs)
                    .vector(vector)
                    .text(str(query_texts_list[idx]))
                )
            else:
                query = table.search([vector], vector_column_name=vector_column_name, **search_kwargs)
            if where_clause is not None:
                query = query.where(where_clause)
            query = query.limit(top_k).refine_factor(refine_factor).nprobes(n_probe)
            if result_fields is not None:
                query = query.select(result_fields)
            results = query.to_list()
            search_results.append(results)

        return search_results
