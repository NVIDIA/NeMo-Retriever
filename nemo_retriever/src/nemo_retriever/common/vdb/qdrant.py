# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qdrant backend for the :class:`~nemo_retriever.common.vdb.adt_vdb.VDB` interface."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

try:
    from qdrant_client import QdrantClient, models
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('The Qdrant VDB backend requires `pip install "nemo-retriever[qdrant]"`.') from exc

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
    IndexCapabilities,
    VDB,
)
from nemo_retriever.common.vdb.hybrid_fusion import DEFAULT_HYBRID_FUSION_POLICY, HybridFusionPolicy
from nemo_retriever.common.vdb.lancedb_schema import infer_vector_dim
from nemo_retriever.common.vdb.records import DEFAULT_VECTOR_DIM, build_dense_rows, build_text_rows

logger = logging.getLogger(__name__)

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "l2": models.Distance.EUCLID,
    "euclid": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
    "manhattan": models.Distance.MANHATTAN,
}
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
_BM25_MODEL = "qdrant/bm25"
_SCROLL_PAGE = 1_000
# Qdrant rejects request bodies over 32 MiB by default.
MAX_REQUEST_BYTES = 16 * 1024 * 1024
QUERIES_PER_REQUEST = 16
# How long reads may reuse a collection's config. Writes always fetch it fresh.
CONFIG_CACHE_SECONDS = 5.0


def _metadata_key(key: str) -> str:
    return f"nemo_retriever.{key}"


_BM25_BACKFILL_KEY = _metadata_key("bm25_backfill")
# Keys that put() and tabular retrieval filter on. Strict-mode servers reject filters on unindexed keys.
FILTER_PAYLOAD_INDEXES = ("id", "metadata.label", "metadata.database_name")


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return list(value)
    tolist = getattr(value, "tolist", None)  # numpy values from Ray/pandas rows
    return tolist() if callable(tolist) else str(value)


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    """Keep metadata and source as objects so filters can address nested keys."""
    payload = {
        "text": row["text"],
        "metadata": row["metadata"] if isinstance(row["metadata"], dict) else {},
        "source": row["source"] if isinstance(row["source"], dict) else {},
        "id": row["id"],
    }
    return json.loads(json.dumps(payload, default=_json_default))


def _dense_params(name: str, config: Any) -> Any | None:
    vectors = config.params.vectors
    if not vectors:
        return None
    if isinstance(vectors, dict) and DENSE_VECTOR_NAME in vectors:
        return vectors[DENSE_VECTOR_NAME]
    raise ValueError(
        f"Qdrant collection {name!r} was not created by NeMo Retriever: its dense vector must be named "
        f"{DENSE_VECTOR_NAME!r}. Overwrite the collection or use another one."
    )


def _has_bm25(config: Any) -> bool:
    return SPARSE_VECTOR_NAME in (config.params.sparse_vectors or {})


def _stored_bm25_options(config: Any) -> dict[str, Any] | None:
    options = (config.metadata or {}).get(_metadata_key("bm25_options"))
    return options if isinstance(options, dict) else None


def _layout(name: str, config: Any) -> str:
    dense = _dense_params(name, config) is not None
    bm25 = _has_bm25(config)
    return {(True, True): "hybrid", (True, False): "dense", (False, True): "sparse"}.get((dense, bm25), "unknown")


def _bm25(text: str, options: dict[str, Any] | None) -> Any:
    return models.Document(text=str(text), model=_BM25_MODEL, options=options)


def _vector_bytes(vector: Any) -> int:
    """Upper bound on the JSON size of a vector, BM25 document, or dict of them."""
    if isinstance(vector, dict):
        return sum(len(key) + 4 + _vector_bytes(value) for key, value in vector.items())
    if isinstance(vector, models.Document):
        return len(json.dumps(vector.text)) + len(json.dumps(vector.options or {})) + 64
    return 25 * len(vector)


def _point_bytes(point: Any) -> int:
    payload = getattr(point, "payload", None)
    return _vector_bytes(point.vector) + (len(json.dumps(payload)) if payload else 0) + 64


def _point_batches(points: Iterable[Any], max_count: int) -> Iterator[list[Any]]:
    """Split points by count and estimated request size. A batch always has at least one point."""
    batch: list[Any] = []
    batch_bytes = 0
    for point in points:
        size = _point_bytes(point)
        if batch and (len(batch) >= max_count or batch_bytes + size > MAX_REQUEST_BYTES):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(point)
        batch_bytes += size
    if batch:
        yield batch


def _scroll_points(client: QdrantClient, name: str, **kwargs: Any) -> Iterator[Any]:
    offset = None
    while True:
        points, offset = client.scroll(name, limit=_SCROLL_PAGE, offset=offset, **kwargs)
        yield from points
        if offset is None:
            return


def _query_batch(client: QdrantClient, name: str, requests: list[models.QueryRequest]) -> list[Any]:
    # A few queries per request keeps large query vectors under the request size limit.
    responses: list[Any] = []
    for start in range(0, len(requests), QUERIES_PER_REQUEST):
        responses.extend(client.query_batch_points(name, requests[start : start + QUERIES_PER_REQUEST]))
    return responses


class Qdrant(VDB):
    """Qdrant server backend with dense, hybrid (dense + server-side BM25), and sparse (BM25-only) collections.

    ``table_name`` is an alias for ``collection_name``. ``bm25_options`` are recorded on the collection so
    writes and queries tokenize text the same way.
    """

    metadata_filter_format = "qdrant"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        overwrite: bool = True,
        metric: str = "cosine",
        vector_dim: int | None = DEFAULT_VECTOR_DIM,
        validate_vector_length: bool = True,
        batch_size: int = 256,
        on_disk: bool = False,
        embedding_model_name: str | None = None,
        embedding_model_revision: str | None = None,
        hybrid: bool = False,
        sparse: bool = False,
        bm25_options: dict[str, Any] | None = None,
        client_kwargs: dict[str, Any] | None = None,
        expiration_cleanup_enabled: bool = True,
        payload_indexes: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the target collection. The client connects on first use.

        ``url``, ``api_key`` and ``client_kwargs`` go to ``QdrantClient``. ``hybrid`` adds a BM25 vector next to the
        dense one and ``sparse`` stores only BM25. ``overwrite`` recreates the collection on the first write, otherwise
        writes append. ``payload_indexes`` adds keyword indexes for filtered keys, which strict-mode servers require.

        Raises:
            ValueError: If the options conflict or are out of range.
        """
        if sparse and hybrid:
            raise ValueError("Qdrant sparse ingest cannot also be hybrid; pass only one retrieval mode.")
        if bm25_options and not (hybrid or sparse):
            raise ValueError("bm25_options requires hybrid=True or sparse=True.")
        if metric.lower() not in _DISTANCES:
            raise ValueError(f"metric must be one of {sorted(_DISTANCES)}; got {metric!r}")
        if vector_dim is not None and vector_dim <= 0:
            raise ValueError(f"vector_dim must be positive; got {vector_dim}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        if {"path", "location"} & set(client_kwargs or {}):
            raise ValueError("The Qdrant backend requires a Qdrant server; pass url instead of path or location.")
        self.collection_name = collection_name or kwargs.pop("table_name", None) or "nemo-retriever"
        connection = {"url": url, "api_key": api_key}
        self.client_kwargs = {**{k: v for k, v in connection.items() if v is not None}, **(client_kwargs or {})}
        self.overwrite = overwrite
        self.metric = metric
        self.vector_dim = vector_dim
        self.validate_vector_length = validate_vector_length
        self.batch_size = batch_size
        self.on_disk = on_disk
        self.embedding_model_name = embedding_model_name
        self.embedding_model_revision = embedding_model_revision
        self.hybrid = bool(hybrid)
        self.sparse = bool(sparse)
        self.bm25_options = dict(bm25_options) if bm25_options else None
        self.payload_indexes = list(dict.fromkeys([*FILTER_PAYLOAD_INDEXES, *(payload_indexes or [])]))
        self.expiration_cleanup_enabled = expiration_cleanup_enabled
        self._init_runtime_state()
        super().__init__(**kwargs)

    def _init_runtime_state(self) -> None:
        self._client: QdrantClient | None = None
        self._client_lock = threading.Lock()
        self._collection_store: Any = None
        self._collection_store_init_failed = False
        self._collection_store_lock = threading.Lock()
        self._info_cache: dict[str, tuple[float, Any]] = {}

    # Ray pickles operators, so the client, locks and caches are rebuilt.
    def __getstate__(self) -> dict[str, Any]:
        runtime = ("_client", "_client_lock", "_collection_store", "_collection_store_init_failed")
        runtime += ("_collection_store_lock", "_info_cache")
        return {key: value for key, value in self.__dict__.items() if key not in runtime}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_runtime_state()

    @property
    def client(self) -> QdrantClient:
        """The shared client, created on first use."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    # Compute BM25 vectors on the server rather than locally with fastembed.
                    self._client = QdrantClient(**{"cloud_inference": True, **self.client_kwargs})
        return self._client

    @property
    def _mode(self) -> str:
        return "sparse" if self.sparse else "hybrid" if self.hybrid else "dense"

    def _name(self, kwargs: dict[str, Any]) -> str:
        # RetrieveVdbOperator merges constructor kwargs into calls and reserves collection_name for the
        # service collection API, so table_name wins.
        return str(kwargs.get("table_name") or kwargs.get("collection_name") or self.collection_name)

    def _info(self, name: str, *, cached: bool = False) -> Any | None:
        """Return the collection info, or ``None`` if it does not exist. Reads may pass ``cached=True``."""
        entry = self._info_cache.get(name) if cached else None
        if entry is not None and entry[0] > time.monotonic():
            return entry[1]
        info = self.client.get_collection(name) if self.client.collection_exists(name) else None
        if info is None:
            self._info_cache.pop(name, None)
        else:
            self._info_cache[name] = (time.monotonic() + CONFIG_CACHE_SECONDS, info)
        return info

    def _config(self, name: str, *, cached: bool = False) -> Any | None:
        info = self._info(name, cached=cached)
        return info.config if info else None

    def _collection_metadata(self) -> dict[str, Any]:
        metadata = {
            _metadata_key("embedding_model_name"): None if self.sparse else self.embedding_model_name,
            _metadata_key("embedding_model_revision"): None if self.sparse else self.embedding_model_revision,
            _metadata_key("bm25_options"): self.bm25_options,
        }
        return {key: value for key, value in metadata.items() if value}

    def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
        """Return a value recorded on the collection, such as ``embedding_model_name``, or ``None``."""
        config = self._config(self._name(kwargs), cached=True)
        value = (config.metadata or {}).get(_metadata_key(key)) if config else None
        if value is None:
            return None
        return str(value).strip() or None

    def index_capabilities(self, **kwargs: Any) -> IndexCapabilities | None:
        """Describe the collection layout, or return ``None`` if the collection does not exist."""
        name = self._name(kwargs)
        config = self._config(name, cached=True)
        if config is None:
            return None
        mode = _layout(name, config)
        dense, bm25 = mode in ("dense", "hybrid"), mode in ("hybrid", "sparse")
        return IndexCapabilities(
            has_vector=dense,
            has_fts=bm25,
            retrieval_mode=mode,
            vector_column=DENSE_VECTOR_NAME if dense else None,
            text_column="text" if bm25 else None,
        )

    def _rows(self, records: list) -> tuple[list[dict[str, Any]], int | None]:
        if self.sparse:
            return build_text_rows(records)[0], None
        dim = self.vector_dim
        if dim is None and not self.overwrite and (config := self._config(self.collection_name)):
            dim = getattr(_dense_params(self.collection_name, config), "size", None)
        if dim is None:
            dim = infer_vector_dim(build_dense_rows(records, expected_dim=None)[0]) or None
        rows, _ = build_dense_rows(records, expected_dim=dim if self.validate_vector_length else None)
        return rows, dim

    def _check_embedding_model(self, name: str, stored: dict[str, Any]) -> None:
        stored_model = str(stored.get(_metadata_key("embedding_model_name")) or "").strip()
        if not self.embedding_model_name or not stored_model:
            return
        if stored_model != self.embedding_model_name:
            raise ValueError(
                f"Qdrant collection {name!r} uses embedding model {stored_model!r}; cannot append vectors "
                f"from {self.embedding_model_name!r}. Use the collection model or overwrite the collection."
            )
        stored_revision = str(stored.get(_metadata_key("embedding_model_revision")) or "").strip()
        if stored_revision and not self.embedding_model_revision:
            raise ValueError(
                f"Qdrant collection {name!r} uses embedding model revision {stored_revision!r}; cannot append "
                "vectors without a known revision. Use the collection revision or overwrite the collection."
            )
        if stored_revision and stored_revision != self.embedding_model_revision:
            raise ValueError(
                f"Qdrant collection {name!r} uses embedding model revision {stored_revision!r}; cannot append "
                f"vectors from revision {self.embedding_model_revision!r}. Use the collection revision or "
                "overwrite the collection."
            )

    def _upload(self, name: str, config: Any, targets: list[tuple[Any, dict[str, Any], dict[str, Any]]]) -> None:
        mode = _layout(name, config)
        options = _stored_bm25_options(config)
        points = []
        for point_id, row, payload in targets:
            if mode == "sparse":
                vector = {SPARSE_VECTOR_NAME: _bm25(row["text"], options)}
            elif mode == "hybrid":
                vector = {DENSE_VECTOR_NAME: row["vector"], SPARSE_VECTOR_NAME: _bm25(row["text"], options)}
            else:
                vector = {DENSE_VECTOR_NAME: row["vector"]}
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
        for batch in _point_batches(points, self.batch_size):
            self.client.upload_points(name, batch, batch_size=len(batch), wait=True)

    def _add_bm25(self, name: str, config: Any) -> None:
        """Add a BM25 vector to a dense collection and index its existing points.

        The ``bm25_backfill`` marker stays ``pending`` until every point is indexed, so an interrupted
        upgrade resumes on the next append.
        """
        if not _has_bm25(config):
            version = self.client.info().version
            if tuple(int(part) for part in re.findall(r"\d+", version)[:2]) < (1, 18):
                raise ValueError(
                    f"Upgrading Qdrant collection {name!r} to hybrid requires Qdrant server 1.18 or later; "
                    f"the server runs {version}. Upgrade the server or re-ingest with overwrite=True."
                )
        options = self.bm25_options or _stored_bm25_options(config)
        pending = {_BM25_BACKFILL_KEY: "pending"}
        if options:
            pending[_metadata_key("bm25_options")] = options
        self.client.update_collection(name, metadata=pending)
        if not _has_bm25(config):
            sparse_config = models.SparseVectorConfig(modifier=models.Modifier.IDF)
            self.client.create_vector_name(
                name, SPARSE_VECTOR_NAME, models.SparseVectorNameConfig(sparse=sparse_config)
            )
        self._info_cache.pop(name, None)
        updates = (
            models.PointVectors(
                id=point.id, vector={SPARSE_VECTOR_NAME: _bm25((point.payload or {}).get("text") or "", options)}
            )
            for point in _scroll_points(self.client, name, with_payload=["text"])
        )
        for batch in _point_batches(updates, self.batch_size):
            self.client.update_vectors(name, points=batch, wait=True)
        self.client.update_collection(name, metadata={_BM25_BACKFILL_KEY: "complete"})
        self._info_cache.pop(name, None)

    def create_index(self, vector_dim: int | None = None, recreate: bool | None = None, **kwargs: Any) -> None:
        """Create the collection, or validate an existing one for appending.

        A hybrid append to a dense collection upgrades it in place.
        """
        name = self.collection_name
        vector_dim = vector_dim or self.vector_dim
        info = self._info(name)
        if info is not None and (self.overwrite if recreate is None else recreate):
            self.client.delete_collection(name)
            self._info_cache.pop(name, None)
            info = None
        config = info.config if info else None

        distance = _DISTANCES[self.metric.lower()]
        if config is not None:
            existing = _layout(name, config)
            compatible = existing == self._mode or (existing, self._mode) in (("hybrid", "dense"), ("dense", "hybrid"))
            if not compatible:
                raise ValueError(
                    f"Cannot append with {self._mode} retrieval to the existing {existing} Qdrant collection "
                    f"{name!r}; use a matching mode or overwrite=True to replace the collection."
                )
            if existing != "sparse":
                vectors = _dense_params(name, config)
                if vector_dim and vectors.size != vector_dim:
                    raise ValueError(
                        f"Qdrant collection {name!r} has vector size {vectors.size}, expected {vector_dim}; "
                        "use overwrite=True to replace the collection."
                    )
                if vectors.distance != distance:
                    raise ValueError(
                        f"Qdrant collection {name!r} uses {vectors.distance.value} distance, but "
                        f"metric={self.metric!r} was requested; use the collection metric or overwrite=True "
                        "to replace the collection."
                    )
                self._check_embedding_model(name, config.metadata or {})
            if existing != "dense" and self.bm25_options and self.bm25_options != _stored_bm25_options(config):
                raise ValueError(
                    f"Qdrant collection {name!r} records bm25_options={_stored_bm25_options(config)!r}; cannot "
                    f"append with {self.bm25_options!r}. Use the collection options or overwrite the collection."
                )
            backfill_pending = (config.metadata or {}).get(_BM25_BACKFILL_KEY) == "pending"
            if (existing, self._mode) == ("dense", "hybrid") or (existing == "hybrid" and backfill_pending):
                self._add_bm25(name, config)
            self._ensure_payload_indexes(name, info, self.payload_indexes)
            return

        dense = None
        if not self.sparse:
            if vector_dim is None:
                raise ValueError("Cannot create a Qdrant collection without vector_dim and no embeddings to infer it.")
            dense = {DENSE_VECTOR_NAME: models.VectorParams(size=vector_dim, distance=distance, on_disk=self.on_disk)}
        bm25 = {SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}
        try:
            self.client.create_collection(
                name,
                vectors_config=dense if dense is not None else {},
                sparse_vectors_config=bm25 if self.hybrid or self.sparse else None,
                metadata=self._collection_metadata() or None,
            )
        except Exception:
            if not self.client.collection_exists(name):
                raise
            # Another writer created it first, so validate it as an append target.
            self.create_index(vector_dim=vector_dim, recreate=False)
            return
        self._ensure_payload_indexes(name, None, self.payload_indexes)

    def _ensure_payload_indexes(self, name: str, info: Any | None, keys: list[str]) -> None:
        present = (info.payload_schema if info else None) or {}
        missing = [key for key in keys if key not in present]
        for key in missing:
            self.client.create_payload_index(name, field_name=key, field_schema=models.PayloadSchemaType.KEYWORD)
        if missing:
            self._info_cache.pop(name, None)

    def write_to_index(self, records: list, rows: list[dict[str, Any]] | None = None, **kwargs: Any) -> int:
        """Upload rows as new points and return the count. Like LanceDB, re-appending duplicates them."""
        if rows is None:
            rows, _ = self._rows(list(records or []))
        config = self._config(self.collection_name)
        if config is None:
            raise FileNotFoundError(f"Qdrant collection {self.collection_name!r} not found; call create_index first.")
        self._upload(self.collection_name, config, [(str(uuid.uuid4()), row, _payload(row)) for row in rows])
        return len(rows)

    def run(self, records: list) -> list:
        """Create or validate the collection, write the records and return them."""
        rows, dim = self._rows(list(records or []))
        self.create_index(vector_dim=dim)
        written = self.write_to_index(records, rows=rows)
        logger.info("Wrote %d point(s) to Qdrant collection %r.", written, self.collection_name)
        return records

    def reindex(self, records: list, **kwargs: Any) -> None:
        """Recreate the collection and write the records."""
        rows, dim = self._rows(list(records or []))
        self.create_index(vector_dim=dim, recreate=True)
        self.write_to_index(records, rows=rows)

    def put(self, records: list, table_name: str | None = None, key: str = "id") -> dict[str, int]:
        """Replace existing points in place, matched on payload ``key``. Never inserts or deletes."""
        if key not in ("id", "text"):
            raise ValueError(f"Qdrant.put: key must be 'id' or 'text'; got {key!r}")
        name = table_name or self.collection_name
        info = self._info(name)
        if info is None:
            raise FileNotFoundError(f"Qdrant.put: collection {name!r} not found; put() only updates existing points.")
        config = info.config
        vectors = _dense_params(name, config)
        if vectors is None:
            rows, counts = build_text_rows(records or [])
        else:
            expected_dim = vectors.size if self.validate_vector_length else None
            rows, counts = build_dense_rows(records or [], expected_dim=expected_dim)
        counts["put"] = 0
        if not rows:
            return counts

        payloads = [_payload(row) for row in rows]
        values = [payload.get(key) for payload in payloads]
        if not all(values):
            raise KeyError(f"Qdrant.put: every row requires a non-empty {key!r} value.")

        self._ensure_payload_indexes(name, info, [key])
        point_ids: dict[Any, list[Any]] = {}
        key_filter = models.Filter(must=[models.FieldCondition(key=key, match=models.MatchAny(any=values))])
        for match in _scroll_points(self.client, name, scroll_filter=key_filter, with_payload=[key]):
            point_ids.setdefault(match.payload[key], []).append(match.id)

        missing = [value for value in dict.fromkeys(values) if value not in point_ids]
        if missing:
            raise KeyError(f"Qdrant.put: point(s) with {key}={missing!r} not found in collection {name!r}.")

        targets = [
            (point_id, row, payload) for row, payload in zip(rows, payloads) for point_id in point_ids[payload[key]]
        ]
        self._upload(name, config, targets)
        counts["put"] = len(rows)
        return counts

    def _search_options(self, kwargs: dict[str, Any]) -> tuple[Any, Any, int]:
        if kwargs.get("where") is not None or kwargs.get("_filter") is not None:
            raise ValueError(
                "Qdrant retrieval does not accept SQL 'where' predicates; pass query_filter as a "
                "qdrant_client.models.Filter (payload keys: text, id, metadata.*, source.*)."
            )
        query_filter = kwargs.get("query_filter")
        if isinstance(query_filter, dict):
            query_filter = models.Filter.model_validate(query_filter)
        params = None
        if kwargs.get("hnsw_ef") is not None or kwargs.get("exact") is not None:
            params = models.SearchParams(hnsw_ef=kwargs.get("hnsw_ef"), exact=bool(kwargs.get("exact")))
        top_k = int(kwargs.get("top_k", 10))
        if top_k <= 0:
            raise ValueError(f"top_k must be positive; got {top_k}")
        return query_filter, params, top_k

    def _search(self, name: str, requests: list[Any], score_key: str) -> list[list[dict[str, Any]]]:
        responses = _query_batch(self.client, name, requests)
        return [[{**(hit.payload or {}), score_key: hit.score} for hit in response.points] for response in responses]

    def _existing_config(self, name: str, usable: Callable[[Any], bool]) -> Any:
        config = self._config(name, cached=True)
        if config is not None and not usable(config):
            config = self._config(name)  # the cached copy may predate another writer's change
        if config is None:
            raise FileNotFoundError(f"Qdrant collection {name!r} not found.")
        return config

    def retrieval(self, vectors: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Search with precomputed query vectors.

        Accepts ``top_k``, ``query_filter`` (a ``models.Filter`` or dict), ``hnsw_ef``, ``exact`` and
        ``table_name``. ``hybrid`` needs ``query_texts`` aligned with ``vectors``, and ``hybrid_fusion`` sets the
        weighted RRF. Unknown kwargs are ignored.
        """
        hybrid = bool(kwargs.get("hybrid", self.hybrid))
        fusion = kwargs.get("hybrid_fusion")
        if fusion is not None and not hybrid:
            raise ValueError("hybrid_fusion requires hybrid=True")
        if fusion is not None and not isinstance(fusion, HybridFusionPolicy):
            raise TypeError(f"hybrid_fusion must be a HybridFusionPolicy or None; got {type(fusion).__name__}")
        query_filter, params, top_k = self._search_options(kwargs)
        vectors = [list(map(float, vector)) for vector in vectors]
        query_texts = kwargs.get("query_texts")
        if hybrid:
            if query_texts is None:
                raise ValueError("Qdrant hybrid retrieval requires query_texts aligned with vectors.")
            query_texts = [query_texts] if isinstance(query_texts, str) else list(query_texts)
            if len(query_texts) != len(vectors):
                raise ValueError(
                    "Qdrant hybrid retrieval requires query_texts length to match vectors length; "
                    f"got query_texts={len(query_texts)} vectors={len(vectors)}."
                )
        if not vectors:
            return []

        name = self._name(kwargs)
        config = self._existing_config(
            name, lambda c: _dense_params(name, c) is not None and (not hybrid or _has_bm25(c))
        )
        dense = _dense_params(name, config)
        if dense is None:
            raise ValueError(f"Qdrant collection {name!r} has no dense vector; use sparse_retrieval for it.")
        if not hybrid:
            requests = [
                models.QueryRequest(
                    query=vector,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    params=params,
                    limit=top_k,
                    with_payload=True,
                )
                for vector in vectors
            ]
            # Euclid and Manhattan scores are distances. The others are similarities.
            distances = (models.Distance.EUCLID, models.Distance.MANHATTAN)
            return self._search(name, requests, "_distance" if dense.distance in distances else "_score")

        if not _has_bm25(config):
            raise ValueError(
                f"Qdrant collection {name!r} has no BM25 vector; ingest with hybrid=True to use hybrid retrieval."
            )
        policy = fusion or DEFAULT_HYBRID_FUSION_POLICY
        depth = max(top_k, policy.candidate_depth)
        rrf = models.Rrf(k=policy.rrf_k, weights=[policy.dense_weight, 1.0 - policy.dense_weight])
        options = _stored_bm25_options(config)
        requests = [
            models.QueryRequest(
                prefetch=[
                    models.Prefetch(
                        query=vector, using=DENSE_VECTOR_NAME, filter=query_filter, params=params, limit=depth
                    ),
                    models.Prefetch(
                        query=_bm25(text, options), using=SPARSE_VECTOR_NAME, filter=query_filter, limit=depth
                    ),
                ],
                query=models.RrfQuery(rrf=rrf),
                limit=top_k,
                with_payload=True,
            )
            for vector, text in zip(vectors, query_texts)
        ]
        return self._search(name, requests, "_relevance_score")

    def sparse_retrieval(self, query_texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Search the BM25 vector with raw query strings. Takes the same options as :meth:`retrieval`."""
        query_filter, _, top_k = self._search_options(kwargs)
        query_texts = [query_texts] if isinstance(query_texts, str) else list(query_texts)
        if not query_texts:
            return []
        name = self._name(kwargs)
        config = self._existing_config(name, _has_bm25)
        if not _has_bm25(config):
            raise ValueError(
                f"Qdrant collection {name!r} has no BM25 vector; ingest with sparse=True or hybrid=True to use "
                "sparse retrieval."
            )
        options = _stored_bm25_options(config)
        requests = [
            models.QueryRequest(
                query=_bm25(text, options),
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            for text in query_texts
        ]
        return self._search(name, requests, "_score")

    def _get_collection_store(self) -> Any:
        store = self._collection_store
        if store is None:
            with self._collection_store_lock:
                store = self._collection_store
                if store is None:
                    from nemo_retriever.common.vdb.qdrant_collections import QdrantCollectionStore

                    try:
                        store = QdrantCollectionStore(self, expiration_cleanup_enabled=self.expiration_cleanup_enabled)
                    except Exception:
                        self._collection_store_init_failed = True
                        raise
                    self._collection_store_init_failed = False
                    self._collection_store = store
        return store

    def health(self) -> dict[str, Any]:
        """Report the collection layout, row count and catalog state.

        Raises:
            RuntimeError: If the collection catalog failed to initialize.
        """
        from nemo_retriever.common.vdb.qdrant_collections import QdrantCollectionStore

        config = self._config(self.collection_name, cached=True)
        mode = None if config is None else _layout(self.collection_name, config)
        if self._collection_store_init_failed:
            raise RuntimeError("Collection catalog initialization failed")
        store = self._collection_store
        return {
            **(store.health() if store is not None else QdrantCollectionStore.empty_health()),
            "total_rows": self.client.count(self.collection_name, exact=True).count if config else 0,
            "table_exists": config is not None,
            "effective_retrieval_mode": mode,
            "retrieval_strategies": [mode] if mode in ("dense", "hybrid", "sparse") else [],
        }

    def create_collection(self, *, scope: str, request: CollectionCreateRequest) -> CollectionInfo:
        """Create a logical collection. See ``QdrantCollectionStore``."""
        return self._get_collection_store().create_collection(scope, request)

    def get_collection(self, *, scope: str, collection_name: str) -> CollectionInfo:
        """Return a logical collection."""
        return self._get_collection_store().get_collection(scope, collection_name)

    def list_collections(self, *, scope: str, limit: int, continuation_token: str | None) -> CollectionPage:
        """List the logical collections in ``scope``."""
        return self._get_collection_store().list_collections(scope, limit, continuation_token)

    def update_collection(
        self, *, scope: str, collection_name: str, request: CollectionUpdateRequest
    ) -> CollectionInfo:
        """Update a logical collection's settings."""
        return self._get_collection_store().update_collection(scope, collection_name, request)

    def delete_collection(self, *, scope: str, collection_name: str, if_exists: bool) -> CollectionDeleteResult:
        """Delete a logical collection and its chunks."""
        return self._get_collection_store().delete_collection(scope, collection_name, if_exists)

    def get_document(self, *, scope: str, collection_name: str, document_id: str) -> DocumentInfo:
        """Return a document in a logical collection."""
        return self._get_collection_store().get_document(scope, collection_name, document_id)

    def list_documents(
        self, *, scope: str, collection_name: str, limit: int, continuation_token: str | None
    ) -> DocumentPage:
        """List the documents in a logical collection."""
        return self._get_collection_store().list_documents(scope, collection_name, limit, continuation_token)

    def delete_document(
        self, *, scope: str, collection_name: str, document_id: str, if_exists: bool
    ) -> DocumentDeleteResult:
        """Delete a document and its chunks."""
        return self._get_collection_store().delete_document(scope, collection_name, document_id, if_exists)

    def write_collection(self, records: list, *, context: CollectionWriteContext) -> CollectionWriteResult:
        """Append or replace documents in a logical collection."""
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
        """Search a logical collection. Returns the hits per query and the retrieval strategies used."""
        return self._get_collection_store().retrieve_collection(
            vectors, scope=scope, collection_name=collection_name, query_texts=query_texts, top_k=top_k, **kwargs
        )

    def reconcile_collections(self) -> dict[str, int]:
        """Finish interrupted writes and deletes, and remove expired collections."""
        return self._get_collection_store().reconcile_collections()
