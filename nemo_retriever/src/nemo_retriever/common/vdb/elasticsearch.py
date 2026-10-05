# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elasticsearch dense-vector backend with optional NVIDIA cuVS indexing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
import base64
import json

import numpy as np
from typing import Any

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
    UnsupportedVDBOperation,
    VDB,
)
from nemo_retriever.common.vdb.lancedb import _create_lancedb_result

_SUPPORTED_INDEX_TYPES = {"hnsw", "int8_hnsw"}
_SUPPORTED_SIMILARITIES = {"cosine", "dot_product", "l2_norm", "max_inner_product"}


class Elasticsearch(VDB):
    """Store and retrieve NRL embeddings through Elasticsearch.

    Elasticsearch uses NVIDIA cuVS only while constructing supported HNSW
    graphs. Search remains CPU based. Set require_gpu=True to reject a cluster
    that cannot confirm GPU vector indexing through the xpack usage API.
    """

    supports_stream_ingest = True

    def __init__(
        self,
        *,
        url: str | Sequence[str] = "http://localhost:9200",
        index_name: str = "nemo-retriever",
        vector_dim: int = 2048,
        index_type: str = "int8_hnsw",
        metric: str = "cosine",
        hnsw_m: int = 64,
        hnsw_ef_construction: int = 2000,
        num_candidates: int = 10_000,
        overwrite: bool = True,
        batch_size: int = 500,
        bulk_workers: int = 16,
        bulk_queue_size: int = 16,
        encode_vectors_base64: bool = True,
        max_chunk_bytes: int = 100 * 1024 * 1024,
        request_timeout: float = 120.0,
        transport_max_retries: int = 3,
        retry_on_timeout: bool = True,
        bulk_max_retries: int = 3,
        bulk_initial_backoff: float = 2.0,
        bulk_max_backoff: float = 600.0,
        delete_index_on_failure: bool = True,
        refresh_on_finish: bool = True,
        require_gpu: bool = True,
        number_of_shards: int | None = None,
        number_of_replicas: int | None = None,
        refresh_interval: str | None = "-1",
        input_format: str = "nrl",
        index_mode: str | None = "vectordb_document",
        use_auto_ids: bool = False,
        api_key: str | tuple[str, str] | None = None,
        basic_auth: tuple[str, str] | None = None,
        ca_certs: str | None = None,
        verify_certs: bool = True,
        client: Any | None = None,
        bulk_helper: Any | None = None,
        **kwargs: Any,
    ) -> None:
        for name, value in (
            ("vector_dim", vector_dim),
            ("hnsw_m", hnsw_m),
            ("hnsw_ef_construction", hnsw_ef_construction),
            ("num_candidates", num_candidates),
            ("batch_size", batch_size),
            ("bulk_workers", bulk_workers),
            ("bulk_queue_size", bulk_queue_size),
            ("max_chunk_bytes", max_chunk_bytes),
            ("transport_max_retries", transport_max_retries),
            ("bulk_max_retries", bulk_max_retries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        index_type = str(index_type).strip().lower()
        if index_type not in _SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(_SUPPORTED_INDEX_TYPES)} for cuVS indexing; got {index_type!r}"
            )
        metric = str(metric).strip().lower()
        if metric not in _SUPPORTED_SIMILARITIES:
            raise ValueError(f"metric must be one of {sorted(_SUPPORTED_SIMILARITIES)}; got {metric!r}")

        self.url = url
        self.index_name = index_name
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.metric = metric
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.num_candidates = num_candidates
        self.overwrite = bool(overwrite)
        self.supports_stream_ingest = True
        self.batch_size = batch_size
        self.bulk_workers = bulk_workers
        self.bulk_queue_size = bulk_queue_size
        self.encode_vectors_base64 = bool(encode_vectors_base64)
        self.max_chunk_bytes = max_chunk_bytes
        self.request_timeout = float(request_timeout)
        self.transport_max_retries = transport_max_retries
        self.retry_on_timeout = bool(retry_on_timeout)
        self.bulk_max_retries = bulk_max_retries
        self.bulk_initial_backoff = float(bulk_initial_backoff)
        self.bulk_max_backoff = float(bulk_max_backoff)
        self.delete_index_on_failure = bool(delete_index_on_failure)
        self.refresh_on_finish = bool(refresh_on_finish)
        self.require_gpu = bool(require_gpu)
        self.number_of_shards = number_of_shards
        self.number_of_replicas = number_of_replicas
        self.refresh_interval = refresh_interval
        self.index_mode = index_mode
        self.use_auto_ids = bool(use_auto_ids)
        self.input_format = str(input_format).strip().lower()
        if self.input_format not in {"nrl", "embedding_parquet"}:
            raise ValueError("input_format must be 'nrl' or 'embedding_parquet'")
        self.supports_prepared_stream_ingest = self.input_format == "embedding_parquet"
        self._client = client
        self._bulk_helper = bulk_helper
        self._client_kwargs = {
            "api_key": api_key,
            "basic_auth": basic_auth,
            "ca_certs": ca_certs,
            "verify_certs": verify_certs,
            "request_timeout": request_timeout,
            "max_retries": transport_max_retries,
            "retry_on_timeout": self.retry_on_timeout,
        }
        super().__init__(**kwargs)

    @property
    def client(self) -> Any:
        """Lazily construct the Elasticsearch client."""
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch as ElasticsearchClient
            except ImportError as exc:
                raise ImportError(
                    "The Elasticsearch VDB backend requires the elasticsearch Python package. "
                    "Install NeMo Retriever Library with the elasticsearch extra."
                ) from exc
            client_kwargs = {key: value for key, value in self._client_kwargs.items() if value is not None}
            self._client = ElasticsearchClient(self.url, **client_kwargs)
        return self._client

    def _gpu_usage(self) -> dict[str, Any]:
        xpack = getattr(self.client, "xpack", None)
        if xpack is not None and callable(getattr(xpack, "usage", None)):
            response = xpack.usage(filter_path="gpu_vector_indexing")
        else:
            response = self.client.perform_request(
                "GET", "/_xpack/usage", params={"filter_path": "gpu_vector_indexing"}
            )
        body = getattr(response, "body", response)
        return dict((body or {}).get("gpu_vector_indexing") or {})

    def _assert_gpu_indexing(self) -> dict[str, Any]:
        usage = self._gpu_usage()
        if self.require_gpu and not (
            usage.get("available") is True
            and usage.get("enabled") is True
            and int(usage.get("nodes_with_gpu") or 0) > 0
        ):
            raise RuntimeError(
                "Elasticsearch cuVS indexing is required, but xpack usage did not report an enabled GPU node. "
                "Configure vectors.indexing.use_gpu=true and a supported CUDA/cuVS runtime."
            )
        return usage

    def _index_definition(self) -> dict[str, Any]:
        settings: dict[str, Any] = {"index.mapping.exclude_source_vectors": True}
        if self.index_mode is not None:
            settings["index.mode"] = str(self.index_mode)
        if self.number_of_shards is not None:
            settings["number_of_shards"] = int(self.number_of_shards)
        if self.number_of_replicas is not None:
            settings["number_of_replicas"] = int(self.number_of_replicas)
        if self.refresh_interval is not None:
            settings["refresh_interval"] = str(self.refresh_interval)
        return {
            "settings": settings,
            "mappings": {
                "_meta": {
                    "nrl_backend": "elasticsearch",
                    "index_type": self.index_type,
                    "hnsw_m": self.hnsw_m,
                    "hnsw_ef_construction": self.hnsw_ef_construction,
                    "num_candidates": self.num_candidates,
                },
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.vector_dim,
                        "element_type": "float",
                        "index": True,
                        "similarity": self.metric,
                        "index_options": {
                            "type": self.index_type,
                            "m": self.hnsw_m,
                            "ef_construction": self.hnsw_ef_construction,
                        },
                    },
                    "text": {"type": "text", "index": False},
                    # Retain these retrieval payloads in _source without
                    # dynamically indexing corpus-specific metadata keys.
                    "metadata": {"type": "object", "enabled": False},
                    "source": {"type": "object", "enabled": False},
                    "id": {"type": "keyword"},
                },
            },
        }

    def create_index(self, **kwargs: Any) -> Any:
        """Create the Elasticsearch index with the configured HNSW mapping."""
        recreate = bool(kwargs.pop("recreate", self.overwrite))
        self._assert_gpu_indexing()
        exists = bool(self.client.indices.exists(index=self.index_name))
        if exists and recreate:
            self.client.indices.delete(index=self.index_name)
            exists = False
        if not exists:
            return self.client.indices.create(index=self.index_name, **self._index_definition())
        return None

    @staticmethod
    def _decode_json_object(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        return {}

    def _row_from_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        row, _ = _create_lancedb_result(record, expected_dim=self.vector_dim)
        if row is None:
            return None
        row["metadata"] = self._decode_json_object(row.get("metadata"))
        row["source"] = self._decode_json_object(row.get("source"))
        if self.encode_vectors_base64:
            vector_bytes = np.asarray(row["vector"], dtype=">f4").tobytes()
            row["vector"] = base64.b64encode(vector_bytes).decode("ascii")
        return row

    def _actions(self, records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for record in records:
            row = self._row_from_record(record)
            if row is None:
                continue
            action: dict[str, Any] = {"_index": self.index_name, "_source": row}
            if row.get("id") and not self.use_auto_ids:
                action["_id"] = row["id"]
            yield action

    def _streaming_bulk(self, actions: Iterable[dict[str, Any]]) -> int:
        helper = self._bulk_helper
        if helper is None:
            try:
                from elasticsearch.helpers import parallel_bulk, streaming_bulk
            except ImportError as exc:
                raise ImportError("The Elasticsearch VDB backend requires the elasticsearch Python package.") from exc
            helper = parallel_bulk if self.bulk_workers > 1 else streaming_bulk
        helper_kwargs = {
            "chunk_size": self.batch_size,
            "max_chunk_bytes": self.max_chunk_bytes,
            "request_timeout": self.request_timeout,
            "raise_on_error": True,
            "raise_on_exception": True,
        }
        if self.bulk_workers > 1:
            helper_kwargs.update(thread_count=self.bulk_workers, queue_size=self.bulk_queue_size)
        else:
            # These retry controls belong to streaming_bulk only. Passing them
            # through parallel_bulk forwards unsupported keywords to client.bulk
            # and can retain failed chunks in the ordered worker result queue.
            helper_kwargs.update(
                max_retries=self.bulk_max_retries,
                initial_backoff=self.bulk_initial_backoff,
                max_backoff=self.bulk_max_backoff,
            )
        written = 0
        for ok, item in helper(
            self.client,
            actions,
            **helper_kwargs,
        ):
            if not ok:
                raise RuntimeError(f"Elasticsearch bulk indexing failed: {item!r}")
            written += 1
        return written

    def write_to_index(self, records: list, **kwargs: Any) -> dict[str, int]:
        """Bulk-index nested canonical NRL record batches."""
        actions = self._actions(record for batch in records for record in batch)
        written = self._streaming_bulk(actions)
        if self.refresh_on_finish:
            self.client.indices.refresh(index=self.index_name)
        return {"written": written}

    def stream_ingest(self, records: Iterable[dict[str, Any]]) -> None:
        """Consume canonical records lazily and send bounded bulk requests."""
        self.create_index()
        try:
            self._streaming_bulk(self._actions(records))
            if self.refresh_on_finish:
                self.client.indices.refresh(index=self.index_name)
        except BaseException:
            # Elasticsearch bulk requests commit incrementally. Remove the
            # incomplete replacement so a late conversion or transport error
            # never leaves a successful-looking prefix behind.
            if self.delete_index_on_failure and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
            raise

    def prepare_embedding_parquet_batch(self, batch: Any) -> Any:
        """Validate and project one staged embedding Parquet Arrow batch."""
        if not self.supports_prepared_stream_ingest:
            raise UnsupportedVDBOperation(
                "Elasticsearch embedding Parquet input requires input_format=embedding_parquet."
            )
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError("Embedding Parquet input requires pyarrow.") from exc
        if isinstance(batch, pa.RecordBatch):
            batch = pa.Table.from_batches([batch])
        if not isinstance(batch, pa.Table):
            raise TypeError(f"Embedding Parquet batches must be pyarrow.Table, got {type(batch).__name__}.")
        required = {"id", "text", "vector"}
        missing = sorted(required.difference(batch.column_names))
        if missing:
            raise ValueError(f"Embedding Parquet batch is missing required column(s): {', '.join(missing)}")
        vector_type = batch.schema.field("vector").type
        if not pa.types.is_fixed_size_list(vector_type) or vector_type.list_size != self.vector_dim:
            raise ValueError(
                f"Embedding Parquet vector type must be fixed_size_list<float32>[{self.vector_dim}]; "
                f"got {vector_type}."
            )
        if vector_type.value_type != pa.float32():
            raise ValueError(f"Embedding Parquet vectors must contain float32 values; got {vector_type.value_type}.")
        optional = [
            name for name in ("document_id", "url", "file_path", "dump", "source_id") if name in batch.column_names
        ]
        selected = batch.select(["id", "text", *optional, "vector"])
        if not self.encode_vectors_base64:
            return selected
        vector_array = selected.column("vector").combine_chunks()
        vector_values = vector_array.values.to_numpy(zero_copy_only=False).reshape(selected.num_rows, self.vector_dim)
        encoded = pa.array(
            [base64.b64encode(np.asarray(vector, dtype=">f4").tobytes()).decode("ascii") for vector in vector_values],
            type=pa.string(),
        )
        return selected.drop(["vector"]).append_column("vector_b64", encoded)

    @staticmethod
    def _arrow_value(column: Any, row_index: int) -> Any:
        value = column[row_index]
        return value.as_py() if value.is_valid else None

    def _prepared_actions(self, batches: Iterable[Any]) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError("Embedding Parquet input requires pyarrow.") from exc
        for value in batches:
            if isinstance(value, pa.Table):
                tables = (value,)
            elif isinstance(value, pa.RecordBatch):
                tables = (pa.Table.from_batches([value]),)
            else:
                raise TypeError(
                    "Prepared Elasticsearch batches must be pyarrow.Table or "
                    f"pyarrow.RecordBatch, got {type(value).__name__}."
                )
            for table in tables:
                columns = {name: table.column(name) for name in table.column_names}
                vector_values = None
                if "vector_b64" not in columns:
                    vector_array = columns["vector"].combine_chunks()
                    vector_values = vector_array.values.to_numpy(zero_copy_only=False).reshape(
                        table.num_rows, self.vector_dim
                    )
                for row_index in range(table.num_rows):
                    raw_id = self._arrow_value(columns["id"], row_index)
                    if raw_id is None:
                        raise ValueError("Embedding Parquet id values must not be null.")
                    doc_id = str(raw_id)
                    if "vector_b64" in columns:
                        vector = self._arrow_value(columns["vector_b64"], row_index)
                    else:
                        assert vector_values is not None
                        vector = vector_values[row_index]
                        if self.encode_vectors_base64:
                            vector = base64.b64encode(np.asarray(vector, dtype=">f4").tobytes()).decode("ascii")
                        else:
                            vector = vector.tolist()
                    metadata: dict[str, Any] = {"id": doc_id, "type": "text"}
                    source: dict[str, Any] = {}
                    for name in ("document_id", "url", "file_path", "dump", "source_id"):
                        if name not in columns:
                            continue
                        item = self._arrow_value(columns[name], row_index)
                        if item is not None:
                            source[name] = item
                            if name == "document_id":
                                metadata[name] = item
                    if source.get("url"):
                        source["source_name"] = source["url"]
                    row = {
                        "id": doc_id,
                        "text": self._arrow_value(columns["text"], row_index),
                        "metadata": metadata,
                        "source": source,
                        "vector": vector,
                    }
                    action: dict[str, Any] = {"_index": self.index_name, "_source": row}
                    if not self.use_auto_ids:
                        action["_id"] = doc_id
                    yield action

    def stream_ingest_prepared_batches(self, batches: Iterable[Any]) -> None:
        """Stream staged embedding Parquet batches into one replacement index."""
        if not self.supports_prepared_stream_ingest:
            raise UnsupportedVDBOperation("Elasticsearch prepared streaming is unavailable for this configuration.")
        self.create_index()
        try:
            self._streaming_bulk(self._prepared_actions(batches))
            if self.refresh_on_finish:
                self.client.indices.refresh(index=self.index_name)
        except BaseException:
            if self.delete_index_on_failure and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
            raise

    def run(self, records: list) -> list:
        self.create_index()
        self.write_to_index(records)
        return records

    def retrieval(self, queries: Iterable[Sequence[float]], **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Search with precomputed vectors and a 10,000-candidate default."""
        top_k = int(kwargs.pop("top_k", 10))
        num_candidates = int(kwargs.pop("num_candidates", self.num_candidates))
        if num_candidates < top_k:
            raise ValueError("num_candidates must be greater than or equal to top_k")
        retrieval_workers = int(kwargs.pop("retrieval_workers", 1))
        if retrieval_workers <= 0:
            raise ValueError("retrieval_workers must be positive")
        result_fields = kwargs.pop("result_fields", None)
        query_filter = kwargs.pop("filter", kwargs.pop("_filter", None))

        def search_one(vector: Sequence[float]) -> list[dict[str, Any]]:
            knn: dict[str, Any] = {
                "field": "vector",
                "query_vector": list(vector),
                "k": top_k,
                "num_candidates": num_candidates,
            }
            if query_filter is not None:
                knn["filter"] = query_filter
            body: dict[str, Any] = {
                "size": top_k,
                "knn": knn,
                "_source": {"excludes": ["vector"]},
            }
            if result_fields is not None:
                body["_source"]["includes"] = list(result_fields)
            response = self.client.search(index=self.index_name, body=body)
            response_body = getattr(response, "body", response)
            query_hits: list[dict[str, Any]] = []
            for hit in response_body.get("hits", {}).get("hits", []):
                item = dict(hit.get("_source") or {})
                item.setdefault("id", str(hit.get("_id", "")))
                if "_score" in hit:
                    item["_score"] = hit["_score"]
                query_hits.append(item)
            return query_hits

        query_vectors = list(queries)
        if retrieval_workers == 1 or len(query_vectors) <= 1:
            return [search_one(vector) for vector in query_vectors]
        worker_count = min(retrieval_workers, len(query_vectors))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="elasticsearch-query") as pool:
            return list(pool.map(search_one, query_vectors))

    def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
        response = self.client.indices.get_mapping(index=self.index_name)
        body = getattr(response, "body", response)
        value = body.get(self.index_name, {}).get("mappings", {}).get("_meta", {}).get(key)
        return str(value) if value is not None else None

    def health(self) -> dict[str, Any]:
        return {"gpu_vector_indexing": self._gpu_usage()}

    @staticmethod
    def _unsupported(operation: str) -> None:
        raise UnsupportedVDBOperation(
            f"Elasticsearch does not yet implement the NRL collection operation {operation!r}."
        )

    def create_collection(self, *, scope: str, request: CollectionCreateRequest) -> CollectionInfo:
        self._unsupported("create_collection")

    def get_collection(self, *, scope: str, collection_name: str) -> CollectionInfo:
        self._unsupported("get_collection")

    def list_collections(self, *, scope: str, limit: int, continuation_token: str | None) -> CollectionPage:
        self._unsupported("list_collections")

    def update_collection(
        self, *, scope: str, collection_name: str, request: CollectionUpdateRequest
    ) -> CollectionInfo:
        self._unsupported("update_collection")

    def delete_collection(self, *, scope: str, collection_name: str, if_exists: bool) -> CollectionDeleteResult:
        self._unsupported("delete_collection")

    def get_document(self, *, scope: str, collection_name: str, document_id: str) -> DocumentInfo:
        self._unsupported("get_document")

    def list_documents(
        self, *, scope: str, collection_name: str, limit: int, continuation_token: str | None
    ) -> DocumentPage:
        self._unsupported("list_documents")

    def delete_document(
        self, *, scope: str, collection_name: str, document_id: str, if_exists: bool
    ) -> DocumentDeleteResult:
        self._unsupported("delete_document")

    def write_collection(self, records: list, *, context: CollectionWriteContext) -> CollectionWriteResult:
        self._unsupported("write_collection")

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
        self._unsupported("retrieve_collection")
