import json
import logging
import os
import time

from datetime import timedelta
from functools import partial
from typing import Any, Final, FrozenSet
from urllib.parse import urlparse

import lancedb
import pyarrow as pa
from lancedb.rerankers import RRFReranker

from nv_ingest_client.util.transport import infer_microservice

from nemo_retriever.vdb.adt_vdb import VDB
from nemo_retriever.vdb.milvus import nv_rerank

logger = logging.getLogger(__name__)


_DEFAULT_VECTOR_DIM: Final[int] = 2048
_VALID_ON_BAD_VECTORS: Final[FrozenSet[str]] = frozenset({"drop", "fill", "null", "error"})


def _normalize_on_bad_vectors(value: str) -> str:
    """Validate and normalize an ``on_bad_vectors`` policy string.

    LanceDB's ``Table.create`` accepts a fixed set of policies for handling rows
    whose vector column does not match the declared fixed-size schema. We
    surface the same vocabulary on this wrapper so callers can configure the
    behavior through ``--vdb-kwargs-json``.

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
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)
    with open(timing_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _get_text_for_element(element):
    """
    Extract searchable text from an element based on document_type.

    Matches Milvus behavior: for images, use caption/OCR text instead of raw content.
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
    """Transform NV-Ingest pipeline results into LanceDB ingestible rows.

    Extracts the appropriate searchable text per ``document_type`` and, when
    ``expected_dim`` is set, validates that each row's embedding is shaped
    consistently with the LanceDB fixed-size-list schema before forwarding it
    to the writer. Rows whose embedding is missing, of the wrong type, or of
    the wrong length are dropped and counted; per-row reasons are emitted at
    ``DEBUG`` and a single structured ``WARNING`` summary is emitted at the
    end of the call when any drops occurred.

    Passing ``expected_dim=None`` disables the length check entirely. Callers
    that prefer to defer to LanceDB's ``on_bad_vectors`` policy on the writer
    side (e.g. ``LanceDB(on_bad_vectors="error")``) should use this mode so
    bad rows reach LanceDB rather than being silently dropped at the wrapper.

    Args:
        results: Iterable of pipeline output result lists, where each element
            is a per-document list of NV-Ingest record dicts.
        expected_dim: Required vector length, or ``None`` to skip the length
            check. Defaults to :data:`_DEFAULT_VECTOR_DIM`.

    Returns:
        ``(rows, counts)`` where ``rows`` is the list of dicts shaped for
        LanceDB ingestion (``vector``, ``text``, ``metadata``, ``source``)
        and ``counts`` is a dict containing ``accepted``,
        ``dropped_no_embedding``, ``dropped_bad_length``, and
        ``dropped_no_text`` keys.
    """
    lancedb_rows: list = []
    accepted = 0
    dropped_no_embedding = 0
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

            if not text:
                dropped_no_text += 1
                source_name = metadata.get("source_metadata", {}).get("source_name", "unknown")
                pg_num = content_meta.get("page_number")
                logger.debug(f"No text found for entity: {source_name} page: {pg_num} type: {doc_type}")
                continue

            lancedb_rows.append(
                {
                    "vector": embedding,
                    "text": text,
                    "metadata": _json_str(content_meta),
                    "source": _json_str(metadata.get("source_metadata", {})),
                }
            )
            accepted += 1

    counts: dict[str, int] = {
        "accepted": accepted,
        "dropped_no_embedding": dropped_no_embedding,
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
        fts_language: str = "English",
        vector_dim: int = _DEFAULT_VECTOR_DIM,
        on_bad_vectors: str = "drop",
        fill_value: float = 0.0,
        validate_vector_length: bool = True,
        **kwargs,
    ):
        if int(vector_dim) <= 0:
            raise ValueError(f"vector_dim must be positive; got {vector_dim}")
        self.uri = uri or "lancedb"
        self.overwrite = overwrite
        self.table_name = table_name
        self.index_type = index_type
        self.metric = metric
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        self.hybrid = hybrid
        self.fts_language = fts_language
        self.vector_dim = int(vector_dim)
        self.on_bad_vectors = _normalize_on_bad_vectors(on_bad_vectors)
        self.fill_value = float(fill_value)
        self.validate_vector_length = bool(validate_vector_length)
        super().__init__(**kwargs)

    def create_index(self, records=None, table_name: str = "nv-ingest", **kwargs):
        """Create a LanceDB table and populate it with transformed records.

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

        if self.validate_vector_length and self.on_bad_vectors != "error":
            expected_dim: int | None = self.vector_dim
        else:
            expected_dim = None

        results, counts = _create_lancedb_results(records, expected_dim=expected_dim)
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("source", pa.string()),
            ]
        )
        create_kwargs: dict[str, Any] = {
            "data": results,
            "schema": schema,
            "mode": "overwrite" if self.overwrite else "append",
            "on_bad_vectors": self.on_bad_vectors,
        }
        if self.on_bad_vectors == "fill":
            create_kwargs["fill_value"] = self.fill_value
        create_start = time.perf_counter()
        table = db.create_table(table_name, **create_kwargs)
        _record_timing(
            "lancedb.create_table",
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
        hybrid: bool = None,
        fts_language: str = None,
        **kwargs,
    ):
        """Create vector and optionally FTS indexes on the LanceDB table."""
        hybrid = hybrid if hybrid is not None else self.hybrid
        fts_language = fts_language or self.fts_language

        vector_index_start = time.perf_counter()
        table.create_index(
            index_type=index_type,
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
        )
        for index_stub in table.list_indices():
            table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))
        _record_timing("lancedb.vector_index_ready", time.perf_counter() - vector_index_start)

        if hybrid:
            fts_index_start = time.perf_counter()
            table.create_fts_index("text", language=fts_language)
            for index_stub in table.list_indices():
                if "text" in index_stub.name.lower() or "fts" in index_stub.name.lower():
                    table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))
            _record_timing("lancedb.fts_index_ready", time.perf_counter() - fts_index_start)

    def run(self, records):
        """Orchestrate index creation and data ingestion."""
        table = self.create_index(records=records, table_name=self.table_name)
        self.write_to_index(
            records,
            table=table,
            index_type=self.index_type,
            metric=self.metric,
            num_partitions=self.num_partitions,
            num_sub_vectors=self.num_sub_vectors,
            hybrid=self.hybrid,
            fts_language=self.fts_language,
        )
        return records

    def retrieval(self, vectors, **kwargs):
        """Search LanceDB with precomputed query vectors."""
        hybrid = kwargs.pop("hybrid", self.hybrid)
        if hybrid:
            raise NotImplementedError("LanceDB hybrid retrieval with precomputed vectors is not implemented yet.")
        table_path = kwargs.pop("table_path", self.uri)
        table_name = kwargs.pop("table_name", self.table_name)

        table = lancedb.connect(uri=table_path).open_table(table_name)

        result_fields = kwargs.pop("result_fields", None)
        top_k = int(kwargs.pop("top_k", 10))
        refine_factor = int(kwargs.pop("refine_factor", 50))
        n_probe = int(kwargs.pop("n_probe", kwargs.pop("nprobes", 64)))
        vector_column_name = str(kwargs.pop("vector_column_name", "vector"))

        search_results = []
        for vector in vectors:
            query = (
                table.search([vector], vector_column_name=vector_column_name)
                .limit(top_k)
                .refine_factor(refine_factor)
                .nprobes(n_probe)
            )
            if result_fields is not None:
                query = query.select(result_fields)
            results = query.to_list()
            search_results.append(results)

        return search_results


def lancedb_retrieval(
    queries,
    table_path: str = None,
    table_name: str = "nv-ingest",
    table=None,
    embedding_endpoint: str = "http://localhost:8012/v1",
    nvidia_api_key: str = None,
    model_name: str = "nvidia/llama-nemotron-embed-1b-v2",
    top_k: int = 10,
    refine_factor: int = 50,
    n_probe: int = 64,
    nv_ranker: bool = False,
    nv_ranker_endpoint: str = None,
    nv_ranker_model_name: str = None,
    nv_ranker_top_k: int = 50,
    result_fields=None,
    **kwargs,
):
    """
    Standalone LanceDB retrieval function.

    This is the primary interface for LanceDB vector search. It embeds queries using the
    specified embedding model and performs vector similarity search against
    a LanceDB table.

    Parameters
    ----------
    queries : list[str]
        Text queries to search for.
    table_path : str, optional
        Path to the LanceDB database directory (the `uri` used during ingestion).
        Required if `table` is not provided.
    table_name : str, optional
        Name of the table within the LanceDB database (default: "nv-ingest").
    table : object, optional
        Pre-opened LanceDB table object. If provided, table_path and table_name
        are ignored. Useful for reusing connections or testing.
    embedding_endpoint : str, optional
        URL of the embedding microservice (default: "http://localhost:8012/v1").
    nvidia_api_key : str, optional
        NVIDIA API key for authentication. If None, no auth is used.
    model_name : str, optional
        Name of the embedding model (default: "nvidia/llama-nemotron-embed-1b-v2").
    top_k : int, optional
        Number of results to return per query (default: 10).
    refine_factor : int, optional
        LanceDB search refine factor for accuracy (default: 50).
    n_probe : int, optional
        Number of partitions to probe during search (default: 64).
    nv_ranker : bool, optional
        Whether to apply NV reranker after retrieval (default: False).
    nv_ranker_endpoint : str, optional
        URL of the reranker microservice.
    nv_ranker_model_name : str, optional
        Name of the reranker model.
    nv_ranker_top_k : int, optional
        Number of candidates to fetch before reranking (default: 50).
    result_fields : list, optional
        List of field names to retrieve from each hit document (default:
        ["text", "metadata", "source"]).
    **kwargs
        Additional keyword arguments (ignored, for API compatibility).

    Returns
    -------
    list[list[dict]]
        For each query, a list of result dicts formatted to match Milvus output
        structure for recall scoring compatibility.
    """
    if table is None:
        if table_path is None:
            raise ValueError("Either table or table_path must be provided")
        db = lancedb.connect(uri=table_path)
        table = db.open_table(table_name)

    if result_fields is None:
        result_fields = ["text", "metadata", "source"]

    embed_start = time.perf_counter()
    embed_model = partial(
        infer_microservice,
        model_name=model_name,
        embedding_endpoint=embedding_endpoint,
        nvidia_api_key=nvidia_api_key,
        input_type="query",
        output_names=["embeddings"],
        grpc=not ("http" in urlparse(embedding_endpoint).scheme),
    )
    query_embeddings = embed_model(queries)
    _record_timing(
        "lancedb.embed_queries",
        time.perf_counter() - embed_start,
        {"query_count": len(queries)},
    )

    search_top_k = nv_ranker_top_k if nv_ranker else top_k

    results = []
    search_durations = []
    for query_embed in query_embeddings:
        search_start = time.perf_counter()
        search_results = (
            table.search([query_embed], vector_column_name="vector")
            .select(result_fields)
            .limit(search_top_k)
            .refine_factor(refine_factor)
            .nprobes(n_probe)
            .to_list()
        )
        search_durations.append(time.perf_counter() - search_start)
        formatted = []
        for r in search_results:
            source = _maybe_parse_json(r.get("source"))
            content_meta = _maybe_parse_json(r.get("metadata"))

            if not isinstance(source, dict):
                source = {"source_id": "" if source is None else str(source)}
            if not isinstance(content_meta, dict):
                # Back-compat: previously this code used `metadata` as `page_number`.
                try:
                    page_number = int(content_meta)  # type: ignore[arg-type]
                except Exception:
                    page_number = -1
                content_meta = {"page_number": page_number}

            formatted.append(
                {
                    "entity": {
                        "source": source,
                        "content_metadata": content_meta,
                        "text": r.get("text"),
                    }
                }
            )
        results.append(formatted)
    if search_durations:
        total_search = sum(search_durations)
        _record_timing(
            "lancedb.vector_search",
            total_search,
            {
                "query_count": len(search_durations),
                "avg_query_s": total_search / len(search_durations),
                "max_query_s": max(search_durations),
            },
        )

    if nv_ranker:
        rerank_start = time.perf_counter()
        rerank_results = []
        num_queries = len(queries)
        for idx, (query, candidates) in enumerate(zip(queries, results), 1):
            if num_queries > 1:
                logger.info(f"Reranking query {idx}/{num_queries}")
            rerank_results.append(
                nv_rerank(
                    query,
                    candidates,
                    reranker_endpoint=nv_ranker_endpoint,
                    model_name=nv_ranker_model_name,
                    topk=top_k,
                )
            )
        results = rerank_results
        _record_timing(
            "lancedb.nv_rerank",
            time.perf_counter() - rerank_start,
            {"query_count": num_queries},
        )

    return results


def lancedb_hybrid_retrieval(
    queries,
    table_path: str = None,
    table_name: str = "nv-ingest",
    table=None,
    embedding_endpoint: str = "http://localhost:8012/v1",
    nvidia_api_key: str = None,
    model_name: str = "nvidia/llama-nemotron-embed-1b-v2",
    top_k: int = 10,
    refine_factor: int = 50,
    n_probe: int = 64,
    nv_ranker: bool = False,
    nv_ranker_endpoint: str = None,
    nv_ranker_model_name: str = None,
    nv_ranker_top_k: int = 50,
    result_fields=None,
    **kwargs,
):
    """
    LanceDB hybrid retrieval combining vector similarity with BM25 full-text search.

    Uses RRF reranking to merge dense and sparse results. Requires an FTS index
    on the text column (set hybrid=True during ingestion).
    """
    if table is None:
        if table_path is None:
            raise ValueError("Either table or table_path must be provided")
        db = lancedb.connect(uri=table_path)
        table = db.open_table(table_name)

    if result_fields is None:
        result_fields = ["text", "metadata", "source"]

    embed_start = time.perf_counter()
    embed_model = partial(
        infer_microservice,
        model_name=model_name,
        embedding_endpoint=embedding_endpoint,
        nvidia_api_key=nvidia_api_key,
        input_type="query",
        output_names=["embeddings"],
        grpc=not ("http" in urlparse(embedding_endpoint).scheme),
    )
    query_embeddings = embed_model(queries)
    _record_timing(
        "lancedb.embed_queries",
        time.perf_counter() - embed_start,
        {"query_count": len(queries)},
    )

    search_top_k = nv_ranker_top_k if nv_ranker else top_k
    reranker = RRFReranker()

    results = []
    search_durations = []
    for query_text, query_embed in zip(queries, query_embeddings):
        search_start = time.perf_counter()
        search_results = (
            table.search(query_type="hybrid")
            .vector(query_embed)
            .text(query_text)
            .select(result_fields)
            .limit(search_top_k)
            .refine_factor(refine_factor)
            .nprobes(n_probe)
            .rerank(reranker)
            .to_list()
        )
        search_durations.append(time.perf_counter() - search_start)
        formatted = []
        for r in search_results:
            formatted.append(
                {
                    "entity": {
                        "source": {"source_id": r.get("source")},
                        "content_metadata": {"page_number": r.get("metadata")},
                        "text": r.get("text"),
                    }
                }
            )
        results.append(formatted)
    if search_durations:
        total_search = sum(search_durations)
        _record_timing(
            "lancedb.hybrid_search",
            total_search,
            {
                "query_count": len(search_durations),
                "avg_query_s": total_search / len(search_durations),
                "max_query_s": max(search_durations),
            },
        )

    if nv_ranker:
        rerank_start = time.perf_counter()
        rerank_results = []
        num_queries = len(queries)
        for idx, (query, candidates) in enumerate(zip(queries, results), 1):
            if num_queries > 1:
                logger.info(f"Reranking query {idx}/{num_queries}")
            rerank_results.append(
                nv_rerank(
                    query,
                    candidates,
                    reranker_endpoint=nv_ranker_endpoint,
                    model_name=nv_ranker_model_name,
                    topk=top_k,
                )
            )
        results = rerank_results
        _record_timing(
            "lancedb.nv_rerank",
            time.perf_counter() - rerank_start,
            {"query_count": num_queries},
        )

    return results
