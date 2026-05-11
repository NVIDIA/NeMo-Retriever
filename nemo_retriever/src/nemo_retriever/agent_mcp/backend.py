# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import shutil
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol

from nemo_retriever.agent_mcp.evidence import normalize_hits
from nemo_retriever.agent_mcp.models import (
    AgentMcpError,
    AgentMcpErrorCode,
    CollectionRecord,
    EvidenceHit,
    IngestJobRecord,
    JobStatus,
)
from nemo_retriever.agent_mcp.paths import PathPolicy, expand_local_paths, group_paths_by_media_type
from nemo_retriever.agent_mcp.registry import CollectionRegistry


CANONICAL_MEDIA_ORDER = ("document", "image", "text", "html", "audio", "video")


class RetrieverBackend(Protocol):
    def list_collections(self) -> list[CollectionRecord]:
        ...

    def create_collection(
        self,
        name: str = "default",
        temporary: bool = False,
        hybrid: bool = False,
    ) -> CollectionRecord:
        ...

    def describe_collection(self, name: str = "default") -> dict[str, Any]:
        ...

    def delete_collection(self, name: str, delete_data: bool = False) -> CollectionRecord:
        ...

    def query_collection(
        self,
        collection: str = "default",
        query: str = "",
        top_k: int = 10,
        hybrid: bool = False,
        rerank: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[EvidenceHit]:
        ...

    def rerank_results(
        self,
        query: str,
        hits: list[EvidenceHit],
        top_n: int | None = None,
    ) -> list[EvidenceHit]:
        ...

    def start_ingestion(
        self,
        collection: str,
        paths: list[str | Path],
        *,
        run_mode: str = "inprocess",
    ) -> IngestJobRecord:
        ...

    def ingest_local_paths(
        self,
        collection: str,
        paths: list[str | Path],
        *,
        wait: bool = True,
        timeout_s: float | None = None,
        run_mode: str = "inprocess",
    ) -> IngestJobRecord:
        ...

    def get_ingestion_status(self, job_id: str) -> IngestJobRecord:
        ...


class InProcessRetrieverBackend:
    def __init__(
        self,
        registry: CollectionRegistry,
        path_policy: PathPolicy,
        retriever_factory: Callable[..., Any] | None = None,
        ingestor_factory: Callable[..., Any] | None = None,
        rerank_fn: Callable[..., list[dict[str, Any]]] | None = None,
        reranker_model: Any | None = None,
        reranker_endpoint: str | None = None,
        reranker_model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
        reranker_api_key: str = "",
        reranker_max_length: int = 8192,
        reranker_batch_size: int = 8,
        rerank_modality: str = "text",
        local_reranker_backend: str = "vllm",
        local_hf_device: str | None = None,
        local_hf_cache_dir: str | Path | None = None,
        reranker_gpu_memory_utilization: float = 0.5,
        max_workers: int = 2,
    ) -> None:
        self.registry = registry
        self.path_policy = path_policy
        self._retriever_factory = retriever_factory
        self._ingestor_factory = ingestor_factory
        self._rerank_fn = rerank_fn
        self._custom_rerank_fn = rerank_fn is not None
        self._reranker_model = reranker_model
        self.reranker_endpoint = reranker_endpoint
        self.reranker_model_name = reranker_model_name
        self.reranker_api_key = reranker_api_key
        self.reranker_max_length = reranker_max_length
        self.reranker_batch_size = reranker_batch_size
        self.rerank_modality = rerank_modality
        self.local_reranker_backend = local_reranker_backend
        self.local_hf_device = local_hf_device
        self.local_hf_cache_dir = Path(local_hf_cache_dir) if local_hf_cache_dir else None
        self.reranker_gpu_memory_utilization = reranker_gpu_memory_utilization
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-mcp-ingest")
        self._futures: dict[str, Future[Any]] = {}
        self._collection_locks_guard = Lock()
        self._collection_locks: dict[str, Any] = {}

    @property
    def retriever_factory(self) -> Callable[..., Any]:
        if self._retriever_factory is None:
            from nemo_retriever.retriever import Retriever

            self._retriever_factory = Retriever
        return self._retriever_factory

    @property
    def ingestor_factory(self) -> Callable[..., Any]:
        if self._ingestor_factory is None:
            from nemo_retriever import create_ingestor

            self._ingestor_factory = create_ingestor
        return self._ingestor_factory

    @property
    def rerank_fn(self) -> Callable[..., list[dict[str, Any]]]:
        if self._rerank_fn is None:
            from nemo_retriever.rerank import rerank_hits

            self._rerank_fn = rerank_hits
        return self._rerank_fn

    @property
    def reranker_model(self) -> Any:
        if self._reranker_model is None:
            from nemo_retriever.model import create_local_reranker

            cache_dir = str(self.local_hf_cache_dir) if self.local_hf_cache_dir else None
            self._reranker_model = create_local_reranker(
                model_name=self.reranker_model_name,
                device=self.local_hf_device,
                hf_cache_dir=cache_dir,
                backend=self.local_reranker_backend,
                gpu_memory_utilization=self.reranker_gpu_memory_utilization,
            )
        return self._reranker_model

    def create_collection(
        self,
        name: str = "default",
        temporary: bool = False,
        hybrid: bool = False,
    ) -> CollectionRecord:
        return self.registry.create_collection(name=name, temporary=temporary, hybrid=hybrid)

    def list_collections(self) -> list[CollectionRecord]:
        return self.registry.list_collections()

    def describe_collection(self, name: str = "default") -> dict[str, Any]:
        record = self.registry.get_collection(name)
        payload = record.model_dump(mode="json")
        payload["recent_jobs"] = [
            job.model_dump(mode="json")
            for job in self.registry.list_jobs(name)[:10]
        ]
        return payload

    def delete_collection(self, name: str, delete_data: bool = False) -> CollectionRecord:
        record = self.registry.get_collection(name)
        if delete_data:
            self._delete_collection_root(record)
        return self.registry.delete_collection(name)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def _delete_collection_root(self, record: CollectionRecord) -> None:
        data_root = self.registry.data_root.expanduser().resolve()
        root_path = Path(record.root_path).expanduser().resolve()
        if root_path == data_root or data_root not in root_path.parents:
            raise AgentMcpError(
                AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT,
                f"Collection root '{root_path}' is outside the registry data root.",
                details={"collection": record.name, "root_path": str(root_path), "data_root": str(data_root)},
            )
        if root_path.exists():
            shutil.rmtree(root_path)

    def query_collection(
        self,
        collection: str = "default",
        query: str = "",
        top_k: int = 10,
        hybrid: bool = False,
        rerank: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[EvidenceHit]:
        record = self.registry.get_collection(collection)
        if not record.queryable:
            raise AgentMcpError(
                AgentMcpErrorCode.COLLECTION_NOT_QUERYABLE,
                f"Collection '{collection}' is not queryable.",
                details={"collection": collection},
            )
        if hybrid and not record.hybrid:
            raise AgentMcpError(
                AgentMcpErrorCode.HYBRID_NOT_AVAILABLE,
                f"Collection '{collection}' was not created with hybrid search enabled.",
                details={"collection": collection},
            )

        retriever = self.retriever_factory(
            top_k=top_k,
            rerank=rerank,
            vdb_kwargs={
                "vdb_op": record.vdb_backend,
                "vdb_kwargs": {"uri": record.vdb_uri, "table_name": record.vdb_table},
            },
            embed_kwargs={
                "model_name": record.embedding_model,
                "embed_model_name": record.embedding_model,
                "embedding_endpoint": record.embedding_endpoint,
            },
        )
        call_vdb_kwargs = dict(filters) if filters else {}
        if hybrid:
            call_vdb_kwargs["hybrid"] = True
        try:
            raw_hits = retriever.query(
                query,
                top_k=top_k,
                vdb_kwargs=call_vdb_kwargs or None,
            )
        except NotImplementedError as exc:
            if not hybrid:
                raise
            raise AgentMcpError(
                AgentMcpErrorCode.HYBRID_NOT_AVAILABLE,
                f"Hybrid query is not available for collection '{collection}'.",
                details={"collection": collection},
            ) from exc
        return normalize_hits(raw_hits)

    def start_ingestion(
        self,
        collection: str,
        paths: list[str | Path],
        *,
        run_mode: str = "inprocess",
    ) -> IngestJobRecord:
        self.registry.get_or_create_collection(collection)
        expanded = expand_local_paths(paths, policy=self.path_policy)
        job = self.registry.create_job(collection, source_count=len(expanded.files) + len(expanded.skipped))
        if expanded.skipped:
            job = self.registry.update_job(
                job.job_id,
                skipped_count=len(expanded.skipped),
                warnings=list(expanded.skipped),
            )
        future = self._executor.submit(self._run_ingestion_job, job.job_id, collection, expanded.files, run_mode)
        self._futures[job.job_id] = future
        return job

    def ingest_local_paths(
        self,
        collection: str,
        paths: list[str | Path],
        *,
        wait: bool = True,
        timeout_s: float | None = None,
        run_mode: str = "inprocess",
    ) -> IngestJobRecord:
        job = self.start_ingestion(collection, paths, run_mode=run_mode)
        if not wait:
            return self.registry.get_job(job.job_id)

        future = self._futures[job.job_id]
        try:
            future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return self.registry.get_job(job.job_id)
        except Exception:
            return self.registry.get_job(job.job_id)
        return self.registry.get_job(job.job_id)

    def get_ingestion_status(self, job_id: str) -> IngestJobRecord:
        return self.registry.get_job(job_id)

    def _collection_lock(self, collection: str) -> Any:
        with self._collection_locks_guard:
            if collection not in self._collection_locks:
                self._collection_locks[collection] = Lock()
            return self._collection_locks[collection]

    def _run_ingestion_job(
        self,
        job_id: str,
        collection: str,
        files: list[Path],
        run_mode: str,
    ) -> None:
        with self._collection_lock(collection):
            self.registry.update_job(job_id, status=JobStatus.RUNNING)
            initial_record = self.registry.get_collection(collection)
            accepted_count = 0
            try:
                grouped = group_paths_by_media_type(files)
                ordered_media_types = [
                    media_type
                    for media_type in CANONICAL_MEDIA_ORDER
                    if media_type in grouped
                ]
                ordered_media_types.extend(sorted(set(grouped) - set(CANONICAL_MEDIA_ORDER)))

                for media_type in ordered_media_types:
                    media_files = grouped[media_type]
                    overwrite = accepted_count == 0 and not initial_record.queryable
                    self._run_media_group(collection, media_type, media_files, run_mode, overwrite=overwrite)
                    accepted_count += len(media_files)

                current_job = self.registry.get_job(job_id)
                final_status = JobStatus.PARTIAL if current_job.skipped_count else JobStatus.COMPLETE
                self.registry.update_job(
                    job_id,
                    status=final_status,
                    accepted_count=accepted_count,
                    row_count=accepted_count if accepted_count else None,
                )
                if accepted_count:
                    self.registry.mark_collection_queryable(collection, row_count=accepted_count)
            except Exception as exc:
                error = AgentMcpError(
                    AgentMcpErrorCode.BACKEND_ERROR,
                    f"Ingestion job '{job_id}' failed.",
                    details={"job_id": job_id, "collection": collection, "cause": str(exc)},
                )
                final_status = JobStatus.PARTIAL if accepted_count else JobStatus.FAILED
                self.registry.update_job(
                    job_id,
                    status=final_status,
                    accepted_count=accepted_count,
                    row_count=accepted_count if accepted_count else None,
                    errors=[error.to_dict()],
                )
                if accepted_count:
                    self.registry.mark_collection_queryable(collection, row_count=accepted_count)
                raise

    def _run_media_group(
        self,
        collection: str,
        media_type: str,
        files: list[Path],
        run_mode: str,
        *,
        overwrite: bool,
    ) -> None:
        record = self.registry.get_collection(collection)
        ingestor = self.ingestor_factory(run_mode=run_mode)
        ingestor = ingestor.files([str(path) for path in files])

        extract_methods = {
            "document": "extract",
            "image": "extract_image_files",
            "text": "extract_txt",
            "html": "extract_html",
            "audio": "extract_audio",
            "video": "extract_video",
        }
        try:
            extract_method = extract_methods[media_type]
        except KeyError as exc:
            raise AgentMcpError(
                AgentMcpErrorCode.UNSUPPORTED_MEDIA_TYPE,
                f"Unsupported media type '{media_type}'.",
                details={"collection": collection, "media_type": media_type},
            ) from exc

        ingestor = getattr(ingestor, extract_method)()
        ingestor = ingestor.embed(
            model_name=record.embedding_model,
            embedding_endpoint=record.embedding_endpoint,
        )
        if record.artifact_root:
            ingestor = ingestor.store(storage_uri=record.artifact_root)
        result = ingestor.ingest()
        self._upload_records(record, result, overwrite=overwrite)

    def _upload_records(self, record: CollectionRecord, result: Any, *, overwrite: bool) -> None:
        from nemo_retriever.vdb import IngestVdbOperator

        operator = IngestVdbOperator(
            vdb_op=record.vdb_backend,
            vdb_kwargs={
                "uri": record.vdb_uri,
                "table_name": record.vdb_table,
                "hybrid": record.hybrid,
                "overwrite": overwrite,
            },
        )
        operator.process(result)

    def rerank_results(
        self,
        query: str,
        hits: list[EvidenceHit],
        top_n: int | None = None,
    ) -> list[EvidenceHit]:
        raw_hits = [self._hit_to_rerank_payload(hit) for hit in hits]
        if self._custom_rerank_fn:
            reranked = self.rerank_fn(query, raw_hits, top_n=top_n)
        else:
            endpoint = (self.reranker_endpoint or "").strip() or None
            reranked = self.rerank_fn(
                query,
                raw_hits,
                model=None if endpoint else self.reranker_model,
                invoke_url=endpoint,
                model_name=str(self.reranker_model_name),
                api_key=(self.reranker_api_key or "").strip(),
                max_length=int(self.reranker_max_length),
                batch_size=int(self.reranker_batch_size),
                top_n=top_n,
                modality=self.rerank_modality,
            )
        return normalize_hits(reranked)

    def _hit_to_rerank_payload(self, hit: EvidenceHit) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": hit.text,
            "metadata": hit.metadata,
        }
        if hit.score is not None:
            payload["score"] = hit.score
        if hit.source_path:
            payload["source_path"] = hit.source_path
            payload["path"] = hit.source_path
        if hit.content_type != "unknown":
            payload["content_type"] = hit.content_type

        locator = hit.locator
        if locator.page_number is not None:
            payload["page_number"] = locator.page_number
        if locator.timestamp_start_s is not None:
            payload["timestamp_start_s"] = locator.timestamp_start_s
        if locator.timestamp_end_s is not None:
            payload["timestamp_end_s"] = locator.timestamp_end_s
        if locator.frame_index is not None:
            payload["frame_index"] = locator.frame_index
        if locator.chunk_id is not None:
            payload["chunk_id"] = locator.chunk_id
        if locator.bbox_xyxy_norm is not None:
            payload["bbox_xyxy_norm"] = json.dumps(locator.bbox_xyxy_norm)

        if hit.artifacts.stored_image_uri is not None:
            payload["stored_image_uri"] = hit.artifacts.stored_image_uri
        if hit.artifacts.thumbnail_uri is not None:
            payload["thumbnail_uri"] = hit.artifacts.thumbnail_uri
        return payload
