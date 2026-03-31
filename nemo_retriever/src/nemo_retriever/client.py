# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python client for the NeMo Retriever REST API.

Provides the same fluent builder API as :class:`nemo_retriever.ingestor` and
the same query API as :class:`nemo_retriever.retriever.Retriever`, but
delegates all processing to a remote server via ``/api/v1/*`` endpoints.

Ingestion::

    from nemo_retriever.client import RemoteIngestor

    results = (
        RemoteIngestor("http://localhost:7670")
        .files("docs/*.pdf")
        .extract(extract_tables=True, method="pdfium")
        .embed(model_name="nvidia/llama-nemotron-embed-1b-v2")
        .vdb_upload(lancedb_uri="lancedb", table_name="my-table")
        .ingest()
    )

Streaming (page-by-page)::

    ingestor = (
        RemoteIngestor("http://localhost:7670")
        .files("report.pdf")
        .extract()
        .embed()
        .vdb_upload()
    )
    for page in ingestor.ingest_stream():
        print(f"Page {page.get('page_number')}: {page.get('text', '')[:80]}")

Async with job handle::

    job = ingestor.ingest_async()
    print(job.job_id)
    job.wait()
    print(job.results())

Retrieval::

    from nemo_retriever.client import RemoteRetriever

    retriever = RemoteRetriever("http://localhost:7670")
    hits = retriever.query("What is machine learning?")
"""

from __future__ import annotations

import glob as _glob
import json
import os
import time
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Ingestor client
# ---------------------------------------------------------------------------


class RemoteIngestor:
    """Fluent ingestor client that mirrors :class:`nemo_retriever.ingestor`.

    Builder methods (``.files``, ``.extract``, ``.embed``, etc.) record
    pipeline configuration.  Terminal methods (``.ingest``,
    ``.ingest_async``, ``.ingest_stream``) POST to the server.

    The stage configuration recorded by each builder call is sent to the
    server as the ``config`` JSON field of the ``/api/v1/ingest`` or
    ``/api/v1/ingest/stream`` endpoint, so the pipeline actually runs
    with the parameters you specify (unlike the legacy ``RemoteIngestor``
    where config was silently ignored).
    """

    RUN_MODE = "remote"

    def __init__(
        self,
        base_url: str = "http://localhost:7670",
        *,
        timeout: float = 600.0,
        documents: Optional[List[str]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._input_documents: List[str] = list(documents or [])
        self._buffers: List[Tuple[str, BytesIO]] = []

        self._extract_config: Optional[Dict[str, Any]] = None
        self._dedup_config: Optional[Dict[str, Any]] = None
        self._caption_config: Optional[Dict[str, Any]] = None
        self._split_config: Optional[Dict[str, Any]] = None
        self._embed_config: Optional[Dict[str, Any]] = None
        self._vdb_config: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Builder: documents
    # ------------------------------------------------------------------

    def files(self, documents: Union[str, List[str]]) -> "RemoteIngestor":
        """Add local file paths or globs for ingestion."""
        if isinstance(documents, str):
            documents = [documents]
        for pattern in documents:
            matches = _glob.glob(pattern, recursive=True)
            if matches:
                resolved = [os.path.abspath(p) for p in matches if os.path.isfile(p)]
                if not resolved:
                    raise FileNotFoundError(f"Pattern matched but no files found: {pattern!r}")
                self._input_documents.extend(resolved)
            elif os.path.isfile(pattern):
                self._input_documents.append(os.path.abspath(pattern))
            else:
                raise FileNotFoundError(f"No files found for: {pattern!r}")
        return self

    def buffers(self, buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]]) -> "RemoteIngestor":
        """Add in-memory byte buffers for ingestion.  Each item is ``(name, BytesIO)``."""
        if isinstance(buffers, (list, tuple)) and len(buffers) == 2 and not isinstance(buffers[0], (list, tuple)):
            buffers = [buffers]  # type: ignore[assignment]
        for name, buf in buffers:  # type: ignore[union-attr]
            self._buffers.append((str(name), buf))
        return self

    def load(self) -> "RemoteIngestor":
        return self

    # ------------------------------------------------------------------
    # Builder: pipeline stages
    # ------------------------------------------------------------------

    def extract(self, params: Any = None, **kwargs: Any) -> "RemoteIngestor":
        """Configure the extraction stage."""
        cfg = _params_to_dict(params, kwargs)
        cfg.setdefault("enabled", True)
        self._extract_config = cfg
        return self

    def extract_image_files(self, params: Any = None, **kwargs: Any) -> "RemoteIngestor":
        """Configure extraction for image files."""
        return self.extract(params, **kwargs)

    def dedup(self, params: Any = None, **kwargs: Any) -> "RemoteIngestor":
        """Configure the deduplication stage."""
        cfg = _params_to_dict(params, kwargs)
        cfg.setdefault("enabled", True)
        self._dedup_config = cfg
        return self

    def caption(self, params: Any = None, **kwargs: Any) -> "RemoteIngestor":
        """Configure the VLM captioning stage."""
        cfg = _params_to_dict(params, kwargs)
        cfg.setdefault("enabled", True)
        self._caption_config = cfg
        return self

    def split(self, params: Any = None, **kwargs: Any) -> "RemoteIngestor":
        """Configure the text chunking/splitting stage."""
        cfg = _params_to_dict(params, kwargs)
        cfg.setdefault("enabled", True)
        self._split_config = cfg
        return self

    def embed(self, params: Any = None, **kwargs: Any) -> "RemoteIngestor":
        """Configure the embedding stage."""
        cfg = _params_to_dict(params, kwargs)
        cfg.setdefault("enabled", True)
        self._embed_config = cfg
        return self

    def vdb_upload(self, params: Any = None, **kwargs: Any) -> "RemoteIngestor":
        """Configure the LanceDB upload stage."""
        cfg = _params_to_dict(params, kwargs)
        if "lancedb" in cfg and isinstance(cfg["lancedb"], dict):
            lancedb = cfg.pop("lancedb")
            cfg.update(lancedb)
        cfg.pop("purge_results_after_upload", None)
        cfg.setdefault("enabled", True)
        self._vdb_config = cfg
        return self

    def save_to_disk(self, output_directory: Optional[str] = None, **_: Any) -> "RemoteIngestor":
        return self

    def save_intermediate_results(self, output_dir: str) -> "RemoteIngestor":
        return self

    def all_tasks(self) -> "RemoteIngestor":
        return self.extract().embed().vdb_upload()

    def filter(self) -> "RemoteIngestor":
        return self

    def store(self) -> "RemoteIngestor":
        return self

    def store_embed(self) -> "RemoteIngestor":
        return self

    def udf(self, *_args: Any, **_kwargs: Any) -> "RemoteIngestor":
        return self

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "RemoteIngestor":
        return self

    # ------------------------------------------------------------------
    # Terminal: execute
    # ------------------------------------------------------------------

    def ingest(self, params: Any = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Submit documents, wait for completion, return results.

        Calls ``POST /api/v1/ingest`` then polls ``GET /api/v1/jobs/{id}``
        until the job finishes.  Returns the list of result records.
        """
        job = self.ingest_async(params=params, **kwargs)
        job.wait(poll_interval=1.0)
        return job.results()

    def ingest_async(self, params: Any = None, **kwargs: Any) -> "RemoteJob":
        """Submit documents asynchronously.  Returns a :class:`RemoteJob`."""
        import httpx

        files_payload = self._build_upload_list()
        if not files_payload:
            raise ValueError("No documents configured. Call .files() or .buffers() first.")

        config_json = json.dumps(self._build_config())
        multipart = [("files", (name, body, mime)) for name, body, mime in files_payload]

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/api/v1/ingest",
                files=multipart,
                data={"config": config_json},
            )
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Ingest submission failed (HTTP {resp.status_code}): {resp.text[:500]}")

        body = resp.json()
        return RemoteJob(
            base_url=self._base_url,
            job_id=body["job_id"],
            timeout=self._timeout,
        )

    def ingest_stream(
        self,
        params: Any = None,
        *,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Stream results page-by-page from ``POST /api/v1/ingest/stream``.

        Yields one dict per processed page/chunk.  Error dicts have
        ``"_error": True``.  The final dict has ``"_summary": True``.

        If *callback* is provided it is called for each dict as well.
        """
        import httpx

        files_payload = self._build_upload_list()
        if not files_payload:
            raise ValueError("No documents configured. Call .files() or .buffers() first.")

        config_json = json.dumps(self._build_config())
        multipart = [("files", (name, body, mime)) for name, body, mime in files_payload]

        with httpx.Client(timeout=self._timeout) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/api/v1/ingest/stream",
                files=multipart,
                data={"config": config_json},
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise RuntimeError(f"Stream ingest failed (HTTP {resp.status_code}): {resp.text[:500]}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if callback is not None:
                        callback(record)
                    yield record

    # ------------------------------------------------------------------
    # Job status helpers
    # ------------------------------------------------------------------

    def completed_jobs(self) -> int:
        return self._count_jobs_by_status("completed")

    def failed_jobs(self) -> int:
        return self._count_jobs_by_status("failed")

    def cancelled_jobs(self) -> int:
        return self._count_jobs_by_status("cancelled")

    def remaining_jobs(self) -> int:
        return self._count_jobs_by_status("queued") + self._count_jobs_by_status("running")

    def get_status(self) -> Dict[str, str]:
        """Return a mapping of all known job IDs to their status string."""
        import httpx

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{self._base_url}/api/v1/jobs", params={"limit": 10000})
        if resp.status_code != 200:
            return {}
        body = resp.json()
        return {j["job_id"]: j["status"] for j in body.get("jobs", [])}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_jobs_by_status(self, status: str) -> int:
        import httpx

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{self._base_url}/api/v1/jobs",
                params={"status": status, "limit": 10000},
            )
        if resp.status_code != 200:
            return 0
        return resp.json().get("total", 0)

    def _build_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self._extract_config is not None:
            config["extract"] = self._extract_config
        if self._dedup_config is not None:
            config["dedup"] = self._dedup_config
        if self._caption_config is not None:
            config["caption"] = self._caption_config
        if self._split_config is not None:
            config["split"] = self._split_config
        if self._embed_config is not None:
            config["embed"] = self._embed_config
        if self._vdb_config is not None:
            config["vdb_upload"] = self._vdb_config
        return config

    def _build_upload_list(self) -> List[Tuple[str, bytes, str]]:
        """Return ``[(filename, body_bytes, mime), ...]`` for all inputs."""
        items: List[Tuple[str, bytes, str]] = []
        for path in self._input_documents:
            with open(path, "rb") as f:
                body = f.read()
            items.append((os.path.basename(path), body, _mime_for_path(path)))
        for name, buf in self._buffers:
            buf.seek(0)
            items.append((name, buf.read(), _mime_for_path(name)))
        return items


# ---------------------------------------------------------------------------
# Job handle
# ---------------------------------------------------------------------------


class RemoteJob:
    """Handle for a running ingest job on the server.

    Returned by :meth:`RemoteIngestor.ingest_async`.
    """

    def __init__(self, *, base_url: str, job_id: str, timeout: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._job_id = job_id
        self._timeout = timeout
        self._cached_status: Optional[Dict[str, Any]] = None

    @property
    def job_id(self) -> str:
        return self._job_id

    def status(self) -> Dict[str, Any]:
        """Poll ``GET /api/v1/jobs/{job_id}`` and return the status dict."""
        import httpx

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{self._base_url}/api/v1/jobs/{self._job_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get job status (HTTP {resp.status_code}): {resp.text[:500]}")
        self._cached_status = resp.json()
        return self._cached_status

    @property
    def is_complete(self) -> bool:
        s = (self._cached_status or {}).get("status", "")
        return s in ("completed", "failed", "cancelled")

    def wait(self, *, poll_interval: float = 2.0, timeout: Optional[float] = None) -> "RemoteJob":
        """Block until the job finishes.  Returns self."""
        deadline = time.monotonic() + (timeout or self._timeout)
        while time.monotonic() < deadline:
            self.status()
            if self.is_complete:
                return self
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {self._job_id} did not complete within {timeout or self._timeout}s")

    def results(self) -> List[Dict[str, Any]]:
        """Fetch results from ``GET /api/v1/jobs/{job_id}/results``."""
        import httpx

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self._base_url}/api/v1/jobs/{self._job_id}/results")
        if resp.status_code == 409:
            raise RuntimeError(f"Job {self._job_id} is not yet completed.")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get results (HTTP {resp.status_code}): {resp.text[:500]}")
        return resp.json().get("records", [])

    def cancel(self) -> None:
        """Cancel or delete the job via ``DELETE /api/v1/jobs/{job_id}``."""
        import httpx

        with httpx.Client(timeout=30.0) as client:
            client.delete(f"{self._base_url}/api/v1/jobs/{self._job_id}")

    def __repr__(self) -> str:
        status = (self._cached_status or {}).get("status", "unknown")
        return f"RemoteJob(job_id={self._job_id!r}, status={status!r})"


# ---------------------------------------------------------------------------
# Retriever client
# ---------------------------------------------------------------------------


class RemoteRetriever:
    """Retriever client that mirrors :class:`nemo_retriever.retriever.Retriever`.

    Delegates to ``POST /api/v1/retrieve`` on a running NeMo Retriever
    service.  Constructor parameters match the local ``Retriever`` fields
    where applicable; those that only make sense locally (device, cache
    dirs) are accepted but ignored.

    ::

        retriever = RemoteRetriever("http://localhost:7670", top_k=5)
        hits = retriever.query("What is machine learning?")
        for hit in hits:
            print(hit["text"], hit.get("_rerank_score"))
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7670",
        *,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
        embedder: str = "nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint: Optional[str] = None,
        embedding_api_key: str = "",
        top_k: int = 10,
        hybrid: bool = False,
        reranker: Optional[bool] = False,
        reranker_model_name: Optional[str] = "nvidia/llama-nemotron-rerank-1b-v2",
        reranker_endpoint: Optional[str] = None,
        reranker_api_key: str = "",
        timeout: float = 120.0,
        # Accepted for signature compatibility with the local Retriever;
        # not used by the remote client.
        **_ignored: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.lancedb_uri = lancedb_uri
        self.lancedb_table = lancedb_table
        self.embedder = embedder
        self.embedding_endpoint = embedding_endpoint
        self.embedding_api_key = embedding_api_key
        self.top_k = top_k
        self.hybrid = hybrid
        self.reranker = reranker
        self.reranker_model_name = reranker_model_name
        self.reranker_endpoint = reranker_endpoint
        self.reranker_api_key = reranker_api_key
        self._timeout = timeout

    def query(
        self,
        query: str,
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval for a single query string."""
        return self.queries(
            [query],
            embedder=embedder,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
        )[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
    ) -> list[list[dict[str, Any]]]:
        """Run retrieval for multiple query strings.

        Issues one ``POST /retrieve`` per query (matching the server's
        single-query endpoint) and collects the results.
        """
        import httpx

        query_list = [str(q) for q in queries]
        if not query_list:
            return []

        uri = lancedb_uri or self.lancedb_uri
        table = lancedb_table or self.lancedb_table

        all_results: list[list[dict[str, Any]]] = []
        with httpx.Client(timeout=self._timeout) as client:
            for q in query_list:
                payload: Dict[str, Any] = {
                    "query": q,
                    "top_k": self.top_k,
                    "hybrid": self.hybrid,
                }
                if uri:
                    payload["lancedb_uri"] = uri
                if table:
                    payload["lancedb_table"] = table

                resp = client.post(f"{self._base_url}/retrieve", json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(f"Retrieval failed for query {q!r} (HTTP {resp.status_code}): {resp.text[:500]}")
                body = resp.json()
                all_results.append(body.get("results", []))

        return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTENSION_MIME: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
}


def _mime_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _EXTENSION_MIME.get(ext, "application/octet-stream")


def _params_to_dict(params: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Pydantic params object (or None) + kwargs into a plain dict.

    Handles ``ExtractParams``, ``EmbedParams``, ``VdbUploadParams``, etc.
    """
    if params is None and not kwargs:
        return {}
    if params is None:
        return dict(kwargs)
    if hasattr(params, "model_dump"):
        merged = params.model_dump(mode="json", exclude_defaults=True)
        merged.update(kwargs)
        return merged
    if isinstance(params, dict):
        merged = dict(params)
        merged.update(kwargs)
        return merged
    return dict(kwargs)
