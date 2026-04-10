"""QA evaluation utilities for the nv-ingest test harness.

The canonical evaluation framework lives in ``nemo_retriever.evaluation``.
This package exists only for ``TopKRetriever``, which depends on harness-specific
recall utilities (``nv_ingest_harness.utils.recall``) and optional ``pymilvus``.

All other evaluation components should be imported directly::

    from nemo_retriever.evaluation.generators import LiteLLMClient
    from nemo_retriever.evaluation.judges import LLMJudge
    from nemo_retriever.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.evaluation.retrievers import FileRetriever
    from nemo_retriever.evaluation.ground_truth import get_qa_dataset_loader
"""

from __future__ import annotations

import os
from typing import Optional

from nemo_retriever.evaluation.types import RetrievalResult


class TopKRetriever:
    """Retriever that queries an existing Milvus or LanceDB collection.

    Requires the collection to exist before construction -- run an e2e
    ingestion pass (or e2e_qa_eval) first.

    This class stays in the harness because it depends on
    ``nv_ingest_harness.utils.recall.get_retrieval_func`` and optional
    ``pymilvus``, neither of which belong in ``nemo_retriever``.
    """

    def __init__(
        self,
        collection_name: str,
        hostname: str = "localhost",
        model_name: Optional[str] = None,
        sparse: bool = False,
        gpu_search: bool = False,
        vdb_backend: str = "milvus",
        table_path: Optional[str] = None,
        hybrid: bool = False,
    ):
        self.collection_name = collection_name
        self.hostname = hostname
        self.model_name = model_name
        self.sparse = sparse
        self.gpu_search = gpu_search
        self.vdb_backend = vdb_backend
        self.table_path = table_path
        self.hybrid = hybrid
        self.embedding_endpoint = f"http://{hostname}:8012/v1"

        from nv_ingest_harness.utils.recall import get_retrieval_func

        self._retrieval_func = get_retrieval_func(
            vdb_backend=vdb_backend,
            table_path=table_path,
            table_name=collection_name,
            hybrid=hybrid,
        )

        self._validate_collection()

    def _validate_collection(self) -> None:
        """Verify the target collection exists before the pipeline starts."""
        if self.vdb_backend == "milvus":
            try:
                from pymilvus import MilvusClient

                client = MilvusClient(uri=f"http://{self.hostname}:19530")
                existing = client.list_collections()
                if self.collection_name not in existing:
                    raise RuntimeError(
                        f"Milvus collection '{self.collection_name}' not found. "
                        f"Run e2e or e2e_qa_eval first to ingest documents. "
                        f"Available collections: {existing}"
                    )
            except ImportError:
                pass
        elif self.vdb_backend == "lancedb":
            if self.table_path and not os.path.exists(self.table_path):
                raise RuntimeError(
                    f"LanceDB path '{self.table_path}' does not exist. "
                    f"Run e2e or e2e_qa_eval first to ingest documents."
                )

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """Retrieve the top-k most relevant chunks for a query."""
        if self.vdb_backend == "lancedb":
            raw = self._retrieval_func(
                [query],
                embedding_endpoint=self.embedding_endpoint,
                model_name=self.model_name,
                top_k=top_k,
            )
        else:
            raw = self._retrieval_func(
                [query],
                self.collection_name,
                hybrid=self.hybrid,
                embedding_endpoint=self.embedding_endpoint,
                model_name=self.model_name,
                top_k=top_k,
                gpu_search=self.gpu_search,
            )

        hits = raw[0] if raw else []
        chunks: list[str] = []
        metadata: list[dict] = []

        for hit in hits:
            entity = hit.get("entity", {})
            text = entity.get("text", "")
            chunks.append(text)

            source = entity.get("source", {})
            content_meta = entity.get("content_metadata", {})
            metadata.append(
                {
                    "source_id": source.get("source_id", ""),
                    "page_number": content_meta.get("page_number", ""),
                    "distance": hit.get("distance"),
                }
            )

        return RetrievalResult(chunks=chunks, metadata=metadata)


__all__ = ["TopKRetriever"]
