"""
Retriever strategy implementations for the QA evaluation pipeline.

TopKRetriever: queries an existing Milvus or LanceDB collection at eval time.
FileRetriever: reads pre-computed retrieval results from a JSON file.

FileRetriever is the primary integration point. Any retrieval method -- vector
search, agentic retrieval, hybrid, reranked, BM25, or a completely custom
pipeline -- can plug into the QA eval harness by writing a single JSON file.
See the FileRetriever class docstring for the minimal required format.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Optional

from nv_ingest_harness.utils.qa.types import RetrievalResult


def _normalize_query(text: str) -> str:
    """Canonical form for query matching: NFKC unicode, stripped, case-folded,
    collapsed whitespace. This makes lookup resilient to trivial formatting
    differences between the ground-truth CSV and the retrieval JSON."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


class TopKRetriever:
    """
    Retriever that queries an existing Milvus or LanceDB collection.

    Requires the collection to exist before construction -- run an e2e
    ingestion pass (or e2e_qa_eval) first.

    Args:
        collection_name: VDB collection / LanceDB table name.
        hostname: Service hostname used for embedding and Milvus endpoints.
        model_name: Embedding model name passed to the embedding service.
        sparse: Enable hybrid sparse-dense retrieval (Milvus only).
        gpu_search: Use GPU acceleration for Milvus search.
        vdb_backend: "milvus" or "lancedb".
        table_path: Path to LanceDB database directory (required for lancedb).
        hybrid: Enable LanceDB hybrid (FTS + vector) retrieval.
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

        # Lazy import: nv_ingest_client / recall.py are not available in the
        # minimal Docker image used for FileRetriever-only smoke tests.
        from nv_ingest_harness.utils.recall import get_retrieval_func

        self._retrieval_func = get_retrieval_func(
            vdb_backend=vdb_backend,
            table_path=table_path,
            table_name=collection_name,
            hybrid=hybrid,
        )

        self._validate_collection()

    def _validate_collection(self) -> None:
        """
        Verify the target collection exists before the pipeline starts.

        Raises RuntimeError with an actionable message if not found,
        rather than letting callers hit a cryptic error mid-run.
        """
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
        """
        Retrieve the top-k most relevant chunks for a query.

        Extracts text from entity.text and stores source/page metadata.
        """
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
                hybrid=self.sparse,
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


class FileRetriever:
    """
    Retriever that reads pre-computed results from a JSON file.

    This is the integration point for **any** retrieval method. Vector search,
    agentic retrieval, hybrid pipelines, BM25, rerankers, or a completely
    custom system -- as long as it produces a JSON file in the format below,
    the QA eval harness will generate answers and judge them identically.
    This makes comparisons across retrieval strategies apples-to-apples.

    Minimal required JSON format
    ----------------------------
    Only ``"queries"`` with ``"chunks"`` is required. Everything else is optional.

    ::

        {
          "queries": {
            "What is the range of the 767?": {
              "chunks": ["First retrieved chunk text...", "Second chunk..."]
            }
          }
        }

    Full format (with optional fields)
    -----------------------------------
    ::

        {
          "metadata": {                         # optional -- ignored by FileRetriever
            "retrieval_method": "agentic",      #   free-form; useful for your records
            "model": "...",
            "top_k": 5
          },
          "queries": {
            "What is the range of the 767?": {
              "chunks": [                       # REQUIRED: list of text strings
                "The 767 has a range of..."     #   (one per retrieved chunk/passage)
              ],
              "metadata": [                     # optional: per-chunk provenance
                {"source_id": "bo767.pdf", "page_number": 24, "distance": 0.42}
              ]
            }
          }
        }

    Rules:
    - The ``"queries"`` key must be a dict mapping query strings to objects.
    - Each object must have ``"chunks"``: a list of plain-text strings.
    - ``"metadata"`` per-chunk is optional; if present it is carried through
      to the results JSON for traceability but is not used for scoring.
    - Query matching is normalized (NFKC unicode, case-folded, whitespace-
      collapsed) so trivial formatting differences between the ground-truth
      CSV and the retrieval JSON do not cause silent misses.
    - The harness respects ``top_k`` at eval time: if your JSON has 10 chunks
      per query but the eval is configured with ``qa_top_k=5``, only the
      first 5 are used. Order matters -- put the best chunks first.

    Args:
        file_path: Path to the JSON file with pre-computed retrieval results.
    """

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"FileRetriever: retrieval results file not found: {file_path}"
            )

        with open(file_path) as f:
            data = json.load(f)

        raw_index: dict[str, dict] = data.get("queries", {})
        if not raw_index:
            raise ValueError(
                f"FileRetriever: no 'queries' key found in {file_path}. "
                "Expected format: {\"queries\": {\"query text\": {\"chunks\": [...], \"metadata\": [...]}}}"
            )

        self._norm_index: dict[str, dict] = {}
        self._raw_keys: dict[str, str] = {}
        for raw_key, value in raw_index.items():
            norm = _normalize_query(raw_key)
            self._norm_index[norm] = value
            self._raw_keys[norm] = raw_key

        self._miss_count = 0

    def check_coverage(self, qa_pairs: list[dict]) -> float:
        """Validate retrieval file covers the ground-truth queries.

        Logs a per-query miss list and returns the coverage fraction.
        Intended to be called once before the pipeline starts so data
        quality issues surface early rather than as hundreds of silent
        warnings mid-run.
        """
        total = len(qa_pairs)
        if total == 0:
            return 1.0

        misses: list[str] = []
        for pair in qa_pairs:
            norm = _normalize_query(pair.get("query", ""))
            if norm not in self._norm_index:
                misses.append(pair.get("query", "")[:80])

        coverage = (total - len(misses)) / total
        if misses:
            print(f"  [FileRetriever] Coverage: {coverage:.1%} "
                  f"({total - len(misses)}/{total} queries matched)")
            for q in misses[:10]:
                print(f"    MISS: {q!r}")
            if len(misses) > 10:
                print(f"    ... and {len(misses) - 10} more")
        else:
            print(f"  [FileRetriever] Coverage: 100% ({total}/{total} queries matched)")

        return coverage

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """
        Look up pre-computed chunks for a query string.

        Uses normalized matching (case-folded, whitespace-collapsed) so
        trivial formatting differences don't cause misses.  Falls back to
        an empty result if the query is truly absent so the pipeline can
        continue and record the miss.
        """
        norm = _normalize_query(query)
        entry = self._norm_index.get(norm)

        if entry is None:
            self._miss_count += 1
            if self._miss_count <= 20:
                print(f"  [FileRetriever] WARNING: query not found in retrieval file: {query!r}")
            elif self._miss_count == 21:
                print("  [FileRetriever] WARNING: suppressing further miss warnings (>20)")
            return RetrievalResult(chunks=[], metadata=[])

        chunks = entry.get("chunks", [])[:top_k]
        metadata = entry.get("metadata", [])[:top_k]
        return RetrievalResult(chunks=chunks, metadata=metadata)
