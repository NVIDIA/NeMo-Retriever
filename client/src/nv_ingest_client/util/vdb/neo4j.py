# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j vector database operator implementing the VDB abstract base class.

Requires the optional ``neo4j`` extra:

    pip install "nv-ingest-client[neo4j]"

Neo4j 5.11+ native vector index support is used for all vector operations.
The Bolt protocol (port 7687) is used for all driver connections.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from nv_ingest_client.util.vdb.adt_vdb import VDB

# Environment variable defaults — set these in your .env file:
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=test
_DEFAULT_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
_DEFAULT_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
_DEFAULT_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4jpassword")

logger = logging.getLogger(__name__)


def _get_text_for_element(element: Dict[str, Any]) -> Optional[str]:
    """Extract searchable text from an NV-Ingest element based on document_type."""
    doc_type = element.get("document_type")
    metadata = element.get("metadata", {})

    if doc_type == "text":
        return metadata.get("content")
    elif doc_type == "structured":
        table_meta = metadata.get("table_metadata", {})
        return table_meta.get("table_content")
    elif doc_type == "image":
        image_meta = metadata.get("image_metadata", {})
        content_meta = metadata.get("content_metadata", {})
        if content_meta.get("subtype") == "page_image":
            return image_meta.get("text")
        return image_meta.get("caption")
    elif doc_type == "audio":
        audio_meta = metadata.get("audio_metadata", {})
        return audio_meta.get("audio_transcript")
    return metadata.get("content")


def _build_neo4j_nodes(results: list) -> List[Dict[str, Any]]:
    """Transform NV-Ingest pipeline results into Neo4j node property maps."""
    nodes: List[Dict[str, Any]] = []
    for result in results:
        for element in result:
            metadata = element.get("metadata", {})
            embedding = metadata.get("embedding")
            if embedding is None:
                continue
            text = _get_text_for_element(element)
            if not text:
                continue
            content_meta = metadata.get("content_metadata", {})
            source_meta = metadata.get("source_metadata", {})
            nodes.append(
                {
                    "embedding": embedding,
                    "text": text,
                    "source_id": source_meta.get("source_id", ""),
                    "source_name": source_meta.get("source_name", ""),
                    "page_number": content_meta.get("page_number", -1),
                    "document_type": element.get("document_type", ""),
                }
            )
    return nodes


class Neo4jVDB(VDB):
    """Neo4j operator implementing the VDB interface.

    Uses Neo4j 5.11+ native vector index support.  All write operations use
    MERGE so that re-running ingestion on the same data is idempotent.

    Parameters
    ----------
    uri:
        Bolt URI of the Neo4j instance (default: ``bolt://localhost:7687``).
    user:
        Neo4j username (default: ``neo4j``).
    password:
        Neo4j password (default: ``neo4jpassword``).
    index_name:
        Name of the vector index to create / query (default: ``nv-ingest``).
    node_label:
        Node label used when merging documents (default: ``Document``).
    dense_dim:
        Dimensionality of stored embeddings (default: ``2048``).
    similarity_function:
        Vector index similarity function – ``cosine`` or ``euclidean``
        (default: ``cosine``).
    batch_size:
        Number of nodes per Cypher write batch (default: ``256``).
    """

    def __init__(
        self,
        uri: str = _DEFAULT_URI,
        user: str = _DEFAULT_USER,
        password: str = _DEFAULT_PASSWORD,
        index_name: str = "nv-ingest",
        node_label: str = "Document",
        dense_dim: int = 2048,
        similarity_function: str = "cosine",
        batch_size: int = 256,
        **kwargs,
    ):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'neo4j' package is required. "
                "Install it with: pip install 'nv-ingest-client[neo4j]'"
            ) from exc

        self.uri = uri
        self.user = user
        self.password = password
        self.index_name = index_name
        self.node_label = node_label
        self.dense_dim = dense_dim
        self.similarity_function = similarity_function
        self.batch_size = batch_size

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session(self):
        return self._driver.session()

    # ------------------------------------------------------------------
    # VDB interface
    # ------------------------------------------------------------------

    def create_index(self, recreate: bool = False, **kwargs) -> None:
        """Create a Neo4j native vector index.

        If *recreate* is True and the index already exists it is dropped and
        recreated.  Otherwise the call is a no-op when the index exists.

        Parameters
        ----------
        recreate:
            Drop and recreate the index if it already exists (default: False).
        """
        with self._session() as session:
            if recreate:
                session.run(
                    "DROP INDEX $name IF EXISTS",
                    name=self.index_name,
                )
                logger.info("Dropped existing Neo4j vector index '%s'.", self.index_name)

            session.run(
                """
                CREATE VECTOR INDEX $name IF NOT EXISTS
                FOR (n:$($label))
                ON n.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: $sim
                    }
                }
                """,
                name=self.index_name,
                label=self.node_label,
                dim=self.dense_dim,
                sim=self.similarity_function,
            )
            logger.info(
                "Created (or confirmed existing) Neo4j vector index '%s' "
                "(dim=%d, similarity=%s).",
                self.index_name,
                self.dense_dim,
                self.similarity_function,
            )

    def write_to_index(self, records: list, **kwargs) -> int:
        """Write NV-Ingest records to Neo4j as Document nodes.

        Nodes are MERGE'd on ``(source_id, page_number)`` so re-ingestion is
        idempotent.  Embeddings and metadata properties are set (or
        overwritten) on each MERGE.

        Parameters
        ----------
        records:
            NV-Ingest pipeline results (list of record-sets).
        batch_size:
            Override the instance-level batch size for this call.

        Returns
        -------
        int
            Total number of nodes written.
        """
        batch_size: int = kwargs.get("batch_size", self.batch_size)
        nodes = _build_neo4j_nodes(records)
        if not nodes:
            logger.warning("No embeddable records found; nothing written to Neo4j.")
            return 0

        total_written = 0
        with self._session() as session:
            for start in range(0, len(nodes), batch_size):
                batch = nodes[start : start + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (n:$($label) {source_id: row.source_id, page_number: row.page_number})
                    SET n.text           = row.text,
                        n.source_name    = row.source_name,
                        n.document_type  = row.document_type,
                        n.embedding      = row.embedding
                    """,
                    batch=batch,
                    label=self.node_label,
                )
                total_written += len(batch)
                logger.debug("Written batch of %d nodes to Neo4j.", len(batch))

        logger.info("Total nodes written to Neo4j: %d", total_written)
        return total_written

    def retrieval(self, queries: list, **kwargs) -> List[List[Dict[str, Any]]]:
        """Perform vector similarity search in Neo4j for a list of text queries.

        The queries are embedded using the configured embedding microservice,
        then queried against the Neo4j vector index via Cypher.

        Parameters
        ----------
        queries:
            List of text strings to search.
        top_k:
            Number of nearest neighbours to return per query (default: 10).
        embedding_endpoint:
            URL of the NIM embedding microservice.
        nvidia_api_key:
            Optional NVIDIA API key for authentication.
        model_name:
            Embedding model name.

        Returns
        -------
        list[list[dict]]
            For each query, a list of result dicts with keys
            ``text``, ``source_id``, ``source_name``, ``page_number``, and
            ``document_type``.
        """
        top_k: int = kwargs.get("top_k", 10)
        embedding_endpoint: str = kwargs.get("embedding_endpoint", "http://localhost:8012/v1")
        nvidia_api_key: Optional[str] = kwargs.get("nvidia_api_key")
        model_name: str = kwargs.get("model_name", "nvidia/llama-nemotron-embed-1b-v2")

        try:
            from nv_ingest_client.util.transport import infer_microservice  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "nv_ingest_client.util.transport is required for retrieval."
            ) from exc

        embed_fn = partial(
            infer_microservice,
            model_name=model_name,
            embedding_endpoint=embedding_endpoint,
            nvidia_api_key=nvidia_api_key,
            input_type="query",
            output_names=["embeddings"],
            grpc=not ("http" in urlparse(embedding_endpoint).scheme),
        )
        query_embeddings = embed_fn(queries)

        all_results: List[List[Dict[str, Any]]] = []
        with self._session() as session:
            for query_embed in query_embeddings:
                cypher_result = session.run(
                    """
                    CALL db.index.vector.queryNodes($index, $k, $embedding)
                    YIELD node, score
                    RETURN node.text          AS text,
                           node.source_id     AS source_id,
                           node.source_name   AS source_name,
                           node.page_number   AS page_number,
                           node.document_type AS document_type,
                           score
                    """,
                    index=self.index_name,
                    k=top_k,
                    embedding=query_embed,
                )
                hits = [
                    {
                        "entity": {
                            "text": record["text"],
                            "source": {
                                "source_id": record["source_id"],
                                "source_name": record["source_name"],
                            },
                            "content_metadata": {"page_number": record["page_number"]},
                            "document_type": record["document_type"],
                        },
                        "score": record["score"],
                    }
                    for record in cypher_result
                ]
                all_results.append(hits)

        return all_results

    def run(self, records: list) -> list:
        """Orchestrate index creation and data ingestion.

        Parameters
        ----------
        records:
            NV-Ingest pipeline results to ingest.

        Returns
        -------
        list
            The original *records* passed in.
        """
        self.create_index()
        self.write_to_index(records)
        return records

    def close(self) -> None:
        """Close the underlying Neo4j driver connection."""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
