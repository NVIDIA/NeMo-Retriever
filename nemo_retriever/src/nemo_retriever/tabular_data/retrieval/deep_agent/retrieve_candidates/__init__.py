"""Retrieve-candidates phase of the Deep Agent pipeline.

Decomposes the user question into typed entities, retrieves per-entity
candidates from LanceDB / Neo4j, and synthesizes SQL expressions for
entities that have no direct match.  Output is a ``RetrievalContext``
consumed by the SQL-generation phase.
"""

from nemo_retriever.tabular_data.retrieval.deep_agent.retrieve_candidates.agent_runtime import (
    run_retrieval_agent,
)

__all__ = ["run_retrieval_agent"]
