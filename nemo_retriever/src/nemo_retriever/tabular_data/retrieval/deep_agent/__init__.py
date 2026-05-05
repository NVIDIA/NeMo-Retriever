"""Deep Agent text-to-SQL pipeline.

Public entrypoint: :func:`get_agent_response` — orchestrates the 3-phase
Retrieval → SQL → Execute pipeline for a natural-language question.
"""

from nemo_retriever.tabular_data.retrieval.deep_agent.main import get_agent_response

__all__ = ["get_agent_response"]
