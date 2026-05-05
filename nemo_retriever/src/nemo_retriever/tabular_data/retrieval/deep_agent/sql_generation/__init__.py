"""SQL-generation phase of the Deep Agent pipeline.

Receives a ``RetrievalContext`` produced by the retrieve-candidates phase
and runs a Deep Agent that plans, generates, and validates SQL.  Output
is a validated SQL string captured in an ``ExecutionStore``.
"""

from nemo_retriever.tabular_data.retrieval.deep_agent.sql_generation.agent_runtime import (
    create_sql_agent,
    extract_structured_answer,
    format_sql_user_prompt,
)

__all__ = ["create_sql_agent", "extract_structured_answer", "format_sql_user_prompt"]
