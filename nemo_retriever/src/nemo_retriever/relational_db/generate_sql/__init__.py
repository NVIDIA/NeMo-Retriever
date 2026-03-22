"""Public entry point: `generate_sql` (SQL string) and `get_sql_tool_response_top_k` (full dict).

Implementation lives in `nemo_retriever.relational_db.sql_tool.generate_sql`.
"""

from nemo_retriever.relational_db.sql_tool.generate_sql import (
    get_sql_tool_response_top_k,
)


def generate_sql(query: str, top_k: int = 15) -> str:
    """Generate SQL for a natural language query; returns the sql_code string."""
    result = get_sql_tool_response_top_k(query, top_k=top_k)
    return (result.get("sql_code") or "").strip() or ""


__all__ = ["generate_sql", "get_sql_tool_response_top_k"]
