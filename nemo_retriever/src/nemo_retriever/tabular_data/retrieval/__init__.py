"""Public entry point: `generate_sql` (SQL string) and `get_sql_tool_response_top_k` (full dict).

Implementation lives in `nemo_retriever.tabular_data.retrieval.generate_sql`.
"""

from nemo_retriever.tabular_data.retrieval.generate_sql import (
    get_sql_tool_response_top_k,
)


def generate_sql(query: str, embedding_api_key: str = "", embedding_http_endpoint: str = "", top_k: int = 15) -> str:
    """Generate SQL for a natural language query; returns the sql_code string."""
    result = get_sql_tool_response_top_k(
        query, embedding_api_key=embedding_api_key, embedding_http_endpoint=embedding_http_endpoint, top_k=top_k
    )
    return (result.get("sql_code") or "").strip() or ""


__all__ = ["generate_sql", "get_sql_tool_response_top_k"]
