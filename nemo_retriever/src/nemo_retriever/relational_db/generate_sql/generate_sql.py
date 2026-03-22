from nemo_retriever.relational_db.sql_tool.generate_sql import get_sql_tool_response_top_k


def generate_sql(query: str, top_k: int = 15) -> str:
    """Generate a SQL query for a given natural language query."""
    return get_sql_tool_response_top_k(query, top_k=top_k)
