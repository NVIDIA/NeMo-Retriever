"""
SQL Execution Agent

Executes validated SQL via the injected DB connector.
"""

import logging
from typing import Any, Dict, Optional

from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentState

logger = logging.getLogger(__name__)


class QueryResponse:
    def __init__(self, result: list[str], sliced: bool, error: Optional[str] = None):
        self.result = result
        self.sliced = sliced
        self.error = error


def _run_sql(sql: str, state: AgentState) -> QueryResponse:
    """Execute SQL via the injected ``connector``."""
    connector = state.get("connector")
    if connector is not None:
        try:
            df = connector.execute(sql)
        except Exception as e:
            logger.exception("SQL execution failed (injected connector)")
            return QueryResponse(result=None, sliced=False, error=str(e))
        payload = df.to_json(orient="records", default_handler=str) if len(df) else "[]"
        return QueryResponse(result=[payload], sliced=False, error=None)


class SQLExecutionAgent(BaseAgent):
    """
    Agent that executes SQL.

    Input:
    - ``path_state["sql_code"]`` (from validation) or ``sql_generation_result.sql_code``
    - ``connector``: injected DB connector.

    Output:
    - ``path_state["sql_response_from_db"]``: :class:`QueryResponse`
    """

    def __init__(self):
        super().__init__("sql_execution")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        sql_code = path_state.get("sql_code")
        if not sql_code or not str(sql_code).strip():
            llm = path_state.get("sql_generation_result")
            sql_code = getattr(llm, "sql_code", None) if llm else None
        if not sql_code or not str(sql_code).strip():
            self.logger.warning("No SQL code found for execution")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        sql_code = path_state.get("sql_code")
        if not sql_code or not str(sql_code).strip():
            llm = path_state.get("sql_generation_result")
            sql_code = getattr(llm, "sql_code", "") if llm else ""

        response_from_db = _run_sql(sql_code, state)

        if response_from_db.error:
            self.logger.info("SQL execution error: %s", response_from_db.error)
            path_state["error"] = response_from_db.error
            return {"decision": "invalid_sql", "path_state": path_state}

        return {
            "decision": "valid_sql",
            "path_state": {**path_state, "sql_response_from_db": response_from_db.result},
        }
