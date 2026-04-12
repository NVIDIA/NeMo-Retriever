"""
SQL Execution Agent

Executes validated SQL via in-process DuckDB (``tabular-dev-tools/duckdb_connector.py``).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict


from nemo_retriever.tabular_data.retrieval.omni_lite.agents.query_executor import QueryResponse
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.query_validation import (
    is_infra_or_auth_error,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentState


logger = logging.getLogger(__name__)




def _run_sql_duckdb(sql: str, state: AgentState) -> QueryResponse:
    """Use ``path_state['db_connector']`` (or legacy ``duckdb_connector``); else ``duckdb_path`` / env."""
    connector = state.get("db_connector") or state.get("duckdb_connector")
    if connector is not None:
        try:
            df = connector.execute(sql)
        except Exception as e:
            logger.exception("DuckDB execute failed (injected connector)")
            return QueryResponse(result=None, sliced=False, error=str(e))
        payload = df.to_json(orient="records", default_handler=str) if len(df) else "[]"
        return QueryResponse(result=[payload], sliced=False, error=None)



class SQLExecutionAgent(BaseAgent):
    """
    Agent that executes SQL against DuckDB.

    Input:
    - ``path_state["sql_code"]`` (from validation) or ``llm_calc_response.sql_code``
    - Optional ``path_state["db_connector"]``: injected DB connector (e.g. :class:`DuckDB`).
    - Optional legacy ``path_state["duckdb_connector"]``: same as ``db_connector``.
    - Optional ``path_state["duckdb_path"]``: path string when no connector is passed.

    Output:
    - ``path_state["sql_response_from_db"]``: :class:`QueryResponse`
    """

    def __init__(self):
        super().__init__("sql_execution")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        sql_code = path_state.get("sql_code")
        if not sql_code or not str(sql_code).strip():
            llm = path_state.get("llm_calc_response")
            sql_code = getattr(llm, "sql_code", None) if llm else None
        if not sql_code or not str(sql_code).strip():
            self.logger.warning("No SQL code found for execution")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        sql_code = path_state.get("sql_code")
        if not sql_code or not str(sql_code).strip():
            llm = path_state.get("llm_calc_response")
            sql_code = getattr(llm, "sql_code", "") if llm else ""

        response_from_db = _run_sql_duckdb(sql_code, state)

        if response_from_db.error:
            self.logger.info("SQL execution error: %s", response_from_db.error)
            if not is_infra_or_auth_error(response_from_db.error):
                path_state["error"] = response_from_db.error
                return {"decision": "invalid_sql", "path_state": path_state}
            self.logger.warning("Infra/auth error during execution: %s", response_from_db.error)
            response_from_db = None

        return {
            "decision": "valid_sql",
            "path_state": {**path_state, "sql_response_from_db": response_from_db},
        }
