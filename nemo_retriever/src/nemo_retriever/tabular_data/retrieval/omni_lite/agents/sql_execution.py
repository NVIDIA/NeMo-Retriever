"""
SQL Execution Agent

This agent executes validated SQL queries against the database.
Handles errors and stores results in path_state.

Responsibilities:
- Execute SQL queries against the database
- Handle execution errors (infra/auth errors vs SQL errors)
- Store query results in path_state
- Tag queries for observability

Design Decisions:
- Only executes if SQL is valid (called after validation)
- Distinguishes infra/auth errors from SQL errors
- Stores results for use in response generation
- Returns decision: "valid_sql" or "invalid_sql"
"""

import asyncio
import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.retrieval.omni_lite.agents.query_executor import run_query
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import AgentState


logger = logging.getLogger(__name__)


class SQLExecutionAgent(BaseAgent):
    """
    Agent that executes SQL queries against the database.

    This agent executes validated SQL and handles errors appropriately.
    Distinguishes between SQL errors (need reconstruction) and infra/auth errors.

    Input Requirements:
    - path_state["llm_calc_response"]: SQL response with sql_code
    - path_state["connection_data"]["connection"]: Database connection info
    - path_state["pii_objects"]: PII objects detected
    - state["account_id"]: Account ID
    - state["source"]: Request source

    Output:
    - path_state["sql_response_from_db"]: Query execution result
    - path_state["pii_objects"]: PII objects (preserved)
    - path_state["connection_data"]: Connection data (preserved)
    - path_state["error"]: Error message if execution failed (SQL error only)
    - decision: "valid_sql" or "invalid_sql"
    """

    def __init__(self):
        super().__init__("sql_execution")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that SQL and connection are available."""
        path_state = state.get("path_state", {})
        llm_response = path_state.get("llm_calc_response")

        if not llm_response:
            self.logger.warning("No LLM response found for execution")
            return False

        sql_code = getattr(llm_response, "sql_code", None)
        connection = getattr(llm_response, "connection", None)

        if not sql_code:
            self.logger.warning("No SQL code found for execution")
            return False
       
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute SQL query against the database.

        Executes the SQL and handles errors. Distinguishes between
        SQL errors (need reconstruction) and infra/auth errors.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains execution result
            - decision: "valid_sql" or "invalid_sql"
        """
        path_state = state.get("path_state", {})
        sql_code = path_state.get("sql_code")

        connection_id = connections[0]["connection"]["id"] # TODO how to run ??
        db_name = connections[0]["db"]["name"]

        # Execute SQL query
        response_from_db = asyncio.run(
            run_query(
                sql_code,
                connection_id,
                db_name,
                state["source"],
            )
        )

        if response_from_db.error:
            self.logger.info(f"SQL execution error: {response_from_db.error}")
            is_infra_error = is_infra_or_auth_error(response_from_db.error)

            if not is_infra_error:
                # SQL error - need reconstruction
                path_state["error"] = response_from_db.error
                return {
                    "decision": "invalid_sql",
                    "path_state": path_state,
                }
            else:
                # Infra/auth error - don't retry, just log
                self.logger.warning(
                    f"Infra/auth error during execution: {response_from_db.error}"
                )
                response_from_db = None

        # Store execution result
        # Preserve connection_data format for backward compatibility
        connection_data = connections[0] if connections else None
        updated_path_state = {
            **path_state,
            "sql_response_from_db": response_from_db,
        }

        self.logger.info("SQL execution completed successfully")

        return {
            "decision": "valid_sql",
            "path_state": updated_path_state,
        }
