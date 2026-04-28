"""
SQL Validation Agent

This agent validates SQL queries before execution.
Checks for logical correctness, not just syntax.

Responsibilities:
- Validate SQL logic (not just syntax)
- Check for common mistakes (self-comparisons, incorrect filters, etc.)
- Handle text-based answers (skip validation)
- Store validation result in path_state

Design Decisions:
- Skips validation for text-based answers (from file contents)
- Uses LLM to validate logical correctness
- Sets connection data based on retrieved tables
- Returns decision: "valid_sql" or "invalid_sql"
"""

import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentState
from nemo_retriever.tabular_data.retrieval.text_to_sql.utils import (
    get_all_schemas_ids,
    get_schemas_by_ids,
    get_custom_analyses_ids,
)

logger = logging.getLogger(__name__)


class SQLValidationAgent(BaseAgent):
    """
    Agent that validates SQL queries before execution.

    This agent performs logical validation of SQL queries, checking for
    common mistakes like self-comparisons, incorrect filters, etc.

    Input Requirements:
    - path_state["sql_generation_result"]: SQL response to validate
    - path_state["relevant_tables"]: Relevant tables used

    Output:
    - path_state["sql_response_from_db"]: None (will be set after execution)
    - path_state["sql_columns"]: Column IDs from SQL
    - path_state["custom_analyses_used"]: Semantic entity IDs used
    - decision: "valid_sql" or "invalid_sql"
    """

    def __init__(self):
        super().__init__("sql_validation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that SQL response is available."""
        path_state = state.get("path_state", {})
        if state.get("decision") == "unconstructable":
            # Skip validation if SQL couldn't be constructed
            return False
        if not path_state.get("sql_generation_result"):
            self.logger.warning("No SQL response found for validation")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate SQL query.

        Performs logical validation using LLM and query_validation function.
        Sets connection data and extracts columns from SQL.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains validation result and extracted data
            - decision: "valid_sql" or "invalid_sql"
        """
        path_state = state.get("path_state", {})
        response = path_state.get("sql_generation_result")
        connector = state.get("connector")
        dialect = connector.dialect
        schemas_ids = get_all_schemas_ids()
        schemas = get_schemas_by_ids(schemas_ids)

        validation_result = self._sql_parse_validation(schemas, response.sql_code, dialect)

        if validation_result.get("error"):
            error_msg = validation_result["error"]
            self.logger.info(f"SQL validation failed: {error_msg}")
            path_state["error"] = error_msg
            return {
                "decision": "invalid_sql",
                "path_state": path_state,
            }

        sql_columns = validation_result.get("sql_columns") or []
        custom_analyses_used = []
        if hasattr(response, "custom_analyses_used"):
            custom_analyses_used = get_custom_analyses_ids(response.custom_analyses_used)

        # Store connection_data in the format expected by execute_sql_query
        # execute_sql_query expects connections as a list
        updated_path_state = {
            **path_state,
            "sql_response_from_db": None,  # Will be set after execution
            "sql_columns": sql_columns,
            "custom_analyses_used": custom_analyses_used,
            "sql_code": response.sql_code,  # Store SQL code for execution
        }

        self.logger.info(f"SQL validation passed, columns: {len(sql_columns)}")

        return {
            "decision": "valid_sql",
            "path_state": updated_path_state,
        }

    @staticmethod
    def _sql_parse_validation(schemas, sql: str, dialect: str) -> dict:
        result: dict = {}
        try:
            parse_query_single(
                sql=sql,
                dialect=dialect,
                schemas=schemas,
            )
            result["success"] = True
        except Exception as error:
            result.update({"error": str(error), "another_try": 1})
        return result
