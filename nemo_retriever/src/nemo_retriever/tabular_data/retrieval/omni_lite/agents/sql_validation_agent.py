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

from nemo_retriever.tabular_data.retrieval.omni_lite.agents.query_validation import query_validation
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import get_semantic_entities_ids


logger = logging.getLogger(__name__)


class SQLValidationAgent(BaseAgent):
    """
    Agent that validates SQL queries before execution.

    This agent performs logical validation of SQL queries, checking for
    common mistakes like self-comparisons, incorrect filters, etc.

    Input Requirements:
    - path_state["llm_calc_response"]: SQL response to validate
    - path_state["relevant_tables"]: Relevant tables used

    Output:
    - path_state["sql_response_from_db"]: None (will be set after execution)
    - path_state["sql_columns"]: Column IDs from SQL
    - path_state["semantic_elements"]: Semantic entity IDs used
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
        if not path_state.get("llm_calc_response"):
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
       
        response = path_state.get("llm_calc_response")
        relevant_tables = path_state.get("relevant_tables", [])
        dialects = state["dialects"]


    
        # Convert schema IDs to schemas dict format (keyed by schema name)
        # relevant_schemas_ids is a set of schema IDs, need to convert to dict format
        schemas = get_schemas_slim(list(relevant_schemas_ids)) # TODO where to get schemas?

        # Validate SQL using query_validation
        # This extracts columns, checks syntax, validates logic
        validation_result = query_validation(
            schemas,
            response.sql_code,
            dialects,
        )

        # query_validation returns a dict, not an object
        if validation_result.get("error"):
            # SQL is invalid
            error_msg = validation_result["error"]
            self.logger.info(f"SQL validation failed: {error_msg}")
            path_state["error"] = error_msg
            return {
                "decision": "invalid_sql",
                "path_state": path_state,
            }

        # SQL is valid, extract columns and semantic elements
        sql_columns = validation_result.get("sql_columns") or []
        candidates_with_entities = path_state.get("candidates", [])

        # Extract just the candidate objects for processing
        candidates = [
            item["candidate"]
            if isinstance(item, dict) and "candidate" in item
            else item
            for item in candidates_with_entities
        ]

        semantic_elements = []
        if hasattr(response, "semantic_elements"):
            semantic_elements = get_semantic_entities_ids(response.semantic_elements)

        # Store connection_data in the format expected by execute_sql_query
        # execute_sql_query expects connections as a list
        updated_path_state = {
            **path_state,
            "sql_response_from_db": None,  # Will be set after execution
            "pii_objects": validation_result.get("pii_objects") or [],
            "sql_columns": sql_columns,
            "semantic_elements": semantic_elements,
            "sql_code": response.sql_code,  # Store SQL code for execution
        }

        self.logger.info(f"SQL validation passed, columns: {len(sql_columns)}")

        return {
            "decision": "valid_sql",
            "path_state": updated_path_state,
        }
