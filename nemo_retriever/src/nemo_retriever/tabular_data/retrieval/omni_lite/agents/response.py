"""
Calculation Response Agent

This agent formats and returns calculation results to the user.
Handles both SQL-based and text-based answers.

Responsibilities:
- Format calculation response with SQL results
- Include SQL code, columns, semantic elements
- Handle both SQL-based and text-based answers
- Format response using helper functions

Design Decisions:
- Used as final step in calculation flow
- Formats response consistently regardless of source (SQL or text)
- Includes all metadata (columns, semantic elements, connection info)
- Returns action="calculation" for consistency
"""

import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import get_semantic_entities_ids



logger = logging.getLogger(__name__)


class CalculationResponseAgent(BaseAgent):
    """
    Agent that formats and returns calculation results.

    This agent formats the final response with SQL results, code, columns,
    and semantic elements. Handles both SQL-based and text-based answers.

    Input Requirements:
    - path_state["llm_calc_response"]: LLM response with SQL code or text answer
    - path_state["sql_response_from_db"]: SQL execution result (if SQL-based)
    - path_state["sql_columns"]: Column IDs from SQL
    - path_state["connection_data"]["connection"]: Connection info
    - path_state["pii_objects"]: PII objects detected
    - path_state["candidates"]: Candidates for formatting
    - state["account_id"]: Account ID

    Output:
    - messages: Final response message with formatted answer
    """

    def __init__(self):
        super().__init__("calculation_response")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that LLM response is available."""
        path_state = state.get("path_state", {})
        llm_calc_response = path_state.get("llm_calc_response")
        if not llm_calc_response:
            self.logger.warning(
                "No LLM response found for calculation response formatting"
            )
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Format and return calculation response.

        Formats the final response with SQL results (if SQL-based) or
        text answer (if text-based). Includes all metadata.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - messages: Final response message
        """
        path_state = state.get("path_state", {})
        llm_response = path_state.get("llm_calc_response")

        # Get formatted response from path_state (already formatted by SQLResponseFormattingAgent)
        formatted_response = path_state.get("formatted_response", "")
        if not formatted_response:
            # Fallback: use response from llm_calc_response if formatted_response not available
            formatted_response = getattr(llm_response, "response", "")

        # Extract SQL columns and semantic elements
        sql_columns = path_state.get("sql_columns", [])
        semantic_elements = []
        if hasattr(llm_response, "semantic_elements"):
            semantic_elements = get_semantic_entities_ids(
                llm_response.semantic_elements
            )

        # Build final response dict (preparation for final_answer node)
        response = {
            "response": formatted_response,
            "sql_code": getattr(llm_response, "sql_code", ""),
            "sql_columns": sql_columns,
            "semantic_elements": semantic_elements,
            "sql_response_from_db": path_state.get("sql_response_from_db"),
        }

        self.logger.info("Calculation response prepared and returned")

        # Return final_response in path_state (for final_answer node to use)
        return {"path_state": {**path_state, "final_response": response}}
