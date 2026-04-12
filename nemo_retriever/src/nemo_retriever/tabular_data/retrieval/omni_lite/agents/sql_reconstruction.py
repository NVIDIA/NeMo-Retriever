"""
SQL Reconstruction Agent

This agent reconstructs SQL queries that failed validation.
Used to fix SQL errors and improve SQL quality based on validation feedback.

Responsibilities:
- Reconstruct SQL that failed validation
- Fix errors based on validation feedback
- Preserve candidate context for reconstruction
- Store reconstructed SQL response in path_state

Design Decisions:
- Used when SQL validation fails
- Uses error message from validation to guide reconstruction
- Preserves relevant candidates and context
- Can handle feedback scenarios differently
"""

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage, SystemMessage
from nemo_retriever.tabular_data.retrieval.omni_lite.ai_services import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.state import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.models import SQLGenerationModel
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import get_semantic_entities_ids



logger = logging.getLogger(__name__)


class SQLReconstructionAgent(BaseAgent):
    """
    Agent that reconstructs SQL queries that failed validation.

    This agent fixes SQL errors by using validation feedback and
    reconstructing the query with corrections.

    Input Requirements:
    - path_state["error"]: Error message from validation
    - path_state["llm_calc_response"]: Previous (incorrect) SQL response
    - path_state["candidates"]: Relevant candidates for context
    - state["initial_question"]: Original user question

    Output:
    - path_state["llm_calc_response"]: Reconstructed SQL response
    - path_state["relevant_tables"]: Relevant tables
    - path_state["semantic_elements"]: Semantic entity IDs used
    - messages: Updated messages with reconstruction
    """

    def __init__(self):
        super().__init__("sql_reconstruction")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that error and previous response are available."""
        path_state = state.get("path_state", {})
        if not path_state.get("error"):
            self.logger.warning("No error found for SQL reconstruction")
            return False
        if not path_state.get("llm_calc_response"):
            self.logger.warning("No previous SQL response found for reconstruction")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Reconstruct SQL based on validation error.

        Uses the validation error message to guide SQL reconstruction,
        preserving relevant candidates and context.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains reconstructed SQL response
            - messages: Updated messages with reconstruction
            - thoughts: Reconstruction reasoning
        """
        path_state = state.get("path_state", {})
        llm = state["llm"]
        error = path_state.get("error", "")
        incorrect_response = path_state.get("llm_calc_response")
        question = get_question_for_processing(state)

        # Build messages list starting from state messages
        messages = state["messages"]

        # Variables to track all tables and FKs for updating path_state
        all_tables = None

        # Build error prompt for reconstruction
        error_prompt = (
            "The following SQL contains an ERROR:\n\n"
            f"```sql\n{incorrect_response.sql_code}\n```\n\n"
            f"Validation failed with the following message:\n{error}\n\n"
            "Please correct the SQL. Do not return the same SQL — it is invalid.\n"
            "Do not explain how you corrected the sql, like you were never wrong. \n"
        )

        # Add specific guidance for unknown column errors (likely wrong schema)
        if "unknown column name" in error.lower() and "in table" in error.lower():
            error_prompt += (
                "\n⚠️ RECONSTRUCTION HINT: The 'Unknown column name in table' error typically indicates "
                "that you're using the correct table name but referencing it from the wrong schema. "
                "Look for the same table name in a different schema from the additional tables provided above. "
                "The column you're looking for likely exists in the same table name under a different schema.\n\n"
                "VERY IMPORTANT: Review the failed SQL and the error message carefully. "
                "Do NOT attempt to use the same column that caused this error in your next attempt. "
                "Either find the column in a different schema's version of the table, or exclude it from your query.\n"
            )

        error_prompt += (
            "\nUse only the tables provided in the history.\n\n"
            "Use only the foreign keys provided in the history. NEVER create new foreign keys! \n"
            f"The original question was: {question}.\n"
            "You must include corrected sql in your final answer.\n"
            "Follow the rules defined in the previous messages for writing the final answer."
        )

        messages = messages + [AIMessage(content=error_prompt)]

        # Choose schema based on context
        # Use SQLGenerationModel for reconstruction (same as from_multiple_snippets)
        # Formatting will be handled by SQLResponseFormattingAgent

        schema = SQLGenerationModel  # Use SQLGenerationModel for all non-feedback cases
            

        # Invoke LLM for reconstruction
        response = invoke_with_structured_output(llm, messages, schema)

        if response is None:
            self.logger.warning("SQL reconstruction returned None — marking unconstructable")
            return {
                "decision": "unconstructable",
                "path_state": path_state,
            }

        sql_preview = (getattr(response, "sql_code", "") or "")[:100]
        self.logger.info(f"SQL reconstructed: {sql_preview}...")

        response_explanation = getattr(
            response, "response", getattr(response, "thought", "No explanation")
        ) or ""
        self.logger.info(f"Reconstruction explanation: {response_explanation[:100]}...")

        # Extract semantic elements
        semantic_elements = []
        if hasattr(response, "semantic_elements"):
            semantic_elements = get_semantic_entities_ids(response.semantic_elements)

        return {
            "messages": messages,
            "path_state": {
                **path_state,
                "llm_calc_response": response,
                "relevant_tables": all_tables
                if all_tables is not None
                else path_state.get("relevant_tables", []),
                "semantic_elements": semantic_elements,
            },
        }
