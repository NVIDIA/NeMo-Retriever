"""
SQL Semantic Validation Agent

This agent validates the semantic correctness of the generated SQL query.
Checks that all required entities are covered, joins are logical, and aggregations are correct.

Responsibilities:
- Validate SQL covers all required entities (from action_input.required_entity_name)
- Validate joins are logical and correct
- Validate aggregation functions (SUM, AVG, COUNT, etc.) are correct
- Return decision: "intent_valid" or "intent_invalid"

Design Decisions:
- Runs after SQLValidationAgent (syntax validation)
- Uses entities from action_input (extracted during routing phase)
- Uses LLM to validate semantic correctness
- If invalid, routes back to reconstruction with specific error
- Graph routing skips this agent after 5 reconstructions to avoid infinite loops
"""

import logging
from typing import Dict, Any
from nemo_retriever.tabular_data.retrieval.text_to_sql.llm_invoke import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import (
    INTENT_VALIDATION_SYSTEM_PROMPT,
    create_intent_validation_prompt,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, AIMessage


logger = logging.getLogger(__name__)


class IntentValidationModel(BaseModel):
    """Model for intent validation response."""

    is_valid: bool = Field(
        description="Whether the SQL query has any CRITICAL issues. Should be True unless there are serious problems."
    )
    missing_entities: list[str] = Field(
        default_factory=list,
        description="List of CRITICAL entity names that are completely missing and essential for the query. Leave EMPTY [] if no entities are missing — do NOT add explanatory text like 'no missing entities'.",
    )
    join_issues: list[str] = Field(
        default_factory=list,
        description="List of CRITICAL join issues that would produce completely wrong results. Leave EMPTY [] if there are no join issues — do NOT add explanatory text like 'no join issues'.",
    )
    aggregation_issues: list[str] = Field(
        default_factory=list,
        description="List of CRITICAL aggregation issues that are clearly wrong (not minor variations). Leave EMPTY [] if there are no aggregation issues — do NOT add explanatory text like 'no aggregation issues'.",
    )




class IntentValidationAgent(BaseAgent):
    """
    Agent that validates semantic correctness of SQL queries.

    This agent performs semantic validation of SQL queries, checking that
    all required entities are covered, joins are logical, and aggregations
    are correct.

    Input Requirements:
    - path_state["sql_generation_result"]: SQL response to validate
    - path_state["action_input"]["required_entity_name"]: Required entities from action input
    - state["initial_question"]: User's question
    - state["llm"]: LLM instance

    Output:
    - decision: "intent_valid" or "intent_invalid"
    - path_state["error"]: Error message if invalid
    """

    def __init__(self):
        super().__init__("intent_validation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that SQL response is available."""
        path_state = state.get("path_state", {})

        # Skip if already marked as invalid from syntax validation
        if state.get("decision") == "invalid_sql":
            self.logger.info("Skipping intent validation - SQL already marked invalid")
            return False

        # Skip if unconstructable
        if state.get("decision") == "unconstructable":
            self.logger.info("Skipping intent validation - SQL unconstructable")
            return False

        if not path_state.get("sql_generation_result"):
            self.logger.warning("No SQL response found for intent validation")
            return False

        # Check if this is a text-based answer (skip intent validation)
        if path_state.get("is_text_based_answer", False):
            self.logger.info("Text-based answer, skipping intent validation")
            return False

        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate SQL query semantic correctness.

        Uses LLM to check that the SQL query:
        - Covers all required entities
        - Has logical joins
        - Uses correct aggregations

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - decision: "intent_valid" or "intent_invalid"
            - path_state: Updated with error if invalid
        """
        path_state = state.get("path_state", {})
        llm = state["llm"]

        # Get SQL response
        response = path_state.get("sql_generation_result")
        sql_code = response.sql_code if hasattr(response, "sql_code") else ""

        if not sql_code or not sql_code.strip():
            self.logger.warning("No SQL code found for intent validation")
            return {
                "decision": "intent_valid",  # Skip validation if no SQL
                "path_state": path_state,
            }

        # Get user's question
        question = get_question_for_processing(state)

        # Get entities from action_input
       
        required_entities = path_state.get("entities_and_concepts", [])

        if required_entities:
            self.logger.info(
                f"Validating intent with required entities: {required_entities}"
            )
            entities_text = "\n".join([f"- {entity}" for entity in required_entities])
        else:
            self.logger.info("No required entities specified for intent validation")
            entities_text = "No specific entities required"

        validation_prompt = create_intent_validation_prompt(question, entities_text, sql_code)

        messages = [
            SystemMessage(content=INTENT_VALIDATION_SYSTEM_PROMPT),
            AIMessage(content=validation_prompt),
        ]

        # Call LLM for validation
        try:
            validation_result = invoke_with_structured_output(
                llm, messages, IntentValidationModel
            )
        except Exception as e:
            self.logger.error(f"Intent validation LLM call failed: {str(e)}")
            # On error, pass through (don't block execution)
            return {
                "decision": "intent_valid",
                "path_state": path_state,
            }

        # Check if valid
        if validation_result.is_valid:
            self.logger.info("SQL validation passed (no critical issues)")
            return {
                "decision": "intent_valid",
                "path_state": path_state,
            }

        has_real_issues = (
            validation_result.missing_entities
            or validation_result.join_issues
            or validation_result.aggregation_issues
        )
        if not has_real_issues:
            self.logger.info("SQL validation passed (is_valid=False but no real issues listed)")
            return {
                "decision": "intent_valid",
                "path_state": path_state,
            }

        # Build detailed error message
        error_parts = ["Critical SQL issues found:"]

        if validation_result.missing_entities:
            error_parts.append(
                f"\n\nCritical missing entities: {', '.join(validation_result.missing_entities)}"
            )

        if validation_result.join_issues:
            error_parts.append(
                "\n\nCritical join issues:\n"
                + "\n".join(f"  - {issue}" for issue in validation_result.join_issues)
            )

        if validation_result.aggregation_issues:
            error_parts.append(
                "\n\nCritical aggregation issues:\n"
                + "\n".join(
                    f"  - {issue}" for issue in validation_result.aggregation_issues
                )
            )

        error_msg = "".join(error_parts)

        self.logger.info(
            f"SQL validation failed (critical issues): {error_msg[:200]}..."
        )

        # Store error and return invalid decision
        updated_path_state = {
            **path_state,
            "error": error_msg,
        }

        return {
            "decision": "intent_invalid",
            "path_state": updated_path_state,
        }
