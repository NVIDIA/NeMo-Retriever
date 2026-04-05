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
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import get_question_for_processing
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
    - path_state["feedback"]: Optional feedback flag

    Output:
    - path_state["llm_calc_response"]: Reconstructed SQL response
    - path_state["relevant_tables"]: Relevant tables
    - path_state["connection"]: Database connection info
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
        is_feedback = path_state.get("feedback", False)
        question = get_question_for_processing(state)

        # Check if this is the first reconstruction attempt (not a second invalid cycle)
        reconstruction_count = path_state.get("reconstruction_count", 0)
        is_first_reconstruction = reconstruction_count == 0

        # Build messages list starting from state messages
        messages = state["messages"]

        # Variables to track all tables and FKs for updating path_state
        all_tables = None
        all_fks = None

        # Add system message with all table groups ONLY on first reconstruction
        if is_first_reconstruction:
            table_groups = path_state.get("table_groups", [])
            if table_groups:
                # Extract all tables and FKs from all groups (not just the best one)
                tables_with_entities = path_state.get("tables_with_entities", [])
                fks_with_entities = path_state.get("fks_with_entities", [])

                # Fallback to all tables and FKs from all groups
                all_tables = [item["table"] for item in tables_with_entities]
                all_fks = [item["fk"] for item in fks_with_entities]

                self.logger.info(
                    f"First reconstruction - using ALL table groups: {len(all_tables)} tables, "
                    f"{len(all_fks)} FKs from {len(table_groups)} groups"
                )

                # Build system message with additional context
                system_message_parts = []

                if all_tables:
                    # Create structured table information
                    table_info_list = []
                    for t in all_tables:
                        table_info = {
                            "database": t.get("db_name", ""),
                            "schema_name": t.get("schema_name", ""),
                            "table_name": t.get("name", "UNKNOWN"),
                            "columns": [],
                        }

                        # Extract column information
                        columns = t.get("columns", [])
                        for col in columns:
                            if isinstance(col, dict):
                                col_info = {
                                    "column_name": col.get("name", "UNKNOWN"),
                                }
                                # Add description if available
                                if col.get("description"):
                                    col_info["description"] = col.get("description")
                                table_info["columns"].append(col_info)
                            elif isinstance(col, str):
                                table_info["columns"].append({"column_name": col})

                        table_info_list.append(table_info)

                    # Format tables for the prompt
                    formatted_tables = []
                    for tbl in table_info_list:
                        full_name = (
                            f"{tbl['database']}.{tbl['schema_name']}.{tbl['table_name']}"
                            if tbl["database"] and tbl["schema_name"]
                            else tbl["table_name"]
                        )
                        table_str = f"- {full_name}"
                        if tbl["columns"]:
                            cols_str = ", ".join(
                                [
                                    f"{c['column_name']}"
                                    + (
                                        f" ({c['description']})"
                                        if c.get("description")
                                        else ""
                                    )
                                    for c in tbl["columns"]
                                ]
                            )
                            table_str += f"\n  Columns: {cols_str}"
                        formatted_tables.append(table_str)

                    system_message_parts.append(
                        f"ADDITIONAL TABLES AVAILABLE (from all table groups):\n"
                        f"{chr(10).join(formatted_tables)}"
                    )

                if all_fks:
                    fk_descriptions = [
                        f"{fk.get('table1')}.{fk.get('column1')} = {fk.get('table2')}.{fk.get('column2')}"
                        for fk in all_fks
                    ]
                    system_message_parts.append(
                        f"ADDITIONAL FOREIGN KEYS AVAILABLE (from all table groups):\n"
                        f"{chr(10).join(fk_descriptions)}"
                    )

                if system_message_parts:
                    system_content = "\n\n".join(system_message_parts)
                    system_content += "\n\nYou may use these additional tables and foreign keys to correct the SQL."
                    messages = messages + [SystemMessage(content=system_content)]
        else:
            self.logger.info(
                f"Second reconstruction attempt (count={reconstruction_count}) - not adding additional table groups"
            )

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

        self.logger.info(
            f"SQL reconstructed: {response.sql_code[:100] if response.sql_code else 'None'}..."
        )
        # Log response explanation (from 'response' for new model, 'thought' for old)
        response_explanation = getattr(
            response, "response", getattr(response, "thought", "No explanation")
        )
        self.logger.info(f"Reconstruction explanation: {response_explanation[:100]}...")

        # Extract semantic elements
        semantic_elements = []
        if hasattr(response, "semantic_elements"):
            semantic_elements = get_semantic_entities_ids(response.semantic_elements)

        return {
            "messages": messages,  # Don't add formatted response here - formatting agent will do it
            "path_state": {
                **path_state,
                "llm_calc_response": response,  # Keep as object (Pydantic model)
                "relevant_tables": all_tables
                if all_tables is not None
                else path_state.get("relevant_tables", []),
                "semantic_elements": semantic_elements,
                "reconstruction_count": reconstruction_count
                + 1,  # Increment for next time
            },
        }
