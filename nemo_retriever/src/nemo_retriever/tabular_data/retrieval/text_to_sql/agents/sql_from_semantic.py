"""
SQL generation from semantic retrieval context.

Builds SQL from graph-backed semantic candidates (custom analyses, columns),
prepared tables, and optional file extraction — not from ad-hoc “snippet”
assembly alone.

Responsibilities:
- Construct SQL using semantic candidates and schema context from CandidatePreparationAgent
- Handle file extraction results (data_for_sql) when present
- Incorporate similar questions from conversation history
- Handle feedback scenarios
- Store SQL response with custom analyses in path_state

Design Decisions:
- Primary path: vector/semantic retrieval + preparation, then LLM SQL synthesis
- Supports text-style answers when the model returns prose instead of SQL
- Optional extracted file data from upstream file steps
"""

import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage

from nemo_retriever.tabular_data.retrieval.text_to_sql.llm_invoke import safe_invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.utils import (
    build_custom_analyses_section,
    get_custom_analyses_ids,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import (
    create_sql_from_candidates_prompt,
    create_sql_user_prompt,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.models import SQLGenerationModel

logger = logging.getLogger(__name__)


def format_tables_for_prompt(tables: list[dict]) -> str:
    """
    Format tables with clear column information to prevent cross-table column confusion.

    Args:
        tables: Table dicts from ``path_state["relevant_tables"]`` — each must expose
            ``columns`` as a list of dicts (from ``_normalize_table_to_relevant_shape`` / prep).

    Returns:
        Formatted string clearly showing which columns belong to each table
    """
    if not tables:
        return "No tables available"

    formatted_tables = []
    for table in tables:
        table_parts = []

        # Table identifier
        table_name = table.get("name", "UNKNOWN")
        table_label = table.get("label", "")
        table_id = table.get("id", "")

        # Database and schema info
        db_name = table.get("db_name", "")
        schema_name = table.get("schema_name", "")

        # Build table header
        if db_name and schema_name:
            full_name = f"{db_name}.{schema_name}.{table_name}"
        else:
            full_name = table_name

        table_parts.append(f"TABLE: {full_name}")
        if table_label and table_label != table_name:
            table_parts.append(f"  Label: {table_label}")
        table_parts.append(f"  ID: {table_id}")

        # Primary key
        if "primary_key" in table:
            table_parts.append(f"  Primary Key: {table['primary_key']}")

        columns = table.get("columns")
        if not isinstance(columns, list):
            columns = []
        if columns:
            table_parts.append("  AVAILABLE COLUMNS (only use these columns for this table):")
            for col in columns:
                # Handle both dict and string column formats
                if isinstance(col, dict):
                    col_name = col.get("name", "UNKNOWN")
                    col_type = col.get("data_type", "UNKNOWN")
                    col_desc = col.get("description", "")

                    col_line = f"    - {col_name} ({col_type})"
                    if col_desc:
                        col_line += f" - {col_desc}"
                    table_parts.append(col_line)
                elif isinstance(col, str):
                    # If column is a string, use it directly
                    table_parts.append(f"    - {col}")
                else:
                    # Unknown format, convert to string
                    table_parts.append(f"    - {str(col)}")

        formatted_tables.append("\n".join(table_parts))

    return "\n\n".join(formatted_tables)


class SQLFromCandidatesAgent(BaseAgent):
    """
    Agent that constructs SQL from semantic retrieval and prepared schema context.

    Uses candidates, table groups, and related signals produced by
    CandidatePreparationAgent, then prompts the LLM to produce SQL

    Input Requirements:
    - path_state["retrieved_candidates"]: Candidate dicts from preparation
    - path_state["relevant_tables"]: schema context
    - path_state["relevant_queries"]: Relevant queries (from CandidatePreparationAgent)

    Output:
    - path_state["sql_generation_result"]: SQL response with SQL code or text answer
    - path_state["relevant_tables"]: Relevant tables used
    - path_state["custom_analyses_used"]: Semantic entity IDs used
    - decision: "constructable" or "unconstructable"
    """

    def __init__(self):
        super().__init__("sql_from_semantic")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that prepared candidates exist for semantic SQL construction."""
        path_state = state.get("path_state", {})
        if not path_state.get("retrieved_candidates"):
            self.logger.warning("No candidates found for SQL construction from semantic context")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Construct SQL from semantic candidates and prepared schema context.

        Uses CandidatePreparationAgent outputs (candidates, tables, queries,
        similar questions). May return a text response when the model does not emit SQL.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains SQL response, tables, connection, custom analyses
            - messages: Adds SQL response to messages
            - decision: "constructable" or "unconstructable"
        """
        path_state = state.get("path_state", {})
        llm = state["llm"]
        connector = state["connector"]
        question = get_question_for_processing(state)

        relevant_tables = path_state.get("relevant_tables", [])
        relevant_queries = path_state.get("relevant_queries", [])
        similar_questions = path_state.get("similar_questions", [])
        custom_analyses = path_state.get("custom_analyses", [])
        custom_analyses_str = path_state.get("custom_analyses_str", [])

        # Format similar questions for prompt
        similar_questions_txt = "\n".join(f"question: {x[0]}\nanswer: {x[1]}" for x in similar_questions)
        self.logger.info(f"Using {len(similar_questions)} similar questions from conversations.")

        def build_messages() -> list:
            """
            Build messages for SQL construction.

            Includes semantic candidate context, similar questions, and optionally
            extracted file data or file excerpts.
            """
            observation_block = f"\nlist of important semantic entities with sql snippets:\n{custom_analyses_str}\n"

            # Build user prompt with formatted tables
            user_prompt = create_sql_user_prompt.format(
                dialect=connector.dialect,
                main_question=question,
                observation_block=observation_block,
                queries=relevant_queries,
                qa_from_conversations=similar_questions_txt,
                tables=format_tables_for_prompt(relevant_tables),
            )

            # Choose system prompt based on context
            system_prompt = create_sql_from_candidates_prompt(custom_analyses)

            messages = state["messages"] + [
                SystemMessage(content=system_prompt),
                AIMessage(content=user_prompt),
            ]

            # Add calendar time window reminder if needed
            if any(phrase in question.lower() for phrase in ["last week", "last month", "last year"]):
                messages.append(
                    SystemMessage(content="Apply only calendar time windows. DO NOT apply rolling time windows.")
                )

            return messages

        # Choose schema based on context
        # Use SQLGenerationModel for new flow (without formatting)
        # Keep old models for feedback scenarios

        schema = SQLGenerationModel

        def run_with_context() -> tuple:
            """Invoke LLM with messages, optionally including file snippets and extracted data."""
            messages = build_messages()
            try:
                response = safe_invoke_with_structured_output(llm, messages, schema)
            except Exception as e:
                self.logger.error(
                    "LLM structured output failed: %s: %s",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                return None, messages
            if response and hasattr(response, "response") and response.response:
                self.logger.info(
                    "LLM response generated: %s...",
                    response.response[:100],
                )
            return response, messages

        MAX_RETRIES = 3
        response, messages = None, []
        for attempt in range(1, MAX_RETRIES + 1):
            response, messages = run_with_context()
            if response is not None:
                break
            self.logger.warning(
                "LLM returned None on attempt %d/%d — retrying.",
                attempt,
                MAX_RETRIES,
            )

        if response is None:
            self.logger.error("LLM returned None after %d attempts.", MAX_RETRIES)
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": "LLM failed to produce a response.",
                },
                "decision": "unconstructable",
            }

        # Check if we have a valid response (either SQL or text-based answer from file contents)
        has_sql = bool(response.sql_code and response.sql_code.strip())
        has_response = bool(response.response and response.response.strip())

        if has_sql:
            custom_analyses_used = []
            if hasattr(response, "custom_analyses_used") and response.custom_analyses_used:
                # Filter custom analyses to keep only those found in candidates
                candidates_ids = {
                    c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
                    for c in path_state["retrieved_candidates"]
                }
                filtered_elements = [
                    elem
                    for elem in response.custom_analyses_used
                    if (elem.id if hasattr(elem, "id") else elem.get("id")) in candidates_ids
                ]
                response.custom_analyses_used = filtered_elements
                custom_analyses_used = get_custom_analyses_ids(response.custom_analyses_used)

            return {
                "messages": messages,  # Don't add formatted response here - formatting agent will do it
                "path_state": {
                    **path_state,
                    "sql_generation_result": response,  # Keep as object (Pydantic model)
                    "relevant_tables": relevant_tables if has_sql else [],
                    "custom_analyses_used": custom_analyses_used,
                },
                "decision": "constructable",
            }
        elif has_response:
            custom_analyses_used = []
            if hasattr(response, "custom_analyses_used"):
                response.response += build_custom_analyses_section(
                    response.custom_analyses_used, path_state["retrieved_candidates"]
                )
                custom_analyses_used = get_custom_analyses_ids(response.custom_analyses_used)

            return {
                "messages": messages + [AIMessage(content=response.response)],
                "path_state": {
                    **path_state,
                    "sql_generation_result": response,
                    "relevant_tables": relevant_tables if has_sql else [],
                    "custom_analyses_used": custom_analyses_used,
                },
                "decision": "constructable",
            }
        else:
            # SQL could not be generated
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": response.response or "Unable to construct response.",
                },
                "decision": "unconstructable",
            }
