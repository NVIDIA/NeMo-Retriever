"""
SQL generation from semantic retrieval context.

Builds SQL from graph-backed semantic candidates (custom analyses, columns),
prepared tables/FKs, and optional file extraction — not from ad-hoc “snippet”
assembly alone.

Responsibilities:
- Construct SQL using semantic candidates and schema context from CandidatePreparationAgent
- Handle file extraction results (data_for_sql) when present
- Incorporate similar questions from conversation history
- Handle feedback scenarios
- Store SQL response with semantic elements in path_state

Design Decisions:
- Primary path: vector/semantic retrieval + preparation, then LLM SQL synthesis
- Supports text-style answers when the model returns prose instead of SQL
- Optional extracted file data from upstream file steps
"""

import logging
from typing import Dict, Any, Optional

from langchain_core.messages import AIMessage, SystemMessage

from search.api.omni.agent.agents.shared.types import AgentState
from search.api.omni.agent.agents.base import BaseAgent
from enrichments.ai_services.llm_invoke import invoke_with_structured_output
from search.api.omni.agent.agents.calculation.utils import (
    build_semantic_items_section,
    get_semantic_entities_ids,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import (
    get_question_for_processing,
)
from search.api.omni.agent.agents.calculation.models import (
    SQLGenerationModel,
    CalculationFinalFeedbackResponseModel,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.prompts import (
    create_sql_from_semantic_prompt,
    create_sql_from_semantic_prompt_non_technical,
)
from search.api.omni.agent.agents.calculation.sql_generation.prompts import (
    create_sql_user_prompt,
    create_sql_feedback_prompt,
)
from search.api.omni.agent.agents.shared.prompts import snowflake_dialect_prompt

logger = logging.getLogger(__name__)


def format_table_groups_for_prompt(table_groups: list[dict]) -> str:
    """
    Format table groups showing which tables are connected and which entities they relate to.

    Args:
        table_groups: List of table group dictionaries with structure:
            {"tables": [...], "entities": [...], "candidate_ids": [...], "fks": [...]}

    Returns:
        Formatted string showing table groups with their connections and entities
    """
    if not table_groups:
        return "No table groups available"

    formatted_groups = []
    for i, group in enumerate(table_groups, 1):
        group_parts = []
        entities = group.get("entities", [])
        tables = group.get("tables", [])
        fks = group.get("fks", [])

        # Group header with entities
        if entities:
            group_parts.append(f"TABLE GROUP {i} (Related to: {', '.join(entities)}):")
        else:
            group_parts.append(f"TABLE GROUP {i}:")

        # Add tables in this group
        for table in tables:
            table_name = table.get("name", "UNKNOWN")
            db_name = table.get("db_name", "")
            schema_name = table.get("schema_name", "")

            if db_name and schema_name:
                full_name = f"{db_name}.{schema_name}.{table_name}"
            else:
                full_name = table_name

            group_parts.append(f"  - {full_name}")

        # Add FKs connecting tables in this group
        if fks:
            group_parts.append("  Foreign Key Relationships:")
            for fk in fks:
                fk_str = f"    {fk.get('table1')}.{fk.get('column1')} = {fk.get('table2')}.{fk.get('column2')}"
                group_parts.append(fk_str)

        formatted_groups.append("\n".join(group_parts))

    return "\n\n".join(formatted_groups)


def format_tables_for_prompt(tables: list[dict]) -> str:
    """
    Format tables with clear column information to prevent cross-table column confusion.

    Args:
        tables: List of table dictionaries with columns, db_name, schema_name

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

        # Foreign key
        if "foreign_key" in table:
            table_parts.append(f"  Foreign Key: {table['foreign_key']}")

        # Columns - MOST IMPORTANT: explicitly list each column
        if "columns" in table and table["columns"]:
            table_parts.append(
                "  AVAILABLE COLUMNS (only use these columns for this table):"
            )
            for col in table["columns"]:
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
        else:
            # Fallback to table_info if columns not parsed
            if "table_info" in table:
                table_parts.append(f"  Info: {table['table_info']}")

        formatted_tables.append("\n".join(table_parts))

    return "\n\n".join(formatted_tables)


class SQLFromSemanticAgent(BaseAgent):
    """
    Agent that constructs SQL from semantic retrieval and prepared schema context.

    Uses candidates, table groups, FKs, and related signals produced by
    CandidatePreparationAgent, then prompts the LLM to produce SQL (or a text
    answer in some feedback/file cases).

    Input Requirements:
    - path_state["candidates"]: Wrapped candidates from preparation (candidate + entity)
    - path_state["tables_with_entities"] / table_groups / fks_with_entities: schema context
    - path_state["relevant_queries"]: Relevant queries (from CandidatePreparationAgent)
    - path_state["similar_questions"]: Similar questions (from CandidatePreparationAgent)
    - path_state["complex_candidates"]: Complex candidates (from CandidatePreparationAgent)
    - path_state["complex_candidates_str"]: String representation (from CandidatePreparationAgent)
    - path_state["extracted_file_data"]: Optional extracted data from files
    - state["initial_question"]: User's question
    - state["dialects"]: SQL dialects to support

    Output:
    - path_state["llm_calc_response"]: SQL response with SQL code or text answer
    - path_state["relevant_tables"]: Relevant tables used
    - path_state["connection"]: Database connection info
    - path_state["semantic_elements"]: Semantic entity IDs used
    - decision: "constructable" or "unconstructable"
    """

    def __init__(self):
        super().__init__("sql_from_semantic")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that prepared candidates exist for semantic SQL construction."""
        path_state = state.get("path_state", {})
        if not path_state.get("candidates"):
            self.logger.warning(
                "No candidates found for SQL construction from semantic context"
            )
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Construct SQL from semantic candidates and prepared schema context.

        Uses CandidatePreparationAgent outputs (candidates, tables, FKs, queries,
        similar questions). May return a text response when the model does not emit SQL.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains SQL response, tables, connection, semantic elements
            - messages: Adds SQL response to messages
            - decision: "constructable" or "unconstructable"
        """
        path_state = state.get("path_state", {})
        llm = state["llm"]
        dialects = state["dialects"]
        question = get_question_for_processing(state)
        candidates_with_entities = path_state["candidates"]

        # Extract just the candidate objects for processing
        candidates = [item["candidate"] for item in candidates_with_entities]


        # Get pre-fetched data from CandidatePreparationAgent
        table_groups = path_state.get("table_groups", [])
        tables_with_entities = path_state.get("tables_with_entities", [])
        fks_with_entities = path_state.get("fks_with_entities", [])
        relevant_queries = path_state.get("relevant_queries", [])
        similar_questions = path_state.get("similar_questions", [])
        complex_candidates = path_state.get("complex_candidates", [])
        complex_candidates_str = path_state.get("complex_candidates_str", [])
        action_input = path_state.get("action_input", {})
        entities = action_input.get("required_entity_name", []) if action_input else []

        # Select the best table group: most entities covered, then most connected tables
        best_group = None
        if table_groups:
            # Score each group by: prioritize entities first (always preferred), then tables
            # Using tuple comparison: (entities, tables) - entities will always be compared first
            def score_group(group):
                num_tables = len(group.get("tables", []))
                num_entities = len(group.get("entities", []))
                return (num_entities, num_tables)

            best_group = max(table_groups, key=score_group)
            self.logger.info(
                f"Selected best table group: {len(best_group.get('tables', []))} tables, "
                f"{len(best_group.get('entities', []))} entities, "
                f"{len(best_group.get('fks', []))} FKs"
            )

        # Extract tables and FKs from the best group
        if len(entities) > 1 and best_group:
            relevant_tables = best_group.get("tables", [])
            relevant_fks = best_group.get("fks", [])
            # Use only the best group for the prompt
            table_groups = [best_group]
        else:
            # Fallback to all tables and FKs if no groups
            relevant_tables = [item["table"] for item in tables_with_entities]
            relevant_fks = [item["fk"] for item in fks_with_entities]

        # Format similar questions for prompt
        similar_questions_txt = "\n".join(
            f"question: {x[0]}\nanswer: {x[1]}" for x in similar_questions
        )
        self.logger.info(
            f"Using {len(similar_questions)} similar questions from conversations."
        )

        def build_messages(
            extracted_data: Optional[str] = None,
        ) -> list:
            """
            Build messages for SQL construction.

            Includes semantic candidate context, FKs, similar questions, and optionally
            extracted file data or file excerpts.
            """
            observation_block = f"\nlist of important semantic entities with sql snippets:\n{complex_candidates_str}\n"

            # Add table groups information to show connected tables and their entities
            if table_groups:
                table_groups_info = format_table_groups_for_prompt(table_groups)
                observation_block += f"\n\nTABLE GROUPS (tables connected by foreign keys):\n{table_groups_info}\n"

            # Add extracted data from files if available (data to use in SQL, not full snippets)
            if extracted_data:
                observation_block += f"\n\nEXTRACTED DATA FROM FILES (use this in SQL construction if needed):\n{extracted_data}\n"

           

            # Build user prompt with formatted tables
            user_prompt = create_sql_user_prompt.format(
                dialects=dialects,
                dialects_prompt=(
                    snowflake_dialect_prompt
                    if any(dialect.lower() == "snowflake" for dialect in dialects)
                    else ""
                ),
                main_question=question,
                observation_block=observation_block,
                fks=[
                    f"{item['table1']}.{item['column1']} = {item['table2']}.{item['column2']}"
                    for item in relevant_fks
                ],
                queries=relevant_queries,
                qa_from_conversations=similar_questions_txt,
                tables=format_tables_for_prompt(relevant_tables),
            )

            # Choose system prompt based on context
            
            system_prompt = create_sql_from_semantic_prompt(complex_candidates)
                    
            messages = state["messages"] + [
                SystemMessage(content=system_prompt),
                AIMessage(content=user_prompt),
            ]

            # Add calendar time window reminder if needed
            if any(
                phrase in question.lower()
                for phrase in ["last week", "last month", "last year"]
            ):
                messages.append(
                    SystemMessage(
                        content="Apply only calendar time windows. DO NOT apply rolling time windows."
                    )
                )

            return messages

        # Choose schema based on context
        # Use SQLGenerationModel for new flow (without formatting)
        # Keep old models for feedback scenarios

        schema = SQLGenerationModel

        def run_with_context(
            extracted_data: Optional[str] = None,
        ) -> tuple:
            """Invoke LLM with messages, optionally including file snippets and extracted data."""
            messages = build_messages(extracted_data)
            response = invoke_with_structured_output(llm, messages, schema)
            # Log response explanation
            # SQLGenerationModel uses 'response' field; feedback models may use 'thought'
            if hasattr(response, "response"):
                self.logger.info(
                    f"LLM response generated, explanation: {response.response[:100]}..."
                )
            elif hasattr(response, "thought"):
                self.logger.info(
                    f"LLM response generated, thought: {response.thought[:100]}..."
                )
            return response, messages

        # Get extracted data from path_state (set by extract_from_file_snippets node)
        extracted_data = path_state.get("extracted_file_data")

        # Call LLM for SQL construction (with extracted data from files if available) 
        response, messages = run_with_context(
            extracted_data=extracted_data
        )

        # Check if we have a valid response (either SQL or text-based answer from file contents)
        has_sql = bool(response.sql_code and response.sql_code.strip())
        has_response = bool(response.response and response.response.strip())

        if has_sql:
            # Extract semantic elements
            semantic_elements = []
            if hasattr(response, "semantic_elements"):
                # Filter semantic elements to keep only those found in candidates
                candidates_ids = {
                    c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
                    for c in candidates
                }
                filtered_elements = [
                    elem
                    for elem in response.semantic_elements
                    if (elem.id if hasattr(elem, "id") else elem.get("id"))
                    in candidates_ids
                ]
                response.semantic_elements = filtered_elements
                semantic_elements = get_semantic_entities_ids(
                    response.semantic_elements
                )

            return {
                "messages": messages,  # Don't add formatted response here - formatting agent will do it
                "path_state": {
                    **path_state,
                    "llm_calc_response": response,  # Keep as object (Pydantic model)
                    "relevant_tables": relevant_tables if has_sql else [],
                    "connection": response.connection,
                    "semantic_elements": semantic_elements,
                },
                "decision": "constructable",
            }
        elif has_response:
            # Fallback for feedback scenarios that still use old model with response field
            semantic_elements = []
            if hasattr(response, "semantic_elements"):
                response.response += build_semantic_items_section(
                    response.semantic_elements, candidates
                )
                semantic_elements = get_semantic_entities_ids(
                    response.semantic_elements
                )

            connection = (
                response.connection
                if (response.connection and response.connection.strip())
                else ""
            )

            return {
                "messages": messages + [AIMessage(content=response.response)],
                "path_state": {
                    **path_state,
                    "llm_calc_response": response,
                    "relevant_tables": relevant_tables if has_sql else [],
                    "connection": connection,
                    "semantic_elements": semantic_elements,
                },
                "decision": "constructable",
            }
        else:
            # SQL could not be generated
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": response.response
                    or "Unable to construct response.",
                },
                "decision": "unconstructable",
            }


__all__ = [
    "SQLFromSemanticAgent",
    "format_table_groups_for_prompt",
    "format_tables_for_prompt",
]
