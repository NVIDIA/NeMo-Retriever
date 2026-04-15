"""
SQL Generation from Tables Agent

This agent generates SQL queries from table schemas when no snippets are available.
Used as a fallback when attribute snippets don't exist or aren't sufficient.

Responsibilities:
- Generate SQL from table schemas and relationships
- Find similar questions from conversation history
- Handle cases where snippets are not available
- Store SQL response in path_state

Design Decisions:
- Used when no suitable snippets are found
- Relies on table schemas and foreign key relationships
- Can incorporate similar questions from history for context
"""

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage, SystemMessage
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.sql_from_semantic import format_tables_for_prompt
from nemo_retriever.tabular_data.retrieval.text_to_sql.llm_invoke import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.models import SQLGenerationModel
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentState, get_question_for_processing
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import create_sql_general_prompt, create_sql_user_prompt
from nemo_retriever.tabular_data.retrieval.text_to_sql.utils import get_relevant_tables



logger = logging.getLogger(__name__)
    

class SQLFromTablesAgent(BaseAgent):
    """
    Agent that generates SQL from table schemas.

    This agent is used when no suitable attribute snippets are available.
    It builds SQL from table schemas, foreign key relationships, and similar questions.

    Input Requirements:
    - path_state["relevant_tables"]: Optional relevant tables (if not provided, will search)
    - path_state["error"]: Optional error from previous attempt (for reconstruction)
    - state["initial_question"]: User's question
    - state["dialect"]: SQL dialect

    Output:
    - path_state["sql_generation_result"]: SQL response with SQL code
    - path_state["relevant_tables"]: Relevant tables used
    - decision: "constructable" or "unconstructable"
    """

    def __init__(self):
        super().__init__("sql_from_tables")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate SQL from table schemas.

        Uses table schemas, foreign keys, and similar questions to generate SQL
        when no attribute snippets are available.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains SQL response, tables, connection
            - messages: Adds SQL response to messages
            - decision: "constructable" or "unconstructable"
        """
        path_state = state.get("path_state", {})
        llm = state["llm"]
        dialect = state["dialect"]
        question = get_question_for_processing(state)

        system_prompt = create_sql_general_prompt

        # Get relevant tables (search if not already available)
        relevant_tables = path_state.get("relevant_tables", [])
        if not relevant_tables:
            relevant_tables, _ = get_relevant_tables(
                question
            )

        # Find similar questions from conversation history
        similar_questions = []

        # Build user prompt with formatted tables
        user_prompt = create_sql_user_prompt.format(
            dialect=dialect,
            main_question=question,
            observation_block="",
            fks=[],  # Foreign keys can be added if needed
            queries=[],  # Relevant queries can be added if needed
            tables=format_tables_for_prompt(relevant_tables),
            qa_from_conversations=similar_questions,
        )

        messages = state["messages"] + [
            SystemMessage(content=system_prompt),
            AIMessage(content=user_prompt),
        ]

        response = invoke_with_structured_output(llm, messages, SQLGenerationModel)

        self.logger.info(
            "SQL generated from tables: %s...",
            response.sql_code[:100] if response and response.sql_code else "None",
        )

        if response and response.sql_code:
            return {
                "messages": messages + [AIMessage(content=response.response)],
                "path_state": {
                    **path_state,
                    "sql_generation_result": response,
                    "relevant_tables": relevant_tables,
                },
                "decision": "constructable",
            }
        else:
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": getattr(response, "response", "LLM failed to produce SQL."),
                },
                "decision": "unconstructable",
            }
