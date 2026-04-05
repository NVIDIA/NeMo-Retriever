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
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.prompts import create_sql_general_prompt
from nemo_retriever.tabular_data.retrieval.omni_lite.state_helpers import get_question_for_processing



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
    - state["dialects"]: SQL dialects to support

    Output:
    - path_state["llm_calc_response"]: SQL response with SQL code
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
        dialects = state["dialects"]
        question = get_question_for_processing(state)
        account_id = state["account_id"]
        user_participants = state["user_participants"]
        user_id = user_participants[0] if user_participants else None

        system_prompt = create_sql_general_prompt

        # Get relevant tables (search if not already available)
        relevant_tables = path_state.get("relevant_tables", [])
        if not relevant_tables:
            relevant_tables, _ = get_relevant_tables(
                account_id, user_participants, question
            )

        # Find similar questions from conversation history
        similar_questions = []
        embeddings_client = get_embeddings(account_id, is_embeddings=True)
        if embeddings_client and user_id:
            similar_questions = find_similar_questions(
                embeddings_client.embed_query(question), user_id
            )

        # Build user prompt with formatted tables
        user_prompt = create_sql_user_prompt.format(
            dialects=dialects,
            dialects_prompt=(
                snowflake_dialect_prompt
                if any(dialect.lower() == "snowflake" for dialect in dialects)
                else ""
            ),
            main_question=question,
            observation_block="",
            fks=[],  # Foreign keys can be added if needed
            queries=[],  # Relevant queries can be added if needed
            tables=format_tables_for_prompt(relevant_tables),
            qa_from_conversations=similar_questions,
        )

        # Add error context if this is a reconstruction attempt
        # Note: Error handling for reconstruction is in SQLReconstructionAgent
        # This agent is for initial generation from tables, not reconstruction

        # Build messages and invoke LLM
        messages = state["messages"] + [
            SystemMessage(content=system_prompt),
            AIMessage(content=user_prompt),
        ]

        response = invoke_with_structured_output(llm, messages, CalcFinalResponseModel)

        self.logger.info(
            f"SQL generated from tables: {response.sql_code[:100] if response.sql_code else 'None'}..."
        )
        self.logger.info(f"Response thought: {response.thought}")

        if response.sql_code:
            # SQL was successfully generated
            return {
                "messages": messages + [AIMessage(content=response.response)],
                "path_state": {
                    **path_state,
                    "llm_calc_response": response,
                    "relevant_tables": relevant_tables,
                    "connection": response.connection,
                },
                "decision": "constructable",
            }
        else:
            # SQL could not be generated
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": response.response,
                },
                "decision": "unconstructable",
            }
