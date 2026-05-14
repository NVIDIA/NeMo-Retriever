# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Calculation Response Agent

Formats SQL generation results into user-friendly markdown, assembles the
final response dict (with DB result, sql_code, custom analyses, etc.),
and stores it in ``path_state["final_response"]``.

Combines the responsibilities of the former SQLResponseFormattingAgent and
ResponseAgent into a single graph node.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentState
from nemo_retriever.tabular_data.retrieval.data_access.response_formatting import format_response

logger = logging.getLogger(__name__)


class ResponseAgent(BaseAgent):
    """
    Final-step agent: format SQL results into markdown, attach DB output,
    and set ``path_state["final_response"]``.

    Input Requirements:
    - path_state["sql_generation_result"]: SQLGenerationModel
    - path_state["sql_response_from_db"]: DB execution result (optional)
    - path_state["relevant_tables"]: table dicts
    - path_state["candidates"]: semantic candidates

    Output:
    - path_state["final_response"]: complete response dict
    - messages: appended AIMessage with formatted text
    """

    def __init__(self):
        super().__init__("calculation_response")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        llm_response = path_state.get("sql_generation_result")
        if not llm_response:
            self.logger.warning("No LLM response found for calculation response")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        llm_response = path_state.get("sql_generation_result")

        sql_code = getattr(llm_response, "sql_code", "")
        response_explanation = getattr(llm_response, "response", "")

        relevant_tables = path_state.get("relevant_tables", [])
        candidates_with_entities = path_state.get("candidates", [])
        candidates = [
            item["candidate"] if isinstance(item, dict) and "candidate" in item else item
            for item in candidates_with_entities
        ]

        # --- formatting  ---
        formatted_response = self._format_sql_response(
            sql_code=sql_code,
            relevant_tables=relevant_tables,
            response_explanation=response_explanation,
        )
        formatted_response = format_response(
            candidates=candidates,
            response=formatted_response,
        )

        # --- final dict assembly ---
        sql_columns = path_state.get("sql_columns", [])

        response = {
            "response": formatted_response,
            "sql_code": sql_code,
            "sql_columns": sql_columns,
            "custom_analyses_used": [],
            "sql_response_from_db": path_state.get("sql_response_from_db"),
        }

        self.logger.info("Calculation response prepared and returned")

        return {
            "messages": state["messages"] + [AIMessage(content=formatted_response)],
            "path_state": {
                **path_state,
                "formatted_response": formatted_response,
                "final_response": response,
            },
        }

    # ---- formatting helpers ----

    def _format_sql_response(
        self,
        sql_code: str,
        relevant_tables: list,
        response_explanation: str,
    ) -> str:
        parts = []

        if response_explanation:
            parts.append(response_explanation.strip())

        parts.append("")
        parts.append("The SQL generated for your question is:")
        parts.append("%%%")
        parts.append(sql_code)
        parts.append("%%%")
        return "\n".join(parts)
