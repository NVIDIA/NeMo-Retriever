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
from nemo_retriever.tabular_data.retrieval.utils import (
    format_response,
    # TODO: re-enable these imports when clickable table/analysis link
    # highlighting is needed in the future.
    # Labels,
    # get_custom_analyses_ids,
    # prepare_link,
)

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

        # TODO: re-enable "Main tables used" section once we can accurately
        # determine which tables were actually referenced in the SQL query.
        # Previously this relied on LLM-returned table UUIDs (tables_ids),
        # which were removed to prevent the LLM from confusing them with
        # real table names.

        # TODO: re-enable clickable table/analysis links when the UI highlight
        # feature is needed. The helpers _extract_table_info,
        # _format_custom_analyses_used, and prepare_link are preserved below
        # for future use.
        #
        # table_info = self._extract_table_info(relevant_tables, tables_ids)
        # if table_info:
        #     parts.append("")
        #     parts.append("**Main tables used**")
        #     for table in table_info:
        #         table_name = table.get("name", "")
        #         table_id = table.get("id", "")
        #         if table_id:
        #             link = prepare_link(table_name, table_id, Labels.TABLE)
        #             parts.append(f"• *<{link}>*")
        #         else:
        #             parts.append(f"• `{table_name}`")
        #
        # if custom_analyses_used and candidates:
        #     formatted_analyses = self._format_custom_analyses_used(
        #         custom_analyses_used, candidates
        #     )
        #     if formatted_analyses:
        #         parts.append("")
        #         parts.append("**Custom analyses used**:")
        #         parts.extend(formatted_analyses)

        return "\n".join(parts)

    # --- commented-out helpers for future clickable-link support ---
    #
    # @staticmethod
    # def _extract_table_info(relevant_tables, tables_ids):
    #     table_info = []
    #     if relevant_tables:
    #         for table in relevant_tables:
    #             table_name = table.get("name") or table.get("table_name") or ""
    #             table_id = table.get("id") or ""
    #             if table_name and table_id and table_id in tables_ids:
    #                 table_info.append({"name": table_name, "id": table_id})
    #     return table_info
    #
    # def _format_custom_analyses_used(self, custom_analyses_used, candidates):
    #     if not custom_analyses_used or not candidates:
    #         return []
    #     candidates_by_id = {}
    #     for candidate in candidates:
    #         cid = candidate.get("id") if isinstance(candidate, dict) else getattr(candidate, "id", None)
    #         if cid:
    #             candidates_by_id[cid] = candidate
    #     def _get(obj, key, default=None):
    #         if hasattr(obj, key):
    #             return getattr(obj, key, default)
    #         if isinstance(obj, dict):
    #             return obj.get(key, default)
    #         return default
    #     formatted_items = []
    #     for elem in custom_analyses_used:
    #         elem_id = _get(elem, "id")
    #         elem_label = _get(elem, "label")
    #         elem_classification = _get(elem, "classification", False)
    #         if not elem_classification or not elem_id:
    #             continue
    #         candidate = candidates_by_id.get(elem_id)
    #         if not candidate:
    #             continue
    #         candidate_name = candidate.get("name") if isinstance(candidate, dict) else getattr(candidate, "name", "")
    #         candidate_label = candidate.get("label") if isinstance(candidate, dict) else getattr(candidate, "label", None)
    #         label_to_use = candidate_label or elem_label
    #         if candidate_name and elem_id:
    #             link = prepare_link(candidate_name, elem_id, label_to_use)
    #             formatted_items.append(f"• *<{link}>*")
    #     return formatted_items
