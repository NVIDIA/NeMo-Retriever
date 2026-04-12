"""
Calculation Response Agent

Formats SQL generation results into user-friendly markdown, assembles the
final response dict (with DB result, sql_code, semantic elements, etc.),
and stores it in ``path_state["final_response"]``.

Combines the responsibilities of the former SQLResponseFormattingAgent and
CalculationResponseAgent into a single graph node.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels,
    format_response,
    get_semantic_entities_ids,
    prepare_link,
)


logger = logging.getLogger(__name__)


class CalculationResponseAgent(BaseAgent):
    """
    Final-step agent: format SQL results into markdown, attach DB output,
    and set ``path_state["final_response"]``.

    Input Requirements:
    - path_state["llm_calc_response"]: SQLGenerationModel
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
        llm_response = path_state.get("llm_calc_response")
        if not llm_response:
            self.logger.warning("No LLM response found for calculation response")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        llm_response = path_state.get("llm_calc_response")

        sql_code = getattr(llm_response, "sql_code", "")
        tables_ids = getattr(llm_response, "tables_ids", [])
        response_explanation = getattr(llm_response, "response", "")
        semantic_elements = getattr(llm_response, "semantic_elements", [])

        relevant_tables = path_state.get("relevant_tables", [])
        candidates_with_entities = path_state.get("candidates", [])

        candidates = [
            item["candidate"]
            if isinstance(item, dict) and "candidate" in item
            else item
            for item in candidates_with_entities
        ]

        # --- formatting (formerly SQLResponseFormattingAgent) ---
        formatted_response = self._format_sql_response(
            sql_code=sql_code,
            tables_ids=tables_ids,
            relevant_tables=relevant_tables,
            response_explanation=response_explanation,
            semantic_elements=semantic_elements,
            candidates=candidates,
        )
        formatted_response = format_response(
            candidates=candidates,
            response=formatted_response,
        )

        # --- final dict assembly (formerly CalculationResponseAgent) ---
        sql_columns = path_state.get("sql_columns", [])
        sem_ids = []
        if hasattr(llm_response, "semantic_elements"):
            sem_ids = get_semantic_entities_ids(llm_response.semantic_elements)

        response = {
            "response": formatted_response,
            "sql_code": sql_code,
            "sql_columns": sql_columns,
            "semantic_elements": sem_ids,
            "sql_response_from_db": path_state.get("sql_response_from_db"),
        }

        self.logger.info("Calculation response prepared and returned")

        return {
            "messages": state["messages"]
            + [AIMessage(content=formatted_response)],
            "path_state": {
                **path_state,
                "formatted_response": formatted_response,
                "final_response": response,
            },
        }

    # ---- formatting helpers (moved from SQLResponseFormattingAgent) ----

    def _format_sql_response(
        self,
        sql_code: str,
        tables_ids: list[str],
        relevant_tables: list,
        response_explanation: str,
        semantic_elements: list,
        candidates: list = None,
    ) -> str:
        parts = []

        if response_explanation:
            parts.append(response_explanation.strip())

        parts.append("")
        parts.append("The SQL generated for your question is:")
        parts.append("%%%")
        parts.append(sql_code)
        parts.append("%%%")

        table_info = self._extract_table_info(relevant_tables, tables_ids)
        if table_info:
            parts.append("")
            parts.append("**Main tables used**")
            for table in table_info:
                table_name = table.get("name", "")
                table_id = table.get("id", "")
                if table_id:
                    link = prepare_link(table_name, table_id, Labels.TABLE)
                    parts.append(f"• *<{link}>*")
                else:
                    parts.append(f"• `{table_name}`")

        if semantic_elements and candidates:
            semantic_items_used = self._format_semantic_elements(
                semantic_elements, candidates
            )
            if semantic_items_used:
                parts.append("")
                parts.append("**Semantic items used**:")
                parts.extend(semantic_items_used)

        return "\n".join(parts)

    @staticmethod
    def _extract_table_info(
        relevant_tables: list, tables_ids: list[str]
    ) -> list[dict]:
        table_info = []
        if relevant_tables:
            for table in relevant_tables:
                table_name = table.get("name") or table.get("table_name") or ""
                table_id = table.get("id") or ""
                if table_name and table_id and table_id in tables_ids:
                    table_info.append({"name": table_name, "id": table_id})
        return table_info

    def _format_semantic_elements(
        self, semantic_elements: list, candidates: list
    ) -> list[str]:
        if not semantic_elements or not candidates:
            return []

        candidates_by_id = {}
        for candidate in candidates:
            candidate_id = (
                candidate.get("id")
                if isinstance(candidate, dict)
                else getattr(candidate, "id", None)
            )
            if candidate_id:
                candidates_by_id[candidate_id] = candidate

        def _get(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key, default)
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default

        formatted_items = []
        for elem in semantic_elements:
            elem_id = _get(elem, "id")
            elem_label = _get(elem, "label")
            elem_classification = _get(elem, "classification", False)

            if not elem_classification or not elem_id:
                continue

            candidate = candidates_by_id.get(elem_id)
            if not candidate:
                self.logger.warning(
                    f"Semantic element {elem_id} (label: {elem_label}) not found in candidates, removing"
                )
                continue

            candidate_name = (
                candidate.get("name")
                if isinstance(candidate, dict)
                else getattr(candidate, "name", "")
            )
            candidate_label = (
                candidate.get("label")
                if isinstance(candidate, dict)
                else getattr(candidate, "label", None)
            )
            label_to_use = candidate_label or elem_label

            if candidate_name and elem_id:
                link = prepare_link(candidate_name, elem_id, label_to_use)
                formatted_items.append(f"• *<{link}>*")
            else:
                self.logger.warning(
                    f"Semantic element {elem_id} found in candidates but missing name, removing"
                )

        return formatted_items
