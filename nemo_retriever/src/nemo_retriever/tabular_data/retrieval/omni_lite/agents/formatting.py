"""
SQL Response Formatting Agent

This agent formats SQL generation results into user-friendly markdown responses.
It takes structured SQL data (sql_code, tables_ids, semantic_elements, etc.) and
generates a formatted explanation with SQL code blocks and component breakdowns.

Responsibilities:
- Format SQL results into markdown
- Generate user-friendly explanations
- Include SQL code blocks with proper formatting
- Explain table usage and SQL components
"""

import logging
from typing import Dict, Any, Optional

from langchain_core.messages import AIMessage
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentState



logger = logging.getLogger(__name__)


class SQLResponseFormattingAgent(BaseAgent):
    """
    Agent that formats SQL generation results into user-friendly markdown.

    Takes structured SQL data from SQL generation agents and formats it into
    a clear, well-structured markdown response with explanations.

    Input Requirements:
    - path_state["llm_calc_response"]: SQLGenerationModel with sql_code, tables_ids, semantic_elements, connection, thought
    - path_state["normalized_question"] or state["initial_question"]: User's question
    - path_state["relevant_tables"]: Optional list of table objects with names/descriptions
    - path_state["extracted_file_data"]: Optional extracted data from files (for mentioning in breakdown)

    Output:
    - path_state["llm_calc_response"].response: Formatted markdown response
    - messages: Adds formatted response to messages
    """

    def __init__(self):
        super().__init__("sql_response_formatting")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that SQL response data is available."""
        path_state = state.get("path_state", {})
        llm_response = path_state.get("llm_calc_response")

        if not llm_response:
            self.logger.warning("No SQL response found for formatting")
            return False

        if not getattr(llm_response, "sql_code", None):
            self.logger.warning("SQL code is missing in response")
            return False

        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Format SQL generation results into user-friendly markdown.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Updated with formatted response
            - messages: Adds formatted response to messages
        """
        path_state = state.get("path_state", {})
        llm_response = path_state.get("llm_calc_response")

        # Get SQL data
        sql_code = getattr(llm_response, "sql_code", "")
        tables_ids = getattr(llm_response, "tables_ids", [])
        response_explanation = getattr(llm_response, "response", "")
        semantic_elements = getattr(llm_response, "semantic_elements", [])

        # Get additional context
        relevant_tables = path_state.get("relevant_tables", [])
        extracted_file_data = path_state.get("extracted_file_data")
        candidates_with_entities = path_state.get("candidates", [])

        # Extract just the candidate objects for processing
        candidates = [
            item["candidate"]
            if isinstance(item, dict) and "candidate" in item
            else item
            for item in candidates_with_entities
        ]

        # Format response using Python (no LLM call)
        formatted_response = self._format_sql_response(
            sql_code=sql_code,
            tables_ids=tables_ids,
            relevant_tables=relevant_tables,
            response_explanation=response_explanation,
            semantic_elements=semantic_elements,
            extracted_file_data=extracted_file_data,
            candidates=candidates,
        )

        # Format response (adds citations, formatting, etc.)
        formatted_response_with_citations = format_response(
            account_id=state["account_id"],
            candidates=candidates,
            response=formatted_response,
        )

        self.logger.info("SQL response formatted successfully")

        return {
            "messages": state["messages"]
            + [AIMessage(content=formatted_response_with_citations)],
            "path_state": {
                **path_state,
                "formatted_response": formatted_response_with_citations,  # Store formatted response in path_state
            },
        }

    def _format_sql_response(
        self,
        sql_code: str,
        tables_ids: list[str],
        relevant_tables: list,
        connection: str,
        response_explanation: str,
        semantic_elements: list,
        extracted_file_data: Optional[str] = None,
        candidates: list = None,
    ) -> str:
        """
        Format SQL response into user-friendly markdown using Python.

        Args:
            sql_code: Generated SQL code
            tables_ids: List of table IDs used
            table_names: List of table names (if available)
            connection: Database connection/dialect
            response_explanation: Short explanation from SQLGenerationModel.response
            semantic_elements: List of semantic elements used
            extracted_file_data: Optional extracted data from files

        Returns:
            Formatted markdown response
        """
        parts = []

        # Parse response_explanation - it should contain all the formatted content
        # The response_explanation from SQLGenerationModel should already be well-formatted
        # We just need to add SQL code block and semantic items

        # 1. Summary and explanation from response_explanation
        if response_explanation:
            parts.append(response_explanation.strip())
        else:
            parts.append(f"This query retrieves data using {connection}.")

        # 2. SQL Code Block
        parts.append("")
        parts.append("The SQL generated for your question is:")
        parts.append("%%%")
        parts.append(sql_code)
        parts.append("%%%")

        # 3. Main tables used (extracted from relevant_tables)
        table_info = self._extract_table_info(relevant_tables, tables_ids)
        if table_info:
            parts.append("")
            parts.append("**Main tables used**")
            for table in table_info:
                table_name = table.get("name", "")
                table_id = table.get("id", "")
                if table_id:
                    # Create link for table
                    link = prepare_link(table_name, table_id, Labels.TABLE)
                    parts.append(f"• *<{link}>*")
                else:
                    parts.append(f"• `{table_name}`")

        # 4. Semantic items used (with links)
        if semantic_elements and candidates:
            semantic_items_used = self._format_semantic_elements(
                semantic_elements, candidates
            )
            if semantic_items_used:
                parts.append("")
                parts.append("**Semantic items used**:")
                parts.extend(semantic_items_used)

        return "\n".join(parts)

    def _extract_table_info(
        self, relevant_tables: list, tables_ids: list[str]
    ) -> list[dict]:
        """Extract table names and IDs from relevant_tables that match tables_ids."""
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
        """Format semantic elements with links using candidates lookup, matching table formatting style.

        Filters out semantic elements that don't have a corresponding candidate (invalid/missing IDs).
        """
        if not semantic_elements or not candidates:
            return []

        # Create a map of candidate id -> candidate object
        candidates_by_id = {}
        for candidate in candidates:
            candidate_id = (
                candidate.get("id")
                if isinstance(candidate, dict)
                else getattr(candidate, "id", None)
            )
            if candidate_id:
                candidates_by_id[candidate_id] = candidate

        formatted_items = []
        for elem in semantic_elements:
            # Handle both Pydantic objects and dicts
            def _get_elem(obj, key, default=None):
                """Safe getter for both Pydantic-style objects and plain dicts."""
                if hasattr(obj, key):
                    return getattr(obj, key, default)
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return default

            elem_id = _get_elem(elem, "id")
            elem_label = _get_elem(elem, "label")
            elem_classification = _get_elem(elem, "classification", False)

            # Only include items that were actually used (classification=True)
            if not elem_classification or not elem_id:
                continue

            # Look up candidate to get name and label
            candidate = candidates_by_id.get(elem_id)
            if not candidate:
                # Candidate not found - skip this semantic element (remove from list)
                self.logger.warning(
                    f"Semantic element {elem_id} (label: {elem_label}) not found in candidates, removing from semantic items"
                )
                continue

            # Get name from candidate
            candidate_name = (
                candidate.get("name")
                if isinstance(candidate, dict)
                else getattr(candidate, "name", "")
            )
            # Get label from candidate (preferred) or use elem_label
            candidate_label = (
                candidate.get("label")
                if isinstance(candidate, dict)
                else getattr(candidate, "label", None)
            )
            label_to_use = candidate_label or elem_label

            if candidate_name and elem_id:
                # Use prepare_link like tables do - it handles semantic labels automatically
                link = prepare_link(candidate_name, elem_id, label_to_use)
                formatted_items.append(f"• *<{link}>*")  # Use bullet point like tables
            else:
                # Missing name but candidate exists - still skip to avoid broken links
                self.logger.warning(
                    f"Semantic element {elem_id} found in candidates but missing name, removing from semantic items"
                )
                continue

        return formatted_items
