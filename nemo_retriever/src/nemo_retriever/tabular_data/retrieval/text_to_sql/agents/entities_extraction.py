"""
Entity extraction for omni-lite retrieval.
It stores:
- normalized_question
- extracted entities/concepts from the question
"""

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
    rules_to_text,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.llm_invoke import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import create_entity_extraction_prompt

logger = logging.getLogger(__name__)


class EntitiesExtractionModel(BaseModel):
    """
    Model for extracting entities/concepts and query without values.
    """

    required_entity_name: list[str] = Field(
        ...,
        description="List of primary entities or concepts mentioned in the question. "
        "Ignore time frames, quantities, or constants. ",
    )
    query_no_values: str = Field(
        ...,
        description="The user's query with all specific values stripped out (dates, numbers, names, etc.).",
    )
    item_search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Generate 2-4 short search phrases to help find relevant database items "
            "(tables, columns, etc.). "
            "Rephrase the question from different angles. Use the domain rules "
            "(if provided) to identify which database concepts and item names "
            "might be relevant. Each phrase should target a different aspect "
            "of the question."
        ),
    )
    relevant_rule_names: list[str] = Field(
        default_factory=list,
        description=(
            "From the domain rules provided, return the names of rules that are "
            "relevant to the user's question. Only include rules whose guidance "
            "is needed to construct the correct SQL. Return an empty list if no "
            "domain rules are provided or none are relevant."
        ),
    )


class EntitiesExtractionAgent(BaseAgent):
    """Extract normalized question and entity/concept terms (calculation-only)."""

    def __init__(self):
        super().__init__("entities_extraction")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that a question is available."""
        question = get_question_for_processing(state)
        if not question:
            self.logger.warning("No question found, skipping entity extraction")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Extract normalized question + entities/concepts, and force calculation decision."""
        llm = state["llm"]
        base_messages = state["messages"]
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        custom_prompts_rules = state.get("custom_prompts_rules", [])
        custom_prompts_text = rules_to_text(custom_prompts_rules)

        try:
            extraction_messages = base_messages + [
                SystemMessage(content=create_entity_extraction_prompt(question, custom_prompts_text))
            ]
            extraction_result = invoke_with_structured_output(llm, extraction_messages, EntitiesExtractionModel)
            entities = extraction_result.required_entity_name or []
            item_search_queries = extraction_result.item_search_queries or []
            relevant_rule_names = extraction_result.relevant_rule_names or []

            combined = entities + item_search_queries

            path_state["query_no_values"] = extraction_result.query_no_values
            path_state["entities"] = combined

            self.logger.info(
                "Extracted %s entities + %s search queries = %s total: %s",
                len(entities),
                len(item_search_queries),
                len(combined),
                combined,
            )

            result: Dict[str, Any] = {"path_state": path_state}

            if custom_prompts_rules and relevant_rule_names:
                names_lower = {n.lower() for n in relevant_rule_names}
                filtered_rules = [
                    r for r in custom_prompts_rules
                    if r.get("name", "").lower() in names_lower
                ]
                result["custom_prompts_rules"] = filtered_rules
                self.logger.info(
                    "Filtered custom_prompts_rules: kept %s/%s rules: %s",
                    len(filtered_rules),
                    len(custom_prompts_rules),
                    [r["name"] for r in filtered_rules],
                )
            else:
                self.logger.info("No rule filtering applied (no rules or no relevant names returned)")

            return result

        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}, using fallback values")
            path_state["query_no_values"] = question
            path_state["entities"] = []

            return {"path_state": path_state}
