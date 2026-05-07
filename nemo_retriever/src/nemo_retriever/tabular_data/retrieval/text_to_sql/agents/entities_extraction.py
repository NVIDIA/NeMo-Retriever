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
        min_length=1,
        description="List of primary entities or concepts mentioned in the question. "
        "Ignore time frames, quantities, or constants. Must contain at least one entity.",
    )
    query_no_values: str = Field(
        ...,
        description="The user's query with all specific values stripped out (dates, numbers, names, etc.).",
    )
    item_search_queries: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Generate 2-4 short search phrases to help find relevant database items "
            "(tables, columns, etc.). "
            "Rephrase the question from different angles. Use the domain rules "
            "(if provided) to identify which database concepts and item names "
            "might be relevant. Each phrase should target a different aspect "
            "of the question."
        ),
    )


class RuleFilterModel(BaseModel):
    """LLM output for filtering domain rules by relevance to a question."""

    relevant_rule_names: list[str] = Field(
        default_factory=list,
        description="Names of domain rules that are relevant to the question.",
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
        """Extract entities, then filter custom prompt rules in a separate LLM call."""
        llm = state["llm"]
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        custom_prompts_rules = state.get("custom_prompts_rules", [])
        custom_prompts_text = rules_to_text(custom_prompts_rules)

        # --- Step 1: Entity extraction (dedicated message list, no base_messages) ---
        try:
            extraction_messages = [
                SystemMessage(content=create_entity_extraction_prompt(question, custom_prompts_text))
            ]
            extraction_result = invoke_with_structured_output(llm, extraction_messages, EntitiesExtractionModel)
            self.logger.debug("Raw extraction result: %s", extraction_result)

            if extraction_result is None:
                self.logger.warning("Entity extraction returned None, using fallback")
                path_state["query_no_values"] = question
                path_state["entities"] = []
                return {"path_state": path_state}

            entities = extraction_result.required_entity_name or []
            item_search_queries = extraction_result.item_search_queries or []

            combined = entities + item_search_queries

            if not combined:
                self.logger.warning("LLM returned empty entities and search queries — using question as fallback entity")
                combined = [question]

            path_state["query_no_values"] = extraction_result.query_no_values or question
            path_state["entities"] = combined

            self.logger.info(
                "Extracted %s entities + %s search queries = %s total: %s",
                len(entities),
                len(item_search_queries),
                len(combined),
                combined,
            )
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}, using fallback values")
            path_state["query_no_values"] = question
            path_state["entities"] = []
            return {"path_state": path_state}

        result: Dict[str, Any] = {"path_state": path_state}

        # --- Step 2: Filter custom prompt rules (separate LLM call) ---
        if custom_prompts_rules:
            filtered = self._filter_custom_prompts(
                llm, question, combined, custom_prompts_rules, custom_prompts_text,
            )
            if filtered is not None:
                result["custom_prompts_rules"] = filtered

        return result

    def _filter_custom_prompts(
        self,
        llm,
        question: str,
        entities: list[str],
        rules: list[dict[str, str]],
        rules_text: str,
    ) -> list[dict[str, str]] | None:
        """Separate LLM call to pick which domain rules apply to this question."""
        available_names = [r["name"] for r in rules]
        prompt = (
            "Given the user's question and extracted entities, decide which "
            "domain rules are relevant.\n\n"
            f"Question: {question}\n"
            f"Extracted entities: {entities}\n\n"
            f"Available domain rules:\n{rules_text}\n"
            "Return ONLY the names of rules that are needed to answer this question correctly. "
            f"Choose from these exact names: {available_names}"
        )

        try:
            filter_result = invoke_with_structured_output(
                llm,
                [SystemMessage(content=prompt)],
                RuleFilterModel,
            )
        except Exception as e:
            self.logger.warning("Rule filter LLM call failed: %s — keeping all rules", e)
            return None

        if filter_result is None:
            self.logger.warning("Rule filter returned None — keeping all rules")
            return None

        relevant_names = filter_result.relevant_rule_names or []
        self.logger.info("Rule filter returned: %s", relevant_names)

        if not relevant_names:
            self.logger.info("No relevant rules identified — keeping all rules")
            return None

        names_lower = {n.lower().strip() for n in relevant_names}
        filtered = [r for r in rules if r["name"].lower().strip() in names_lower]

        if not filtered:
            self.logger.warning(
                "Rule filter names %s did not match available rules %s — keeping all rules",
                relevant_names,
                available_names,
            )
            return None

        self.logger.info(
            "Filtered custom_prompts_rules: kept %d/%d rules: %s",
            len(filtered),
            len(rules),
            [r["name"] for r in filtered],
        )
        return filtered
