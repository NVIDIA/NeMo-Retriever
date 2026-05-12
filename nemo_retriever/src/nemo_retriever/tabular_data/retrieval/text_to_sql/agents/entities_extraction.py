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
    """Extract entities from a question (no domain rules)."""

    required_entity_name: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Concepts explicitly mentioned in the question that refer to "
            "database entities. Only extract what the question actually says. "
            "Ignore values, dates, numbers, and constants."
        ),
    )


class EntitiesWithRulesModel(BaseModel):
    """Extract entities and rule-based search phrases from a question."""

    required_entity_name: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Concepts explicitly mentioned in the question that refer to "
            "database entities. Only extract what the question actually says. "
            "Ignore values, dates, numbers, and constants."
        ),
    )
    item_search_queries: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Generate 2-4 short search phrases  (1-3 words each) to help find relevant database items "
            "(tables, columns, etc.). "
            "Use the domain rules "
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
        """Filter domain rules first, then extract entities guided only by kept rules."""
        llm = state["llm"]
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        custom_prompts_rules = state.get("custom_prompts_rules", [])
        custom_prompts_text = rules_to_text(custom_prompts_rules)

        result: Dict[str, Any] = {"path_state": path_state}

        # --- Step 1: Filter domain rules based on question alone ---
        kept_rules_text = ""
        if custom_prompts_rules:
            filtered = self._filter_custom_prompts(
                llm, question, custom_prompts_rules, custom_prompts_text,
            )
            if filtered:
                result["custom_prompts_rules"] = filtered
                kept_rules_text = rules_to_text(filtered)
            else:
                result["custom_prompts_rules"] = []

        # --- Step 2: Extract entities (schema depends on whether rules exist) ---
        model_cls = EntitiesWithRulesModel if kept_rules_text else EntitiesExtractionModel
        try:
            extraction_messages = [
                SystemMessage(content=create_entity_extraction_prompt(question, kept_rules_text))
            ]
            extraction_result = invoke_with_structured_output(llm, extraction_messages, model_cls)
            self.logger.debug("Raw extraction result: %s", extraction_result)

            if extraction_result is None:
                self.logger.warning("Entity extraction returned None, using fallback")
                path_state["entities"] = []
                return result

            entities = extraction_result.required_entity_name or []
            item_search_queries = getattr(extraction_result, "item_search_queries", None) or []

            combined = entities + item_search_queries

            if not combined:
                self.logger.warning("LLM returned empty entities — using question as fallback")
                combined = [question]

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
            path_state["entities"] = []

        return result

    def _filter_custom_prompts(
        self,
        llm,
        question: str,
        rules: list[dict[str, str]],
        rules_text: str,
    ) -> list[dict[str, str]] | None:
        """Pick which domain rules apply to this question (before entity extraction)."""
        available_names = [r["name"] for r in rules]
        prompt = (
            "Which domain rules are relevant to answering this question?\n\n"
            f"Question: {question}\n\n"
            f"Available domain rules:\n{rules_text}\n"
            "Return ONLY the names of rules that are needed. "
            "If none are relevant, return an empty list.\n"
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
            self.logger.info("No relevant rules identified — no domain context for entity extraction")
            return []

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
