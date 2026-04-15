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
from nemo_retriever.tabular_data.retrieval.omni_lite.state import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.ai_services import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.omni_lite.prompts import create_entity_extraction_prompt

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

        try:
            extraction_messages = base_messages + [
                SystemMessage(content=create_entity_extraction_prompt(question))
            ]
            extraction_result = invoke_with_structured_output(
                llm, extraction_messages, EntitiesExtractionModel
            )
            entities_and_concepts = extraction_result.required_entity_name or []

            path_state["query_no_values"] = extraction_result.query_no_values
            path_state["entities_and_concepts"] = entities_and_concepts

            self.logger.info(
                "Extracted %s entities/concepts from normalized question: %s",
                len(entities_and_concepts),
                entities_and_concepts,
            )
            return {"path_state": path_state}

        except Exception as e:
            self.logger.warning(
                f"Entity extraction failed: {e}, using fallback values"
            )
            path_state["query_no_values"] = question
            path_state["entities_and_concepts"] = []

            return {"path_state": path_state}
