"""
Entity extraction for omni-lite retrieval.

This simplified agent is calculation-only. It stores:
- normalized_question
- extracted entities/concepts from the question
"""

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.ai_services import invoke_with_structured_output

logger = logging.getLogger(__name__)


class EntitiesExtractionModel(BaseModel):
    """
    Model for extracting entities/concepts and query without values.
    """

    required_entity_name: list[str] = Field(
        ...,
        description="List of primary entities or concepts mentioned in the question. "
        "Ignore time frames, quantities, or constants. "
        "Exclude system-level labels like 'term', 'metric', etc.",
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
            normalized_question = (question or "").strip()
            extraction_prompt = f"""
You are extracting entities and concepts from a user question for SQL calculation.

User Question:
{normalized_question}

Extract:
1) required_entity_name: list of entities/concepts mentioned in the question.
- extract ALL entities that most likely refer to a specific entity in the database.
   - Ignore time frames, quantities, or constants.
   - Examples: ["Customer", "Order"], ["Product", "Price"]

2) query_no_values: same question with specific values stripped.
- Remove dates, numbers, names, specific identifiers
   - Keep the structure and intent
   - Example: "What is the average order value in 2023?" → "What is the average order value?"

"""
            extraction_messages = base_messages + [SystemMessage(content=extraction_prompt)]
            extraction_result = invoke_with_structured_output(
                llm, extraction_messages, EntitiesExtractionModel
            )
            entities_and_concepts = extraction_result.required_entity_name or []

            path_state["normalized_question"] = normalized_question
            path_state["query_no_values"] = extraction_result.query_no_values
            path_state["entities_and_concepts"] = entities_and_concepts

            self.logger.info(
                f"Extracted {len(entities_and_concepts)} entities/concepts from normalized question"
            )
            return {"path_state": path_state}

        except Exception as e:
            self.logger.warning(
                f"Entity extraction failed: {e}, using fallback values"
            )
            normalized_question = (question or "").strip()
            path_state["normalized_question"] = normalized_question
            path_state["query_no_values"] = normalized_question
            path_state["entities_and_concepts"] = []

            return {"path_state": path_state}
