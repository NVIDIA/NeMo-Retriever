# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.llm_invoke import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import create_entity_extraction_prompt

logger = logging.getLogger(__name__)


class EntitiesExtractionModel(BaseModel):
    """Extract entities from a question."""

    required_entity_name: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Concepts explicitly mentioned in the question that refer to "
            "database entities. Only extract what the question actually says. "
            "Ignore values, dates, numbers, and constants."
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
        """Extract entities from the question only (no domain rules)."""
        llm = state["llm"]
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)

        result: Dict[str, Any] = {"path_state": path_state}

        try:
            extraction_messages = [SystemMessage(content=create_entity_extraction_prompt(question))]
            extraction_result = invoke_with_structured_output(
                llm,
                extraction_messages,
                EntitiesExtractionModel,
            )
            self.logger.debug("Raw extraction result: %s", extraction_result)

            if extraction_result is None:
                self.logger.warning("Entity extraction returned None, using fallback")
                path_state["entities"] = []
                return result

            entities = extraction_result.required_entity_name or []

            if not entities:
                self.logger.warning("LLM returned empty entities — using question as fallback")
                entities = [question]

            path_state["entities"] = entities

            self.logger.info("Extracted %d entities: %s", len(entities), entities)
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}, using fallback values")
            path_state["entities"] = []

        return result
