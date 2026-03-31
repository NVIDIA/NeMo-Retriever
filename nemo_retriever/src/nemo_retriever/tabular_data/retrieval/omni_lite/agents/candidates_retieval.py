"""
Candidate Retrieval Agent

This agent performs semantic search to retrieve relevant candidates from the graph.
It's a foundational agent that provides evidence for routing decisions.

Responsibilities:
- Perform semantic search for graph entities (custom analyses)
- Clean and expand candidate properties
- Build summary for routing agent

Design Decisions:
- Generic retrieval 
- Results by CalculationAgent
- Categorization helps routing agent understand what's available
"""

import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.retrieval.omni_lite.graph import AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from search.api.omni.agent.agents.shared.helpers import (
    get_semantic_candidates_information,
    clean_results,
    update_candidate_properties,
    categorize_attribute_candidates,
    fetch_attribute_file_snippets,
)
from search.api.omni.dal import expand_info
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels as OmniLiteLabels,
    extract_candidates,
)

logger = logging.getLogger(__name__)

def get_question_for_processing(state: Dict[str, Any]) -> str:
    """
    Get the appropriate question for SQL processing.

    Uses normalized_question if available in path_state (for calculation_sql flow),
    otherwise falls back to initial_question.

    Args:
        state: Agent state containing path_state and initial_question

    Returns:
        The question to use for processing (normalized or original)
    """
    path_state = state.get("path_state", {})
    normalized_question = path_state.get("normalized_question")

    if normalized_question:
        return normalized_question

    initial_question = state.get("initial_question", "")
    return initial_question



class CandidateRetrievalAgent(BaseAgent):
    """
    Agent that retrieves candidates from semantic search.

    This agent performs the initial retrieval that informs routing decisions.
    It searches for relevant graph entities and categorizes them.

    Retrieval Strategy:
    - Semantic search over all entity types (terms, metrics, attributes, analyses)
    - Top 10 candidates (balance between coverage and latency)
    - Clean and expand candidate properties
    - Categorize into graph_supported vs file_attributes

    Input Requirements:
    - path_state["normalized_question"]: Normalized English question (from LanguageDetectionAgent)
    - state["account_id"]: Account ID for data access
    - state["user_participants"]: User and group IDs for access control

    Output:
    - path_state["retrieved_candidates"]: List of all candidates (for reuse by downstream agents)
    - path_state["graph_supported_candidates"]: Candidates that can answer from graph
    - path_state["file_attribute_candidates"]: Candidates that need file lookup
    - path_state["candidate_summary"]: Human-readable summary for routing agent
    """

    def __init__(self):
        super().__init__("candidate_retrieval")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that question is available."""
        question = get_question_for_processing(state)
        if not question:
            self.logger.warning("No question available for retrieval")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve candidates from semantic search.

        Performs semantic search, cleans results, expands properties, and
        categorizes candidates for routing decisions.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains retrieved candidates and categorization
        """
        path_state = state.get("path_state", {})

        question = get_question_for_processing(state)

        try:
            # Run semantic search for graph entities
            # k=10 balances coverage with latency
            # Only search semantic entities (terms, attributes, metrics, analyses) for routing decisions
            # Columns/tables are data elements, not semantic entities needed for routing
            entities = state.get("entities_and_concepts", []) 
            query_no_values = path_state.get(
                "query_no_values", ""
            )

            candidates_with_entities = extract_candidates(
                entities,
                query_no_values,
            )


            raw_candidates = get_semantic_candidates_information(
                account_id,
                user_participants,
                question,
                k=10,
                list_of_semantic=[OmniLiteLabels.CUSTOM_ANALYSIS],
            )

            # Merge both candidate sources and remove duplicates.
            merged_raw_candidates = []
            seen_keys = set()
            for candidate in (candidates_with_entities or []) + (raw_candidates or []):
                key = (candidate.get("label"), str(candidate.get("id")))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_raw_candidates.append(candidate)

            # Clean and expand candidates
            cleaned_candidates = clean_results(account_id, merged_raw_candidates)
            ids_and_labels = [
                {"label": c["label"], "id": c["id"]} for c in cleaned_candidates
            ]
            candidates_properties = expand_info(
                account_id, user_participants, ids_and_labels
            )

            # Update candidate properties
            for candidate in cleaned_candidates:
                update_candidate_properties(
                    account_id, candidate, candidates_properties
                )

            # Store in path_state
            path_state["retrieved_candidates"] = cleaned_candidates

            self.logger.info(
                f"Retrieved {len(cleaned_candidates)} candidates"
            )

            return {"path_state": path_state}

        except Exception as e:
            # Fallback: empty candidates (routing agent will handle)
            self.logger.warning(
                f"Candidate retrieval failed: {e}, returning empty candidates"
            )
            path_state["retrieved_candidates"] = []

            return {"path_state": path_state}
