"""
Candidate Retrieval Agent

This agent performs semantic search (custom analysis and columns) to retrieve relevant candidates from the graph.

Responsibilities:
- Perform semantic search for graph entities (custom analyses and columns)
- Clean and expand candidate properties

"""

import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.retrieval.omni_lite.state import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent


from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels,
    clean_results,
    expand_info,
    extract_candidates,
)

logger = logging.getLogger(__name__)


class CandidateRetrievalAgent(BaseAgent):
    """
    Agent that retrieves candidates from semantic search (custom analyses and columns).

    Retrieval Strategy:
    - Semantic search over custom analyses and columns.
    - Clean and expand candidate properties

    Output:
    - path_state["retrieved_custom_analyses"]: Cleaned custom_analysis stream.
    - path_state["retrieved_column_candidates"]: Cleaned column stream.
    - path_state["retrieved_candidates"]: Concatenation of both.
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

        Performs semantic search, cleans results, expands properties.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains retrieved candidates.
        """
        path_state = state.get("path_state", {})

        question = get_question_for_processing(state)

        try:
            # Semantic search: custom analyses + columns (see extract_candidates).
            entities_and_concepts = state.get("entities_and_concepts", []) 
            query_no_values = path_state.get(
                "query_no_values", ""
            )

            extracted = extract_candidates(
                entities_and_concepts,
                query_no_values,
                question,
            )

            # Primary path: tuple (custom_analysis_candidates, column_candidates) — keep streams separate.
            if isinstance(extracted, tuple) and len(extracted) == 2:
                custom_raw, column_raw = extracted
                retrieved_custom_analyses = clean_results(list(custom_raw or []))
                retrieved_column_candidates = clean_results(list(column_raw or []))
            else:
                # Legacy: flat list[{"candidate": ..., "entity": ...}] or raw dicts — merge then split by label.
                merged_raw_candidates = []
                for item in extracted or []:
                    merged_raw_candidates.append(item.get("candidate", item))
                cleaned_mixed = clean_results(merged_raw_candidates)
                retrieved_custom_analyses = [
                    c
                    for c in cleaned_mixed
                    if c.get("label") == Labels.CUSTOM_ANALYSIS
                ]
                retrieved_column_candidates = [
                    c
                    for c in cleaned_mixed
                    if c.get("label") == Labels.COLUMN
                ]

            ids_and_labels = [
                {"label": c["label"], "id": c["id"]}
                for c in (retrieved_custom_analyses + retrieved_column_candidates)
            ]
            candidates_properties = expand_info(ids_and_labels) if ids_and_labels else {}

            for candidate in retrieved_custom_analyses + retrieved_column_candidates:
                cid = candidate.get("id")
                if cid is None:
                    continue
                extra = candidates_properties.get(cid) or candidates_properties.get(str(cid))
                if isinstance(extra, dict):
                    candidate.update(extra)

            path_state["retrieved_custom_analyses"] = retrieved_custom_analyses
            path_state["retrieved_column_candidates"] = retrieved_column_candidates
            path_state["retrieved_candidates"] = (
                retrieved_custom_analyses + retrieved_column_candidates
            )

            n_custom = len(retrieved_custom_analyses)
            n_column = len(retrieved_column_candidates)
            self.logger.info(
                f"Retrieved {n_custom} custom_analysis and {n_column} column candidates "
                f"(combined total {n_custom + n_column} in retrieved_candidates)"
            )

            return {"path_state": path_state}

        except Exception as e:
            # Fallback: empty candidates (routing agent will handle)
            self.logger.warning(
                f"Candidate retrieval failed: {e}, returning empty candidates"
            )
            path_state["retrieved_candidates"] = []
            path_state["retrieved_custom_analyses"] = []
            path_state["retrieved_column_candidates"] = []

            return {"path_state": path_state}
