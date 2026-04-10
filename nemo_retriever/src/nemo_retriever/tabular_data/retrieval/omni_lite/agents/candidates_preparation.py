"""
Candidate Preparation Agent

This agent prepares and fetches all candidates needed for SQL construction.
It runs before SQL generation agents to gather all necessary context.

Responsibilities:
- Fetch relevant tables and foreign keys from candidates
- Retrieve relevant queries for context
- Find similar questions from conversation history
- Filter and process complex candidates (with SQL snippets, metrics, analyses)
- Store all prepared data in path_state for downstream agents

Design Decisions:
- Runs before SQL generation to separate data fetching from SQL construction logic
- Stores fetched data in path_state for reusability across multiple SQL agents
- Handles embeddings and conversation history lookup
"""

import logging
from typing import Dict, Any

import networkx as nx

from nemo_retriever.tabular_data.retrieval.omni_lite.state import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels,
    _apply_foreign_key_hints,
    dedupe_merge_relevant_tables,
    get_relevant_fks_from_candidates_tables,
    get_relevant_queries,
    get_relevant_tables,
)

logger = logging.getLogger(__name__)


class CandidatePreparationAgent(BaseAgent):
    """
    Agent that prepares and fetches all candidates for SQL construction.

    This agent gathers all necessary context before SQL generation:
    - Relevant tables and foreign keys
    - Relevant queries for context
    - Similar questions from conversation history


    Output:
    - path_state["candidates"]: Flat list of candidate dicts (same as retrieved, enriched)
    - path_state["tables_rows"]: Output of ``get_relevant_fks_from_candidates_tables``
        (same per-table dict shape as ``get_relevant_tables``)
    - path_state["relevant_fks"]: Flat list of FK relationship dicts
    - path_state["relevant_queries"]: Relevant queries for context
    - path_state["similar_questions"]: Similar questions from history
    - path_state["complex_candidates"]: Filtered complex candidates
    - path_state["complex_candidates_str"]: String representation for prompts
    """

    def __init__(self):
        super().__init__("candidate_preparation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that retrieval (or legacy candidates) produced at least one hit."""
        path_state = state.get("path_state", {})
        if not path_state.get("retrieved_candidates"):
            self.logger.warning(
                "No candidates for preparation: set retrieved_custom_analyses / "
                "retrieved_column_candidates, retrieved_candidates, or legacy candidates"
            )
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare and fetch all candidates for SQL construction.

        Gathers tables, foreign keys, queries, similar questions, and processes complex candidates.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains all prepared candidate data
        """
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        candidates = list(path_state.get("retrieved_candidates") or [])

        relevant_tables , relevant_fks = get_relevant_fks_from_candidates_tables(candidates)

        additional_tables, additional_fks = get_relevant_tables(
            question,
            k=5,
        )

        relevant_tables.extend(additional_tables)

        relevant_fks.extend(additional_fks)

        relevant_tables = dedupe_merge_relevant_tables(relevant_tables)
        _apply_foreign_key_hints(relevant_tables, relevant_fks)

        self.logger.info(
            f"Found {len(relevant_tables)} relevant tables and {len(relevant_fks)} foreign keys"
        )


        relevant_queries = get_relevant_queries(
            candidates,
        )
        self.logger.info(f"Found {len(relevant_queries)} relevant queries")

        # Find similar questions from conversation history
        # similar_questions = []
        # embeddings_client = _optional_embeddings_client()  #TODO store and retrieve from somewhere
        # similar_questions = find_similar_questions(
        #     embeddings_client.embed_query(question), 
        # )
        # self.logger.info(
        #     f"Found {len(similar_questions)} similar questions from conversations"
        # )

        # Filter complex candidates ()
        complex_candidates = [
            {
                "name": x["name"],
                "sql": x.get("sql") or "",
                "label": x["label"],
                "description": x.get("description") or "",
            }
            for x in candidates
            if x.get("label") == Labels.CUSTOM_ANALYSIS
        ]
        self.logger.info(f"Filtered {len(complex_candidates)} complex candidates")

        # Build string representation of complex candidates for prompts
        complex_candidates_str = self._build_complex_candidates_str(candidates)
        self.logger.info(
            f"Built string representation with {len(complex_candidates_str)} entries"
        )

        # Store all prepared data in path_state
        return {
            "path_state": {
                **path_state,
                "relevant_tables": relevant_tables,
                "relevant_fks": relevant_fks,
                "relevant_queries": relevant_queries,
                # "similar_questions": similar_questions,
                "complex_candidates": complex_candidates,
                "complex_candidates_str": complex_candidates_str,
            }
        }


    def _build_complex_candidates_str(self, candidates: list) -> list[str]:
        """
        Build string representation of complex candidates for prompts.

        Prioritizes certified candidates by sorting them first and including
        certification status in the string representation.

        Args:
            candidates: List of all candidates

        Returns:
            List of formatted candidate strings (certified ones first)
        """
        complex_candidates = []
        for x in candidates:
            if (
                x.get("complex_attribute", False)
                or x.get("label") == Labels.CUSTOM_ANALYSIS
                or x.get("certified", "pending") == "certified"
                or len(x.get("documents", [])) > 0
            ):
                complex_candidates.append(x)

        # Sort to prioritize certified candidates
        # Certified candidates come first, then others sorted by score
        def sort_key(candidate):
            is_certified = candidate.get("certified", "pending") == "certified"
            score = candidate.get("score", 0)
            # Return tuple: (not_certified, -score) so certified=True sorts first
            # Then by score descending
            return (not is_certified, -score)

        complex_candidates.sort(key=sort_key)

        complex_candidates_str = []
        for x in complex_candidates:
            is_certified = x.get("certified", "pending") == "certified"
            certified_marker = " [CERTIFIED]" if is_certified else ""

            preview = self._get_cleaned_sql(x)
            if preview:
                complex_candidates_str.append(
                    f"name: {x['name']}, label: {x['label']}, id: {x['id']}{certified_marker}, sql_snippet: {preview}"
                )
            else:
                # No SQL preview available; still include basic metadata
                complex_candidates_str.append(
                    f"name: {x['name']}, label: {x['label']}, id: {x['id']}{certified_marker}"
                )
        return complex_candidates_str

    def _get_cleaned_sql(self, candidate: dict) -> str:
        """
        Build a short, clean SQL preview for prompts.

        - Uses the first sql snippet's `sql_code` when available.
        - Avoids dumping full Python list/dict repr with heavy escaping.

        Args:
            candidate: Candidate dictionary

        Returns:
            Cleaned SQL string
        """
        sql_entries = candidate.get("sql") or []
        if isinstance(sql_entries, list) and sql_entries:
            raw = (
                sql_entries[0].get("sql_code")
                or sql_entries[0].get("snippet")
                or sql_entries[0].get("sql_snippet")
                or ""
            )
            if not isinstance(raw, str):
                raw = str(raw)
            # Light cleanup: reduce common escaping that confuses the model
            cleaned = raw.replace('\\"', '"')
            # Turn escaped newlines into real newlines for readability
            cleaned = cleaned.replace("\n", " ")
            return cleaned
        return ""
