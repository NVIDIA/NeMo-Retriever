"""
Candidate Preparation Agent

This agent prepares and fetches all candidates needed for SQL construction.
It runs before SQL generation agents to gather all necessary context.

Responsibilities:
- Fetch relevant tables and foreign keys from candidates
- Retrieve relevant queries for context
- Filter and process complex candidates (custom analyses)
- Store all prepared data in path_state for downstream agents

Design Decisions:
- Runs before SQL generation to separate data fetching from SQL construction logic
- Stores fetched data in path_state for reusability across multiple SQL agents
- Handles embeddings and conversation history lookup
"""

import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.utils import (
    Labels,
    _apply_foreign_key_hints,
    dedupe_merge_relevant_tables,
    get_relevant_fks_from_candidates_tables,
    get_relevant_tables,
)

logger = logging.getLogger(__name__)


def _extract_relevant_queries(candidates: list) -> list[str]:
    queries = []
    for candidate in candidates:
        if candidate.get("label", "") == Labels.CUSTOM_ANALYSIS:
            sql = (candidate.get("sql") or "").strip()
            if sql and sql not in queries:
                queries.append(sql)
    return queries


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
    - path_state["custom_analyses"]: Filtered complex candidates
    - path_state["custom_analyses_str"]: String representation for prompts
    """

    def __init__(self):
        super().__init__("candidate_preparation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that retrieval produced at least one hit."""
        path_state = state.get("path_state", {})
        if not path_state.get("retrieved_candidates"):
            self.logger.warning(
                "No candidates for preparation: set retrieved_custom_analyses / "
                "retrieved_column_candidates, retrieved_candidates"
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

        relevant_tables, relevant_fks = get_relevant_fks_from_candidates_tables(candidates)

        additional_tables, additional_fks = get_relevant_tables(
            state["retriever"],
            question,
            k=5,
        )

        relevant_tables.extend(additional_tables)
        relevant_fks.extend(additional_fks)
        relevant_tables = dedupe_merge_relevant_tables(relevant_tables)
        _apply_foreign_key_hints(relevant_tables, relevant_fks)
        self.logger.info(f"Found {len(relevant_tables)} relevant tables and {len(relevant_fks)} foreign keys")

        relevant_queries = _extract_relevant_queries(
            candidates,
        )
        self.logger.info(f"Found {len(relevant_queries)} relevant queries")

        custom_analyses = [x for x in candidates if x.get("label") == Labels.CUSTOM_ANALYSIS]
        self.logger.info(f"Filtered {len(custom_analyses)} custom analyses")

        custom_analyses_str = self._build_custom_analyses_str(custom_analyses)
        self.logger.info(f"Built string representation with {len(custom_analyses_str)} entries")

        return {
            "path_state": {
                **path_state,
                "relevant_tables": relevant_tables,
                "relevant_fks": relevant_fks,
                "relevant_queries": relevant_queries,
                "custom_analyses": custom_analyses,
                "custom_analyses_str": custom_analyses_str,
            }
        }

    def _build_custom_analyses_str(self, custom_analyses: list) -> list[str]:
        """Build string representation of custom analyses for prompts."""
        sorted_analyses = sorted(custom_analyses, key=lambda c: -c.get("score", 0))

        return [
            f"name: {x['name']}, label: {x['label']}, id: {x['id']}"
            + (f", sql_snippet: {p}" if (p := self._get_sql_preview_from_sql(x)) else "")
            for x in sorted_analyses
        ]

    def _get_sql_preview_from_sql(self, candidate: dict) -> str:
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
            raw = sql_entries[0].get("sql_code") or ""
            if not isinstance(raw, str):
                raw = str(raw)
            # Light cleanup: reduce common escaping that confuses the model
            cleaned = raw.replace('\\"', '"').replace("\n", " ")
            return cleaned
        return ""
