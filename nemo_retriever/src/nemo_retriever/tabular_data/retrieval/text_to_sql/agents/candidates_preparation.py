"""
Candidate Preparation Agent

This agent prepares and fetches all candidates needed for SQL construction.
It runs before SQL generation agents to gather all necessary context.

Responsibilities:
- Fetch relevant tables from candidates
- Filter tables by LLM-based relevance check
- Retrieve relevant queries for context
- Filter and process complex candidates (custom analyses)
- Store all prepared data in path_state for downstream agents

Design Decisions:
- Runs before SQL generation to separate data fetching from SQL construction logic
- Stores fetched data in path_state for reusability across multiple SQL agents
- Handles embeddings and conversation history lookup
- LLM relevance filter removes noise tables before SQL construction
"""

import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from nemo_retriever.tabular_data.retrieval.llm_invoke import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.text_to_sql.models import TableRelevanceModel
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import TABLE_RELEVANCE_FILTER_PROMPT
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import rules_to_text
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.utils import (
    Labels,
    dedupe_merge_relevant_tables,
    get_relevant_tables,
    get_relevant_tables_from_candidates,
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
    - Relevant tables
    - Relevant queries for context
    - Similar questions from conversation history


    Output:
    - path_state["candidates"]: Flat list of candidate dicts (same as retrieved, enriched)
    - path_state["relevant_tables"]: Deduplicated list of relevant table dicts
        (same per-table dict shape as ``get_relevant_tables``)
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

        Gathers tables, queries, similar questions, and processes complex candidates.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains all prepared candidate data
        """
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        candidates = list(path_state.get("retrieved_candidates") or [])

        relevant_tables = get_relevant_tables_from_candidates(candidates)

        additional_tables = get_relevant_tables(
            state["retriever"],
            question,
            k=15,
        )
        relevant_tables.extend(additional_tables)
        relevant_tables = dedupe_merge_relevant_tables(relevant_tables)[:20]
        self.logger.info(f"Found {len(relevant_tables)} relevant tables (after dedupe, capped at 20)")

        relevant_tables, table_relevance_reasoning = self._filter_tables_by_relevance(
            state, question, relevant_tables,
        )
        self.logger.info(
            "Kept %d relevant tables (after relevance filter): %s",
            len(relevant_tables),
            [t["name"] for t in relevant_tables],
        )

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
                "relevant_queries": relevant_queries,
                "custom_analyses": custom_analyses,
                "custom_analyses_str": custom_analyses_str,
                "table_relevance_reasoning": table_relevance_reasoning,
            }
        }

    def _filter_tables_by_relevance(
        self,
        state: AgentState,
        question: str,
        tables: list[dict],
    ) -> tuple[list[dict], str]:
        """Use the LLM to decide which candidate tables are actually needed.

        Sends table names and descriptions to the LLM alongside the user's
        question and domain rules.  Returns ``(filtered_tables, reasoning)``.
        On any failure the full list is returned unchanged with empty reasoning.
        """
        if len(tables) <= 2:
            return tables, ""

        try:
            llm = state["llm"]
        except KeyError:
            self.logger.warning("No LLM in state — skipping relevance filter")
            return tables, ""

        tables_summary = "\n".join(
            f"- {t['name']}: {t.get('description', '(no description)')}"
            for t in tables
        )

        custom_prompts_text = rules_to_text(state.get("custom_prompts_rules", []))
        domain_rules = ""
        if custom_prompts_text:
            domain_rules = (
                "Domain-specific rules (use these to decide relevance):\n"
                f"{custom_prompts_text}\n"
            )

        prompt_text = TABLE_RELEVANCE_FILTER_PROMPT.format(
            question=question,
            tables_summary=tables_summary,
            domain_rules=domain_rules,
        )

        messages = [
            SystemMessage(content="You are a database schema expert that filters candidate tables."),
            HumanMessage(content=prompt_text),
        ]

        try:
            result = invoke_with_structured_output(llm, messages, TableRelevanceModel)
        except Exception as e:
            top_n = tables[:10]
            self.logger.warning(
                "Table relevance LLM call failed: %s — falling back to top %d/%d tables",
                e, len(top_n), len(tables),
            )
            return top_n, ""

        if result is None:
            top_n = tables[:10]
            self.logger.warning(
                "Table relevance filter returned None (LLM parsing failed). "
                "Falling back to top %d/%d tables by retrieval order. "
                "Check ERROR logs above for parsing/validation details.",
                len(top_n),
                len(tables),
            )
            return top_n, ""

        reasoning = (result.reasoning or "").strip()
        kept_names = {name.lower() for name in result.relevant_table_names}

        filtered = [t for t in tables if t["name"].lower() in kept_names]

        removed = [t["name"] for t in tables if t["name"].lower() not in kept_names]
        self.logger.info("Relevance filter reasoning: %s", reasoning if reasoning else "(empty)")
        if removed:
            self.logger.info("Relevance filter removed tables: %s", removed)

        if not filtered:
            top_n = tables[:10]
            self.logger.warning(
                "Relevance filter removed ALL tables — falling back to top %d/%d tables",
                len(top_n), len(tables),
            )
            return top_n, reasoning

        return filtered, reasoning

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
