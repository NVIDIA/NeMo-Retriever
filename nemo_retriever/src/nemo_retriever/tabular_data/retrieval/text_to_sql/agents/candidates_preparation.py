# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from nemo_retriever.tabular_data.retrieval.text_to_sql.models import (
    CustomAnalysisRelevanceModel,
    TableRelevanceModel,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import (
    CUSTOM_ANALYSIS_RELEVANCE_FILTER_PROMPT,
    TABLE_RELEVANCE_FILTER_PROMPT,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
    rules_to_text,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.retrieval.data_access.relevant_tables import (
    dedupe_merge_relevant_tables,
    get_relevant_tables,
    get_relevant_tables_from_candidates,
)


def _qualified_name(t: dict) -> str:
    """Build schema-qualified table name (e.g. 'public.users') for dedup/filtering."""
    schema = t.get("schema_name", "")
    name = t.get("name", "")
    return f"{schema}.{name}" if schema else name


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

        # --- 1. Filter custom analyses first ---
        custom_analyses = [x for x in candidates if x.get("label") == Labels.CUSTOM_ANALYSIS]
        self.logger.info("Retrieved %d custom analyses", len(custom_analyses))

        custom_analyses = self._filter_custom_analyses_by_relevance(state, question, custom_analyses)
        self.logger.info(
            "Kept %d custom analyses (after relevance filter): %s",
            len(custom_analyses),
            [x.get("name") for x in custom_analyses],
        )

        relevant_queries = _extract_relevant_queries(custom_analyses)
        self.logger.info("Found %d relevant queries from custom analyses", len(relevant_queries))

        custom_analyses_str = self._build_custom_analyses_str(custom_analyses)

        # --- 2. Retrieve relevant tables ---
        relevant_tables = get_relevant_tables_from_candidates(candidates)
        self.logger.info("Tables from candidates: %s", [t["name"] for t in relevant_tables])

        connector = state.get("connector")
        database_name = getattr(connector, "database_name", None) if connector else None

        additional_tables = []
        search_queries = [question] + path_state.get("entities", [])
        k_per_query = max(1, 10 // len(search_queries))
        for query in search_queries:
            try:
                tables = get_relevant_tables(
                    state["retriever"],
                    query,
                    k=k_per_query,
                    database_name=database_name,
                )
                additional_tables.extend(tables)
            except Exception:
                self.logger.warning("Table retrieval failed for query: %s", query, exc_info=True)
        additional_tables = dedupe_merge_relevant_tables(additional_tables)[:10]
        relevant_tables.extend(additional_tables)

        relevant_tables = dedupe_merge_relevant_tables(relevant_tables)[:20]
        self.logger.info(
            "Found %d relevant tables (after dedupe, capped at 20): %s",
            len(relevant_tables),
            [t["name"] for t in relevant_tables],
        )

        # --- 3. Filter tables by relevance ---
        relevant_tables, table_relevance_reasoning = self._filter_tables_by_relevance(
            state,
            question,
            relevant_tables,
            custom_analyses,
        )
        self.logger.info(
            "Kept %d relevant tables (after relevance filter): %s",
            len(relevant_tables),
            [t["name"] for t in relevant_tables],
        )

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

    def _filter_custom_analyses_by_relevance(
        self,
        state: AgentState,
        question: str,
        analyses: list[dict],
    ) -> list[dict]:
        """Use the LLM to decide which retrieved custom analyses are relevant.

        On any failure the full list is returned unchanged (safe fallback).
        """
        if len(analyses) <= 1:
            return analyses

        try:
            llm = state["llm"]
        except KeyError:
            self.logger.warning("No LLM in state — skipping custom analysis relevance filter")
            return analyses

        analyses_summary = "\n".join(
            f"- {a.get('name', '(unnamed)')}: "
            f"{(a.get('description') or '(no description)').strip()}"
            f"{('  SQL: ' + a['sql'].strip()) if a.get('sql') else ''}"
            for a in analyses
        )

        prompt_text = CUSTOM_ANALYSIS_RELEVANCE_FILTER_PROMPT.format(
            question=question,
            analyses_summary=analyses_summary,
        )

        messages = [
            SystemMessage(content="You are a database domain expert that filters custom analyses."),
            HumanMessage(content=prompt_text),
        ]

        try:
            result = invoke_with_structured_output(llm, messages, CustomAnalysisRelevanceModel)
        except Exception as e:
            self.logger.warning(
                "Custom analysis relevance LLM call failed: %s — keeping all",
                e,
            )
            return analyses

        if result is None:
            self.logger.warning("Custom analysis relevance filter returned None — keeping all")
            return analyses

        names_to_remove = {name.lower() for name in result.analyses_to_remove}

        filtered = [a for a in analyses if (a.get("name") or "").lower() not in names_to_remove]
        removed = [a.get("name") for a in analyses if (a.get("name") or "").lower() in names_to_remove]

        reasoning = (result.reasoning or "").strip()
        self.logger.info("Custom analysis filter reasoning: %s", reasoning if reasoning else "(empty)")
        if removed:
            self.logger.info("Custom analysis filter removed: %s", removed)
        self.logger.info("Custom analysis filter kept: %s", [a.get("name") for a in filtered])

        if not filtered:
            self.logger.warning("Custom analysis filter removed ALL analyses — keeping all")
            return analyses

        return filtered

    def _filter_tables_by_relevance(
        self,
        state: AgentState,
        question: str,
        tables: list[dict],
        custom_analyses: list[dict] | None = None,
    ) -> tuple[list[dict], str]:
        """Use the LLM to decide which candidate tables are actually needed.

        Sends table names and descriptions to the LLM alongside the user's
        question, domain rules, and selected custom analyses (so the LLM
        knows which tables the analyses' SQL requires).
        Returns ``(filtered_tables, reasoning)``.
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
            f"- {_qualified_name(t)}: {t.get('description', '(no description)')}" for t in tables
        )

        domain_rules_text = rules_to_text(state.get("domain_rules", []))
        domain_rules_section = ""
        if domain_rules_text:
            domain_rules_section = "Domain-specific rules (use these to decide relevance):\n" f"{domain_rules_text}\n"

        ca_section = ""
        if custom_analyses:
            ca_lines = []
            for a in custom_analyses:
                line = f"- {a.get('name', '(unnamed)')}"
                desc = (a.get("description") or "").strip()
                if desc:
                    line += f": {desc}"
                sql = (a.get("sql") or "").strip()
                if sql:
                    line += f"  SQL: {sql}"
                ca_lines.append(line)
            ca_section = (
                "Selected custom analyses (their SQL references tables that MUST be kept):\n"
                + "\n".join(ca_lines)
                + "\n\n"
            )

        prompt_text = TABLE_RELEVANCE_FILTER_PROMPT.format(
            question=question,
            tables_summary=tables_summary,
            domain_rules=domain_rules_section,
            custom_analyses=ca_section,
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
                e,
                len(top_n),
                len(tables),
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
        names_to_remove = {name.lower() for name in result.tables_to_remove}

        filtered = [t for t in tables if _qualified_name(t).lower() not in names_to_remove]
        removed = [_qualified_name(t) for t in tables if _qualified_name(t).lower() in names_to_remove]

        self.logger.info("Relevance filter reasoning: %s", reasoning if reasoning else "(empty)")
        if removed:
            self.logger.info("Relevance filter removed tables: %s", removed)

        if not filtered:
            self.logger.warning("Relevance filter removed ALL tables — keeping all")
            return tables, reasoning

        return filtered, reasoning

    def _build_custom_analyses_str(self, custom_analyses: list) -> list[str]:
        """Build string representation of custom analyses for prompts."""
        sorted_analyses = sorted(custom_analyses, key=lambda c: -c.get("score", 0))

        parts_list: list[str] = []
        for x in sorted_analyses:
            entry = f"name: {x['name']}"
            desc = (x.get("description") or "").strip()
            if desc:
                entry += f", description: {desc}"
            sql = (x.get("sql") or "").strip()
            if sql:
                entry += f", sql: {sql}"
            parts_list.append(entry)
        return parts_list
