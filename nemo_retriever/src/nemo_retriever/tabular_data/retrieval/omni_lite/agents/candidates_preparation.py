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

from nemo_retriever.tabular_data.retrieval.omni_lite.graph import (
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels,
    get_relevant_fks_from_candidates_tables,
    get_relevant_queries,
    get_relevant_tables,
    find_similar_questions,
)

logger = logging.getLogger(__name__)




def _optional_embeddings_client(account_id: str):
    """Embeddings are wired in some deployments; absent in the open-source stub."""
    try:
        from nemo_retriever.tabular_data.retrieval.omni_lite import utils as omni_utils

        ge = getattr(omni_utils, "get_embeddings", None)
        if callable(ge):
            return ge(account_id, is_embeddings=True)
    except Exception:
        pass
    return None


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
    - path_state["table_groups"]: Connected table groups with FKs
        Each group: {"tables": [...], "fks": [...]}
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

        self.logger.info(
            f"Found {len(relevant_tables)} relevant tables and {len(relevant_fks)} foreign keys"
        )

        table_groups = self._group_tables_by_fks(relevant_tables, relevant_fks)
        self.logger.info(
            f"Grouped {len(relevant_tables)} tables into {len(table_groups)} connected groups"
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

        # Filter complex candidates (SQL snippets, certified assets, rich column context)
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
                "table_groups": table_groups,  # Grouped tables by FK relationships
                "relevant_queries": relevant_queries,
                # "similar_questions": similar_questions,
                "complex_candidates": complex_candidates,
                "complex_candidates_str": complex_candidates_str,
            }
        }

    def _group_tables_by_fks(
        self,
        tables_rows: list[dict],
        fks: list[dict],
    ) -> list[dict]:
        """
        Group tables by foreign key relationships using networkx.

        Args:
            tables_rows: List of table dicts (see ``get_relevant_tables`` / ``get_relevant_fks_from_candidates_tables``).
            fks: Flat list of FK relationship dicts (table1, table2, columns, ...)

        Returns:
            List of {"tables": [...], "fks": [...]}.
        """
        if not tables_rows:
            return []

        table_by_name = {}
        for table in tables_rows:
            table_name = table.get("schema_name", "") + "." + table.get("name", "")
            if table_name:
                if table_name not in table_by_name:
                    table_by_name[table_name] = {"table": table}

        # Create graph
        G = nx.Graph()

        # # Add all tables as nodes (using table names)
        # for table_name in table_by_name.keys():
        #     G.add_node(table_name)

        covered_nodes = set()

        for fk in fks:
            table1 = fk.get("table1")
            table2 = fk.get("table2")
            if table1 not in covered_nodes:
                G.add_node(table1)
                covered_nodes.add(table1)
            if table2 not in covered_nodes:
                G.add_node(table2)
                covered_nodes.add(table2)

            # if table1 and table2 and table1 in table_by_name and table2 in table_by_name:
            G.add_edge(table1, table2)

        # Find connected components
        connected_components = list(nx.connected_components(G))

        # Group tables from table_by_name by connected_components
        grouped_tables_by_component = []
        for component in connected_components:
            # Get tables that exist in both component and table_by_name
            tables_in_component = [
                {
                    "table_name": table_name,
                    "table": table_by_name[table_name]["table"],
                }
                for table_name in component
                if table_name in table_by_name
            ]
            if tables_in_component:  # Only add non-empty groups
                grouped_tables_by_component.append(
                    {"component": component, "tables": tables_in_component}
                )

        table_groups = []
        for component in connected_components:
            # Get all tables in this component
            group_tables = [
                table_by_name[table_name]["table"]
                for table_name in component
                if table_name in table_by_name
            ]

            group_fks = []
            for fk in fks:
                table1 = fk.get("table1")
                table2 = fk.get("table2")
                if table1 in component and table2 in component:
                    group_fks.append(fk)

            table_groups.append(
                {
                    "tables": group_tables,
                    "fks": group_fks,
                }
            )

        # add all tables that are not in any component to the table_groups
        for table_name in table_by_name.keys():
            if table_name not in covered_nodes:
                table_groups.append(
                    {
                        "tables": [table_by_name[table_name]["table"]],
                        "fks": [],
                    }
                )

        # Sort groups by size (largest first) for consistency
        table_groups.sort(key=lambda g: len(g["tables"]), reverse=True)

        return table_groups

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
