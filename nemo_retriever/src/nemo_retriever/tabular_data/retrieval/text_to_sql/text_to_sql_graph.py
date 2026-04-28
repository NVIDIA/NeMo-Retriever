import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import (
    AgentPayload,
    AgentState,
    get_question_for_processing,
)
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.candidates_preparation import CandidatePreparationAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.candidates_retrieval import CandidateRetrievalAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.entities_extraction import EntitiesExtractionAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.intent_validation import IntentValidationAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.response import ResponseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.sql_execution import SQLExecutionAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.sql_from_semantic import SQLFromCandidatesAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.sql_from_tables import SQLFromTablesAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.sql_reconstruction import SQLReconstructionAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.sql_unconstructable import SQLUnconstructableAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.agents.sql_parse_validation import SQLValidationAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.base import agent_wrapper

logger = logging.getLogger(__name__)


def route_sql_validation(state: AgentState) -> str:
    """
    Route based on SQL validation result.

    Handles SQL validation attempts and fallback logic:
    - "skip_intent_validation" if SQL is valid but reconstruction_count > 5 (skip intent validation)
    - "valid_sql" if SQL is valid (routes to intent validation)
    - "invalid_sql" if invalid (with retry logic)
    - "fallback" after 4 attempts (try constructing from tables)
    - "unconstructable" after 8 attempts (give up)

    Args:
        state: Current agent state

    Returns:
        Routing decision based on validation result and attempt count
    """
    if state["decision"] == "invalid_sql":
        attempts = state["path_state"].get("sql_attempts", 0)
        logger.info(f"Construct sql attempt: {attempts}")
        state["path_state"]["sql_attempts"] = attempts + 1
        if attempts == 4:
            logger.info("Can not construct sql from snippets, try from relevant tables. Fallback.")
            return "fallback"  # try constructing from tables, not only snippets
        elif attempts < 8:
            return "invalid_sql"
        elif attempts == 8:
            logger.error("SQL construction failed after 8 attempts")
            return "unconstructable"

    else:
        # SQL is valid - check if we should skip intent validation
        reconstruction_count = state["path_state"].get("reconstruction_count", 0)
        if reconstruction_count > 5:
            logger.info(f"Skipping intent validation after {reconstruction_count} reconstructions")
            return "skip_intent_validation"
        return "valid_sql"


def route_intent_validation(state: AgentState) -> str:
    """
    Route based on intent validation result.

    Handles intent validation:
    - "intent_valid" if SQL addresses user's intent (proceed to formatting)
    - "intent_invalid" if SQL doesn't address intent (retry with reconstruction)

    Args:
        state: Current agent state

    Returns:
        Routing decision based on intent validation result
    """
    decision = state.get("decision", "")

    if decision == "intent_invalid":
        attempts = state["path_state"].get("sql_attempts", 0)
        logger.info(f"Intent validation failed at attempt: {attempts}")
        # Route back to reconstruction to fix intent issues
        return "invalid_sql"
    else:
        # Intent is valid, proceed to formatting
        return "valid_sql"


def route_translation(state: AgentState) -> str:
    """
    Route to translation or graph END based on target language and context.

    This function does NOT modify state. It only decides whether we need
    to translate the final_response or end the graph.

    Returns:
        - "translate": if translation is needed
        - "end": if no translation is needed
    """
    language = state.get("language", "english") or "english"
    if language.lower() != "english":
        # Non-English target language → request translation
        return "translate"

    # English (or unknown) → no translation
    return "end"


def route_decision(state: AgentState) -> str:
    """
    Generic router — returns the current ``decision`` value from state,
    applying optional aliases first.

    Each caller's ``add_conditional_edges`` edge-map defines which values
    are legal; LangGraph will raise if the returned string isn't a key in
    that map, so no extra guard-rail set is needed here.

    Alias mappings (agent-specific convenience):
    - "constructable" → "validate_sql_query"
    """
    decision = state.get("decision", "") or ""

    decision_mapping = {
        "constructable": "validate_sql_query",
    }

    mapped = decision_mapping.get(decision, decision)

    if decision != mapped:
        logger.debug("Mapped decision '%s' → '%s'", decision, mapped)

    return mapped


def _make_node(name, fn):
    """
    Create a node with logging wrapper.

    For agents, use agent_wrapper instead.
    For simple functions, use this wrapper.
    """
    return RunnableLambda(wrap_node_with_logging(name, fn))


def log_node_visit(state, node_name: str):
    """
    Track how many times each graph node was visited during a run.
    """
    path_state = state.get("path_state", {})
    counts = path_state.get("node_visit_counts", {})
    counts[node_name] = counts.get(node_name, 0) + 1
    path_state["node_visit_counts"] = counts
    state["path_state"] = path_state
    total = sum(counts.values())
    logger.info(f"🔁 Node visits: {counts} | Total visits this run: {total}")


def wrap_node_with_logging(node_name: str, fn):
    """
    Wrap a node callable so it logs node visits automatically.
    """

    def wrapped(state):
        log_node_visit(state, node_name)
        return fn(state)

    return wrapped


def create_graph():

    # ==================== CREATE AGENT INSTANCES ====================

    # Routing agents
    entities_extraction_agent = EntitiesExtractionAgent()
    retrieval_agent = CandidateRetrievalAgent()
    candidate_preparation_agent = CandidatePreparationAgent()
    sql_from_tables_agent = SQLFromTablesAgent()
    sql_from_candidates_agent = SQLFromCandidatesAgent()
    sql_reconstruction_agent = SQLReconstructionAgent()
    sql_validation_agent = SQLValidationAgent()
    intent_validation_agent = IntentValidationAgent()
    sql_execution_agent = SQLExecutionAgent()
    response_agent = ResponseAgent()
    sql_unconstructable_agent = SQLUnconstructableAgent()

    # ==================== CREATE NODES ====================

    # Routing nodes (using agent_wrapper)

    entities_extraction_node = _make_node("entities_extraction", agent_wrapper(entities_extraction_agent))
    retrieve_candidates_node = _make_node("retrieve_candidates", agent_wrapper(retrieval_agent))
    prepare_candidates_node = _make_node("prepare_candidates", agent_wrapper(candidate_preparation_agent))
    construct_sql_not_from_snippets_node = _make_node(
        "construct_sql_not_from_snippets", agent_wrapper(sql_from_tables_agent)
    )
    construct_sql_from_candidates_node = _make_node(
        "construct_sql_from_candidates",
        agent_wrapper(sql_from_candidates_agent),
    )
    reconstruct_sql_node = _make_node("reconstruct_sql", agent_wrapper(sql_reconstruction_agent))

    validate_sql_query_node = _make_node("validate_sql_query", agent_wrapper(sql_validation_agent))
    validate_intent_node = _make_node("validate_intent", agent_wrapper(intent_validation_agent))
    execute_sql_query_node = _make_node("execute_sql_query", agent_wrapper(sql_execution_agent))
    format_and_respond_node = _make_node("format_and_respond", agent_wrapper(response_agent))
    unconstructable_sql_response_node = _make_node(
        "unconstructable_sql_response", agent_wrapper(sql_unconstructable_agent)
    )

    # ==================== CREATE GRAPH ====================

    graph = StateGraph(AgentState)

    # -----------------    ENTRY POINT   ------------------
    graph.set_entry_point("entities_extraction")

    # Add only nodes instantiated above.
    graph.add_node("entities_extraction", entities_extraction_node)
    graph.add_node("retrieve_candidates", retrieve_candidates_node)
    graph.add_node("prepare_candidates", prepare_candidates_node)
    graph.add_node("construct_sql_not_from_snippets", construct_sql_not_from_snippets_node)
    graph.add_node("construct_sql_from_candidates", construct_sql_from_candidates_node)
    graph.add_node("reconstruct_sql", reconstruct_sql_node)
    graph.add_node("validate_sql_query", validate_sql_query_node)
    graph.add_node("validate_intent", validate_intent_node)
    graph.add_node("execute_sql_query", execute_sql_query_node)
    graph.add_node("format_and_respond", format_and_respond_node)
    graph.add_node("unconstructable_sql_response", unconstructable_sql_response_node)

    # Minimal flow using only the defined nodes.
    graph.add_edge("entities_extraction", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "prepare_candidates")
    graph.add_edge("prepare_candidates", "construct_sql_from_candidates")

    graph.add_conditional_edges(
        "construct_sql_from_candidates",
        route_decision,
        {
            "validate_sql_query": "validate_sql_query",
            "unconstructable": "unconstructable_sql_response",
        },
    )

    # SQL validation → route
    graph.add_conditional_edges(
        "validate_sql_query",
        route_sql_validation,
        {
            "valid_sql": "validate_intent",  # Validate intent after syntax validation succeeds
            "skip_intent_validation": "execute_sql_query",  # Skip intent validation after 5+ reconstructions
            "invalid_sql": "reconstruct_sql",
            "fallback": "construct_sql_not_from_snippets",
            "unconstructable": "unconstructable_sql_response",
        },
    )

    # Intent validation → route
    graph.add_conditional_edges(
        "validate_intent",
        route_intent_validation,
        {
            "valid_sql": "execute_sql_query",  # Format after both validations succeed
            "invalid_sql": "reconstruct_sql",  # Reconstruct if intent is invalid
        },
    )

    # SQL execution → route
    graph.add_conditional_edges(
        "execute_sql_query",
        route_decision,
        {
            "valid_sql": "format_and_respond",
            "invalid_sql": "reconstruct_sql",
        },
    )

    graph.add_edge("construct_sql_not_from_snippets", "validate_sql_query")
    graph.add_edge("reconstruct_sql", "validate_sql_query")

    graph.add_edge("unconstructable_sql_response", END)
    graph.add_edge("format_and_respond", END)

    return graph


__all__ = [
    "AgentPayload",
    "AgentState",
    "create_graph",
    "get_question_for_processing",
]
