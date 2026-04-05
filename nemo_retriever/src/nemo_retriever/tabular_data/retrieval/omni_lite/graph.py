from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.candidates_preparation import CandidatePreparationAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.candidates_retieval import CandidateRetrievalAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.entities_extraction import EntitiesExtractionAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.formatting import SQLResponseFormattingAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.intent_validation import IntentValidationAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.response import CalculationResponseAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.sql_execution import SQLExecutionAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.sql_from_semantic import SQLFromSemanticAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.sql_from_tables import SQLFromTablesAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.sql_reconstruction import SQLReconstructionAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.sql_validation_agent import SQLValidationAgent
from nemo_retriever.tabular_data.retrieval.omni_lite.base import agent_wrapper
from langchain_openai import ChatOpenAI
from typing import Literal, Optional, TypedDict




import logging
logger = logging.getLogger(__name__)




class AgentState(TypedDict):
    """State object passed through the LangGraph."""

    llm: ChatOpenAI
    pg_connection: object  # TODO lancedb
    initial_question: str
    messages: list[HumanMessage]
    decision: str  # Generalized to support all graph node names
    intermediate_output: (
        str  # DEPRECATED: Use path_state instead. Kept for backward compatibility.
    )
    thoughts: str  # to log agent reasoning
    path_state: dict  # scoped to logic-specific states like retries
    language: str


def get_question_for_processing(state: AgentState) -> str:
    """
    Question string for retrieval, SQL, and validation.

    Uses ``path_state["normalized_question"]`` when set (e.g. after entity extraction),
    otherwise ``initial_question``.
    """
    path_state = state.get("path_state", {})
    normalized_question = path_state.get("normalized_question")
    if normalized_question:
        return normalized_question
    return state.get("initial_question", "")


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
            logger.info(
                "⚠️ Can not construct sql from snippets, try from relevant tables. Fallback."
            )
            return "fallback"  # try constructing from tables, not only snippets
        elif attempts < 8:
            return "invalid_sql"
        elif attempts == 8:
            logger.error("❌ SQL construction failed after 8 attempts")
            return "unconstructable"

    else:
        # SQL is valid - check if we should skip intent validation
        reconstruction_count = state["path_state"].get("reconstruction_count", 0)
        if reconstruction_count > 5:
            logger.info(
                f"⚠️ Skipping intent validation after {reconstruction_count} reconstructions"
            )
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
    Route based on state decision.

    This function is used by multiple nodes with different valid edges:
    - calculation node: "calculation_search", "construct_sql_chosen_snippet"
    - construct_sql_* nodes: "validate_sql_query", "extract_from_file_snippets", "ambiguity_check", "unconstructable"
    - ambiguity_check node: "ask_user_clarification", "validate_sql_query"

    Maps agent decisions to valid graph edges:
    - "constructable" → "validate_sql_query" (from SQL construction agents)
    - "unconstructable" → "unconstructable" (from SQL construction agents)
    - Preserves other valid decisions as-is

    Args:
        state: Current agent state

    Returns:
        Valid graph edge name
    """
    decision = state.get("decision", "")

    # Map agent-specific decisions to graph edges
    # SQL construction agents return "constructable" which needs to map to "validate_sql_query"
    decision_mapping = {
        "constructable": "validate_sql_query",
        # "unconstructable" maps to "unconstructable" (valid edge from construct_sql_from_semantic)
        # Other decisions are already valid edge names, so pass through
    }

    # Apply mapping if needed, otherwise return decision as-is
    # This preserves valid decisions like "calculation_search", "construct_sql_chosen_snippet", etc.
    mapped_decision = decision_mapping.get(decision, decision)

    # Log if we're mapping a decision (for debugging)
    if decision != mapped_decision:
        logger.debug(f"Mapped decision '{decision}' → '{mapped_decision}'")

    return mapped_decision



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
    sql_from_semantic_agent = SQLFromSemanticAgent()  
    sql_reconstruction_agent = SQLReconstructionAgent()
    sql_formatting_agent = SQLResponseFormattingAgent()
    sql_validation_agent = SQLValidationAgent()
    intent_validation_agent = IntentValidationAgent()
    sql_execution_agent = SQLExecutionAgent()
    calculation_response_agent = CalculationResponseAgent()
    calculation_unconstructable_agent = CalculationUnconstructableAgent()


    # ==================== CREATE NODES ====================

    # Routing nodes (using agent_wrapper)

    retrieve_candidates_node = _make_node(
        "retrieve_candidates", agent_wrapper(retrieval_agent)
    )
    extract_action_input_node = _make_node(
        "extract_action_input", agent_wrapper(entities_extraction_agent)
    )
    calculation_search_node = _make_node(
        "calculation_search", agent_wrapper(calculation_search_agent)
    )
    prepare_candidates_node = _make_node(
        "prepare_candidates", agent_wrapper(candidate_preparation_agent)
    )
    construct_sql_not_from_snippets_node = _make_node(
        "construct_sql_not_from_snippets", agent_wrapper(sql_from_tables_agent)
    )
    construct_sql_from_semantic_node = _make_node(
        "construct_sql_from_semantic",
        agent_wrapper(sql_from_semantic_agent),
    )
    format_sql_response_node = _make_node(
        "format_sql_response", agent_wrapper(sql_formatting_agent)
    )
    reconstruct_sql_node = _make_node(
        "reconstruct_sql", agent_wrapper(sql_reconstruction_agent)
    )

    validate_sql_query_node = _make_node(
        "validate_sql_query", agent_wrapper(sql_validation_agent)
    )
    validate_intent_node = _make_node(
        "validate_intent", agent_wrapper(intent_validation_agent)
    )
    execute_sql_query_node = _make_node(
        "execute_sql_query", agent_wrapper(sql_execution_agent)
    )
    calc_respond_node = _make_node(
        "calc_respond", agent_wrapper(calculation_response_agent)
    )
    unconstructable_sql_response_node = _make_node(
        "unconstructable_sql_response", agent_wrapper(calculation_unconstructable_agent)
    )




    # ==================== CREATE GRAPH ====================

    graph = StateGraph(AgentState)

    # -----------------    ENTRY POINT   ------------------
    graph.set_entry_point("retrieve_candidates")

    # Add only nodes instantiated above.
    graph.add_node("retrieve_candidates", retrieve_candidates_node)
    graph.add_node("extract_action_input", extract_action_input_node)
    graph.add_node("calculation_search", calculation_search_node)
    graph.add_node("prepare_candidates", prepare_candidates_node)
    graph.add_node("construct_sql_not_from_snippets", construct_sql_not_from_snippets_node)
    graph.add_node("construct_sql_from_semantic", construct_sql_from_semantic_node)
    graph.add_node("format_sql_response", format_sql_response_node)
    graph.add_node("reconstruct_sql", reconstruct_sql_node)
    graph.add_node("validate_sql_query", validate_sql_query_node)
    graph.add_node("validate_intent", validate_intent_node)
    graph.add_node("execute_sql_query", execute_sql_query_node)
    graph.add_node("calc_respond", calc_respond_node)
    graph.add_node("unconstructable_sql_response", unconstructable_sql_response_node)

    # Minimal flow using only the defined nodes.
    graph.add_edge("retrieve_candidates", "extract_action_input")
    graph.add_edge("extract_action_input", "calculation_search")
    graph.add_edge("calculation_search", "prepare_candidates")
    graph.add_edge("prepare_candidates", "construct_sql_from_semantic")
    


    graph.add_conditional_edges(
        "construct_sql_from_semantic",
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
            "skip_intent_validation": "format_sql_response",  # Skip intent validation after 5+ reconstructions
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
            "valid_sql": "format_sql_response",  # Format after both validations succeed
            "invalid_sql": "reconstruct_sql",  # Reconstruct if intent is invalid
        },
    )



    # Full flow using all defined nodes.
    graph.add_edge("construct_sql_not_from_snippets", "validate_sql_query")
    
   
    graph.add_edge("format_sql_response", "execute_sql_query")
    graph.add_edge("execute_sql_query", "calc_respond")
    graph.add_edge("reconstruct_sql", "validate_sql_query")

    graph.add_edge("unconstructable_sql_response", END)
    graph.add_edge("calc_respond", END)


    return graph


__all__ = ["AgentState", "create_graph", "get_question_for_processing"]







