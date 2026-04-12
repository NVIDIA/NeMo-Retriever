"""
LangGraph agent state and API payload types.

Kept separate from ``graph.py`` to avoid circular imports (agents import state;
``graph`` imports agents).
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class AgentPayload(TypedDict):
    """Payload received from the API."""

    question: str
    history: list[dict[str, str]]
    path_state: NotRequired[dict]
    dialects: NotRequired[list[str]]
    db_connector: NotRequired[Any]


class AgentState(TypedDict):
    """State object passed through the LangGraph."""

    llm: ChatNVIDIA
    initial_question: str
    messages: list[HumanMessage]
    decision: str
    dialects: list[str]
    db_connector: Any
    path_state: dict


def get_question_for_processing(state: AgentState) -> str:
    """
    Question string for retrieval, SQL, and validation.

    Uses ``path_state["normalized_question"]`` when set (e.g. after entity extraction),
    otherwise ``initial_question``.
    """
    path_state = state.get("path_state", {})
    # normalized_question = path_state.get("normalized_question")
    #TODO remove normalized question, for question : give me all student from seattle , seattle was removed
    normalized_question = path_state.get("initial_question")
    if normalized_question:
        return normalized_question
    return state.get("initial_question", "")


__all__ = ["AgentPayload", "AgentState", "get_question_for_processing"]
