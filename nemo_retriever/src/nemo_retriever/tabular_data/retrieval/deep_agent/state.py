"""
LangGraph agent state and API payload types.

Kept separate from ``graph.py`` to avoid circular imports (agents import state;
``graph`` imports agents).

Also re-exports ``RetrievalContext`` and ``EntityCoverage`` from ``context.py``
so downstream code can import everything from one place::

    from nemo_retriever.tabular_data.retrieval.deep_agent.state import (
        AgentPayload, AgentState, RetrievalContext, EntityCoverage,
    )
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from nemo_retriever.retriever import Retriever
from nemo_retriever.tabular_data.retrieval.deep_agent.context import (  # noqa: F401
    EntityCoverage,
    RetrievalContext,
)


class AgentPayload(TypedDict):
    """Payload received from the API.

    Mirrors :class:`nemo_retriever.tabular_data.retrieval.text_to_sql.state.AgentPayload`
    so callers can use a single payload shape across both pipelines.
    """

    question: str
    retriever: Retriever
    path_state: NotRequired[dict]
    dialect: NotRequired[str]
    connector: NotRequired[Any]
    acronyms: NotRequired[str]
    custom_prompts: NotRequired[str]


class AgentState(TypedDict):
    """State object passed through the LangGraph."""

    llm: ChatNVIDIA
    initial_question: str
    messages: list[HumanMessage]
    decision: str
    dialect: str
    connector: Any
    path_state: dict
    retriever: Retriever
    custom_prompts: str


def get_question_for_processing(state: AgentState) -> str:
    """
    Question string for retrieval, SQL, and validation.

    Uses ``path_state["normalized_question"]`` when set (e.g. after entity extraction),
    otherwise ``initial_question``.
    """
    path_state = state.get("path_state", {})
    # normalized_question = path_state.get("normalized_question")
    # TODO remove normalized question, for question : give me all student from seattle , seattle was removed
    normalized_question = path_state.get("initial_question")
    if normalized_question:
        return normalized_question
    return state.get("initial_question", "")


__all__ = [
    "AgentPayload",
    "AgentState",
    "EntityCoverage",
    "RetrievalContext",
    "get_question_for_processing",
]
