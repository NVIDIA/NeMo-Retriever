"""
Base Agent Class

This module provides the base class for all agents in the Omni system.
All agents should inherit from BaseAgent to ensure consistent behavior,
error handling, and observability.

Design Principles:
- Single Responsibility: Each agent has one clear purpose
- Error Handling: All agents handle errors gracefully
- State Management: Agents operate on AgentState and return state updates
- Observability: All agent executions are logged
- Testability: Agents can be tested independently
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents in the Omni system.

    All agents inherit from this class to ensure:
    - Consistent error handling
    - Standardized logging
    - State validation
    - Graceful degradation

    Usage:
        class MyAgent(BaseAgent):
            def execute(self, state: AgentState) -> Dict[str, Any]:
                # Agent logic here
                return {"path_state": {...}}
    """

    def __init__(self, agent_name: str):
        """
        Initialize the agent.

        Args:
            agent_name: Human-readable name for logging and debugging
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")

    @abstractmethod
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute the agent's main logic.

        This is the core method that all agents must implement.
        It receives the current state and returns state updates.

        Args:
            state: Current agent state

        Returns:
            Dictionary containing state updates. Should include:
            - "path_state": Updates to path_state (dict merge)
            - "messages": New messages to add (list)
            - "decision": Next routing decision (optional, str)
            - Other state fields as needed

        Raises:
            AgentExecutionError: If agent execution fails (caught by wrapper)
        """
        pass

    def validate_input(self, state: AgentState) -> bool:
        """
        Validate input state before execution.

        Override this method to add custom validation logic.
        Return False to skip execution (will log warning and return empty state).

        Args:
            state: Current agent state

        Returns:
            True if input is valid, False otherwise
        """
        return True

    def handle_error(self, error: Exception, state: AgentState) -> Dict[str, Any]:
        """
        Handle errors during agent execution.

        Override this method for custom error handling.
        Default behavior: Log error and return error state.

        Args:
            error: The exception that occurred
            state: Current agent state (may be partially modified)

        Returns:
            Error state to return (should include error information in path_state)
        """
        self.logger.error(
            f"Agent execution failed: {error}",
            exc_info=True,
            extra={
                "agent_name": self.agent_name,
                "account_id": state.get("account_id"),
            },
        )

        path_state = state.get("path_state", {})
        path_state["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "agent": self.agent_name,
        }

        return {"path_state": path_state}

    def log_execution_start(self, state: AgentState) -> None:
        """Log the start of agent execution."""
        account_id = state.get("account_id", "unknown")
        self.logger.info(
            f"Executing agent: {self.agent_name}",
            extra={"agent_name": self.agent_name, "account_id": account_id},
        )

    def log_execution_end(self, state: AgentState, result: Dict[str, Any]) -> None:
        """Log the end of agent execution."""
        account_id = state.get("account_id", "unknown")
        decision = result.get("decision", "none")
        self.logger.info(
            f"Agent completed: {self.agent_name} -> {decision}",
            extra={
                "agent_name": self.agent_name,
                "account_id": account_id,
                "decision": decision,
            },
        )


class AgentExecutionError(Exception):
    """
    Custom exception for agent execution errors.

    Use this to distinguish agent errors from other exceptions.
    """

    pass


def agent_wrapper(agent: BaseAgent):
    """
    Wrapper function for agents that provides:
    - Input validation
    - Error handling
    - Logging
    - State management

    Usage in orchestrator/graph.py:
        graph.add_node("node_name", agent_wrapper(MyAgent("my_agent")))

    Args:
        agent: BaseAgent instance

    Returns:
        Wrapped function compatible with LangGraph
    """

    def wrapped(state: AgentState) -> Dict[str, Any]:
        """
        Wrapped agent execution function.

        This function is called by LangGraph for each agent node.
        It provides error handling and logging around the agent's execute method.
        """
        # Validate input
        if not agent.validate_input(state):
            agent.logger.warning(
                f"Input validation failed for {agent.agent_name}, skipping execution"
            )
            return {}

        # Log execution start
        agent.log_execution_start(state)

        try:
            # Execute agent
            result = agent.execute(state)

            # Ensure result is a dict
            if not isinstance(result, dict):
                agent.logger.error(
                    f"Agent {agent.agent_name} returned non-dict result: {result}"
                )
                return {}

            # Log execution end
            agent.log_execution_end(state, result)

            return result

        except Exception as e:
            # Handle error
            error_result = agent.handle_error(e, state)
            return error_result

    # Preserve agent name for debugging
    wrapped.__name__ = f"{agent.agent_name}_wrapped"
    wrapped.__doc__ = f"Wrapped {agent.agent_name} agent execution"

    return wrapped
