"""
SQL Unconstructable Response Agent

This agent generates a response when SQL cannot be constructed from available data.
"""

import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentState

logger = logging.getLogger(__name__)


class SQLUnconstructableAgent(BaseAgent):
    """
    Agent that generates a response when SQL construction fails.

    This agent returns a message explaining that SQL cannot be constructed
    from the available data, optionally including a detailed explanation.

    Input Requirements:
    - path_state["unconstructable_explanation"]: Optional explanation text

    Output:
    - messages: Unconstructable response message
    """

    def __init__(self):
        super().__init__("sql_unconstructable")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate unconstructable SQL response.

        Returns a message explaining that SQL cannot be constructed,
        using the explanation from path_state if available.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - messages: Unconstructable response message
        """
        path_state = state.get("path_state", {})
        unconstructable = path_state.get("unconstructable_explanation", "")

        response_text = unconstructable if unconstructable else "SQL can't be constructed from the data."

        response = {
            "response": response_text,
        }

        self.logger.info(f"Generated unconstructable SQL response: {response_text[:50]}...")

        return {"messages": response}
