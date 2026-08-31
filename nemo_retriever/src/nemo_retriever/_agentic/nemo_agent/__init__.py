# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic retrieval core: a self-contained agent loop with pluggable LLM backends and tools.

Internal API. This package is driven by the public graph operators (see
:mod:`nemo_retriever.operators.graph_ops`); prefer those over importing from here
directly. Typical integration::

    from nemo_retriever._agentic.nemo_agent import Agent, AgentConfig, create_retrieve_tool
    from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

    llm = create_llm(create_llm_config("litellm", model=...))
    retrieve = create_retrieve_tool("default", my_retriever_fn)
    agent = Agent(config=AgentConfig(mode="select"), llm=llm, retrieve_tool=retrieve)
    result = agent.run_sync("what is ...", raw_log_dir=None)
    result.final_doc_ids  # [] + result.error when the run failed

Standalone selection over candidate documents (an explicit ``scores`` ranking
arms the context-overflow shrink retry; without it there is one attempt)::

    sel = SelectionAgent(config=SelectionAgentConfig(target_top_k=10), llm=llm)
    top = sel.select_sync("what is ...", documents, scores=rrf_scores)
"""

from .agent import Agent, AgentConfig, ToolExecutionError
from .loop import KNOWN_LLM_ERRORS
from .results import (
    ERROR_BAD_FINISH_REASON,
    ERROR_CONTENT_POLICY,
    ERROR_CONTEXT_LIMIT,
    ERROR_LLM_CALL_FAILED,
    ERROR_MAX_STEPS,
    ERROR_TOOL_FAILED,
    ERROR_UNEXPECTED,
    AgentError,
    AgentRunResult,
)
from .selection_agent import SelectionAgent, SelectionAgentConfig
from .tools import (
    BaseEndTool,
    BaseRetrieveTool,
    BaseTool,
    FinalResults,
    LogSelectedDocs,
    ReasoningAugmentedRetrieveTool,
    RetrieveContext,
    RetrieveTool,
    SelectionThinkTool,
    ThinkTool,
    ToolContractError,
    ToolError,
    create_retrieve_tool,
)

__all__ = [
    "ERROR_BAD_FINISH_REASON",
    "ERROR_CONTENT_POLICY",
    "ERROR_CONTEXT_LIMIT",
    "ERROR_LLM_CALL_FAILED",
    "ERROR_MAX_STEPS",
    "ERROR_TOOL_FAILED",
    "ERROR_UNEXPECTED",
    "KNOWN_LLM_ERRORS",
    "Agent",
    "AgentConfig",
    "AgentError",
    "AgentRunResult",
    "BaseEndTool",
    "BaseRetrieveTool",
    "BaseTool",
    "FinalResults",
    "LogSelectedDocs",
    "ReasoningAugmentedRetrieveTool",
    "RetrieveContext",
    "RetrieveTool",
    "SelectionAgent",
    "SelectionAgentConfig",
    "SelectionThinkTool",
    "ThinkTool",
    "ToolContractError",
    "ToolError",
    "ToolExecutionError",
    "create_retrieve_tool",
]
