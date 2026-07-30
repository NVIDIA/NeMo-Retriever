# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools the agent can call, and the bases for plugging in your own.

One module per tool family:

- ``base_tool.py`` — :class:`BaseTool`; :class:`ToolError` (LLM-recoverable)
  and :class:`ToolContractError` (integration bug).
- ``retrieve.py`` — :class:`BaseRetrieveTool` / :class:`RetrieveContext`, the
  concrete :class:`RetrieveTool` and :class:`ReasoningAugmentedRetrieveTool`
  (wrap a plain ``retriever_fn(query, top_k)``), and
  :func:`create_retrieve_tool` picking between them by name.
- ``end_tools.py`` — :class:`BaseEndTool` plus the standard end tools:
  :class:`FinalResults` (the agent's end tool) and :class:`LogSelectedDocs`
  (the selection agent's per-run end tool).
- ``think_tool.py`` — :class:`ThinkTool`, the optional scratchpad, and
  :class:`SelectionThinkTool`, its selection-flavored variant.
"""

from .base_tool import BaseTool, ToolContractError, ToolError
from .end_tools import BaseEndTool, FinalResults, LogSelectedDocs
from .retrieve import (
    BaseRetrieveTool,
    ReasoningAugmentedRetrieveTool,
    RetrieveContext,
    RetrieveTool,
    create_retrieve_tool,
)
from .think_tool import SelectionThinkTool, ThinkTool

__all__ = [
    "BaseEndTool",
    "BaseRetrieveTool",
    "BaseTool",
    "FinalResults",
    "LogSelectedDocs",
    "ReasoningAugmentedRetrieveTool",
    "RetrieveContext",
    "RetrieveTool",
    "SelectionThinkTool",
    "ThinkTool",
    "ToolContractError",
    "ToolError",
    "create_retrieve_tool",
]
