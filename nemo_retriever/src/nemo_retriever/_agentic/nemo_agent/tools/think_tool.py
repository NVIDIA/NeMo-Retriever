# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scratchpad tool that lets the LLM think with output tokens.

:class:`ThinkTool` carries retrieval-flavored use-case bullets (the main
agent's version); :class:`SelectionThinkTool` swaps in selection-flavored
bullets. Both descriptions are LLM-visible; only the bullet list differs.
"""

from __future__ import annotations

from typing import List

from .base_tool import BaseTool


class ThinkTool(BaseTool):
    """Tool that allows the LLM to think with output tokens."""

    _use_case_lines: List[str] = [
        "- When processing a complex query, use this tool to organize your thoughts and think about the sub queries that you need to search for to find the relevant information",  # noqa: E501
        "- If a query is vague is very difficult to find information for it, you can use this tool to think about clues in the query that you can use to narrow down the search and spot relevant pieces of information.",  # noqa: E501
        "- When finding related documents that help you create better search queries in the next step, use this tool to think about what pieces of information from these documents are helpful to search for.",  # noqa: E501
        "- When you fail to find any related information to the query, use this tool to think about other search strategies that you can take to retrieve the related documents",  # noqa: E501
    ]

    def __init__(self, extended_relevance: bool = False):
        if extended_relevance:
            ext_lines = [
                "- When it is difficult to understand what is the intent of the user and what they are trying to find with this query, use this tool to think about potential definitions of relevance that could be meaningful/useful to the user for this task.",  # noqa: E501
                "- If the intention of the user is vague especially given the available documents, use this tool to think how you should decide what documents are relevant and what the metric of relevance is.",  # noqa: E501
            ]
            ext = "\n".join(ext_lines) + "\n"
        else:
            ext = ""
        description = (
            "Use the tool to think about something. It will not obtain new information or make any changes, "
            "but just log the thought. Use it when complex reasoning or brainstorming is needed.\n"
            "\n"
            "Common use cases:\n"
            f"{ext}" + "\n".join(self._use_case_lines) + "\n"
            "\n"
            "The tool simply logs your thought process for better transparency and does not make any changes."
        )
        self.spec_dict = {
            "type": "function",
            "function": {
                "name": "think",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "The thought to log.",
                        }
                    },
                    "required": ["thought"],
                },
            },
        }

    def _spec(self) -> dict:
        return self.spec_dict

    def _call(self, thought: str) -> str:
        return "Your thought has been logged."

    async def _acall(self, thought: str) -> str:
        return "Your thought has been logged."


class SelectionThinkTool(ThinkTool):
    """The selection agent's think tool: same contract, selection-flavored bullets."""

    _use_case_lines: List[str] = [
        "- When processing a complex query, use this tool to organize your thoughts and think about how each document might be related to the given query.",  # noqa: E501
        "- If a query is vague or hard to understand, you can use this tool to think about clues in the query that help you identify the connections between a document and the query.",  # noqa: E501
        "- You can use this tool to think what pieces of information in each document are the most important or relevant for the given query.",  # noqa: E501
    ]
