# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieve tools: the base contract plus ready-made function-backed tools.

- :class:`BaseRetrieveTool` — subclass for full control; the agent detects
  retrieve tools by isinstance and applies its over-fetch/dedup/exclusion/
  logging machinery around :meth:`~BaseRetrieveTool.acall`.
- :class:`RetrieveTool` — wraps a plain ``retriever_fn(query, top_k)``.
- :class:`ReasoningAugmentedRetrieveTool` — same, but prepends the run's
  original question and/or the LLM's latest reasoning trace to the query.
- :func:`create_retrieve_tool` — picks one of the two classes by name.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from .base_tool import BaseTool, ToolContractError, ToolError, tool_error_text


@dataclass(frozen=True)
class RetrieveContext:
    """Per-call context the agent hands to every retrieve tool.

    ``global_query`` is the run's original question. ``reasoning`` is the most
    recent turn's reasoning trace (``None`` when the turn exposed none; the
    agent overwrites it every step, so it is never stale).
    """

    global_query: str
    reasoning: Optional[str] = None


class BaseRetrieveTool(BaseTool):
    """A tool that retrieves documents.

    Implement :meth:`_spec` and :meth:`_acall` (async-only; the standard tool
    hook, here with the fixed signature ``(query, top_k, context)``). ``top_k``
    is the number of documents the caller needs back (the agent's value
    already accounts for over-fetching); ``context`` carries the run's global
    query and latest reasoning — ignore it if unused.

    Result contract (validated here; violations raise
    :class:`ToolContractError` naming your tool): a list of dicts, each with
    ``id`` (coerced to str), ``score`` (coerced to float), and ``text``
    (coerced to str; may be empty); ``image`` optional; extra keys are shown
    to the LLM verbatim.
    Raise :class:`ToolError` for LLM-recoverable failures; let genuine
    failures propagate. Tools may be called from concurrent runs — keep them
    stateless.
    """

    default_top_k: int = 20

    @abstractmethod
    async def _acall(self, query: str, top_k: int, context: RetrieveContext) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def acall(
        self, query: str, top_k: Optional[int] = None, context: Optional[RetrieveContext] = None
    ) -> Union[List[Dict[str, Any]], str]:
        """Retrieve and validate — do not override; implement :meth:`_acall`.

        Narrows :meth:`BaseTool.acall` to the retrieve signature. ``top_k``
        defaults to ``default_top_k``; ``context`` defaults to an empty
        context built from the query. Unlike the generic wrapper, only
        :class:`ToolError` becomes LLM-visible error text — the agent
        validates LLM-supplied arguments before calling, so a ``TypeError``
        here is a tool bug and propagates.
        """
        if top_k is None:
            top_k = self.default_top_k
        if context is None:
            context = RetrieveContext(global_query=str(query))
        try:
            result = await self._acall(query, top_k, context)
        except ToolError as e:
            return tool_error_text(self.name, e)
        return self._validate_result(result)

    def _validate_result(self, result: Any) -> List[Dict[str, Any]]:
        cls_name = type(self).__name__
        if not isinstance(result, list):
            raise ToolContractError(
                f"{cls_name}: retrieve result must be a list of document dicts, " f"got {type(result).__name__}."
            )
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(result):
            if not isinstance(item, dict):
                raise ToolContractError(f"{cls_name}: result item {i} must be a dict, got {type(item).__name__}.")
            missing = [k for k in ("id", "score", "text") if k not in item]
            if missing:
                raise ToolContractError(
                    f"{cls_name}: result item {i} is missing required key(s) {missing}. "
                    "Each document must carry 'id', 'score', and 'text'."
                )
            doc = dict(item)
            doc["id"] = str(doc["id"])
            try:
                doc["score"] = float(doc["score"])
            except (TypeError, ValueError):
                raise ToolContractError(
                    f"{cls_name}: result item {i} has a non-numeric 'score': {item['score']!r}."
                ) from None
            if not isinstance(doc["text"], str):
                doc["text"] = "" if doc["text"] is None else str(doc["text"])
            out.append(doc)
        return out


class RetrieveTool(BaseRetrieveTool):
    """Retrieve tool backed by a plain ``retriever_fn(query, top_k)``.

    ``retriever_fn`` may be sync (runs via ``asyncio.to_thread`` — it may
    block on network/disk/inference) or async, and is called positionally.
    """

    def __init__(
        self,
        retriever_fn: Callable[..., Any],
        *,
        name: str = "retrieve",
        description: str = "Search for documents related to a query using dense retrieval.",
        default_top_k: int = 20,
    ) -> None:
        if not callable(retriever_fn):
            raise TypeError(f"retriever_fn must be callable, got {type(retriever_fn).__name__}.")
        self._fn = retriever_fn
        self._fn_is_async = inspect.iscoroutinefunction(retriever_fn)
        self._name = str(name)
        self._description = str(description)
        self.default_top_k = int(default_top_k)

    def _spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve.",
                            "default": self.default_top_k,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def _acall(self, query: str, top_k: int, context: RetrieveContext) -> List[Dict[str, Any]]:
        return await self._call_retriever(query, top_k)

    async def _call_retriever(self, query: str, top_k: int) -> Any:
        if self._fn_is_async:
            return await self._fn(query, top_k)
        return await asyncio.to_thread(self._fn, query, top_k)


class ReasoningAugmentedRetrieveTool(RetrieveTool):
    """RetrieveTool that adds run context to the query before retrieving.

    Composes the query body from ``body_template`` (or the ``context_mode``
    default) with the run's original question (``{global_query}``) and the
    LLM's latest reasoning trace (``{reasoning}``); the retriever function
    stays a plain ``(query, top_k)``. When a turn exposed no reasoning,
    ``empty_reasoning_policy`` either substitutes a placeholder
    (``"substitute"``, default) or sends the raw query unchanged (``"skip"``).
    """

    BODY_TEMPLATES: Dict[str, str] = {
        "reasoning": "Reasoning: {reasoning}\n\nQuery: {query}",
        "global_query": "Original question: {global_query}\n\nQuery: {query}",
        "both": "Original question: {global_query}\nReasoning: {reasoning}\n\nQuery: {query}",
    }
    EMPTY_REASONING_PLACEHOLDER = "Empty"

    def __init__(
        self,
        retriever_fn: Callable[..., Any],
        *,
        context_mode: str = "both",
        body_template: Optional[str] = None,
        empty_reasoning_policy: str = "substitute",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever_fn, **kwargs)
        if body_template is None:
            if context_mode not in self.BODY_TEMPLATES:
                raise ValueError(f"context_mode must be one of {sorted(self.BODY_TEMPLATES)}, got {context_mode!r}.")
            body_template = self.BODY_TEMPLATES[context_mode]
        try:
            body_template.format(query="", reasoning="", global_query="")
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"body_template has an unsupported placeholder ({e!s}); "
                "supported placeholders are {query}, {reasoning}, {global_query}."
            ) from e
        if empty_reasoning_policy not in ("substitute", "skip"):
            raise ValueError(f"empty_reasoning_policy must be 'substitute' or 'skip', got {empty_reasoning_policy!r}.")
        self._body_template = body_template
        self._empty_reasoning_policy = empty_reasoning_policy

    async def _acall(self, query: str, top_k: int, context: RetrieveContext) -> List[Dict[str, Any]]:
        return await self._call_retriever(self._build_query(query, context), top_k)

    def _build_query(self, query: str, context: RetrieveContext) -> str:
        reasoning = (context.reasoning or "").strip()
        if "{reasoning}" in self._body_template and not reasoning:
            if self._empty_reasoning_policy == "skip":
                return str(query)
            reasoning = self.EMPTY_REASONING_PLACEHOLDER
        return self._body_template.format(
            query=str(query),
            reasoning=reasoning,
            global_query=(context.global_query or "").strip(),
        )


def create_retrieve_tool(kind: str, retriever_fn: Callable[..., Any], **kwargs: Any) -> BaseRetrieveTool:
    """Create a retrieve tool from a plain retriever function.

    ``kind="default"`` builds a :class:`RetrieveTool`;
    ``kind="reasoning_augmented"`` builds a
    :class:`ReasoningAugmentedRetrieveTool`. ``kwargs`` go to the class
    constructor. For anything fancier (e.g. reading ``context`` raw), subclass
    :class:`RetrieveTool` (or :class:`BaseRetrieveTool`) and override ``_acall``.
    """
    if kind == "default":
        return RetrieveTool(retriever_fn, **kwargs)
    if kind == "reasoning_augmented":
        return ReasoningAugmentedRetrieveTool(retriever_fn, **kwargs)
    raise ValueError(f"kind must be 'default' or 'reasoning_augmented', got {kind!r}.")


def retrieve_output_to_msg_content(output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a validated retrieve output list into LLM message content blocks."""
    content_list: List[Dict[str, Any]] = []
    for doc_in in output:
        doc = {**doc_in}
        if str(doc.get("text") or "").strip() == "":
            doc.pop("text", None)
        img = doc.pop("image", None)
        content_list.append({"type": "text", "text": json.dumps(doc)})
        if img is not None:
            content_list.append({"type": "image_url", "image_url": {"url": img}})
    return content_list
