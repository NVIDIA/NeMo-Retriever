# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator that re-ranks retrieved documents using an LLM-based selection agent."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator

# openai is imported lazily in _ensure_client() so the operator stays
# serialisable for Ray workers without requiring it at import time.


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a document re-ranker agent, which is the final stage in an information retrieval pipeline.

<ROLE>
You are given a search query and a list of retrieved candidate documents that are potentially
relevant to the query. Your goal is to identify the {top_k} most relevant documents and rank
them from most to least relevant.
</ROLE>

<WORKFLOW>
1. Read the query carefully and understand what the user is looking for.
2. Compare each candidate document against the query to assess its relevance.
3. Select exactly {top_k} documents (or fewer if there are fewer candidates) and order them
   by decreasing relevance — the first document should be the most relevant.
4. Call log_selected_documents with your ranked list and a brief explanation.
</WORKFLOW>

<THINKING>
Use the think tool to reason through complex queries or to compare documents before committing
to a final ranking. Think out loud about which documents best address the query intent.
</THINKING>"""


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class SelectionAgentOperator(AbstractOperator, CPUOperator):
    """Re-rank a set of retrieved documents using an LLM-based selection agent.

    For each ``query_id`` group in the input DataFrame, the operator runs an
    agentic LLM loop that reads the query and all candidate documents, then
    calls a ``log_selected_documents`` tool to report the final ranked list.
    The loop also has access to a ``think`` tool for extended reasoning.

    Input DataFrame schema
    ----------------------
    query_id   : str  — unique query identifier
    query_text : str  — original query text shown to the LLM
    doc_id     : str  — unique document identifier
    text       : str  — document text content shown to the LLM
    (any additional columns are ignored)

    Output DataFrame schema
    -----------------------
    query_id : str  — same ``query_id`` as the input
    doc_id   : str  — selected document ID
    rank     : int  — 1-indexed rank (1 = most relevant)
    message  : str  — LLM explanation of the selection

    Parameters
    ----------
    llm_model : str
        OpenAI model identifier, e.g. ``"gpt-4o"``.
    top_k : int
        Number of documents to select per query.  If fewer candidate documents
        exist for a query the LLM selects all of them.  Defaults to ``5``.
    api_key : str, optional
        Literal API key **or** an ``"os.environ/VAR_NAME"`` reference resolved
        at call time.  Omit to rely on the ``OPENAI_API_KEY`` environment variable.
    base_url : str, optional
        Custom endpoint URL — useful for NIM deployments or self-hosted models.
    max_tokens : int, optional
        Upper bound on tokens in each LLM response.
    max_steps : int
        Maximum agentic loop iterations per query before giving up.
        Defaults to ``10``.
    system_prompt_override : str, optional
        Fully custom system prompt.  Use ``{top_k}`` as a placeholder.
        When provided, the built-in prompt is ignored.
    text_truncation : int
        Maximum characters of each document's text shown to the LLM.
        Long documents are truncated to avoid exceeding context limits.
        Defaults to ``2000``.

    Examples
    --------
    Standalone use::

        import pandas as pd
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        op = SelectionAgentOperator(llm_model="gpt-4o", top_k=5)
        df = pd.DataFrame({
            "query_id":   ["q1", "q1", "q1"],
            "query_text": ["What causes inflation?"] * 3,
            "doc_id":     ["d1", "d2", "d3"],
            "text":       ["Doc about monetary policy...", "Doc about supply chains...", "Unrelated doc..."],
        })
        result = op.run(df)
        # result has up to 5 rows per query_id, ordered by rank

    In the end-to-end pipeline::

        from nemo_retriever.graph import InprocessExecutor
        from nemo_retriever.graph.pipeline_graph import Graph
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        op = SelectionAgentOperator(llm_model="gpt-4o", top_k=5)
        graph = Graph()
        graph >> op
        executor = InprocessExecutor(graph, show_progress=False)
        ranked_df = executor.ingest(candidates_df)
    """

    def __init__(
        self,
        *,
        llm_model: str,
        top_k: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_steps: int = 10,
        system_prompt_override: Optional[str] = None,
        text_truncation: int = 2000,
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._top_k = top_k
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._max_steps = max_steps
        self._system_prompt_override = system_prompt_override
        self._text_truncation = text_truncation

        # Lazily initialised — keeps the operator serialisable for Ray workers.
        self._client: Any = None

    # ------------------------------------------------------------------
    # AbstractOperator interface
    # ------------------------------------------------------------------

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        """Validate that *data* is a DataFrame with the required columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"SelectionAgentOperator expects a pd.DataFrame, got {type(data).__name__!r}.")
        required = {"query_id", "query_text", "doc_id", "text"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required column(s): {sorted(missing)}. " f"Expected: {sorted(required)}."
            )
        return data.copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Run the selection agent loop for each query group."""
        self._ensure_client()

        rows: List[dict[str, Any]] = []

        for query_id, group in data.groupby("query_id", sort=False):
            query_text = str(group["query_text"].iloc[0])
            docs = [{"id": str(row["doc_id"]), "text": str(row["text"])} for _, row in group.iterrows()]
            result = self._select_documents(query_text, docs)
            message = result.get("message", "")
            for rank, doc_id in enumerate(result.get("doc_ids", []), 1):
                rows.append(
                    {
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "rank": rank,
                        "message": message,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["query_id", "doc_id", "rank", "message"])

        return pd.DataFrame(rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Lazily create the OpenAI client (once per instance)."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "SelectionAgentOperator requires 'openai'. " "Install it with:  pip install 'openai>=1.0'"
            ) from exc

        api_key = self._api_key
        if api_key is not None and api_key.strip().startswith("os.environ/"):
            var = api_key.strip().removeprefix("os.environ/")
            api_key = os.environ[var]

        self._client = OpenAI(
            api_key=api_key,
            **({"base_url": self._base_url} if self._base_url is not None else {}),
        )

    def _build_system_prompt(self, top_k: int) -> str:
        template = self._system_prompt_override or _SYSTEM_PROMPT
        return template.format(top_k=top_k)

    def _build_tools(self, top_k: int, valid_doc_ids: List[str]) -> List[dict[str, Any]]:
        """Return the two tool specs for the selection agent loop."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "think",
                    "description": (
                        "Use this tool to think through complex analysis before making a decision. "
                        "It logs your reasoning without making any changes. Use it to compare "
                        "documents against the query or to reason about relevance."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your reasoning or analysis.",
                            }
                        },
                        "required": ["thought"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "log_selected_documents",
                    "description": (
                        f"Records the {top_k} most relevant documents and ends the task. "
                        f"Call this when you have finished evaluating all candidate documents. "
                        f"The doc_ids list must be sorted from most to least relevant. "
                        f"Valid document IDs are: {valid_doc_ids}."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["doc_ids", "message"],
                        "properties": {
                            "doc_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    f"The IDs of the {top_k} most relevant documents, sorted from "
                                    "most to least relevant. Must be valid document IDs from the candidates."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": ("A brief explanation of your selection and the relevance ordering."),
                            },
                        },
                    },
                },
            },
        ]

    def _build_user_message(self, query_text: str, docs: List[dict[str, Any]]) -> dict[str, Any]:
        """Format query + candidate documents as a multi-part user message."""
        content: List[dict[str, Any]] = [
            {"type": "text", "text": f"Query:\n{query_text}"},
            {"type": "text", "text": "Candidate Documents:"},
        ]
        seen: set[str] = set()
        for doc in docs:
            doc_id = doc["id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            content.append({"type": "text", "text": f"Doc ID: {doc_id}"})
            text = doc.get("text", "").strip()
            if text:
                truncated = text[: self._text_truncation]
                if len(text) > self._text_truncation:
                    truncated += "..."
                content.append({"type": "text", "text": f"Doc Text: {truncated}"})
        return {"role": "user", "content": content}

    def _select_documents(
        self,
        query_text: str,
        docs: List[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the agentic selection loop for a single query.

        Returns a dict with ``doc_ids`` (ranked list) and ``message`` (LLM explanation).
        On failure or timeout returns an empty ``doc_ids`` list.
        """
        valid_ids = list(dict.fromkeys(d["id"] for d in docs))  # deduplicated, order-preserving
        feasible_k = min(self._top_k, len(valid_ids))

        system_prompt = self._build_system_prompt(feasible_k)
        tools = self._build_tools(feasible_k, valid_ids)
        valid_id_set = set(valid_ids)

        messages: List[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            self._build_user_message(query_text, docs),
        ]

        call_kwargs: dict[str, Any] = dict(
            model=self._llm_model,
            tools=tools,
            tool_choice="auto",
        )
        if self._max_tokens is not None:
            call_kwargs["max_tokens"] = self._max_tokens

        for _step in range(self._max_steps):
            response = self._client.chat.completions.create(messages=messages, **call_kwargs)
            choice = response.choices[0]
            msg = choice.message

            # Append the assistant turn to the history
            assistant_turn: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_turn["content"] = msg.content
            if msg.tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_turn)

            if choice.finish_reason == "stop" or not msg.tool_calls:
                # LLM stopped without calling a tool — nudge it to finish
                messages.append(
                    {
                        "role": "user",
                        "content": "Please call log_selected_documents to report your final selection.",
                    }
                )
                continue

            # Process tool calls
            tool_messages: List[dict[str, Any]] = []
            should_end = False
            end_kwargs: dict[str, Any] = {}

            for tc in msg.tool_calls:
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Error: could not parse tool arguments.",
                        }
                    )
                    continue

                if tc.function.name == "think":
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Your thought has been logged.",
                        }
                    )

                elif tc.function.name == "log_selected_documents":
                    doc_ids: List[str] = fn_args.get("doc_ids", [])
                    # Filter to only valid candidates
                    doc_ids = [d for d in doc_ids if d in valid_id_set][:feasible_k]
                    end_kwargs = {
                        "doc_ids": doc_ids,
                        "message": fn_args.get("message", ""),
                    }
                    should_end = True

                else:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error: unknown tool '{tc.function.name}'.",
                        }
                    )

            if should_end:
                return end_kwargs

            messages.extend(tool_messages)

        # Max steps reached without a successful call
        return {
            "doc_ids": [],
            "message": "Selection agent reached max steps without completing.",
        }
