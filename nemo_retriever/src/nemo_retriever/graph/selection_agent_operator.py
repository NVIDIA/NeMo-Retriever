# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator that re-ranks retrieved documents using an LLM-based selection agent."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.nim.chat_completions import invoke_chat_completion_step

# ---------------------------------------------------------------------------
# Prompt rendering  (verbatim content of 01_v0.j2, rendered via Python)
# ---------------------------------------------------------------------------

_ROLE = """\
You are a document re-ranker agent, which is the final stage in an information retrieval pipeline.

<ROLE>
You are given a search query and a list of retrieved candidate documents that are potentially relevant to \
the given query. Your goal is to help the users identify the most relevant documents to the given query \
from the list of candidate documents.
</ROLE>"""

_RELEVANCE_DEFINITION = """\

<RELEVANCE_DEFINITION>
- You should be careful, in the context of this task, what it means to be a "query", "document", and \
"relevant" can sometimes be very complex and might not follow the traditional definition of these terms \
in standard re-ranking and retrieval.
- In standard re-ranking/retrieval, a query is usually a user question (like a web search query), the \
document is some sort of content that provides information (e.g., a web page), and these two are considered \
relevant if the document provides information that answers the user's query.
- However, in our setting, this could be different. Here are some examples:
    * the query is a programming problem and documents are programming language syntax references. A document \
is relevant if it contains the reference for the programming syntax used for solving the problem.
    * both query and documents are descriptions programming problems and a query and document are relevant if \
the same approach is used to solve them.
    * the query is a math problem and documents are theorems. Relevant documents (theorems) are the ones \
that are useful for solving the math problem.
    * the query and document are both math problems. A query and a document are relevant if the same theorem \
is used for solving them.
    * the query is a task description (e.g., for an API programmer) and documents are descriptions of \
available APIs. Relevant documents (e.g., APIs) are the ones needed for completing the task.
- This is not an exhaustive list. These are just some examples to show you the complexity of queries, \
documents, and the concept of relevance in this task.
- Note that even here, the relevant documents are still the ones that are useful for a user who is \
searching for the given query. But the relation is more nuanced.
- You should analyze the query and the available documents. And then reason about what could be a meaningful \
definition of relevance in this case, and what the user could be looking for.
- Moreover, sometimes, the query could be even a prompt that is given to a Large Language Model (LLM) and \
the user wants to find the useful documents for the LLM that help answering/solving this prompt.
</RELEVANCE_DEFINITION>"""

_WORKFLOW_TEMPLATE = """\

<WORKFLOW>
* You are given a search query and a list of candidate documents. You have access to the ID and content of \
each candidate document.
* You should read the query carefully and understand it.
{extended_relevance_line}\
* Then you should compare the query with each one of the candidate documents. In this comparison, you want \
to identify if the document is relevant/useful for the given query and to what extent.
* Select the {top_k} most relevant candidate documents for the given query.
* Note that just selecting the most relevant documents is not enough. You should identify the relative level \
of relevance between the query and selected documents. This helps you sort the selected documents later \
based on how relevant they are to the query.
* Once you have this information, you should call the "log_selected_documents" function to report the final \
results and signal the completion of the task.
* Note that the selected document IDs must be reported in the decreasing level of relevance. I.e., The \
first document in the list is the most relevant, the second is the second most relevant, and so on. This \
is similar to what a search engine (e.g., Google Search) does (it shows you the relevant results in a \
sorted order, where the most relevant results appear on top of the list).
</WORKFLOW>"""

_THINKING_TIPS = """

<THINKING TIPS>
* you have access to a "think" tool that you can use for complex thinking and analysis. Here are examples \
of cases where the think tool might be useful:
    - complex analysis and thinking to understand the meaning and intent of the query. E.g., what is the \
user trying to find with this query? what kind of information is helpful for the user?
    - extended thinking to analyze how each candidate document could or could not be relevant to the given query.
    - reasoning to identify the relative level of relevance between the query and selected documents. It \
helps you sort the documents correctly when reporting the final answer.
</THINKING TIPS>"""


def _render_selection_prompt(top_k: int, *, extended_relevance: bool = False) -> str:
    """Render the selection agent system prompt (verbatim 01_v0.j2 logic)."""
    parts = [_ROLE]
    if extended_relevance:
        parts.append(_RELEVANCE_DEFINITION)
    ext_line = (
        "* As explained above, reason and figure out what the meaning of relevance is in this case, "
        "and what could be relevant and useful information for the given query.\n"
        if extended_relevance
        else ""
    )
    parts.append(_WORKFLOW_TEMPLATE.format(top_k=top_k, extended_relevance_line=ext_line))
    parts.append(_THINKING_TIPS)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class SelectionAgentOperator(AbstractOperator, CPUOperator):
    """Re-rank a set of retrieved documents using an LLM-based selection agent.

    For each ``query_id`` group in the input DataFrame, the operator runs an
    agentic LLM loop that reads the query and all candidate documents, then
    calls a ``log_selected_documents`` tool to report the final ranked list.
    The loop also has access to a ``think`` tool for extended reasoning.

    The system prompt matches the retrieval-bench ``01_v0.j2`` template verbatim,
    with an optional ``extended_relevance`` mode for complex retrieval tasks.

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
        Model identifier forwarded to the endpoint.
    invoke_url : str
        Full ``/v1/chat/completions`` endpoint URL.
    top_k : int
        Number of documents to select per query.  Defaults to ``5``.
    api_key : str, optional
        Literal API key **or** an ``"os.environ/VAR_NAME"`` reference.
    max_tokens : int, optional
        Upper bound on tokens in each LLM response.
    max_steps : int
        Maximum agentic loop iterations per query.  Defaults to ``10``.
    extended_relevance : bool
        When ``True``, include the ``<RELEVANCE_DEFINITION>`` block in the
        system prompt for tasks with non-standard relevance definitions.
        Defaults to ``False``.
    system_prompt_override : str, optional
        Fully custom system prompt.  Use ``{top_k}`` as a placeholder.
    text_truncation : int
        Maximum characters of each document's text shown to the LLM.
        Defaults to ``2000``.
    base_url : str, optional
        Deprecated alias for ``invoke_url``.  Prefer ``invoke_url``.
    """

    _NVIDIA_BUILD_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        *,
        llm_model: str,
        invoke_url: Optional[str] = None,
        top_k: int = 5,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_steps: int = 10,
        extended_relevance: bool = False,
        system_prompt_override: Optional[str] = None,
        text_truncation: int = 2000,
        parallel_tool_calls: bool = True,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._top_k = top_k
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._max_steps = max_steps
        self._extended_relevance = extended_relevance
        self._system_prompt_override = system_prompt_override
        self._text_truncation = text_truncation
        self._parallel_tool_calls = parallel_tool_calls

        if invoke_url is not None:
            self._invoke_url = invoke_url
        elif base_url is not None:
            import warnings

            warnings.warn(
                "SelectionAgentOperator: 'base_url' is deprecated, use 'invoke_url' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._invoke_url = base_url.rstrip("/") + "/v1/chat/completions"
        else:
            self._invoke_url = self._NVIDIA_BUILD_ENDPOINT

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
        rows: List[Dict[str, Any]] = []

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

    def _resolve_api_key(self) -> Optional[str]:
        api_key = self._api_key
        if api_key is not None and api_key.strip().startswith("os.environ/"):
            var = api_key.strip().removeprefix("os.environ/")
            value = os.environ.get(var)
            if value is None:
                raise ValueError(
                    f"Environment variable '{var}' is not set. "
                    f"Set it with: export {var}=<your-api-key>"
                )
            return value
        return api_key

    def _build_system_prompt(self, top_k: int) -> str:
        if self._system_prompt_override:
            return self._system_prompt_override.format(top_k=top_k)
        return _render_selection_prompt(top_k, extended_relevance=self._extended_relevance)

    def _build_tools(self, top_k: int, valid_doc_ids: List[str]) -> List[Dict[str, Any]]:
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
                                "description": "A brief explanation of your selection and the relevance ordering.",
                            },
                        },
                    },
                },
            },
        ]

    def _build_user_message(self, query_text: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format query + candidate documents as a multi-part user message."""
        content: List[Dict[str, Any]] = [
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
        docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the agentic selection loop for a single query."""
        valid_ids = list(dict.fromkeys(d["id"] for d in docs))
        feasible_k = min(self._top_k, len(valid_ids))

        system_prompt = self._build_system_prompt(feasible_k)
        tools = self._build_tools(feasible_k, valid_ids)
        valid_id_set = set(valid_ids)
        api_key = self._resolve_api_key()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            self._build_user_message(query_text, docs),
        ]

        extra_body: Dict[str, Any] = {}
        if not self._parallel_tool_calls:
            extra_body["parallel_tool_calls"] = False

        for _step in range(self._max_steps):
            try:
                response = invoke_chat_completion_step(
                    invoke_url=self._invoke_url,
                    messages=messages,
                    model=self._llm_model,
                    api_key=api_key,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self._max_tokens,
                    extra_body=extra_body or None,
                )
            except Exception as exc:
                logger.warning(
                    "SelectionAgentOperator: LLM call failed on step %d for query %r: %s", _step, query_text, exc
                )
                break

            if not response.get("choices"):
                logger.warning("SelectionAgentOperator: empty choices in API response on step %d", _step)
                break
            choice = response["choices"][0]
            msg = choice["message"]
            finish_reason = choice.get("finish_reason")

            # Append the assistant turn to history
            assistant_turn: Dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                assistant_turn["content"] = msg["content"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant_turn["tool_calls"] = tool_calls
            messages.append(assistant_turn)

            if finish_reason == "stop" or not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Please call log_selected_documents to report your final selection.",
                    }
                )
                continue

            tool_messages: List[Dict[str, Any]] = []
            should_end = False
            end_kwargs: Dict[str, Any] = {}

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                fn = tc.get("function", {})
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": "Error: could not parse tool arguments."}
                    )
                    continue

                if fn.get("name") == "think":
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": "Your thought has been logged."}
                    )

                elif fn.get("name") == "log_selected_documents":
                    raw_doc_ids = fn_args.get("doc_ids", [])
                    if isinstance(raw_doc_ids, str):
                        try:
                            raw_doc_ids = json.loads(raw_doc_ids)
                        except json.JSONDecodeError:
                            raw_doc_ids = []
                    doc_ids = [d for d in raw_doc_ids if d in valid_id_set][:feasible_k]
                    if not doc_ids and raw_doc_ids:
                        logger.warning(
                            "SelectionAgentOperator: LLM returned %d doc_id(s) for query %r "
                            "but none matched the candidate set — possible hallucination. "
                            "Returned IDs: %s",
                            len(raw_doc_ids),
                            query_text,
                            raw_doc_ids[:10],
                        )
                    end_kwargs = {"doc_ids": doc_ids, "message": fn_args.get("message", "")}
                    should_end = True

                else:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": f"Error: unknown tool '{fn.get('name')}'.",
                        }
                    )

            if should_end:
                return end_kwargs

            messages.extend(tool_messages)

        return {
            "doc_ids": [],
            "message": "Selection agent reached max steps without completing.",
        }
