# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools that terminate the agent loop: the base plus the standard end tools.

:class:`FinalResults` ends an agent run with the chosen document IDs;
:class:`LogSelectedDocs` ends a selection-agent run with the top-k subset of
the run's candidate documents.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .base_tool import BaseTool, ToolError, tool_error_text


class BaseEndTool(BaseTool):
    """A tool whose successful call ends the agent run.

    The agent identifies end tools structurally (``isinstance(tool,
    BaseEndTool)``) and calls :meth:`try_end` instead of ``acall``: a valid
    call hands the agent the validated payload at call time; an invalid call
    produces the usual LLM-visible error text so the model retries.

    Subclasses implement :meth:`_validate_payload` with an explicit signature
    matching their spec (so unexpected LLM-supplied kwargs raise ``TypeError``
    naturally). Raise ``TypeError`` / :class:`ToolError` for invalid calls;
    return the normalized payload dict on success. The agent's result exposes
    the payload's ``doc_ids`` key as a typed convenience, with the full
    payload available verbatim.
    """

    success_message: str = "The results have been successfully logged and the interaction ended."

    @abstractmethod
    def _validate_payload(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate the end-call arguments; return the normalized payload."""
        raise NotImplementedError

    def try_end(self, **kwargs: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        """Return ``(payload, tool_message_text)``; payload is ``None`` for invalid calls."""
        try:
            payload = self._validate_payload(**kwargs)
        except (TypeError, ToolError) as e:
            return None, tool_error_text(self.name, e)
        return payload, self.success_message

    def salvage_payload(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Best-effort payload from an INVALID (non-terminating) end call.

        A fallback only: when a run ends in error without ever making a valid
        end call, the agent's last invalid attempt still carries the model's
        intended output. Returns a lenient subset — a non-empty ``doc_ids``
        list of strings and/or a non-empty ``answer`` string, plus a non-blank
        ``message`` — or ``None`` when nothing usable was supplied. The strict
        contract lives in :meth:`_validate_payload`; this never raises and is
        deliberately generic (each end tool only ever supplies its own keys).
        """
        out: Dict[str, Any] = {}
        doc_ids = kwargs.get("doc_ids")
        if isinstance(doc_ids, list) and doc_ids and all(isinstance(i, str) for i in doc_ids):
            out["doc_ids"] = list(doc_ids)
        answer = kwargs.get("answer")
        if isinstance(answer, str) and answer.strip():
            out["answer"] = answer
        # Only worth salvaging when there is a primary result; a bare
        # ``message`` with no doc_ids/answer carries nothing usable.
        if not out:
            return None
        message = kwargs.get("message")
        if isinstance(message, str) and message.strip():
            out["message"] = message
        return out

    def _call(self, **kwargs: Any) -> str:
        self._validate_payload(**kwargs)
        return self.success_message

    async def _acall(self, **kwargs: Any) -> str:
        return self._call(**kwargs)


class FinalResults(BaseEndTool):
    """Tool for logging selected document IDs and signaling the end of the interaction."""

    def __init__(self, top_k: Optional[int] = None, include_msg: bool = True):
        self.top_k = top_k
        self.include_msg = include_msg

        tk_ins = ""
        if top_k is not None:
            tk_ins = f"- You must choose exactly {top_k} document IDs when calling this function."

        required: List[str] = []
        properties: Dict[str, Any] = {}

        desc = """Signals the completion of the search process for the current query.

Use this tool when:
- You have found all the relevant documents to the query.
- Despite several attempts, you cannot find good documents for the given query.

"""
        if self.include_msg:
            desc += """The message should include:
- A brief summary of your exploration and the results
- Explanation if the search was unsuccessful

"""
        desc += f"""When reporting the selected document IDs, make sure:
- the list of document IDs is sorted in the decreasing level of relevance to the query. I.e., the first document in the list is the most relevant to the query, the second is the second most relevant to the query, and so on.
{tk_ins}

The successful_search field should be set to true if you believed you have found the most relevant documents to the user's query, and false otherwise. And partial if it is in between."""

        if self.include_msg:
            required.append("message")
            properties["message"] = {
                "type": "string",
                "description": "A message for the user to explain why you think you found all the related documents and there is no related document is missing. Also, include a short description of your exploration process. If your attempts to find related documents were unsuccessful, explain why.",
            }
        required.append("doc_ids")
        required.append("search_successful")
        properties["doc_ids"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of document IDs that are relevant to the user's query sorted descending by their level of relevance to the user's query. I.e., the first document is the most relevant to the query, the second is the second most relevant to the query, and so on.",
        }
        properties["search_successful"] = {
            "type": "string",
            "enum": ["true", "false", "partial"],
            "description": "Whether you managed to find all the related documents to the query.",
        }

        self.spec_dict = {
            "type": "function",
            "function": {
                "name": "final_results",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "required": required,
                    "properties": properties,
                },
            },
        }

    def _spec(self) -> dict:
        return self.spec_dict

    def _validate_payload(
        self, doc_ids: List[str], search_successful: str, message: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.include_msg:
            if message is None:
                raise TypeError("The `message` argument is required.")
            if not isinstance(message, str):
                raise TypeError(f"The `message` argument must be a string. Got `{type(message)}` type.")
        if not isinstance(doc_ids, list):
            raise TypeError(f"The `doc_ids` argument must be a list. Got `{type(doc_ids)}` type.")
        if len(doc_ids) == 0:
            raise ToolError("`doc_ids` cannot be empty. You must choose at least one relevant document.")
        if not all(isinstance(i, str) for i in doc_ids):
            raise TypeError("Items in `doc_ids` must be of type string (i.e., python's `str` type).")
        if not isinstance(search_successful, str):
            raise TypeError(f"The `search_successful` argument must be a string. Got `{type(search_successful)}` type.")
        if search_successful not in ["true", "false", "partial"]:
            raise ToolError(
                f"`search_successful` must be one of `true`, `false`, or `partial`. Got `{search_successful}` instead."
            )
        if self.top_k is not None and len(doc_ids) != self.top_k:
            raise ToolError(
                f"`doc_ids` must contain exactly {self.top_k} documents. But got {len(doc_ids)} document IDs instead."
            )
        payload: Dict[str, Any] = {"doc_ids": list(doc_ids), "search_successful": search_successful}
        if message is not None:
            payload["message"] = message
        return payload


class LogSelectedDocs(BaseEndTool):
    """Tool for reporting the selected document IDs and ending a selection run.

    This is the end tool of the :class:`~nemo_agent.selection_agent.SelectionAgent`.
    It is constructed per run, not per agent: the required count (``top_k``)
    and the allowed ids (``candidate_docids``) depend on the run's candidate
    documents, and per-run construction keeps concurrent runs on one agent
    instance isolated.
    """

    def __init__(self, top_k: int, candidate_docids: List[str], include_msg: bool = True):
        self.top_k = int(top_k)
        self.allowed_doc_ids = {str(i) for i in candidate_docids}
        self.include_msg = include_msg

        desc = (
            "Records the selected documents and signals the end of the task.\n\n"
            "Use this tool when you have carefully considered the candidate "
            f"documents and have selected exactly {self.top_k} of the most "
            "relevant documents to the query.\n\n"
        )
        if self.include_msg:
            desc += (
                "The message argument should explain your reasoning and "
                "justification for selecting this specific set of documents as "
                "the most relevant to the query.\n\n"
            )
        desc += (
            "**Note**: the list of document IDs passed as the `doc_ids` "
            "argument must be sorted in the decreasing level of relevance. "
            "In other words, the first document in `doc_ids` list is the "
            "most relevant to the query, the second document is the second "
            "most relevant document, and so on."
        )

        required: List[str] = []
        properties: Dict[str, Any] = {}
        if self.include_msg:
            required.append("message")
            properties["message"] = {
                "type": "string",
                "description": "A message for the user to explain why you think the selected are the most relevant to the query. Also, explain why this specific order of document IDs satisfies the most to least relevant ordering criteria.",
            }
        required.append("doc_ids")
        properties["doc_ids"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                f"The IDs of the {self.top_k} most relevant documents to the given query. "
                "The IDs must be sorted in the decreasing "
                "level of relevance. I.e., the first document must be the "
                "most relevant to the query."
            ),
        }

        self.spec_dict = {
            "type": "function",
            "function": {
                "name": "log_selected_documents",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "required": required,
                    "properties": properties,
                },
            },
        }

    def _spec(self) -> dict:
        return self.spec_dict

    def _validate_payload(self, doc_ids: List[str], message: Optional[str] = None) -> Dict[str, Any]:
        if self.include_msg:
            if message is None:
                raise TypeError("The `message` argument is required.")
            if not isinstance(message, str):
                raise TypeError(f"The `message` argument must be a string. Got `{type(message)}` type.")
        if not isinstance(doc_ids, list):
            raise TypeError(f"The `doc_ids` argument must be a list. Got `{type(doc_ids)}` type.")
        if len(doc_ids) != self.top_k:
            raise ToolError(f"You must select exactly {self.top_k} documents. Got {len(doc_ids)} documents.")
        if not all(isinstance(i, str) for i in doc_ids):
            raise TypeError("Items in `doc_ids` must be of type string (i.e., python's `str` type).")
        for i in doc_ids:
            if i not in self.allowed_doc_ids:
                raise ToolError(f"Document with ID `{i}` is not among the candidate documents.")
        payload: Dict[str, Any] = {"doc_ids": list(doc_ids)}
        if message is not None:
            payload["message"] = message
        return payload
