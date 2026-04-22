# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator that runs a ReAct agentic retrieval loop per query."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.nim.chat_completions import invoke_chat_completion_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt rendering  (verbatim content of 02_v1.j2, rendered via Python)
# ---------------------------------------------------------------------------

_GOAL = """\
You are a retrieval agent that finds all documents related to a given query.

<Goal>
You are given a search query and a list of documents retrieved for that query. Your task is to write new \
queries and use the given search tool to find *ALL* the related and somewhat related documents to the given \
query (i.e., maximize recall).
If the user's query is a question, you should not answer the question yourself. Instead, you should find \
the related documents for the given query.
</Goal>"""

_RELEVANCE_DEFINITION = """

<RELEVANCE_DEFINITION>
- You should be careful, in the context of this task, what it means to be a "query", "document", and \
"relevant" can sometimes be very complex and might not follow the traditional definition of these terms \
in standard information retrieval.
- In standard retrieval, a query is usually a user question (like a web search query), the document is \
some sort of content that provides information (e.g., a web page), and these two are considered relevant if \
the document provides information that answers the user's query.
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
- You should analyze the query and some of the available documents. And then reason about what could be a \
meaningful definition of relevance in this case, and what the user could be looking for.
- Moreover, sometimes, the query could be even a prompt that is given to a Large Language Model (LLM) and \
the user wants to find the useful documents for the LLM that help answering/solving this prompt.
</RELEVANCE_DEFINITION>"""

_WORKFLOW_TEMPLATE = """
<WORKFLOW>
- You are given a retrieval tool, powered by a dense embedding model, that takes a text query and returns \
the most similar documents.
{extended_relevance_line}\
- You can call the search tool multiple times.
- Search for related documents to the user's query from different angles.
- If needed, revise your search queries based on the documents you find in previous steps.
- Once you are confident that you have found all the related and somewhat related documents and there are \
no more related documents in the corpus, call the "final_results" tool to finish the task.
{enforce_top_k_line}\
- When calling the "final_results" tool, the list of documents must be sorted in the decreasing level of \
relevance to the query. I.e., the first document is the most relevant to the query, the second document is \
the second most relevant to the query, and so on.
</WORKFLOW>"""

_BEST_PRACTICES_TEMPLATE = """

<BEST_PRACTICES>
- You should be thorough and find all related and somewhat related documents.
- The goal is to increase the **Recall** of your search attempt. So, if multiple documents are relevant \
to the given query, you should find and report all of them even if only a subset of them is enough \
for answering the query.
{with_init_docs_line}\
</BEST_PRACTICES>"""


def _render_react_agent_prompt(
    top_k: int,
    *,
    with_init_docs: bool = True,
    enforce_top_k: bool = True,
    extended_relevance: bool = False,
) -> str:
    """Render the ReAct agent system prompt (verbatim 02_v1.j2 logic)."""
    parts = [_GOAL]
    if extended_relevance:
        parts.append(_RELEVANCE_DEFINITION)

    ext_line = (
        "- As explained above, reason and figure out what the meaning of relevance is in this case, "
        "and what could be relevant and useful information for the given query.\n"
        if extended_relevance
        else ""
    )
    enforce_line = (
        f'- When calling "final_results", you must select exactly the {top_k} most relevant documents '
        "among all documents you have retrieved.\n"
        if enforce_top_k
        else ""
    )
    parts.append(
        _WORKFLOW_TEMPLATE.format(
            extended_relevance_line=ext_line,
            enforce_top_k_line=enforce_line,
        )
    )

    init_docs_line = (
        "- **TIP**: you can look at the list of documents retrieved using the original query and think "
        "what other queries you can use to find the potentially related documents that are missing in these results.\n"
        if with_init_docs
        else ""
    )
    parts.append(_BEST_PRACTICES_TEMPLATE.format(with_init_docs_line=init_docs_line))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tool specs  (verbatim from retrieval_bench/nemo_agentic/tool_helpers.py)
# ---------------------------------------------------------------------------


def _make_think_tool_spec(extended_relevance: bool = False) -> Dict[str, Any]:
    ext = ""
    if extended_relevance:
        ext = (
            "- When it is difficult to understand what is the intent of the user and what they are trying "
            "to find with this query, use this tool to think about potential definitions of relevance that "
            "could be meaningful/useful to the user for this task.\n"
            "- If the intention of the user is vague especially given the available documents, use this tool "
            "to think how you should decide what documents are relevant and what the metric of relevance is.\n"
        )
    description = (
        "Use the tool to think about something. It will not obtain new information or make any changes, "
        "but just log the thought. Use it when complex reasoning or brainstorming is needed.\n\n"
        "Common use cases:\n"
        f"{ext}"
        "- When processing a complex query, use this tool to organize your thoughts and think about "
        "the sub queries that you need to search for to find the relevant information\n"
        "- If a query is vague is very difficult to find information for it, you can use this tool to think "
        "about clues in the query that you can use to narrow down the search and spot relevant pieces of information.\n"
        "- When finding related documents that help you create better search queries in the next step, use this "
        "tool to think about what pieces of information from these documents are helpful to search for.\n"
        "- When you fail to find any related information to the query, use this tool to think about other "
        "search strategies that you can take to retrieve the related documents\n\n"
        "The tool simply logs your thought process for better transparency and does not make any changes."
    )
    return {
        "type": "function",
        "function": {
            "name": "think",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string", "description": "The thought to log."}},
                "required": ["thought"],
            },
        },
    }


def _make_retrieve_tool_spec(top_k: int) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search for documents relevant to the given query using a dense embedding retrieval system. "
                "Returns the most semantically similar documents from the corpus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to retrieve documents for.",
                    },
                },
                "required": ["query"],
            },
        },
    }


def _make_final_results_tool_spec(top_k: Optional[int]) -> Dict[str, Any]:
    tk_ins = ""
    if top_k is not None:
        tk_ins = f"- You must choose exactly {top_k} document IDs when calling this function.\n"

    description = (
        "Signals the completion of the search process for the current query.\n\n"
        "Use this tool when:\n"
        "- You have found all the relevant documents to the query.\n"
        "- Despite several attempts, you cannot find good documents for the given query.\n\n"
        "The message should include:\n"
        "- A brief summary of your exploration and the results\n"
        "- Explanation if the search was unsuccessful\n\n"
        "When reporting the selected document IDs, make sure:\n"
        "- the list of document IDs is sorted in the decreasing level of relevance to the query. "
        "I.e., the first document in the list is the most relevant to the query, the second is the "
        "second most relevant to the query, and so on.\n"
        f"{tk_ins}"
        "\nThe successful_search field should be set to true if you believed you have found the most "
        "relevant documents to the user's query, and false otherwise. And partial if it is in between."
    )
    return {
        "type": "function",
        "function": {
            "name": "final_results",
            "description": description,
            "parameters": {
                "type": "object",
                "required": ["doc_ids", "message", "search_successful"],
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "A message for the user to explain why you think you found all the related "
                            "documents and there is no related document is missing. Also, include a short "
                            "description of your exploration process. If your attempts to find related "
                            "documents were unsuccessful, explain why."
                        ),
                    },
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of document IDs that are relevant to the user's query sorted descending "
                            "by their level of relevance to the user's query. I.e., the first document is "
                            "the most relevant to the query, the second is the second most relevant to the "
                            "query, and so on."
                        ),
                    },
                    "search_successful": {
                        "type": "string",
                        "enum": ["true", "false", "partial"],
                        "description": "Whether you managed to find all the related documents to the query.",
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

#: Message sent when the LLM produces a stop without calling any tool.
_AUTO_USER_MSG = (
    "continue with the task. Do not re-read the query. Do not summarize your progress. "
    "If you believe you have done all the required steps, call the `final_results` tool"
)


class ReActAgentOperator(AbstractOperator, CPUOperator):
    """Run an iterative ReAct retrieval loop per query and emit the full retrieval log.

    Each query row is processed independently by an LLM-driven ReAct loop
    (Reason + Act) that has access to three tools: ``think``, ``retrieve``,
    and ``final_results``.  The operator emits one output row per retrieved
    document per retrieval step, enabling downstream
    :class:`RRFAggregatorOperator` to fuse the ranked lists with Reciprocal
    Rank Fusion.

    The system prompt is a verbatim Python rendering of the retrieval-bench
    ``02_v1.j2`` template, including optional ``extended_relevance`` and
    ``enforce_top_k`` blocks.

    Input DataFrame schema
    ----------------------
    query_id   : str  — unique query identifier
    query_text : str  — the search query text
    (additional columns are ignored)

    Output DataFrame schema
    -----------------------
    query_id   : str  — same ``query_id`` as the input
    query_text : str  — same ``query_text`` (passed through for downstream)
    step_idx   : int  — 0 = initial seed retrieval; 1 … N = per-loop retrieve calls
    doc_id     : str  — retrieved document identifier
    text       : str  — document text
    rank       : int  — 1-indexed rank within this step (1 = most relevant)

    Parameters
    ----------
    invoke_url : str
        Full ``/v1/chat/completions`` endpoint URL.
    llm_model : str
        Model identifier forwarded to the endpoint.
    retriever_fn : Callable[[str, int], list[dict]]
        ``(query_text, top_k) → [{doc_id: str, text: str, ...}]``.
        The callable is invoked for every retrieve tool call the agent makes.
        Each returned dict must contain ``doc_id`` and ``text`` keys.
    retriever_top_k : int
        Number of documents fetched per retrieve call.  Defaults to ``500``.
    target_top_k : int
        Number of final documents to select, communicated to the LLM via the
        system prompt and ``final_results`` tool spec.  Defaults to ``10``.
    enforce_top_k : bool
        When ``True``, the system prompt instructs the LLM to select exactly
        ``target_top_k`` documents in its ``final_results`` call.
        Defaults to ``True``.
    user_msg_type : {"with_results", "simple"}
        ``"with_results"`` (default): make one upfront retrieval call with the
        original query and include those documents in the first user message,
        mirroring the retrieval-bench ``with_results`` mode.
        ``"simple"``: start the loop with just the query text.
    extended_relevance : bool
        Include the ``<RELEVANCE_DEFINITION>`` block in the system prompt for
        tasks with non-standard relevance definitions.  Defaults to ``False``.
    max_steps : int
        Maximum ReAct loop iterations per query before forced exit.
        Defaults to ``10``.
    num_concurrent : int
        Number of queries processed concurrently via ``ThreadPoolExecutor``.
        Defaults to ``8``.
    api_key : str, optional
        Literal API key **or** an ``"os.environ/VAR_NAME"`` reference.
    max_tokens : int, optional
        Upper bound on tokens in each LLM response.

    Notes
    -----
    ``retriever_fn`` must be serialisable when used with ``RayDataExecutor``.
    Prefer module-level functions or picklable callable objects over lambdas.

    Examples
    --------
    ::

        from nemo_retriever.graph.react_agent_operator import ReActAgentOperator
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator
        from nemo_retriever.graph.executor import InprocessExecutor

        def my_retriever(query_text: str, top_k: int) -> list[dict]:
            # Returns [{doc_id, text, score?}, ...]
            ...

        pipeline = (
            ReActAgentOperator(
                invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
                llm_model="nvidia/llama-3.3-nemotron-super-49b-v1",
                retriever_fn=my_retriever,
                retriever_top_k=500,
                target_top_k=10,
            )
            >> RRFAggregatorOperator(k=60)
            >> SelectionAgentOperator(
                invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
                llm_model="nvidia/llama-3.3-nemotron-super-49b-v1",
                top_k=10,
            )
        )

        result_df = InprocessExecutor(pipeline).ingest(query_df)
    """

    _NVIDIA_BUILD_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        *,
        invoke_url: Optional[str] = None,
        llm_model: str,
        retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
        retriever_top_k: int = 500,
        target_top_k: int = 10,
        enforce_top_k: bool = True,
        user_msg_type: Literal["with_results", "simple"] = "with_results",
        extended_relevance: bool = False,
        max_steps: int = 10,
        num_concurrent: int = 8,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        parallel_tool_calls: bool = True,
    ) -> None:
        super().__init__()
        self._invoke_url = invoke_url or self._NVIDIA_BUILD_ENDPOINT
        self._llm_model = llm_model
        self._retriever_fn = retriever_fn
        self._retriever_top_k = retriever_top_k
        self._target_top_k = target_top_k
        self._enforce_top_k = enforce_top_k
        self._user_msg_type = user_msg_type
        self._extended_relevance = extended_relevance
        self._max_steps = max_steps
        self._num_concurrent = num_concurrent
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._parallel_tool_calls = parallel_tool_calls

    # ------------------------------------------------------------------
    # AbstractOperator interface
    # ------------------------------------------------------------------

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"ReActAgentOperator expects a pd.DataFrame, got {type(data).__name__!r}.")
        required = {"query_id", "query_text"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required column(s): {sorted(missing)}. " f"Expected: {sorted(required)}."
            )
        return data[["query_id", "query_text"]].copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Run the ReAct loop for each query, concurrently up to num_concurrent."""
        api_key = self._resolve_api_key()
        rows: List[Dict[str, Any]] = []

        query_rows = [(str(r["query_id"]), str(r["query_text"])) for _, r in data.iterrows()]

        if len(query_rows) == 1:
            # Fast path: single query, no threading overhead
            qid, qtxt = query_rows[0]
            rows.extend(self._run_single_query(qid, qtxt, api_key))
        else:
            with ThreadPoolExecutor(max_workers=min(self._num_concurrent, len(query_rows))) as executor:
                futures = {
                    executor.submit(self._run_single_query, qid, qtxt, api_key): (qid, qtxt) for qid, qtxt in query_rows
                }
                for future in as_completed(futures):
                    try:
                        rows.extend(future.result())
                    except Exception as exc:
                        qid, qtxt = futures[future]
                        logger.warning("ReActAgentOperator: query %r failed: %s", qid, exc, exc_info=True)

        if not rows:
            return pd.DataFrame(columns=["query_id", "query_text", "step_idx", "doc_id", "text", "rank"])

        return pd.DataFrame(rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    # ------------------------------------------------------------------
    # Internal: single query ReAct loop
    # ------------------------------------------------------------------

    def _run_single_query(
        self,
        query_id: str,
        query_text: str,
        api_key: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run the full ReAct loop for one query; return a list of row dicts."""
        with_init_docs = self._user_msg_type == "with_results"

        system_prompt = _render_react_agent_prompt(
            self._target_top_k,
            with_init_docs=with_init_docs,
            enforce_top_k=self._enforce_top_k,
            extended_relevance=self._extended_relevance,
        )
        tools = [
            _make_think_tool_spec(self._extended_relevance),
            _make_retrieve_tool_spec(self._retriever_top_k),
            _make_final_results_tool_spec(self._target_top_k if self._enforce_top_k else None),
        ]

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Retrieval log: one list per step, each item is {doc_id, text, score?}
        retrieval_log: List[List[Dict[str, Any]]] = []
        seen_doc_ids: set[str] = set()
        step_counter = 0

        # ------ optional initial retrieval (with_results mode) ------
        if with_init_docs:
            init_docs = self._call_retriever(query_text, seen_doc_ids, api_key)
            retrieval_log.append(init_docs)
            step_counter += 1
            for d in init_docs:
                seen_doc_ids.add(d["doc_id"])

            doc_content = _docs_to_message_content(init_docs)
            user_msg_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"Query:\n{query_text}\n\nRetrieved Documents:"}
            ] + doc_content
            messages.append({"role": "user", "content": user_msg_content})
        else:
            messages.append({"role": "user", "content": f"Query:\n{query_text}"})

        final_doc_ids: Optional[List[str]] = None

        # ------ main ReAct loop ------
        for _step in range(self._max_steps):
            logger.debug("query=%r loop_step=%d seen_docs=%d", query_id, _step, len(seen_doc_ids))
            try:
                response = invoke_chat_completion_step(
                    invoke_url=self._invoke_url,
                    messages=messages,
                    model=self._llm_model,
                    api_key=api_key,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self._max_tokens,
                    extra_body={"parallel_tool_calls": False} if not self._parallel_tool_calls else None,
                )
            except Exception as exc:
                logger.warning("ReActAgentOperator: LLM call failed on step %d for query %r: %s", _step, query_id, exc, exc_info=True)
                break

            if not response.get("choices"):
                logger.warning("ReActAgentOperator: empty choices in API response on step %d for query %r", _step, query_id)
                break
            choice = response["choices"][0]
            msg = choice["message"]
            finish_reason = choice.get("finish_reason")
            tool_calls = msg.get("tool_calls") or []

            # Append assistant turn
            assistant_turn: Dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                assistant_turn["content"] = msg["content"]
            if tool_calls:
                assistant_turn["tool_calls"] = tool_calls
            messages.append(assistant_turn)

            if finish_reason == "stop" or not tool_calls:
                messages.append({"role": "user", "content": _AUTO_USER_MSG})
                continue

            tool_messages: List[Dict[str, Any]] = []
            loop_done = False

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": "Error: could not parse tool arguments."}
                    )
                    continue

                if fn_name == "think":
                    logger.debug("query=%r step=%d [think] %s", query_id, _step, str(fn_args.get("thought", ""))[:120])
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": "Your thought has been logged."}
                    )

                elif fn_name == "retrieve":
                    subquery = str(fn_args.get("query", query_text))
                    logger.debug("query=%r step=%d [retrieve] subquery=%r", query_id, _step, subquery)
                    retrieved = self._call_retriever(subquery, seen_doc_ids, api_key)
                    logger.debug("query=%r step=%d [retrieve] got %d new docs", query_id, _step, len(retrieved))
                    retrieval_log.append(retrieved)
                    step_counter += 1
                    for d in retrieved:
                        seen_doc_ids.add(d["doc_id"])
                    doc_content = _docs_to_message_content(retrieved)
                    tool_content: List[Dict[str, Any]] = [
                        {"type": "text", "text": f"Retrieved {len(retrieved)} documents:"}
                    ] + doc_content
                    tool_messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_content})

                elif fn_name == "final_results":
                    raw_ids: List[str] = fn_args.get("doc_ids", [])
                    logger.debug("query=%r step=%d [final_results] doc_ids=%s", query_id, _step, raw_ids)
                    if isinstance(raw_ids, list) and raw_ids:
                        final_doc_ids = [str(d) for d in raw_ids]
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": "The results have been successfully logged and the interaction ended.",
                        }
                    )
                    loop_done = True

                else:
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Error: unknown tool '{fn_name}'."}
                    )

            messages.extend(tool_messages)
            if loop_done:
                break

        return _build_output_rows(query_id, query_text, retrieval_log, final_doc_ids)

    def _call_retriever(
        self,
        query_text: str,
        seen_doc_ids: set[str],
        api_key: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Call retriever_fn, over-fetching to ensure new results after dedup."""
        fetch_k = self._retriever_top_k + len(seen_doc_ids)
        try:
            raw = self._retriever_fn(query_text, fetch_k)
        except Exception as exc:
            logger.warning("ReActAgentOperator: retriever_fn failed for query %r: %s", query_text, exc, exc_info=True)
            return []

        # Filter already-seen and normalise keys
        results: List[Dict[str, Any]] = []
        for item in raw:
            doc_id = str(item.get("doc_id", item.get("id", "")))
            text = str(item.get("text", ""))
            score = float(item.get("score", 0.0))
            if doc_id and doc_id not in seen_doc_ids:
                results.append({"doc_id": doc_id, "text": text, "score": score})
            if len(results) >= self._retriever_top_k:
                break

        return results

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _docs_to_message_content(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of doc dicts to LLM message content blocks."""
    content: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = doc.get("doc_id", "")
        text = doc.get("text", "").strip()
        entry: Dict[str, Any] = {"id": doc_id}
        if text:
            entry["text"] = text
        score = doc.get("score")
        if score is not None:
            entry["score"] = score
        content.append({"type": "text", "text": json.dumps(entry)})
    return content


def _build_output_rows(
    query_id: str,
    query_text: str,
    retrieval_log: List[List[Dict[str, Any]]],
    final_doc_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Convert the retrieval log to one row per (step_idx, rank, doc_id)."""
    rows: List[Dict[str, Any]] = []
    for step_idx, step_docs in enumerate(retrieval_log):
        for rank, doc in enumerate(step_docs, 1):
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "step_idx": step_idx,
                    "doc_id": doc.get("doc_id", ""),
                    "text": doc.get("text", ""),
                    "rank": rank,
                }
            )

    # If final_results was called, also emit those as a synthetic final step
    # (step_idx = len(retrieval_log)) so RRF can optionally weight it.
    # These are already covered by the existing steps, so we skip deduplication
    # here — RRF will naturally up-weight docs that appeared in final_results
    # because they were retrieved in earlier steps.
    return rows
