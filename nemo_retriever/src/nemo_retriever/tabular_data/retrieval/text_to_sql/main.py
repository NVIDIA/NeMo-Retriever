# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from datetime import datetime
from typing import Generator

from langchain_core.messages import HumanMessage, SystemMessage

from nemo_retriever.tabular_data.retrieval.text_to_sql.text_to_sql_graph import create_graph
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentPayload, AgentState
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import main_system_prompt_template
from nemo_retriever.tabular_data.retrieval.data_access.custom_analyses import fetch_custom_analyses
from nemo_retriever.tabular_data.retrieval.llm_invoke import get_llm_client

logger = logging.getLogger(__name__)

try:
    llm_client = get_llm_client()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None

graph = create_graph()
app = graph.compile()


def _build_state(payload: AgentPayload) -> AgentState:
    custom_prompts = payload.get("custom_prompts", "")
    acronyms = payload.get("acronyms", [])
    connector = payload.get("connector")
    if connector is None:
        raise ValueError(
            "AgentPayload is missing required 'connector'. "
            "Provide a database connector with a valid 'dialect' attribute."
        )

    retriever = payload.get("retriever")
    if retriever is None:
        raise ValueError(
            "AgentPayload is missing required 'retriever' (nemo_retriever.retriever.Retriever "
            "instance). Construct a Retriever once at startup and pass it in the payload."
        )

    custom_prompts_text = f"{custom_prompts}\n\n" if custom_prompts else ""
    domain_rules = fetch_custom_analyses() + list(acronyms or [])

    initial_path_state = dict(payload.get("path_state") or {})

    main_system_prompt = main_system_prompt_template.format(
        date=datetime.now(),
        custom_prompts=custom_prompts_text,
        dialect=connector.dialect,
    )
    messages = [
        SystemMessage(content=main_system_prompt),
        HumanMessage(content=payload["question"]),
    ]

    return {
        "llm": llm_client,
        "initial_question": payload["question"],
        "connector": connector,
        "messages": messages,
        "path_state": initial_path_state,
        "retriever": retriever,
        "decision": "",
        "domain_rules": domain_rules,
    }


def _extract_answer(final_state: dict) -> dict:
    path_state = final_state.get("path_state", {})
    final_response = path_state.get("final_response")

    if final_response is None:
        messages_out = final_state.get("messages") or []
        if isinstance(messages_out, list) and messages_out:
            final_response = messages_out[-1]
        else:
            final_response = ""

    if isinstance(final_response, dict):
        return final_response
    return {"response": str(final_response)}


def stream_agent_response(
    payload: AgentPayload,
) -> Generator[dict, None, None]:
    """Yield ``{"type": "step", "node": ...}`` for each graph node,
    then ``{"type": "result", "answer": ...}`` with the final answer.
    On error yields ``{"type": "error", "message": ...}``."""
    t0 = time.perf_counter()

    state = _build_state(payload)
    final_state = dict(state)

    try:
        for step in app.stream(state, config={"recursion_limit": 45}):
            logger.info("--- AGENT STEP ---")
            for node_name, node_output in step.items():
                logger.info("Node: %s", node_name)
                yield {"type": "step", "node": node_name}

                if node_output:
                    if "path_state" in node_output:
                        if "path_state" not in final_state:
                            final_state["path_state"] = {}
                        final_state["path_state"].update(node_output["path_state"])
                    for key, value in node_output.items():
                        if key != "path_state":
                            final_state[key] = value

        answer = _extract_answer(final_state)
        elapsed = time.perf_counter() - t0
        logger.info("Final answer (%.2fs):\n%s", elapsed, answer)
        yield {"type": "result", "answer": answer}

    except Exception as exc:
        logger.exception("Error during agent stream")
        yield {"type": "error", "message": f"Agent failed: {exc}"}


def get_agent_response(payload: AgentPayload) -> dict:
    """Non-streaming convenience wrapper around ``stream_agent_response``."""
    for event in stream_agent_response(payload):
        if event["type"] == "result":
            return event["answer"]
        if event["type"] == "error":
            raise RuntimeError(event["message"])
    return {"response": "SQL can't be constructed.", "sql_code": "", "result": None}


__all__ = [
    "get_agent_response",
    "stream_agent_response",
    "app",
    "graph",
    "llm_client",
]
