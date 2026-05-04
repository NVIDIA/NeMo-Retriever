import logging
from datetime import datetime
import time

from langchain_core.messages import HumanMessage, SystemMessage
from nemo_retriever.tabular_data.retrieval.text_to_sql.text_to_sql_graph import create_graph
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentPayload, AgentState
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import main_system_prompt_template
from nemo_retriever.tabular_data.retrieval.text_to_sql.utils import get_llm_client

logger = logging.getLogger(__name__)

try:
    llm_client = get_llm_client()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None

graph = create_graph()
app = graph.compile()


def get_agent_response(payload: AgentPayload):
    t0 = time.perf_counter()
    acronyms = payload.get("acronyms", "")
    custom_prompts = payload.get("custom_prompts", "")
    connector = payload.get("connector")

    acronyms_text = f"Acronyms:\n{acronyms}\n\n" if acronyms else ""
    custom_prompts_text = f"{custom_prompts}\n\n" if custom_prompts else ""

    main_system_prompt = main_system_prompt_template.format(
        date=datetime.now(),
        acronyms=acronyms_text,
        custom_prompts=custom_prompts_text,
        dialect=connector.dialect,
    )
    messages = [
        SystemMessage(content=main_system_prompt),
        HumanMessage(content=payload["question"]),
    ]

    initial_path_state = dict(payload.get("path_state") or {})

    retriever = payload.get("retriever")
    if retriever is None:
        raise ValueError(
            "AgentPayload is missing required 'retriever' (nemo_retriever.retriever.Retriever "
            "instance). Construct a Retriever once at startup and pass it in the payload."
        )

    state: AgentState = {
        "llm": llm_client,
        "initial_question": payload["question"],
        "connector": connector,
        "messages": messages,
        "path_state": initial_path_state,
        "retriever": retriever,
        "decision": "",
    }

    final_state = state.copy()
    for step in app.stream(state, config={"recursion_limit": 45}):
        logger.info("--- AGENT STEP ---")
        for node_name, node_output in step.items():
            logger.info("Node: %s", node_name)
            if node_output:
                if "path_state" in node_output:
                    if "path_state" not in final_state:
                        final_state["path_state"] = {}
                    final_state["path_state"].update(node_output["path_state"])
                for key, value in node_output.items():
                    if key != "path_state":
                        final_state[key] = value

    path_state = final_state.get("path_state", {})
    final_response = path_state.get("final_response")
    if final_response is None:
        messages_out = final_state.get("messages", [])
        if messages_out:
            if isinstance(messages_out, dict):
                final_response = messages_out
            elif isinstance(messages_out[-1], dict):
                final_response = messages_out[-1]
            else:
                final_response = str(messages_out[-1])
        else:
            final_response = ""

    if isinstance(final_response, dict):
        answer = final_response
    else:
        answer = {"response": str(final_response)}

    elapsed = time.perf_counter() - t0
    logger.info("Final answer to user (%.2fs):\n%s", elapsed, answer)
    return answer


__all__ = ["get_agent_response", "app", "graph", "llm_client"]
