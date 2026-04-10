import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from nemo_retriever.tabular_data.retrieval.deep_agent.prompts import main_system_prompt_template
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import create_graph
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentPayload, AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.prompts import ONTOLOGY, get_ontology_prompt
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import _make_llm

logger = logging.getLogger(__name__)

try:
    llm_client = _make_llm()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None

graph = create_graph()
app = graph.compile()

# Graph PNG next to this module (omni_lite/omni_agent_graph.png)
_OMNI_LITE_DIR = Path(__file__).resolve().parent
graph_image_path = str(_OMNI_LITE_DIR / "omni_agent_graph.png")
Path(os.path.dirname(graph_image_path)).mkdir(parents=True, exist_ok=True)


def render_with_mmdc(out_path: str, mermaid_text: str) -> None:
    import pathlib
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mmd") as tmp:
        pathlib.Path(tmp.name).write_text(mermaid_text, encoding="utf-8")
        subprocess.run(
            [
                "mmdc",
                "-i",
                tmp.name,
                "-o",
                out_path,
                "-s",
                "4",
                "-w",
                "3200",
                "-H",
                "2400",
            ],
            check=True,
        )


try:
    import importlib

    importlib.import_module("pyppeteer")

    png_bytes = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER
    )  # pyright: ignore[reportUndefinedVariable]
    with open(graph_image_path, "wb") as f:
        f.write(png_bytes)
    logger.info("Graph visualization saved (local Pyppeteer)")
except Exception:
    try:
        mermaid_src = app.get_graph().draw_mermaid()
        render_with_mmdc(graph_image_path, mermaid_src)
        logger.info("Graph visualization saved (mmdc CLI)")
    except FileNotFoundError:
        logger.warning(
            "mmdc CLI not found. To install: npm install -g @mermaid-js/mermaid-cli"
        )
        logger.warning("Skipping graph visualization generation...")
    except Exception as e_cli:
        logger.warning("Failed to generate graph visualization with mmdc: %s", e_cli)
        logger.warning("Skipping graph visualization generation...")


def get_agent_response(payload: AgentPayload):
    now = datetime.now()
    main_system_prompt = main_system_prompt_template.format(
        date=now,
        ontology_prompt=get_ontology_prompt(ONTOLOGY),
        dialects=payload.get("dialects"),
    )
    messages = [SystemMessage(content=main_system_prompt)]
    if payload.get("history"):
        chat_history = ChatMessageHistory()
        for history in payload.get("history", []):
            chat_history.add_user_message(history["question"])
            chat_history.add_ai_message(history["response"])
        messages.extend(chat_history.messages)
    messages.append(HumanMessage(content=payload["question"]))

    initial_path_state = dict(payload.get("path_state") or {})

    state: AgentState = {
        "llm": llm_client,
        "initial_question": payload["question"],
        "dialects": payload.get("dialects"),
        "db_connector": payload.get("db_connector"),
        "messages": messages,
        "decision": "",
        "path_state": initial_path_state,
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

    logger.info("💬 Final answer to user:\n%s", answer)
    return answer


__all__ = ["get_agent_response", "app", "graph", "llm_client"]
