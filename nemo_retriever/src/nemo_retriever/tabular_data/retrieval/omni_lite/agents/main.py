import os
import logging
from datetime import datetime
from pathlib import Path
import subprocess
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import _make_llm
from nemo_retriever.tabular_data.retrieval.omni_lite.graph import AgentPayload, create_graph



logger = logging.getLogger("langchain agent main")
# Get LLM client with automatic provider selection and fallback (LangChain)

try:
    llm_client = _make_llm()
except ValueError as e:
    logger.error(f"Failed to initialize LLM client: {e}")
    llm_client = None

graph = create_graph()
app = graph.compile()

if os.environ["LMX_ENV"] == "development":
    # Generate graph visualization using local rendering method
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_image_path = os.path.abspath(
        os.path.join(current_dir, "..", "omni_agent_graph.png")
    )
    Path(os.path.dirname(graph_image_path)).mkdir(parents=True, exist_ok=True)

    def render_with_mmdc(out_path: str, mermaid_text: str):
        # Use mmdc CLI to render mermaid diagrams (no-browser fallback)
        # Use scale factor of 4 and larger dimensions for high resolution
        import tempfile
        import pathlib

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
                    "4",  # Scale factor of 4 for high resolution
                    "-w",
                    "3200",  # Width of 3200px
                    "-H",
                    "2400",  # Height of 2400px
                ],
                check=True,
            )

    # Try pyppeteer first, but fall back to mmdc if it fails
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
        # Pyppeteer failed or not available, try mmdc CLI fallback
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
            logger.warning(f"Failed to generate graph visualization with mmdc: {e_cli}")
            logger.warning("Skipping graph visualization generation...")


def get_agent_response(account_id: str, payload: AgentPayload):
    now = datetime.now()
    ontology = get_account_ontology(account_id, user_participants)
    main_system_prompt = main_system_prompt_template.format(
        date=now,
        ontology_prompt=get_ontology_prompt(ontology),
    )
    messages = [SystemMessage(content=main_system_prompt)]
    if payload.get("history"):
        chat_history = ChatMessageHistory()
        for history in payload.get("history", []):
            chat_history.add_user_message(history["question"])
            chat_history.add_ai_message(history["response"])
        messages.extend(chat_history.messages)
    messages.append(HumanMessage(content=payload["question"]))
    dialects = get_dialects_by_account_id(account_id, is_omni=True)
    state: AgentState = {
        "pg_connection": get_postgres_conn(),
        "llm": llm_client,
        "account_id": account_id,
        "user_participants": user_participants,
        "auto_choose": payload.get("auto_choose_option"),
        "pii_access": omni_settings[OmniSettings.PII_ACCESS],
        "dialects": dialects,
        "is_technical": payload.get("is_technical"),
        "source": payload.get("source"),
        "initial_question": payload["question"],
        "messages": messages,
        "decision": "decide",
        "intermediate_output": "",  # Deprecated - use path_state instead
        "thoughts": "",
        "path_state": {
            "action": payload.get("action"),
            "candidate_id": payload.get("candidate_id"),
            "candidate_name": payload.get("candidate_name"),
            "candidate_label": payload.get("candidate_label"),
            "candidate_zone_id": payload.get("candidate_zone_id"),
            "initial_response": payload.get("initial_response"),
            "unstructured": payload.get("unstructured"),
            "not_to_check_in_files": payload.get("not_to_check_in_files", False),
        },
    }
    # Build file name mapping once and store in state
    file_names_to_display = {}
    if payload.get("unstructured"):
        # For regular unstructured, use files from payload
        for file_obj in payload.get("unstructured", []):
            internal_name = os.path.basename(file_obj["name"])
            display_name = file_obj.get("displayName", internal_name)
            ext = (file_obj.get("type") or "").lower() or os.path.splitext(
                display_name
            )[1].lstrip(".").lower()
            file_names_to_display[internal_name] = f"{display_name}.{ext}"
    state["path_state"]["file_names_to_display"] = file_names_to_display

    # Stream through the graph and accumulate state
    final_state = state.copy()
    for step in app.stream(state, config={"recursion_limit": 45}):
        logger.info("--- AGENT STEP ---")
        for node_name, node_output in step.items():
            logger.info(f"Node: {node_name}")
            # Accumulate state updates
            if node_output:
                # Merge path_state updates
                if "path_state" in node_output:
                    if "path_state" not in final_state:
                        final_state["path_state"] = {}
                    final_state["path_state"].update(node_output["path_state"])
                # Merge other state updates
                for key, value in node_output.items():
                    if key != "path_state":
                        final_state[key] = value

    # Get final result from path_state (set by domain-specific/formatting agents)
    path_state = final_state.get("path_state", {})
    final_response = path_state.get("final_response")
    if final_response is None:
        # Fallback to messages if final_response not found (for backward compatibility)
        messages = final_state.get("messages", [])
        if messages:
            if isinstance(messages, dict):
                final_response = messages
            elif isinstance(messages[-1], dict):
                final_response = messages[-1]
            else:
                final_response = str(messages[-1])
        else:
            final_response = ""

    # Normalize to outer answer dict expected by router.py
    if isinstance(final_response, dict):
        answer = final_response
    else:
        # Plain string → wrap into dict
        answer = {"response": str(final_response)}

    logger.info(f"💬 Final answer to user:\n{answer}")
    return answer
