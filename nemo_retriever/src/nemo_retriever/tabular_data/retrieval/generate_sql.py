# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging

from nemo_retriever.tabular_data.retrieval.text_to_sql.utils import _get_llm_client

logger = logging.getLogger(__name__)

# Load .env from current working directory so NVIDIA_API_KEY, LLM_INVOKE_URL are set (run from repo root)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("PYDEVD_WARN_EVALUATION_TIMEOUT", "60")


def _read_sql_string_literal(text: str, start: int) -> tuple[str, int] | None:
    """Read a single-quoted SQL string from text starting at start (after the opening quote).
    '' is treated as escaped quote. Returns (unescaped_content, index_after_closing_quote) or None."""
    if start >= len(text) or text[start] != "'":
        return None
    i = start + 1
    parts = []
    while i < len(text):
        if text[i] == "'":
            if i + 1 < len(text) and text[i + 1] == "'":
                parts.append("'")
                i += 2
                continue
            return ("".join(parts), i + 1)
        parts.append(text[i])
        i += 1
    return None


def _extract_json_from_sql_object(text: str) -> dict | None:
    """Extract sql_code, answer, result from SQL JSON_OBJECT('sql_code', '...', 'answer', ..., 'result', '...')."""
    import re

    text = (text or "").strip()
    # Find JSON_OBJECT( and then parse key-value pairs (allowing nested parens in values)
    obj_start = re.search(r"JSON_OBJECT\s*\(", text, re.IGNORECASE)
    if not obj_start:
        return None
    start = obj_start.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    inner = text[start : i - 1]
    out = {"sql_code": "", "answer": "", "result": None}
    # Find 'sql_code', <value> and 'result', <value> (string literals); 'answer' can be expression
    for key in ("sql_code", "answer", "result"):
        key_pattern = re.escape(f"'{key}'") + r"\s*,\s*"
        m = re.search(key_pattern, inner, re.IGNORECASE)
        if not m:
            continue
        pos = m.end()
        if pos < len(inner) and inner[pos] == "'":
            lit = _read_sql_string_literal(inner, pos)
            if lit:
                out[key], _ = lit
        else:
            # Non-string value (e.g. COUNT(...)); take until next 'sql_code', 'answer', 'result' or end
            next_key = re.search(r"'\s*(?:sql_code|answer|result)\s*'\s*,\s*", inner[pos:], re.IGNORECASE)
            end = pos + next_key.start() if next_key else len(inner)
            out[key] = inner[pos:end].strip().rstrip(",").strip()
    if out["sql_code"] or out["answer"] or out["result"] is not None:
        return out
    return None


def _extract_json_from_markdown(text: str) -> dict | None:
    """Extract a JSON object from markdown content (e.g. inside ```json ... ``` or ``` ... ```).
    Also handles ```sql blocks where the content is SQL with JSON_OBJECT('sql_code', ..., 'answer', ..., 'result', ...).
    """
    import re

    # 1) Try parsing the whole content as JSON
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) Look for ```json or ``` or ```sql ... ``` block
    match = re.search(r"```(?:\w*)\s*\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if match:
        block = match.group(1).strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 2b) Block might be SQL containing JSON_OBJECT(...)
        obj = _extract_json_from_sql_object(block)
        if obj is not None:
            return obj
    # 3) Look for any {...} that might be the JSON object (last occurrence, likely the intended one)
    brace = text.rfind("{")
    if brace != -1:
        depth = 0
        end = -1
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                obj = json.loads(text[brace : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    # 4) No code block matched; try JSON_OBJECT anywhere in text (e.g. raw SQL response)
    obj = _extract_json_from_sql_object(text)
    if obj is not None:
        return obj
    return None


def _parse_markdown_explanation_sql_thought(content: str) -> dict | None:
    """Parse LLM output that uses **Explanation** / **Final Response**, **SQL Code** (```sql...```), **Thought**.
    Used when the model returns markdown instead of JSON (e.g. OutputParserException)."""
    import re

    if not (content and isinstance(content, str)):
        return None
    text = content.strip()
    sql_code = ""
    explanation = ""
    thought = ""

    # Extract ```sql ... ``` block
    sql_match = re.search(r"```\s*sql\s*\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if sql_match:
        sql_code = sql_match.group(1).strip()

    # Extract **Explanation** or **Final Response** ... (until **SQL Code** or **Thought** or end)
    expl_match = re.search(
        r"\*\*(?:Explanation|Final Response)\*\*\s*\n+([\s\S]*?)(?=\n\s*\*\*(?:SQL Code|Thought)\*\*|\Z)",
        text,
        re.IGNORECASE,
    )
    if expl_match:
        explanation = expl_match.group(1).strip()

    # Extract **Thought** ...
    thought_match = re.search(r"\*\*Thought\*\*\s*\n+([\s\S]*)", text, re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    if sql_code or explanation or thought:
        return {
            "sql_code": sql_code,
            "answer": explanation or text[:500],
            "result": thought or None,
        }
    return None


def _parse_sql_response_content(content: str) -> dict | None:
    """Parse LLM response into {sql_code, answer, result}. Handles raw JSON or JSON inside markdown."""
    if not (content and isinstance(content, str)):
        return None
    parsed = _extract_json_from_markdown(content)
    if parsed is None:
        return None
    required = {"sql_code", "answer", "result"}
    if not required.issubset(parsed.keys()):
        # Allow missing "result" by normalizing
        if "sql_code" in parsed and "answer" in parsed:
            return {
                "sql_code": parsed.get("sql_code", ""),
                "answer": parsed.get("answer", ""),
                "result": parsed.get("result"),
            }
        return None
    return {
        "sql_code": parsed.get("sql_code", ""),
        "answer": parsed.get("answer", ""),
        "result": parsed.get("result"),
    }


# JSON schema for structured output (dict return; more reliable than Pydantic when parser returns None)
CALC_FINAL_RESPONSE_JSON_SCHEMA = {
    "title": "CalcFinalResponse",
    "description": "Final SQL answer with explanation and thought",
    "type": "object",
    "properties": {
        "response": {
            "type": "string",
            "description": (
                "The final response with your explanations and the final sql query " "that answers the user's question."
            ),
        },
        "sql_code": {
            "type": "string",
            "description": (
                "The sql code that answers the user's question "
                "based on chosen snippet/s and appropriate joins (if present)."
            ),
        },
        "thought": {
            "type": "string",
            "description": "A short thought for your answer.",
        },
    },
    "required": ["response", "sql_code", "thought"],
}


def _dict_to_sql_result(d: dict | None) -> dict:
    """Map structured output dict (or Pydantic-like) to {sql_code, answer, result}."""
    if not d or not isinstance(d, dict):
        return {"sql_code": "", "answer": "", "result": None}
    return {
        "sql_code": (d.get("sql_code") or "").strip() or " ",
        "answer": (d.get("response") or d.get("answer") or "").strip() or " ",
        "result": (d.get("thought") or d.get("result") or "").strip() or None,
    }


def get_sql_tool_response_top_k(
    question: str,
    embedding_api_key: str = "",
    embedding_http_endpoint="",
    top_k: int = 15,
) -> dict:
    """Retrieve top_k tables from LanceDB, then generate SQL via LLM (JSON schema + markdown fallbacks).

    Returns a dict with keys: sql_code, answer, result.
    """
    from nemo_retriever.retriever import Retriever

    retriever = Retriever(
        lancedb_table="nv-ingest-tabular",
        top_k=top_k,
        embedding_api_key=embedding_api_key,
        embedding_http_endpoint=embedding_http_endpoint,
    )
    hits = retriever.query(question)

    table_context_lines = []
    for i, hit in enumerate(hits, 1):
        text = (hit.get("text") or "").strip()
        if text:
            table_context_lines.append(f"{i}. {text}")
    table_context = "\n".join(table_context_lines)

    prompt = (
        "You are a SQL benchmark assistant.\n\n"
        f"The following {len(table_context_lines)} most relevant tables were retrieved for this question:\n"
        f"{table_context}\n\n"
        f"User question: {question}\n\n"
        "Use only the tables listed above when writing your SQL query.\n\n"
        "You must respond with a single JSON object only (no markdown, no **Explanation** or **SQL Code** sections). "
        "The JSON must have exactly these three keys:\n"
        '- "response": string with your explanation and the final SQL in words\n'
        '- "sql_code": string with the executable SQL query only\n'
        '- "thought": string with a short thought about your answer\n'
        'Example: {"response": "We use table X to", "sql_code": "SELECT ... FROM x;", "thought": "The query joins"}'
    )

    result_dict = None
    try:
        llm_client = _get_llm_client()
    except ValueError as e:
        logger.error("Failed to initialize LLM client: %s", e)
        return _dict_to_sql_result(None)

    # Invoke with structured output. On parse failure, extract raw output from exception.
    structured_llm = llm_client.with_structured_output(CALC_FINAL_RESPONSE_JSON_SCHEMA)
    try:
        result = structured_llm.invoke(prompt)
        if isinstance(result, dict) and (result.get("sql_code") or result.get("response")):
            result_dict = _dict_to_sql_result(result)
    except Exception as e:
        err_str = str(e)
        if "Invalid json output:" in err_str:
            content = err_str.split("Invalid json output:", 1)[-1].strip()
            if "For troubleshooting" in content:
                content = content.split("For troubleshooting")[0].strip()
            result_dict = _parse_sql_response_content(content) or _parse_markdown_explanation_sql_thought(content)
        if result_dict is None:
            response = llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            result_dict = _parse_sql_response_content(content) or _parse_markdown_explanation_sql_thought(content)

    if result_dict is None:
        result_dict = _dict_to_sql_result(None)
    return result_dict
