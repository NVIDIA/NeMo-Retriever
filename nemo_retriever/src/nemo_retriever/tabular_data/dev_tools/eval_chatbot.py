"""Evaluate the text-to-SQL agent against a chatbot evaluation JSON file.

For every entry in the JSON array (each containing ``question_id``,
``question``, ``SQL`` (expected), and ``answer_raw`` (expected user-facing
result)), this script:

1. Calls ``get_agent_response`` with the question (same path as
   ``ingest_postgres.run_retrieve``).
2. Scores the agent's SQL against the expected SQL by **executing both**
   queries against the live Postgres connector and comparing the resulting
   row sets.  Failure to execute either side yields score 0.
3. Scores the agent's answer text against ``answer_raw`` via difflib
   similarity and a normalised substring check.
4. Writes one row per question to a CSV.  Any per-question exception is
   logged, recorded in the ``error`` column, and scored 0 — execution
   continues with the next question.

Usage::

    PYTHONPATH=nemo_retriever/src uv run --no-sync python \
        nemo_retriever/tabular-dev-tools/eval_chatbot.py \
        [--input PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nemo_retriever.tabular_data.dev_tools.postgres_connector import PostgresDatabase
from nemo_retriever.params import EmbedParams, VdbUploadParams
from nemo_retriever.retriever import Retriever
from nemo_retriever.tabular_data.retrieval.text_to_sql.main import get_agent_response
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentPayload

logger = logging.getLogger("eval_chatbot")

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not _NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export NVIDIA_API_KEY='nvapi-...'\n\n"
        "Get your key at https://build.nvidia.com"
    )

EMBED_PARAMS = EmbedParams(
    embed_invoke_url="https://integrate.api.nvidia.com/v1",
    model_name="nvidia/llama-nemotron-embed-1b-v2",
    api_key=_NVIDIA_API_KEY,
    embed_modality="text",
)

VDB_PARAMS = VdbUploadParams(
    vdb_op="lancedb",
    vdb_kwargs={
        "lancedb_uri": "lancedb",
        "table_name": "nv-ingest-tabular",
        "overwrite": False,
        "create_index": False,
    },
)

DATABASE: str = os.environ.get("POSTGRES_DB", "testdb")

_DEFAULT_INPUT = Path(__file__).parent / "chatbot_evaluation.json"
_DEFAULT_OUTPUT = Path(__file__).parent / "chatbot_evaluation_scores.csv"


def _conn_string(db: str) -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _build_retriever() -> Retriever:
    lancedb_kwargs = VDB_PARAMS.vdb_kwargs
    return Retriever(
        vdb="lancedb",
        vdb_kwargs={
            "uri": lancedb_kwargs["lancedb_uri"],
            "table_name": lancedb_kwargs["table_name"],
        },
        top_k=15,
        embedding_api_key=_NVIDIA_API_KEY,
        embedding_http_endpoint=EMBED_PARAMS.embed_invoke_url,
    )


# -----------------------------------------------------------------------------
# Scoring helpers
# -----------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, drop trailing semicolons."""
    if s is None:
        return ""
    s = str(s).strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _sql_text_similarity(expected: str, actual: str) -> float:
    a = _normalize_text(expected)
    b = _normalize_text(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _df_values_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Compare two DataFrames by row-multiset of values, ignoring column names/order."""
    try:
        if a.shape != b.shape:
            return False
        a_rows = sorted(tuple(_canonical(v) for v in row) for row in a.values.tolist())
        b_rows = sorted(tuple(_canonical(v) for v in row) for row in b.values.tolist())
        return a_rows == b_rows
    except Exception:
        return False


def _canonical(value: Any) -> Any:
    """Make a value hashable and comparable across small numeric/string drift."""
    if value is None:
        return None
    if isinstance(value, float):
        # Round to mitigate float jitter from aggregations
        return round(value, 4)
    return str(value).strip().lower()


def _execute_sql(connector: PostgresDatabase, sql: str) -> Tuple[Optional[pd.DataFrame], str]:
    if not sql or not sql.strip():
        return None, "empty SQL"
    try:
        df = connector.execute(sql)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return df, ""
    except Exception as exc:  # pragma: no cover - tooling script
        return None, f"{type(exc).__name__}: {exc}"


def _score_sql(connector: PostgresDatabase, expected: str, actual: str) -> Dict[str, Any]:
    text_sim = _sql_text_similarity(expected, actual)
    expected_df, expected_err = _execute_sql(connector, expected)
    actual_df, actual_err = _execute_sql(connector, actual)
    exec_match = 0
    if expected_df is not None and actual_df is not None:
        exec_match = 1 if _df_values_equal(expected_df, actual_df) else 0
    return {
        "sql_text_similarity": round(text_sim, 4),
        "sql_exec_match": exec_match,
        "expected_sql_error": expected_err,
        "returned_sql_error": actual_err,
    }


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> List[float]:
    if not text:
        return []
    out = []
    for tok in _NUM_RE.findall(str(text)):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _parse_markdown_table(md: str) -> Optional[pd.DataFrame]:
    """Parse a simple markdown table into a DataFrame, or None on failure."""
    if not md:
        return None
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    # Need at least header + separator + one data row
    if len(lines) < 3:
        return None
    data_lines = [ln for ln in lines if not re.match(r"^\|[\s:_-]+\|$", ln)]
    if len(data_lines) < 2:
        return None
    header = [c.strip() for c in data_lines[0].strip("|").split("|")]
    rows = []
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def _db_result_to_df(value: str) -> Optional[pd.DataFrame]:
    """Try to parse the stringified DB result into a DataFrame."""
    if not value:
        return None
    text = str(value).strip()
    # Unwrap outer list wrapper like ['[{"count":712}]']
    if text.startswith("[") and text.endswith("]"):
        try:
            outer = json.loads(text)
            if isinstance(outer, list) and len(outer) == 1 and isinstance(outer[0], str):
                text = outer[0]
        except (json.JSONDecodeError, TypeError):
            pass
    # Try JSON array of objects
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        if isinstance(parsed, dict):
            return pd.DataFrame([parsed])
    except (json.JSONDecodeError, TypeError):
        pass
    # Try CSV
    try:
        from io import StringIO

        df = pd.read_csv(StringIO(text))
        if not df.empty:
            return df
    except Exception:
        pass
    return None


def _score_answer(expected_raw: str, returned_db_str: str) -> Dict[str, Any]:
    """Score the agent's answer against the expected ``answer_raw`` markdown table.

    Strategy (in priority order):
    1. Parse both sides into DataFrames and compare row-multisets (structural match).
    2. Compare the multiset of numeric values (survives formatting differences).
    3. Fall back to fuzzy text similarity.
    """
    expected_df = _parse_markdown_table(expected_raw)
    actual_df = _db_result_to_df(returned_db_str)

    structural_match = 0
    if expected_df is not None and actual_df is not None:
        structural_match = 1 if _df_values_equal(expected_df, actual_df) else 0

    haystack = str(returned_db_str or "")

    expected_nums = sorted(round(n, 4) for n in _extract_numbers(expected_raw))
    actual_nums = sorted(round(n, 4) for n in _extract_numbers(haystack))
    nums_match = 1 if expected_nums and expected_nums == actual_nums else 0

    sim = (
        difflib.SequenceMatcher(None, _normalize_text(expected_raw), _normalize_text(haystack)).ratio()
        if expected_raw and haystack
        else 0.0
    )
    if structural_match:
        sim = 1.0
    elif nums_match:
        sim = max(sim, 1.0)

    return {
        "answer_text_similarity": round(sim, 4),
        "answer_numbers_match": nums_match,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of questions, got {type(data).__name__}")
    return data


def _stringify_db_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.DataFrame):
        return value.to_csv(index=False)
    return str(value)


CSV_FIELDS = [
    "row_index",
    "question_id",
    "difficulty",
    "question",
    "expected_sql",
    "returned_sql",
    "sql_text_similarity",
    "sql_exec_match",
    "expected_sql_error",
    "returned_sql_error",
    "expected_answer_raw",
    "returned_answer",
    "answer_text_similarity",
    "answer_numbers_match",
    "runtime_seconds",
    "error",
]


def _print_agent_result(qid: Any, question: str, agent_result: Dict[str, Any] | None, expected_sql: str = "") -> None:
    """Pretty-print the agent result to stdout for quick visual inspection."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Question {qid}: {question}")
    print(sep)
    if expected_sql:
        print("\n  [expected_sql]")
        for line in expected_sql.splitlines():
            print(f"    {line}")
    if not agent_result:
        print("  (no result)")
        print(sep)
        return
    for key in ("sql_code", "response", "sql_response_from_db"):
        val = agent_result.get(key)
        if val is None:
            continue
        print(f"\n  [{key}]")
        for line in str(val).splitlines():
            print(f"    {line}")
    remaining = {k: v for k, v in agent_result.items() if k not in ("sql_code", "response", "sql_response_from_db")}
    if remaining:
        print("\n  [other keys]")
        for k, v in remaining.items():
            print(f"    {k}: {v}")
    print(sep)


def evaluate(input_path: Path, output_path: Path, start_index: int = 0, end_index: int | None = None) -> None:
    # questions = [{"question_id": 0, "question": "Show me the details for component 670-14039-0072-TS5", "SQL": ""}]
    all_questions = _load_questions(input_path)
    questions = all_questions[start_index:end_index]
    logger.info(
        "Running questions %d–%d (%d of %d total) from %s",
        start_index,
        start_index + len(questions) - 1,
        len(questions),
        len(all_questions),
        input_path,
    )

    connector = PostgresDatabase(_conn_string(DATABASE))
    retriever = _build_retriever()

    resuming = start_index > 0 and output_path.exists()
    mode = "a" if resuming else "w"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not resuming:
            writer.writeheader()

        for idx, item in enumerate(questions, start=start_index):
            qid = item.get("question_id", idx)
            question = item.get("question", "")
            expected_sql = item.get("SQL", "")
            expected_answer = item.get("answer_raw", "")
            difficulty = item.get("difficulty", "")
            logger.info("[%d/%d] q%s: %s", idx + 1, len(questions), qid, question)

            row: Dict[str, Any] = {
                "row_index": idx,
                "question_id": qid,
                "difficulty": difficulty,
                "question": question,
                "expected_sql": expected_sql,
                "returned_sql": "",
                "sql_text_similarity": 0.0,
                "sql_exec_match": 0,
                "expected_sql_error": "",
                "returned_sql_error": "",
                "expected_answer_raw": expected_answer,
                "returned_answer": "",
                "answer_text_similarity": 0.0,
                "answer_numbers_match": 0,
                "runtime_seconds": "",
                "error": "",
            }

            t0 = time.perf_counter()
            try:
                payload: AgentPayload = {
                    "question": question,
                    "retriever": retriever,
                    "connector": connector,
                    "path_state": {},
                    "custom_prompts": [
                        {
                            "name": "Responsible Users",
                            "description": (
                                "When asked who is responsible, involved, assigned, working on, "
                                "or blocking a request or request task, "
                                "always use this UNION pattern to get ALL assigned users:\n"
                                "  SELECT users.name FROM request_tasks\n"
                                "    JOIN users ON request_tasks.pic_id = users.id\n"
                                "    WHERE request_tasks.request_id = <ID> AND <filters>\n"
                                "  UNION\n"
                                "  SELECT users.name FROM request_tasks\n"
                                "    JOIN request_task_collaborators "
                                "ON request_task_collaborators.request_task_id = request_tasks.id\n"
                                "    JOIN users ON request_task_collaborators.user_id = users.id\n"
                                "    WHERE request_tasks.request_id = <ID> AND <filters>\n"
                                "The first half selects primary responsible users (pic_id), "
                                "the second half selects collaborators (user_id). "
                                "Only skip the second half if the question explicitly asks "
                                "for the primary responsible user only."
                            ),
                        },
                        {
                            "name": "Task vs Request Task",
                            "description": (
                                "When a question refers to 'tasks' generically "
                                "(e.g. 'all tasks', 'tasks by frequency', 'task rollback'), "
                                "it means task definitions from the `tasks` table "
                                "(which has `name`, `id`), NOT individual request_tasks. "
                                "To connect events or request-level data back to task definitions, "
                                "use the join chain:\n"
                                "  request_task_events → request_tasks (via request_task_id)\n"
                                "    → filter_rule_group_tasks (via filter_rule_group_task_id)\n"
                                "    → workflow_tasks (via workflow_task_id)\n"
                                "    → tasks (via task_id)\n"
                                "A `request_task` is a specific instance of a task for a specific request. "
                                "Always GROUP BY tasks.id and SELECT tasks.name when aggregating across task types."
                            ),
                        },
                        {
                            "name": "Request Attributes",
                            "description": (
                                "When a question mentions trainstop, quality level, or process by name, "
                                "the actual names live in separate lookup tables, not in the requests table itself. "
                                "The requests table only stores foreign key IDs. You must JOIN to get names:\n"
                                "- Trainstop: JOIN trainstops ON requests.trainstop_id = trainstops.id → use trainstops.name\n"
                                "- Quality level: JOIN quality_levels "
                                "ON requests.quality_level_id = quality_levels.id → use quality_levels.name\n"
                                "- Process: JOIN request_processes ON request_processes.request_id = requests.id "
                                "JOIN processes ON request_processes.process_id = processes.id → use processes.name"
                            ),
                        },
                        {
                            "name": "Product Hierarchy",
                            "description": (
                                "Pcodes, products, and business units form a hierarchy:\n"
                                "- A pcode belongs to a product: JOIN products ON pcodes.product_id = products.id\n"
                                "- A product belongs to a business unit (BU): JOIN bus ON products.bu_id = bus.id\n"
                                "When a question asks about pcodes and business units together, "
                                "always join through products."
                            ),
                        },
                        {
                            "name": "GPU MODS Version",
                            "description": (
                                "GPU MODS version values are stored in the request_attribute_values table "
                                "where field_name = 'modsVersion'. The actual version value is in the text_value column."
                            ),
                        },
                    ],
                    "acronyms": "",
                }
                agent_result = get_agent_response(payload)
                _print_agent_result(qid, question, agent_result, expected_sql)
                returned_sql = (agent_result or {}).get("sql_code", "") or ""
                returned_db = (agent_result or {}).get("sql_response_from_db")
                returned_db_str = _stringify_db_result(returned_db)

                row["returned_sql"] = returned_sql
                row["returned_answer"] = returned_db_str

                row.update(_score_sql(connector, expected_sql, returned_sql))
                row.update(_score_answer(expected_answer, returned_db_str))
            except Exception as exc:
                logger.exception("Question %s failed", qid)
                row["error"] = f"{type(exc).__name__}: {exc}"
                # Truncate traceback into the cell to keep the CSV diff-friendly.
                row["error"] += " | " + traceback.format_exc().replace("\n", " | ")[:1000]
            finally:
                row["runtime_seconds"] = round(time.perf_counter() - t0, 2)
                writer.writerow(row)
                f.flush()

    logger.info("Wrote scores to %s", output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input JSON path (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


START_INDEX = 0
END_INDEX = None  # None = run to the end

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    evaluate(args.input, args.output, START_INDEX, END_INDEX)
