# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt rendering + output parsing for agent_eval.

The task prompt is identical across profiles (clean A/B): it states the task,
the mounted corpus path, and the required ``output.json`` contract
(``final_answer`` + ranked ``selected_chunks``). The only difference between
baseline and skill is the *environment* (skill present + index + retriever on
PATH, vs blocked), not the instructions.

Pure stdlib.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Default rewrites: manifest prompts hard-code dataset paths like
# ``test-data/vidorev3-fda/pdfs/...``; the corpus is mounted at ./pdfs/.
_DEFAULT_REWRITES = [
    (re.compile(r"test-data/[^\s`'\"]*?/pdfs/"), "./pdfs/"),
    (re.compile(r"test-data/[^\s`'\"]*?/pdfs\b"), "./pdfs"),
]

_TEMPLATE = """\
You have a corpus of PDF documents mounted at ./pdfs/ in your working directory.

Task:
{prompt}

When you are done, write your result to ./output.json (in the current working
directory) with EXACTLY this JSON structure and nothing else:

{{
  "query_id": "{query_id}",
  "final_answer": "<concise answer synthesized from the documents. Include the exact fact asked for. If the answer is NOT supported by the corpus, say so explicitly — do not invent or guess.>",  # noqa: E501
  "selected_chunks": [
    {{"rank": 1, "doc_id": "<PDF filename without .pdf>", "modality": "pdf_page", "locator": {{"page_number": {page_value_hint}}}, "content": "<the text of that page / passage you used as evidence>", "score": <optional float>}}  # noqa: E501
  ]
}}

Rules:
{page_rule}
- Provide up to {top_k} entries in "selected_chunks", ranked best-first, reflecting
  the actual evidence you used to answer.
- "doc_id" is the PDF filename without the .pdf extension.
- After writing ./output.json, stop. Do not print the file contents.
"""

# Page-number convention per emission base. The harness normalizes to the
# 0-indexed gold using run_config["page_index_base"], so the agent never has to
# do arithmetic — it copies a value verbatim.
_PAGE_RULES = {
    0: (
        "<0-indexed int>",
        '- "page_number" is 0-indexed: the first page of a PDF is page 0.',
    ),
    1: (
        "<the retriever hit's page_number, copied verbatim>",
        '- "page_number": copy the retriever hit\'s `page_number` field EXACTLY as '
        "returned. Do NOT convert to 0-indexed and do NOT add or subtract anything, "
        "even if other notes suggest converting — emit the raw value.",
    ),
}


_SETUP_TEMPLATE = """\
You have a folder of PDF documents at ./pdfs/ in your working directory.

Build a searchable index over these PDFs in the current working directory so they
can be queried later, using whatever retrieval tooling is available to you.

When the index is built, STOP — do not answer any question yet, do not run test
queries. This is a one-time setup step.
"""


def render_setup_prompt() -> str:
    return _SETUP_TEMPLATE


_FUNCTIONAL_SUFFIX = (
    "\n\nThe corpus paths above are mounted in your current working directory at the same "
    "relative paths. Use the available retrieval tooling. Report the result of the task in "
    "your final message (include citations to source files where the task asks for them)."
)


def render_functional_prompt(prompt: str) -> str:
    """Functional tests are self-contained (the prompt states the task and names its
    corpus path). We append a short note that paths are mounted locally; we do NOT
    impose the selected_chunks/output.json schema — grading reads the final message and
    workdir artifacts."""
    return (prompt or "").strip() + _FUNCTIONAL_SUFFIX


def rewrite_paths(text: str, extra_rewrites: list[tuple[str, str]] | None = None) -> str:
    out = text
    for pat, repl in _DEFAULT_REWRITES:
        out = pat.sub(repl, out)
    for pat, repl in extra_rewrites or []:
        out = re.sub(pat, repl, out)
    return out


def render_prompt(query_id: str, prompt: str, *, top_k: int, page_index_base: int = 0, extra_rewrites=None) -> str:
    page_value_hint, page_rule = _PAGE_RULES.get(page_index_base, _PAGE_RULES[0])
    return _TEMPLATE.format(
        prompt=rewrite_paths(prompt, extra_rewrites),
        query_id=query_id,
        top_k=top_k,
        page_value_hint=page_value_hint,
        page_rule=page_rule,
    )


def parse_output_json(workdir: Path) -> dict[str, Any] | None:
    """Read and normalize ./output.json written by the agent.

    Returns a dict {final_answer, selected_chunks:[{rank,doc_id,modality,locator,
    content,score}]} or None if missing/malformed.
    """
    path = workdir / "output.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    chunks_out: list[dict[str, Any]] = []
    for i, c in enumerate(raw.get("selected_chunks") or []):
        if not isinstance(c, dict):
            continue
        loc = c.get("locator")
        if not isinstance(loc, dict):
            # tolerate a bare page_number at top level
            pn = c.get("page_number")
            loc = {"page_number": int(pn)} if isinstance(pn, (int, float)) else {}
        chunks_out.append(
            {
                "rank": int(c.get("rank", i + 1) or i + 1),
                "doc_id": str(c.get("doc_id", "")),
                "modality": str(c.get("modality", "pdf_page")),
                "locator": loc,
                "content": str(c.get("content", "")),
                "score": c.get("score") if isinstance(c.get("score"), (int, float)) else None,
            }
        )
    return {
        "final_answer": str(raw.get("final_answer", "")),
        "selected_chunks": chunks_out,
    }
