# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator that expands a query DataFrame into sub-queries via an LLM."""

from __future__ import annotations

import json
import os
from typing import Any, List, Literal, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator

# openai is imported lazily in _ensure_client() so the operator stays
# serialisable for Ray workers without requiring it at import time.


# ---------------------------------------------------------------------------
# Built-in strategy prompts
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    "decompose": """\
You are a query decomposition assistant for a retrieval system.

Given a search query, break it down into up to {max_subqueries} distinct sub-queries that \
together cover all aspects and angles of the original query. Generate as many sub-queries as \
are genuinely useful — do not pad with redundant ones just to hit the maximum. Each sub-query \
should target a specific facet, making it easier for a dense retrieval system to find all \
relevant documents.

Rules:
- Each sub-query must be self-contained and meaningful on its own.
- Sub-queries should be diverse and complementary, not redundant.
- Use clear, precise language suited for dense embedding retrieval.
- Output a JSON array of strings only — no explanation, no markdown fences.""",

    "hyde": """\
You are a Hypothetical Document Embedding (HyDE) assistant for a retrieval system.

Given a search query, generate up to {max_subqueries} short hypothetical document passages \
(2–4 sentences each) that would directly answer or address the query. Generate as many as are \
genuinely useful — fewer is fine if the query is simple. These passages will be used as queries \
to a dense retrieval system to find real, similar documents.

Rules:
- Each passage should read like a real document excerpt that answers the query.
- Vary the style and perspective across passages (e.g., academic, technical, narrative).
- Be factually plausible; focus on covering the query intent.
- Output a JSON array of strings only — no explanation, no markdown fences.""",

    "multi_perspective": """\
You are a multi-perspective query expansion assistant for a retrieval system.

Given a search query, generate up to {max_subqueries} reformulations from different angles, \
perspectives, or levels of specificity to maximise recall in a dense retrieval system. Only \
generate reformulations that add genuine coverage — do not pad.

Rules:
- Vary terminology: use synonyms, technical vs. casual language, acronyms vs. full names.
- Vary scope: broad overview queries alongside narrow, specific ones.
- Vary form: declarative statements, questions, and entity-focused queries.
- Each reformulation must have a meaningfully different surface form from the others.
- Output a JSON array of strings only — no explanation, no markdown fences.""",
}


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class SubQueryGeneratorOperator(AbstractOperator, CPUOperator):
    """Expand each query row into sub-query rows using an LLM.

    The operator calls an LLM (via the OpenAI SDK) once per input query and
    explodes the result into one output row per generated sub-query.  The LLM
    decides how many sub-queries to generate up to ``max_subqueries``.  This
    makes it a natural upstream stage for a retrieval operator: the downstream
    operator can retrieve documents independently for every sub-query row, and
    a subsequent aggregation step (e.g. RRF) can merge the per-sub-query
    ranked lists back into a single ranking per ``query_id``.

    Input DataFrame schema
    ----------------------
    query_id   : str  — unique identifier for the query
    query_text : str  — the search query text
    (any additional columns are passed through unchanged)

    Output DataFrame schema
    -----------------------
    query_id      : str  — same ``query_id`` as the input row
    query_text    : str  — original query text (preserved for context)
    subquery_idx  : int  — 0-based position within the generated sub-query group
    subquery_text : str  — the generated sub-query text
    (additional input columns are forwarded to every expanded row)

    Parameters
    ----------
    llm_model : str
        OpenAI model identifier, e.g. ``"gpt-4o"``.
    max_subqueries : int
        Maximum number of sub-queries the LLM may generate per query.
        The LLM will generate fewer if the query does not warrant the maximum.
        Defaults to ``4``.
    strategy : {"decompose", "hyde", "multi_perspective"}
        Built-in sub-query generation strategy.

        ``"decompose"``
            Break the query into complementary sub-aspects (default).
        ``"hyde"``
            Generate hypothetical answer passages (HyDE).
        ``"multi_perspective"``
            Rewrite the query from diverse angles to maximise recall.
    api_key : str, optional
        Literal API key **or** an ``"os.environ/VAR_NAME"`` reference that is
        resolved at call time.  Omit to rely on the ``OPENAI_API_KEY``
        environment variable.
    base_url : str, optional
        Custom endpoint URL — useful for NIM deployments or self-hosted models.
    max_tokens : int, optional
        Upper bound on tokens in the LLM response.
    system_prompt_override : str, optional
        Fully custom system prompt.  Use ``{max_subqueries}`` as a placeholder.
        When provided, ``strategy`` is ignored.

    Examples
    --------
    Standalone use::

        import pandas as pd
        from nemo_retriever.graph.subquery_operator import SubQueryGeneratorOperator

        op = SubQueryGeneratorOperator(llm_model="gpt-4o", max_subqueries=5)
        df = pd.DataFrame({
            "query_id":   ["q1", "q2"],
            "query_text": ["What causes inflation?", "How do vaccines work?"],
        })
        result = op.run(df)
        # result has up to 10 rows: ≤5 sub-queries × 2 original queries

    Composing into a graph::

        from nemo_retriever.graph import InprocessExecutor
        from nemo_retriever.graph.subquery_operator import SubQueryGeneratorOperator

        graph = (
            SubQueryGeneratorOperator(llm_model="gpt-4o", max_subqueries=4)
            >> RetrievalOperator(retriever=my_retriever)
            >> RRFAggregatorOperator()
        )
        executor = InprocessExecutor(graph)
        result_df = executor.ingest(query_df)
    """

    def __init__(
        self,
        *,
        llm_model: str,
        max_subqueries: int = 4,
        strategy: Literal["decompose", "hyde", "multi_perspective"] = "decompose",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        system_prompt_override: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._max_subqueries = max_subqueries
        self._strategy = strategy
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._system_prompt_override = system_prompt_override

        # OpenAI client is created lazily so the operator stays serialisable for Ray.
        self._client: Any = None

    # ------------------------------------------------------------------
    # AbstractOperator interface
    # ------------------------------------------------------------------

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        """Normalise *data* to a DataFrame with ``query_id`` / ``query_text`` columns.

        Accepted input types
        --------------------
        ``pd.DataFrame``
            Must contain at least ``query_id`` and ``query_text`` columns.
        ``list[str]``
            Plain query strings; ``query_id`` values are auto-assigned as
            ``"q0"``, ``"q1"``, …
        ``list[tuple[str, str]]`` or ``list[list[str, str]]``
            ``(query_id, query_text)`` pairs.
        """
        if isinstance(data, pd.DataFrame):
            missing = {"query_id", "query_text"} - set(data.columns)
            if missing:
                raise ValueError(
                    f"Input DataFrame is missing required column(s): {sorted(missing)}. "
                    "Expected at minimum: 'query_id' and 'query_text'."
                )
            return data.copy()

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, str):
                return pd.DataFrame(
                    {
                        "query_id": [f"q{i}" for i in range(len(data))],
                        "query_text": list(data),
                    }
                )
            if isinstance(first, (tuple, list)) and len(first) == 2:
                return pd.DataFrame(data, columns=["query_id", "query_text"])

        raise TypeError(
            f"Unsupported input type {type(data).__name__!r}. "
            "Pass a pd.DataFrame with 'query_id' and 'query_text' columns, "
            "a list[str], or a list[tuple[str, str]]."
        )

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate sub-queries for every row and explode to one row per sub-query."""
        self._ensure_client()
        system_prompt = self._build_system_prompt()

        passthrough_cols = [c for c in data.columns if c not in ("query_id", "query_text")]
        rows: List[dict[str, Any]] = []

        for _, row in data.iterrows():
            subqueries = self._generate_one(row["query_text"], system_prompt)
            for idx, sq in enumerate(subqueries):
                new_row: dict[str, Any] = {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "subquery_idx": idx,
                    "subquery_text": sq,
                }
                for col in passthrough_cols:
                    new_row[col] = row[col]
                rows.append(new_row)

        if not rows:
            return pd.DataFrame(columns=["query_id", "query_text", "subquery_idx", "subquery_text"])

        return pd.DataFrame(rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Lazily create the OpenAI client (once per instance)."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "SubQueryGeneratorOperator requires 'openai'. "
                "Install it with:  pip install 'openai>=1.0'"
            ) from exc

        api_key = self._api_key
        if api_key is not None and api_key.strip().startswith("os.environ/"):
            var = api_key.strip().removeprefix("os.environ/")
            api_key = os.environ[var]

        self._client = OpenAI(
            api_key=api_key,
            **({"base_url": self._base_url} if self._base_url is not None else {}),
        )

    def _build_system_prompt(self) -> str:
        template = self._system_prompt_override or _PROMPTS[self._strategy]
        return template.format(max_subqueries=self._max_subqueries)

    def _generate_one(self, query: str, system_prompt: str) -> List[str]:
        """Call the LLM and return a list of sub-query strings for *query*."""
        call_kwargs: dict[str, Any] = dict(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"},
            ],
        )
        if self._max_tokens is not None:
            call_kwargs["max_tokens"] = self._max_tokens

        response = self._client.chat.completions.create(**call_kwargs)
        raw = response.choices[0].message.content.strip()
        return _parse_json_list(raw, fallback=query)


# ---------------------------------------------------------------------------
# Module-level helpers (no instance state — easier to test in isolation)
# ---------------------------------------------------------------------------


def _parse_json_list(raw: str, *, fallback: str) -> List[str]:
    """Parse a JSON array from *raw*, stripping markdown fences if present.

    Returns *[fallback]* when parsing fails so downstream stages always
    receive at least one sub-query row.
    """
    text = raw
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            break
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed and all(isinstance(s, str) for s in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    return [fallback]
