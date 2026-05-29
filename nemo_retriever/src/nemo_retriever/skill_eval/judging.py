# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario-aware LLM-as-judge for skill_eval.

Routes each trial through one of two prompts based on whether the
dataset entry carries a ``scoring_mode``:

- ``simple``  -> batch_1 ``llm_scorer_prompt.md`` style (answer-grading).
- ``scenario`` -> batch_2 ``llm_scenario_scorer_prompt.md`` style (handles
  ingest_only / extract_only / refusal / capability_gap / dispatcher_prompt /
  skip in addition to answerable_retrieval / ingest_plus_answer).

Both prompts live on disk in the SDG run directory that produced the
manifest, so this module never vendors prompt text. See
``cli._build_judge`` for how paths are resolved.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol

if TYPE_CHECKING:
    from nemo_retriever.skill_eval.dataset import DatasetEntry
    from nemo_retriever.skill_eval.runner import TrialResult


JudgeMode = Literal["simple", "scenario"]


SIMPLE_SUB_SCORES: tuple[str, ...] = ("answer_correctness", "citation_quality", "faithfulness")
SCENARIO_SUB_SCORES: tuple[str, ...] = (
    "action_correctness",
    "routing_correctness",
    "answer_correctness",
    "citation_quality",
    "faithfulness",
    "refusal_correctness",
)
SIMPLE_FLAGS: tuple[str, ...] = ("is_answer_correct", "is_citation_correct")
SCENARIO_FLAGS: tuple[str, ...] = ("is_action_correct", "is_answer_correct", "is_citation_correct")
LIST_KEYS: tuple[str, ...] = ("major_errors", "missing_facts", "unsupported_claims")


@dataclass
class ScenarioJudgeResult:
    """Structured judge output, shared between simple and scenario prompts."""

    mode: JudgeMode
    sub_scores: dict[str, Optional[int]] = field(default_factory=dict)
    flags: dict[str, Optional[bool]] = field(default_factory=dict)
    lists: dict[str, list[str]] = field(default_factory=dict)
    rationale: str = ""
    error: Optional[str] = None


@functools.lru_cache(maxsize=8)
def _read_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _select_mode(entry: "DatasetEntry") -> JudgeMode:
    """Pick the prompt mode for ``entry`` based on its ``scoring_mode`` field."""
    return "scenario" if entry.scoring_mode else "simple"


def _render_pages(entry: "DatasetEntry") -> str:
    parts = [f"{p.doc_id}:{p.page_number}" for p in entry.ground_truth_pages]
    return ", ".join(parts)


def _render_cited(result: "TrialResult") -> str:
    parts = [f"{item.get('doc_id')}:{item.get('page_number')}" for item in result.ranked_retrieved]
    return ", ".join(parts)


def _render_raw_answers(entry: "DatasetEntry") -> str:
    return "; ".join(entry.raw_answers)


# Cap ``agent_answer`` at ~20K chars before stuffing into the judge prompt.
# A handful of trials (notably ``extract_only`` scenarios where the agent
# dumps the whole corpus into its answer) otherwise blow past the judge
# model's context window. 20K chars ≈ 5-6K tokens leaves plenty of headroom
# for the system instructions + prompt template + reference content.
JUDGE_ANSWER_CHAR_CAP = 20000
_TRUNCATION_MARKER = f"\n\n... [truncated for judge: original answer exceeded {JUDGE_ANSWER_CHAR_CAP} chars]"


def truncate_for_judge(text: str, cap: int = JUDGE_ANSWER_CHAR_CAP) -> str:
    """Cap ``text`` at ``cap`` chars with a visible marker.

    Both the new judge (via ``_format_prompt``) and the legacy ``LLMJudge``
    pass through this helper so the cap is applied identically across both
    scoring paths. The marker makes the truncation visible to the judge so
    its rationale can flag the gap rather than silently mis-scoring.
    """
    if len(text) <= cap:
        return text
    head_chars = max(0, cap - len(_TRUNCATION_MARKER))
    return text[:head_chars] + _TRUNCATION_MARKER


def _format_prompt(template: str, entry: "DatasetEntry", result: "TrialResult") -> str:
    """Substitute the prompt slots for ``entry`` / ``result`` into ``template``.

    Uses literal-token replacement for each known slot, NOT ``str.format_map``.
    The SDG-shipped judge prompts contain literal JSON-schema blocks like
    ``{ "action_correctness": <0-5>, ... }`` which a Python format-string
    parser would misinterpret as substitution slots with invalid format
    specifiers. With ``str.replace`` those literals pass through verbatim and
    only our enumerated slot tokens are substituted.

    Slots that the template doesn't reference are simply ignored. Slots the
    template does reference but that we don't know about are left as literal
    ``{slot_name}`` in the output — the judge LLM is robust to that, and
    leaving them visible aids debugging compared to silently rendering empty.
    """
    substitutions: list[tuple[str, str]] = [
        ("{query}", entry.original_query),
        ("{reference_answer}", entry.ground_truth_answer),
        ("{raw_answers}", _render_raw_answers(entry)),
        ("{relevant_pages}", _render_pages(entry)),
        ("{agent_answer}", truncate_for_judge(result.final_answer)),
        ("{agent_cited_pages}", _render_cited(result)),
        ("{cited_evidence}", ""),
        ("{scenario_prompt}", entry.paraphrased_prompt),
        ("{category}", entry.category),
        ("{phase}", entry.phase),
        ("{scoring_mode}", entry.scoring_mode),
        ("{expected_action}", entry.expected_action),
        ("{expected_output_shape}", entry.expected_output_shape),
        ("{validation_signal}", entry.validation_signal),
    ]
    out = template
    for token, value in substitutions:
        out = out.replace(token, value)
    return out


class _ChatClient(Protocol):
    """Minimal slice of LiteLLMClient that this module needs."""

    def complete(self, messages: list[dict], max_tokens: Optional[int] = None) -> tuple[str, float]: ...


_SYSTEM_INSTRUCTION = "Respond with ONLY valid JSON matching the schema in the user message. No prose, no markdown."


def _strip_envelope(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def _validate_subscore(value: Any) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"sub-score must be an integer, got {type(value).__name__}: {value!r}")
    score = value
    if not 0 <= score <= 5:
        raise ValueError(f"sub-score {score} out of range 0-5")
    return score


def _validate_flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _validate_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [str(x) for x in value]


def _parse_response(raw: str, mode: JudgeMode) -> ScenarioJudgeResult:
    """Parse a judge JSON response into a ``ScenarioJudgeResult``.

    Tolerates ``<think>`` blocks and code fences. Returns a result whose
    ``error`` is set when JSON parsing or sub-score validation fails;
    sub-scores then all default to ``None``.
    """
    sub_keys = SIMPLE_SUB_SCORES if mode == "simple" else SCENARIO_SUB_SCORES
    flag_keys = SIMPLE_FLAGS if mode == "simple" else SCENARIO_FLAGS
    empty = ScenarioJudgeResult(
        mode=mode,
        sub_scores={k: None for k in sub_keys},
        flags={k: None for k in flag_keys},
        lists={k: [] for k in LIST_KEYS},
        rationale="",
    )

    try:
        data = json.loads(_strip_envelope(raw))
    except json.JSONDecodeError as exc:
        empty.error = f"parse_failure: {exc}: {raw[:200]!r}"
        return empty
    if not isinstance(data, dict):
        empty.error = f"parse_failure: top-level must be a JSON object, got {type(data).__name__}"
        return empty

    try:
        sub_scores = {k: _validate_subscore(data.get(k)) for k in sub_keys}
    except (TypeError, ValueError) as exc:
        empty.error = f"parse_failure: {exc}"
        return empty

    return ScenarioJudgeResult(
        mode=mode,
        sub_scores=sub_scores,
        flags={k: _validate_flag(data.get(k)) for k in flag_keys},
        lists={k: _validate_list(data.get(k)) for k in LIST_KEYS},
        rationale=str(data.get("brief_rationale") or ""),
    )


def evaluate_entry(
    *,
    client: _ChatClient,
    entry: "DatasetEntry",
    result: "TrialResult",
    simple_prompt_path: Optional[str],
    scenario_prompt_path: Optional[str],
) -> ScenarioJudgeResult:
    """Score ``result`` against ``entry`` using the prompt selected by ``_select_mode``.

    ``simple_prompt_path`` / ``scenario_prompt_path`` are the on-disk
    locations of the SDG-shipped judge prompts. Only the path for the
    selected mode is required; the other may be ``None``.
    """
    mode = _select_mode(entry)
    chosen_path = scenario_prompt_path if mode == "scenario" else simple_prompt_path
    if not chosen_path:
        raise ValueError(
            f"judge mode={mode!r} requires {('scenario_prompt_path' if mode == 'scenario' else 'simple_prompt_path')}; "
            "set judge.<mode>_prompt_path in the config or co-locate the prompt file with the manifest."
        )

    template = _read_prompt(chosen_path)
    user = _format_prompt(template, entry, result)
    messages = [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": user},
    ]
    try:
        raw, _ = client.complete(messages)
    except Exception as exc:
        sub_keys = SIMPLE_SUB_SCORES if mode == "simple" else SCENARIO_SUB_SCORES
        flag_keys = SIMPLE_FLAGS if mode == "simple" else SCENARIO_FLAGS
        return ScenarioJudgeResult(
            mode=mode,
            sub_scores={k: None for k in sub_keys},
            flags={k: None for k in flag_keys},
            lists={k: [] for k in LIST_KEYS},
            rationale="",
            error=f"judge_api_error: {exc}",
        )
    return _parse_response(raw, mode=mode)
