# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM-as-judge scoring.

LLMJudge uses a strong LLM to score generated answers on a 1-5 scale
against a ground-truth reference answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from nemo_retriever.llm.clients.litellm import LiteLLMClient
from nemo_retriever.llm.types import JudgeResult
from nemo_retriever.params.models import LLMInferenceParams, LLMRemoteClientParams

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for factual question answering.

You will receive a QUESTION, a REFERENCE answer, and a CANDIDATE answer.

Step 1 -- Identify required facts:
  Break the REFERENCE into its key terms: specific numbers, names, dates,
  percentages, units, or short phrases that constitute the factual core.
  Example: "16% of adults" -> required facts = ["16%", "adults"].

Step 2 -- Check each required fact in the CANDIDATE:
  - Allow numeric equivalence: "16.00%" = "16%", "1,000" = "1000".
  - Allow paraphrasing: "Peers" matches "Peers of those adults".
  - Allow additional correct detail: extra facts do NOT reduce the score.
  - Short but correct answers are fine: "Peers" is valid for "Peers".

Step 3 -- Score on a 1-5 scale based on the fraction of required facts present:
  5 - All required facts present. Answer is fully correct.
  4 - Nearly all required facts present. One minor fact may differ trivially.
  3 - Most required facts present but at least one non-trivial fact is missing
      or slightly wrong.
  2 - Some required facts present but the core answer is incomplete or has a
      significant factual error.
  1 - None or almost none of the required facts present. Includes: wrong answer,
      irrelevant response, or stating "the context does not contain this
      information" when the reference answer exists.

Respond ONLY with valid JSON:
{"score": <integer 1-5>, "reasoning": "<one sentence citing which required facts were matched or missed>"}

No text outside the JSON object."""

_JUDGE_USER_TEMPLATE = """\
Question: {query}

Reference answer: {reference}

Candidate answer: {candidate}"""


class LLMJudge:
    """LLM-as-judge that scores candidate answers on a 1-5 scale.

    Configuration is split into two Pydantic objects:

    * ``transport``: :class:`~nemo_retriever.params.LLMRemoteClientParams`
      owns the endpoint, api_key, retries, and timeout for the judge model.
    * ``sampling``: :class:`~nemo_retriever.params.LLMInferenceParams`
      owns ``temperature`` / ``top_p`` / ``max_tokens``. Defaults to
      ``temperature=0.0, max_tokens=256`` for deterministic scoring.

    Use :meth:`from_kwargs` for a flat, backwards-compatible constructor.
    """

    _DEFAULT_MODEL: str = "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
    _DEFAULT_SAMPLING: LLMInferenceParams = LLMInferenceParams(temperature=0.0, max_tokens=256)

    def __init__(
        self,
        transport: LLMRemoteClientParams,
        sampling: Optional[LLMInferenceParams] = None,
    ):
        self._client = LiteLLMClient(
            transport=transport,
            sampling=sampling if sampling is not None else self._DEFAULT_SAMPLING,
        )

    @property
    def model(self) -> str:
        """Return the judge model identifier from the transport params."""
        return self._client.transport.model

    @classmethod
    def from_kwargs(
        cls,
        *,
        model: str = _DEFAULT_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
    ) -> "LLMJudge":
        """Flat-kwarg constructor for zero-churn migration from the old signature.

        Sampling is left at the class default (deterministic 0.0 temperature,
        256 max tokens). Use the two-arg constructor to override sampling.
        """
        transport = LLMRemoteClientParams(
            model=model,
            api_base=api_base,
            api_key=api_key,
            num_retries=num_retries,
            timeout=timeout,
            extra_params=extra_params or {},
        )
        return cls(transport=transport)

    def judge(self, query: str, reference: str, candidate: str) -> JudgeResult:
        """Score a candidate answer against the reference answer."""
        if not candidate or not candidate.strip():
            return JudgeResult(score=None, reasoning="Candidate answer was empty.", error="empty_candidate")

        user_content = _JUDGE_USER_TEMPLATE.format(
            query=query,
            reference=reference,
            candidate=candidate,
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw, _ = self._client.complete(messages, max_tokens=256)
            return _parse_judge_response(raw)
        except Exception as exc:
            return JudgeResult(score=None, reasoning="", error=f"judge_api_error: {exc}")


def _parse_judge_response(raw: str) -> JudgeResult:
    """Parse the judge's JSON response into a JudgeResult."""
    text = raw.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        data = json.loads(text)
        score = int(data["score"])
        if not (1 <= score <= 5):
            raise ValueError(f"score {score} out of range 1-5")
        return JudgeResult(score=score, reasoning=str(data.get("reasoning", "")))
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    score_match = re.search(r'"score"\s*:\s*([1-5])', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    if score_match:
        score = int(score_match.group(1))
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        return JudgeResult(score=score, reasoning=reasoning)

    return JudgeResult(
        score=None,
        reasoning="",
        error=f"parse_failure: could not extract score from response: {raw[:200]!r}",
    )
