"""
LLM-as-judge scoring for the QA evaluation pipeline.

LLMJudge uses a strong LLM to score generated answers on a 1-5 scale
against a ground-truth reference answer. Scores are returned as structured
JSON so they can be aggregated programmatically.

Default judge model: nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1
"""

from __future__ import annotations

import json
import re

from typing import Any, Optional

from nv_ingest_harness.utils.qa.generators import LiteLLMClient
from nv_ingest_harness.utils.qa.types import JudgeResult

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
    """
    LLM-as-judge that scores candidate answers on a 1-5 scale.

    Uses a LiteLLMClient under the hood so the judge model is swappable
    via the same model-string convention as the generator.

    Args:
        model: litellm model string for the judge LLM.
        api_base: Override endpoint URL for private / local deployments.
        api_key: Explicit API key (prefer env vars).
        extra_params: Additional kwargs forwarded to litellm.completion.
    """

    def __init__(
        self,
        model: str = "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ):
        self._client = LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=0.0,
            max_tokens=256,
            extra_params=extra_params or {},
            num_retries=3,
        )
        self.model = model

    def judge(self, query: str, reference: str, candidate: str) -> JudgeResult:
        """
        Score a candidate answer against the reference answer.

        Uses self._client.complete() so retry logic, auth, and extra_params
        flow through LiteLLMClient -- no litellm calls here directly.

        Args:
            query: The original question.
            reference: Ground-truth reference answer.
            candidate: The LLM-generated answer to evaluate.

        Returns:
            JudgeResult with score (1-5) and one-sentence reasoning.
            Returns score=0 if parsing fails, so the pipeline can continue.
        """
        if not candidate or not candidate.strip():
            return JudgeResult(score=0, reasoning="Candidate answer was empty.", error="empty_candidate")

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
            return JudgeResult(score=0, reasoning="", error=f"judge_api_error: {exc}")


def _parse_judge_response(raw: str) -> JudgeResult:
    """
    Parse the judge's JSON response into a JudgeResult.

    Attempts strict JSON parsing first, then falls back to regex extraction
    to handle models that wrap the JSON in prose or code fences.
    """
    text = raw.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Strict JSON parse
    try:
        data = json.loads(text)
        score = int(data["score"])
        if not (1 <= score <= 5):
            raise ValueError(f"score {score} out of range 1-5")
        return JudgeResult(score=score, reasoning=str(data.get("reasoning", "")))
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fallback: regex extraction
    score_match = re.search(r'"score"\s*:\s*([1-5])', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    if score_match:
        score = int(score_match.group(1))
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        return JudgeResult(score=score, reasoning=reasoning)

    return JudgeResult(
        score=0,
        reasoning="",
        error=f"parse_failure: could not extract score from response: {raw[:200]!r}",
    )
