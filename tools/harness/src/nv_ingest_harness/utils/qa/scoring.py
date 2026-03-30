"""
Programmatic scoring functions for multi-tier QA evaluation.

All functions in this module are pure computation with no LLM dependencies.
They provide Tier-1 (retrieval quality) and Tier-2 (answer quality) metrics
that complement the Tier-3 LLM-as-judge score.

Functions:
  answer_in_context  -- Tier 1: are reference answer keywords in the chunks?
  token_f1           -- Tier 2: SQuAD-style token precision/recall/F1
  classify_failure   -- Per-query failure mode classification
"""

from __future__ import annotations

import re
import string
from typing import Optional

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up", "it",
    "its", "that", "this", "these", "those", "what", "which", "who", "whom",
})

_NO_CONTEXT_PATTERNS = [
    r"does not contain",
    r"do not contain",
    r"no information",
    r"not enough information",
    r"cannot be determined",
    r"not mentioned",
    r"not provided",
    r"context does not",
    r"unable to find",
    r"no relevant",
]
_NO_CONTEXT_RE = re.compile("|".join(_NO_CONTEXT_PATTERNS), re.IGNORECASE)


def _normalize(text: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _normalize_numeric(text: str) -> str:
    """Normalize numeric formats so '16.00%' matches '16%', '1,000' matches '1000'."""
    text = re.sub(r",(\d{3})", r"\1", text)
    text = re.sub(r"(\d+)\.0+(%?)", r"\1\2", text)
    return text


def _content_words(text: str) -> list[str]:
    """Extract content words (non-stopwords) from normalized text."""
    normalized = _normalize(_normalize_numeric(text))
    return [w for w in normalized.split() if w not in _STOP_WORDS]


def answer_in_context(reference: str, chunks: list[str]) -> bool:
    """
    Tier-1 retrieval quality check.

    Returns True if >= 50% of the content words in the reference answer
    appear in the concatenated chunk text. Case-insensitive with numeric
    normalization (e.g. '16.00%' matches '16%').
    """
    ref_words = _content_words(reference)
    if not ref_words:
        return True

    chunk_text = _normalize(_normalize_numeric(" ".join(chunks)))
    found = sum(1 for w in ref_words if w in chunk_text)
    return found / len(ref_words) >= 0.5


def token_f1(reference: str, candidate: str) -> dict:
    """
    Tier-2 SQuAD-style token-level scoring after normalization.

    Strips <think> tags from candidate before scoring. Returns a dict with:
      exact_match: bool -- perfect string match after normalization
      f1: float         -- harmonic mean of precision and recall
      precision: float  -- fraction of candidate tokens in reference
      recall: float     -- fraction of reference tokens in candidate
    """
    from nv_ingest_harness.utils.qa.generators import strip_think_tags

    candidate = strip_think_tags(candidate)

    norm_ref = _normalize(_normalize_numeric(reference))
    norm_cand = _normalize(_normalize_numeric(candidate))

    exact = norm_ref == norm_cand

    ref_tokens = norm_ref.split()
    cand_tokens = norm_cand.split()

    if not ref_tokens and not cand_tokens:
        return {"exact_match": True, "f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not ref_tokens or not cand_tokens:
        return {"exact_match": exact, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    common = set(ref_tokens) & set(cand_tokens)
    num_common = sum(min(ref_tokens.count(t), cand_tokens.count(t)) for t in common)

    if num_common == 0:
        return {"exact_match": exact, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = num_common / len(cand_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "exact_match": exact,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def classify_failure(
    ref_in_chunks: bool,
    judge_score: Optional[int],
    gen_error: Optional[str],
    candidate: str,
) -> str:
    """
    Classify the failure mode for a single query + model combination.

    Returns one of:
      "correct"             -- judge score >= 4
      "partial"             -- judge score 2-3
      "retrieval_miss"      -- reference not in chunks AND score <= 2
      "generation_miss"     -- reference in chunks but score <= 2
      "thinking_truncated"  -- generator flagged truncated reasoning
      "no_context"          -- candidate says 'no information' AND score <= 2
    """
    if gen_error == "thinking_truncated":
        return "thinking_truncated"

    if judge_score is None:
        return "thinking_truncated"

    if judge_score >= 4:
        return "correct"

    if 2 <= judge_score <= 3:
        if _NO_CONTEXT_RE.search(candidate):
            return "no_context"
        return "partial"

    if judge_score <= 1:
        if _NO_CONTEXT_RE.search(candidate):
            return "no_context"
        if not ref_in_chunks:
            return "retrieval_miss"
        return "generation_miss"

    return "partial"
