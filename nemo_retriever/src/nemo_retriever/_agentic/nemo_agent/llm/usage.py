# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-usage attribution and aggregation for LLM backends.

Attribution
-----------
Two independent ContextVars label every LLM call:

- **query id** — bound once per query by the pipeline/driver
  (:func:`bind_query_id`).
- **stage** — bound per call site by agent code (:func:`bind_stage`), e.g.
  ``"main_agent"`` or ``"top5_agent"``.

``BaseLLMBackend`` reads both after every successful call and deep-merges
``CompletionResult.usage`` into ``usage[query_id][stage]``. Unbound values fall
back to the public sentinels :data:`UNSET_QUERY` / :data:`UNSET_STAGE` (these
names appear in persisted traces — treat them as schema).

Always prefer the context-manager binders over raw ``ContextVar.set``: they
reset to the *previous* value on exit, so nested scopes compose and nothing
leaks across queries within a task.

Threading caveat
----------------
``contextvars`` propagate into ``asyncio`` tasks and ``asyncio.to_thread``
automatically, but **worker threads do NOT inherit the spawning thread's
context** (e.g. ``ThreadPoolExecutor``). Bind inside the thread/task that makes
the LLM call, or wrap submissions with ``contextvars.copy_context().run``.
Otherwise usage silently lands in the ``UNSET_*`` buckets.

Aggregation rules (contract of :func:`deep_merge_usage`)
---------------------------------------------------------
Usage dicts across providers are heterogeneous (nested ``*_tokens_details``,
cache tiers that appear mid-run, ``None`` placeholders). For each key ``k``
across the running aggregate and an incoming sample:

1. ``sample[k]`` is ``None`` and ``acc[k]`` missing or ``None`` -> store/keep ``None``.
2. ``sample[k]`` is ``None`` and ``acc[k]`` already typed (int or dict) -> no-op.
3. ``sample[k]`` non-``None`` and ``acc[k]`` missing or ``None`` -> deep-copy
   ``sample[k]`` into ``acc[k]`` (the type-lock event).
4. dict + dict -> recurse.
5. int + int -> sum.
6. Type mismatch (locked-int meets dict or vice versa) -> :class:`TypeError`
   carrying the dotted key path. A real schema regression should be loud.

Booleans are int subclasses in Python; they are explicitly excluded from the
"int sum" branch so a ``True`` never becomes ``1`` of someone's
``prompt_tokens``. Other scalar types are out of contract; the helper keeps the
most recent value rather than crashing on a provider extension.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator, Mapping, Optional

UNSET_QUERY = "<unset_query>"
"""Usage bucket for calls made with no bound query id (e.g. preflight probes).

Angle-bracketed so it cannot collide with a real query id; reserved — do not
bind it as an actual query id.
"""

UNSET_STAGE = "<unset_stage>"
"""Usage bucket for calls made with no bound stage. Reserved, like UNSET_QUERY."""

_MISSING = object()
"""Sentinel distinguishing 'key not yet seen' from 'key seen as None'."""

_QUERY_ID: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "agent_core_llm_query_id",
    default=None,
)
_STAGE: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "agent_core_llm_stage",
    default=None,
)


def get_query_id() -> Optional[str]:
    """Return the ambient usage query id, or ``None`` when unbound."""
    return _QUERY_ID.get()


def get_stage() -> Optional[str]:
    """Return the ambient usage stage, or ``None`` when unbound."""
    return _STAGE.get()


@contextmanager
def bind_query_id(query_id: Optional[str]) -> Iterator[None]:
    """Scoped binding of the usage query id (typically once per query)."""
    token = _QUERY_ID.set(query_id)
    try:
        yield
    finally:
        _QUERY_ID.reset(token)


@contextmanager
def bind_stage(stage: Optional[str]) -> Iterator[None]:
    """Scoped binding of the usage stage (typically around each LLM call site)."""
    token = _STAGE.set(stage)
    try:
        yield
    finally:
        _STAGE.reset(token)


def deep_merge_usage(
    acc: Optional[Dict[str, Any]],
    sample: Optional[Mapping[str, Any]],
    *,
    _path: str = "",
) -> Dict[str, Any]:
    """Aggregate ``sample`` into ``acc`` in place; return ``acc``.

    See the module docstring for the full rule contract.

    Parameters
    ----------
    acc:
        Running aggregate. ``None`` is treated as an empty dict (a fresh one is
        created and returned).
    sample:
        New usage snapshot. ``None`` and empty mappings are no-ops.

    Raises
    ------
    TypeError
        On a type-lock conflict; the message includes the dotted key path so
        schema regressions are easy to localize.
    """
    if acc is None:
        acc = {}
    if sample is None:
        return acc
    for k, v in sample.items():
        path = f"{_path}.{k}" if _path else k
        cur = acc.get(k, _MISSING)

        if v is None:
            # Rule 1: nothing seen yet -> remember the slot exists, value None.
            # Rule 2: already typed -> no-op.
            if cur is _MISSING:
                acc[k] = None
            continue

        if isinstance(v, dict):
            if cur is _MISSING or cur is None:
                # Rule 3: type-lock event for a dict slot.
                acc[k] = deepcopy(v)
            elif isinstance(cur, dict):
                # Rule 4: dict + dict -> recurse.
                deep_merge_usage(cur, v, _path=path)
            else:
                # Rule 6: int slot, sample is dict.
                raise TypeError(
                    f"deep_merge_usage: type lock conflict at {path!r}: " f"acc is {type(cur).__name__}, sample is dict"
                )
            continue

        # int (excluding bool, which is an int subclass).
        if isinstance(v, int) and not isinstance(v, bool):
            if cur is _MISSING or cur is None:
                # Rule 3: type-lock event for an int slot.
                acc[k] = int(v)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                # Rule 5: int + int -> sum.
                acc[k] = cur + int(v)
            else:
                # Rule 6: dict (or other) slot, sample is int.
                raise TypeError(
                    f"deep_merge_usage: type lock conflict at {path!r}: " f"acc is {type(cur).__name__}, sample is int"
                )
            continue

        # Out-of-contract scalar (float, str, bool, list, ...): keep the latest
        # value as a safety net rather than crashing on a provider extension.
        acc[k] = v

    return acc


def deep_merge_usage_breakdown(
    acc: Optional[Dict[str, Any]],
    sample: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Deep-merge a stage-keyed usage breakdown (``{stage: usage}``) into ``acc``."""
    if acc is None:
        acc = {}
    if sample is None:
        return acc
    for stage, usage in sample.items():
        if not isinstance(stage, str) or not stage:
            continue
        if not isinstance(usage, Mapping) or not usage:
            continue
        stage_acc = acc.get(stage)
        if not isinstance(stage_acc, dict):
            stage_acc = {}
            acc[stage] = stage_acc
        deep_merge_usage(stage_acc, usage)
    return acc


def sum_usage_breakdown(usage_by_stage: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collapse a stage-keyed usage breakdown into one total usage dict."""
    total: Dict[str, Any] = {}
    if usage_by_stage is None:
        return total
    for usage in usage_by_stage.values():
        if isinstance(usage, Mapping) and usage:
            deep_merge_usage(total, usage)
    return total


def usage_integer(usage: Mapping[str, Any], *keys: str) -> Optional[int]:
    """Return the first integer usage value found under ``keys``."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def cache_read_tokens(usage: Mapping[str, Any]) -> Optional[int]:
    """Return provider-reported cache-read input tokens, when available."""
    cached = usage_integer(
        usage,
        "cached_tokens",
        "cached_input_tokens",
        "cache_read_input_tokens",
    )
    if cached is not None:
        return cached
    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, Mapping):
            cached = usage_integer(details, "cached_tokens")
            if cached is not None:
                return cached
    return None


def normalize_usage_breakdown(usage_by_stage: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return stable token totals plus the exact provider stage breakdown.

    Usage schemas use both ``prompt_tokens`` / ``completion_tokens`` and
    ``input_tokens`` / ``output_tokens`` names. Values are copied from the
    provider aggregate. Separately reported cache-write and cache-read input
    counters are included in the input total. ``cache_tokens`` sums observed
    provider-reported cache reads; cache creation remains available only in the
    stage breakdown because it has different pricing semantics. ``total_tokens``
    is only filled by exact addition when both component totals are present.
    """
    if not usage_by_stage:
        return {}

    stages = deepcopy(dict(usage_by_stage))

    def _token_count(
        usage: Mapping[str, Any],
        aliases: tuple[str, ...],
        *,
        additive_keys: tuple[str, ...] = (),
    ) -> Optional[int]:
        value = usage_integer(usage, *aliases)
        additions = [usage_integer(usage, key) for key in additive_keys]
        if not any(addition is not None for addition in additions):
            return value
        if value is None:
            return None
        return value + sum(addition or 0 for addition in additions)

    input_by_stage: list[int | None] = []
    cache_by_stage: list[int | None] = []
    output_by_stage: list[int | None] = []
    total_by_stage: list[int | None] = []
    for usage in usage_by_stage.values():
        if not isinstance(usage, Mapping) or not usage:
            continue
        stage_input = _token_count(
            usage,
            ("input_tokens", "prompt_tokens"),
            additive_keys=("cache_creation_input_tokens", "cache_read_input_tokens"),
        )
        stage_cache = cache_read_tokens(usage)
        stage_output = _token_count(usage, ("output_tokens", "completion_tokens"))
        stage_total = usage_integer(usage, "total_tokens")
        if stage_total is None and stage_input is not None and stage_output is not None:
            stage_total = stage_input + stage_output
        input_by_stage.append(stage_input)
        cache_by_stage.append(stage_cache)
        output_by_stage.append(stage_output)
        total_by_stage.append(stage_total)

    def _complete_sum(values: list[int | None]) -> Optional[int]:
        return (
            sum(value for value in values if value is not None)
            if values and all(v is not None for v in values)
            else None
        )

    def _observed_sum(values: list[int | None]) -> Optional[int]:
        observed = [value for value in values if value is not None]
        return sum(observed) if observed else None

    return {
        "input_tokens": _complete_sum(input_by_stage),
        "cache_tokens": _observed_sum(cache_by_stage),
        "output_tokens": _complete_sum(output_by_stage),
        "total_tokens": _complete_sum(total_by_stage),
        "stages": stages,
    }


def coerce_usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion of a provider usage object to a plain dict.

    Handles pydantic v2 models (``model_dump``), pydantic v1 / namedtuple-ish
    objects (``dict``), and ``SimpleNamespace`` / plain objects (``__dict__``),
    in that order. ``None`` and unsupported shapes return ``None`` so callers
    can short-circuit.
    """
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    # pydantic v2
    md = getattr(usage, "model_dump", None)
    if callable(md):
        try:
            out = md()
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    # pydantic v1 / namedtuple-ish
    d = getattr(usage, "dict", None)
    if callable(d):
        try:
            out = d()
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    # SimpleNamespace / plain object
    raw = getattr(usage, "__dict__", None)
    if isinstance(raw, dict):
        return dict(raw)
    return None
