# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed result and error records returned by the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Error categories. Downstream code branches on these constants (via
# ``AgentRunResult.error.category``), never on error-message text.
ERROR_MAX_STEPS = "max_steps"
ERROR_BAD_FINISH_REASON = "bad_finish_reason"
ERROR_CONTEXT_LIMIT = "context_limit"
ERROR_CONTENT_POLICY = "content_policy"
ERROR_LLM_CALL_FAILED = "llm_call_failed"
ERROR_TOOL_FAILED = "tool_failed"
ERROR_UNEXPECTED = "unexpected"


@dataclass
class AgentError:
    """Why an agent run ended without a successful end-tool call.

    Attributes
    ----------
    category:
        One of the ``ERROR_*`` constants in this module.
    message:
        Human-readable description (also appended to the trajectory as a
        ``role="agent_error"`` message). Do not branch on it.
    exception_class:
        Class name of the underlying exception, or ``None`` for the two
        normal terminations (``max_steps``, ``bad_finish_reason``).
    """

    category: str
    message: str
    exception_class: Optional[str] = None


@dataclass
class AgentRunResult:
    """Everything one agent run produces.

    ``final_doc_ids``, ``answer`` and ``citations`` are conveniences
    extracted from ``end_payload``; which are filled follows the agent's mode.
    ``citations`` separates ``[]`` (the model reported no supporting document)
    from ``None`` (it never gave a usable list). On a successful run ``end_payload`` is the full
    *validated* end-tool arguments. When a run FAILS without ever making a
    valid end call, it falls back to the agent's last invalid end attempt (a
    lenient best-effort subset) so callers still get the model's final intent;
    ``error`` still says why the run failed and ``succeeded`` stays ``False``.
    It is ``None`` only when no usable end attempt was ever made.

    The verbose fields (``trajectory``, ``retrieval_log``, ``extra_data``)
    are always populated. ``atif_trace`` is the lightweight ATIF rendering of
    the run when trace construction succeeded.
    """

    final_doc_ids: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    citations: Optional[List[str]] = None
    end_payload: Optional[Dict[str, Any]] = None
    error: Optional[AgentError] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_log: List[Dict[str, Any]] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    atif_trace: Optional[Dict[str, Any]] = None

    @property
    def succeeded(self) -> bool:
        """True iff the run ended via a successful end-tool call."""
        return self.error is None
