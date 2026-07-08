# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import operator
import re
from typing import Any, Callable, Mapping, Sequence

from nemo_retriever.harness.contracts import (
    EXIT_INVALID,
    EXIT_METRIC_GATE_FAILURE,
    FailurePayload,
    HarnessRunError,
)

_GATE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<|=)\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*$")
_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "=": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class MetricGate:
    key: str
    op: str
    threshold: float

    def expression(self) -> str:
        threshold = int(self.threshold) if self.threshold.is_integer() else self.threshold
        return f"{self.key}{self.op}{threshold}"


def parse_metric_gates(requirements: Sequence[str]) -> tuple[MetricGate, ...]:
    gates: list[MetricGate] = []
    for raw in requirements:
        match = _GATE_RE.match(str(raw))
        if not match:
            raise HarnessRunError(
                EXIT_INVALID,
                FailurePayload(
                    failed_phase="resolve",
                    failure_reason="invalid_metric_gate",
                    retryable=False,
                    message=f"Metric gate must look like recall_5>=0.80, got: {raw}",
                ),
            )
        key, op, threshold = match.groups()
        gates.append(MetricGate(key=key, op=op, threshold=float(threshold)))
    return tuple(gates)


def enforce_metric_gates(
    summary_metrics: Mapping[str, Any],
    requirements: Sequence[str],
    *,
    skip_missing: bool = False,
) -> tuple[str, ...]:
    gates = parse_metric_gates(requirements)
    failures: list[str] = []
    skipped: list[str] = []
    for gate in gates:
        raw_value = summary_metrics.get(gate.key)
        if skip_missing and raw_value is None:
            skipped.append(gate.expression())
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            failures.append(f"{gate.expression()} failed: {gate.key} is {raw_value!r}")
            continue
        if not _OPS[gate.op](value, gate.threshold):
            failures.append(f"{gate.expression()} failed: observed {value:g}")

    if failures:
        raise HarnessRunError(
            EXIT_METRIC_GATE_FAILURE,
            FailurePayload(
                failed_phase="evaluate",
                failure_reason="metric_gate_failed",
                retryable=False,
                message="; ".join(failures),
                debug_artifacts=(),
            ),
        )
    return tuple(skipped)
