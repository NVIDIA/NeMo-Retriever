# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-operator span emission from inside Ray Data map-batches actors.

Ray actors do not see the driver's active OpenTelemetry context, and the
map-batches API provides no built-in hook for wrapping each batch call.
The span wrapper class in this module solves these problems. It
constructs the real operator inside the actor and wraps every batch in
an operator span parented to the driver's pipeline span via a W3C trace
context carrier. Actor-side provider bootstrap is handled separately by
opentelemetry-distro, which reads the standard OTEL environment
variables forwarded through Ray's runtime environment.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from nemo_retriever.observability.propagate import extract_context
from nemo_retriever.observability.spans import operator_span


# Broad prefix so custom OTEL_* settings (sampler, propagator, log level)
# all flow through to actors.
_OTEL_ENV_PREFIXES = ("OTEL_",)


def collect_otel_env() -> dict[str, str]:
    """Snapshot ``OTEL_*`` env vars for merging into Ray ``runtime_env.env_vars``."""
    return {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in _OTEL_ENV_PREFIXES)}


class RayOperatorSpanWrapper:
    """Wraps each Ray map-batches call in an operator span.

    Deliberately not an AbstractOperator subclass: map-batches only needs
    a callable, and inheritance would pull in unwanted constructor and
    run-method side effects.
    """

    def __init__(
        self,
        *,
        _otel_operator_class: type,
        _otel_node_name: str,
        _otel_node_index: int,
        _otel_parent_carrier: Mapping[str, str] | None = None,
        _otel_operator_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._operator = _otel_operator_class(**dict(_otel_operator_kwargs or {}))
        self._operator_class_name = _otel_operator_class.__name__
        self._node_name = _otel_node_name
        self._node_index = int(_otel_node_index)
        # Extract once at construction instead of re-parsing per batch.
        self._parent_ctx = extract_context(_otel_parent_carrier or {})

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        batch_size = len(data) if hasattr(data, "__len__") else None
        with operator_span(
            self._node_name,
            run_mode="batch",
            operator_class=self._operator_class_name,
            operator_index=self._node_index,
            batch_size=batch_size,
            parent_context=self._parent_ctx,
        ):
            return self._operator(data, **kwargs)
