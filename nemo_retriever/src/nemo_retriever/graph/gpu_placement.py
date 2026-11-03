# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report where fractional-GPU actors landed and warn when the spread across GPUs is uneven."""

from __future__ import annotations

from collections import Counter
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_LEDGER_NAME = "nrl_gpu_placement_ledger"
_POLL_S = 10.0
_CHECK_TIMEOUT_S = 900.0
_RPC_TIMEOUT_S = 5.0


class _PlacementLedger:
    def __init__(self) -> None:
        self._rows: list[tuple[str, str]] = []

    def record(self, stage: str, gpu: str) -> None:
        self._rows.append((stage, gpu))

    def rows(self) -> list[tuple[str, str]]:
        return list(self._rows)


def report_gpu_placement(stage: str) -> None:
    """Record this actor's GPU in the running placement check, if there is one."""
    import ray

    try:
        ledger = ray.get_actor(_LEDGER_NAME)
    except ValueError:
        return
    node = ray.get_runtime_context().get_node_id()[:8]
    ledger.record.remote(stage, f"{node}:" + ",".join(str(gpu) for gpu in ray.get_gpu_ids()))


class _ReportingOperator:
    _stage = ""

    def __init__(self, operator_class: type, operator_kwargs: dict[str, Any]) -> None:
        report_gpu_placement(self._stage)
        self._operator = operator_class(**operator_kwargs)

    def __call__(self, batch: Any) -> Any:
        return self._operator(batch)


def with_placement_report(operator_class: type, stage: str) -> type:
    """Wrap a Ray Data callable class so each actor reports its GPU when it starts."""
    return type(operator_class.__name__, (_ReportingOperator,), {"_stage": stage})


def placement_problems(rows: list[tuple[str, str]], expected: dict[str, int], gpus: int) -> list[str]:
    """Return pending actors and uneven per-GPU counts for pools whose size divides evenly over the GPUs."""
    problems = []
    for stage, count in sorted(expected.items()):
        placed = Counter(gpu for name, gpu in rows if name == stage)
        started = sum(placed.values())
        if started < count:
            problems.append(f"{stage} has {count - started} of {count} actors pending")
        elif count % gpus == 0 and sorted(placed.values()) != [count // gpus] * gpus:
            per_gpu = dict(sorted(placed.items()))
            problems.append(f"{stage} actors per GPU {per_gpu}, expected {count // gpus} on each of {gpus} GPUs")
    return problems


class GpuPlacementCheck:
    """Collect actor GPU ids in a named Ray actor and log the per-GPU spread once every pool has started."""

    def __init__(self, ledger: Any, gpus: int) -> None:
        self._ledger = ledger
        self._gpus = gpus
        self._expected: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nrl-gpu-placement-check", daemon=True)
        self._thread.start()

    @classmethod
    def start(cls, gpus: int) -> GpuPlacementCheck | None:
        """Create the ledger and start polling it for a cluster with ``gpus`` GPUs.

        Returns None, and the stages are not checked, when another ingest's check already owns the ledger.
        """
        import ray

        try:
            ledger = ray.remote(num_cpus=0)(_PlacementLedger).options(name=_LEDGER_NAME).remote()
        except ValueError:
            logger.debug("Another ingest is checking GPU placement; skipping this check")
            return None
        return cls(ledger, gpus)

    def expect(self, stage: str, count: int) -> None:
        """Register ``count`` actors for ``stage``; a later call for the same stage replaces the count."""
        self._expected[stage] = count

    def stop(self) -> None:
        """Report the placement seen so far, even when the ingest ends before the next poll."""
        self._stop.set()
        self._thread.join(timeout=_RPC_TIMEOUT_S + 1)

    def _run(self) -> None:
        import ray

        deadline = time.monotonic() + _CHECK_TIMEOUT_S
        try:
            while True:
                stopped = self._stop.wait(_POLL_S)
                try:
                    rows = ray.get(self._ledger.rows.remote(), timeout=_RPC_TIMEOUT_S)
                except ray.exceptions.GetTimeoutError:
                    if stopped or time.monotonic() >= deadline:
                        return
                    continue
                expected = dict(self._expected)
                started = Counter(stage for stage, _gpu in rows)
                if (
                    stopped
                    or time.monotonic() >= deadline
                    or all(started[stage] >= count for stage, count in expected.items())
                ):
                    problems = placement_problems(rows, expected, self._gpus)
                    if problems:
                        logger.warning(
                            "GPU actors are unevenly placed: %s. Check whether another process holds these GPUs, "
                            "or set explicit worker counts and GPU shares.",
                            "; ".join(problems),
                        )
                    else:
                        logger.info(
                            "GPU actors per GPU: %s",
                            "; ".join(
                                f"{stage} {dict(sorted(Counter(g for s, g in rows if s == stage).items()))}"
                                for stage in sorted(expected)
                            ),
                        )
                    return
        except ray.exceptions.RayError as exc:
            logger.debug("GPU placement check stopped: %s", exc)
        finally:
            ray.kill(self._ledger)
