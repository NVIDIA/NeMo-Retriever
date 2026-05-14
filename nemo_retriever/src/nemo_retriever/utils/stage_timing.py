# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage / per-batch timing for the Ray batch executor.

When the ``NR_STAGE_TIMING=1`` environment variable is set, the
``RayDataExecutor`` starts a detached named Ray actor that collects
records emitted by :class:`AbstractOperator.run` on every worker.
After the pipeline materialises, the executor pulls the records,
combines them with ``ds.stats()`` text, and writes a human-readable
report (and optional JSON dump via ``NR_STAGE_TIMING_REPORT_PATH``).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

COLLECTOR_NAME = "nr_stage_timing_collector"
ENABLED_ENV = "NR_STAGE_TIMING"
REPORT_PATH_ENV = "NR_STAGE_TIMING_REPORT_PATH"

# Driver-side counter of RayDataExecutor.ingest() invocations within this process.
# Used by Phase 1 diagnostics to identify each graph execution distinctly.
_INGEST_CALL_COUNTER = 0
_INGEST_CALL_COUNTER_LOCK = threading.Lock()

# Captured once per process so every report file from one run sorts together.
_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_report_path(call_index: Optional[int], graph_label: Optional[str]) -> Optional[Path]:
    """Compute the JSON output path for one timing report, or ``None`` if file writes are disabled.

    The ``NR_STAGE_TIMING_REPORT_PATH`` env var is treated as a *base*:

    * If it points to an existing directory (or ends with ``/``), files land
      inside it as ``timing_<ts>_<NN>_<label>.json``.
    * Otherwise it is split into a parent directory and a stem; files are
      written next to it as ``<stem>_<ts>_<NN>_<label>.json``.  This means
      ``NR_STAGE_TIMING_REPORT_PATH=/tmp/timing.json`` still works — you
      get ``/tmp/timing_<ts>_<NN>_<label>.json`` instead of a single
      overwritten ``/tmp/timing.json``.
    """
    env_value = os.environ.get(REPORT_PATH_ENV)
    if not env_value:
        return None
    p = Path(env_value)
    if env_value.endswith(os.sep) or env_value.endswith("/") or (p.exists() and p.is_dir()):
        out_dir = p
        stem = "timing"
    else:
        out_dir = p.parent if str(p.parent) else Path(".")
        stem = p.stem if p.suffix else p.name
        if not stem:
            stem = "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = f"{call_index:02d}" if call_index is not None else "00"
    label = graph_label or "graph"
    return out_dir / f"{stem}_{_RUN_TIMESTAMP}_{idx}_{label}.json"


def next_call_index() -> int:
    """Increment and return the per-process ingest-call counter (starts at 1)."""
    global _INGEST_CALL_COUNTER
    with _INGEST_CALL_COUNTER_LOCK:
        _INGEST_CALL_COUNTER += 1
        return _INGEST_CALL_COUNTER


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def make_named_operator_class(base_cls: type, stage_name: str) -> type:
    """Return a per-node subclass of *base_cls* carrying ``_nr_stage_name=stage_name``.

    Two pipeline nodes can share the same operator class (e.g. duplicate
    embedders).  Setting the attribute on the shared class would let one
    node clobber another's stage label; per-node subclasses avoid that.
    cloudpickle (which Ray Data uses) can serialise such dynamically
    created subclasses by-value.
    """
    return type(
        base_cls.__name__,
        (base_cls,),
        {"_nr_stage_name": stage_name, "__module__": base_cls.__module__},
    )


def slugify_graph_label(node_names: Iterable[str], *, max_len: int = 40) -> str:
    """Build a short, filesystem-safe label from the ordered node names of a graph."""
    parts = []
    for name in node_names:
        slug = _SLUG_RE.sub("-", str(name).lower()).strip("-")
        if slug:
            parts.append(slug)
    label = "-".join(parts) if parts else "graph"
    if len(label) > max_len:
        label = label[: max_len - 1].rstrip("-") + "+"
    return label


@dataclass
class StageRecord:
    stage: str
    n_rows_in: int
    n_rows_out: int
    preprocess_ms: float
    process_ms: float
    postprocess_ms: float
    total_ms: float
    worker_pid: int
    wallclock_start: float
    extra: Dict[str, Any] = field(default_factory=dict)


def is_enabled() -> bool:
    """Return True if stage timing is enabled via the env var."""
    return os.environ.get(ENABLED_ENV, "").lower() in ("1", "true", "yes", "on")


# Cache the actor handle per-worker so we don't pay the ray.get_actor cost per batch.
_collector_cache: Dict[str, Any] = {"handle": None, "tried": False}


def _get_collector() -> Any:
    if _collector_cache["tried"]:
        return _collector_cache["handle"]
    _collector_cache["tried"] = True
    try:
        import ray  # local import: ray may not be installed in non-batch contexts

        _collector_cache["handle"] = ray.get_actor(COLLECTOR_NAME)
    except Exception:
        _collector_cache["handle"] = None
    return _collector_cache["handle"]


def record_timing(**fields: Any) -> None:
    """Fire-and-forget submission of a :class:`StageRecord` to the collector.

    Safe to call from any worker. If the collector actor cannot be found
    (e.g. timing disabled, or running outside Ray), this is a no-op.
    """
    handle = _get_collector()
    if handle is None:
        return
    try:
        handle.record.remote(StageRecord(**fields))
    except Exception:
        # Never let timing break the pipeline.
        pass


def _build_collector_class() -> Any:
    import ray

    @ray.remote(num_cpus=0)
    class StageTimingCollector:
        def __init__(self) -> None:
            self._records: List[StageRecord] = []

        def record(self, rec: StageRecord) -> None:
            self._records.append(rec)

        def dump(self) -> List[Dict[str, Any]]:
            return [asdict(r) for r in self._records]

        def clear(self) -> None:
            self._records.clear()

    return StageTimingCollector


def start_collector() -> Any:
    """Create the named, detached collector actor. Idempotent."""
    import ray

    try:
        return ray.get_actor(COLLECTOR_NAME)
    except Exception:
        pass
    cls = _build_collector_class()
    return cls.options(name=COLLECTOR_NAME, lifetime="detached").remote()


def stop_collector(handle: Any) -> None:
    """Kill the collector actor if it exists. Safe to call with None."""
    if handle is None:
        return
    try:
        import ray

        ray.kill(handle)
    except Exception:
        pass


def _fmt_int(n: int) -> str:
    return f"{n:,}" if n >= 0 else "-"


def format_report(
    records: List[Dict[str, Any]],
    ray_stats_text: Optional[str] = None,
    *,
    call_index: Optional[int] = None,
    graph_label: Optional[str] = None,
    node_names: Optional[Iterable[str]] = None,
) -> str:
    """Build a human-readable per-stage timing report from collected records."""
    if not records:
        body = "(no stage timing records were collected)"
    else:
        by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_stage[r["stage"]].append(r)

        cols = ("stage", "batches", "rows_in", "total_s",
                "pre_ms/b", "proc_ms/b", "post_ms/b", "ms/row")
        widths = (34, 9, 11, 10, 11, 11, 11, 10)
        header = "".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(cols, widths)))
        sep = "-" * len(header)
        lines = [header, sep]

        pipeline_total_ms = 0.0
        for stage, recs in sorted(by_stage.items(), key=lambda kv: -sum(r["total_ms"] for r in kv[1])):
            n_batches = len(recs)
            rows_in = sum(max(r["n_rows_in"], 0) for r in recs)
            total_ms = sum(r["total_ms"] for r in recs)
            pre_avg = sum(r["preprocess_ms"] for r in recs) / n_batches
            proc_avg = sum(r["process_ms"] for r in recs) / n_batches
            post_avg = sum(r["postprocess_ms"] for r in recs) / n_batches
            ms_per_row = total_ms / rows_in if rows_in > 0 else 0.0
            pipeline_total_ms += total_ms
            row = (
                stage[: widths[0]].ljust(widths[0])
                + _fmt_int(n_batches).rjust(widths[1])
                + _fmt_int(rows_in).rjust(widths[2])
                + f"{total_ms / 1000:.2f}".rjust(widths[3])
                + f"{pre_avg:.2f}".rjust(widths[4])
                + f"{proc_avg:.2f}".rjust(widths[5])
                + f"{post_avg:.2f}".rjust(widths[6])
                + f"{ms_per_row:.3f}".rjust(widths[7])
            )
            lines.append(row)
        lines.append(sep)
        lines.append(f"sum of stage wall-time (worker-side, parallel): {pipeline_total_ms / 1000:.2f} s")
        body = "\n".join(lines)

    title = "NeMo Retriever - Stage Timing Report"
    if call_index is not None:
        title += f"  (graph #{call_index:02d}"
        if graph_label:
            title += f": {graph_label}"
        title += ")"
    header_lines = [
        "=" * 96,
        title,
        "=" * 96,
        "",
        "Per-stage totals and per-batch / per-row averages (worker-side timing).",
        "preprocess/process/postprocess columns are mean milliseconds per batch.",
        "ms/row = total_ms / sum(rows_in)  - the per-chunk average across all batches.",
    ]
    if node_names:
        nodes_str = " -> ".join(node_names)
        if len(nodes_str) > 200:
            nodes_str = nodes_str[:197] + "..."
        header_lines.append(f"graph nodes: {nodes_str}")
    header_lines.append("")
    out = header_lines + [body]
    if ray_stats_text:
        out += [
            "",
            "=" * 96,
            "Ray Data ds.stats() (driver-side wall time per stage)",
            "=" * 96,
            ray_stats_text.rstrip(),
        ]
    return "\n".join(out)


def write_report(
    records: List[Dict[str, Any]],
    ray_stats_text: Optional[str] = None,
    *,
    call_index: Optional[int] = None,
    graph_label: Optional[str] = None,
    node_names: Optional[Iterable[str]] = None,
) -> str:
    """Format and emit the report. Returns the report text."""
    node_list = list(node_names) if node_names is not None else None
    text = format_report(
        records,
        ray_stats_text,
        call_index=call_index,
        graph_label=graph_label,
        node_names=node_list,
    )
    logger.info("\n%s", text)
    out_path = resolve_report_path(call_index, graph_label)
    if out_path is not None:
        try:
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "call_index": call_index,
                        "graph_label": graph_label,
                        "node_names": node_list,
                        "run_timestamp": _RUN_TIMESTAMP,
                        "records": records,
                        "ray_stats": ray_stats_text,
                        "report": text,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info("Stage timing report written to %s", out_path)
        except Exception as exc:
            logger.warning("Failed to write stage timing report to %s: %s", out_path, exc)
    return text
