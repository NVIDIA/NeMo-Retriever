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
    # Memory metrics (host-side, per worker process). All zero when psutil is
    # unavailable. ``rss_peak_mb`` is the worker's process-lifetime high-water
    # mark sampled at the end of the batch -- it only grows, so the diff across
    # consecutive batches tells you which batch pushed the peak up.
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_peak_mb: float = 0.0
    avail_before_mb: float = 0.0
    avail_after_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSample:
    """A single driver-side snapshot of host + Ray-worker memory.

    Memory fields use **PSS** (proportional set size) on Linux so that shared
    library / mmap pages are not double-counted across processes -- the sum
    of PSS across processes ~= actual physical memory used.  When PSS is
    unavailable (non-Linux), the sampler transparently falls back to RSS;
    field names stay the same to keep the JSON schema stable.
    """

    t_rel_s: float
    driver_pss_mb: float
    workers_pss_mb: float
    workload_pss_mb: float  # driver + workers (PSS)
    sys_used_mb: float  # host-wide MemUsed from psutil.virtual_memory().used
    sys_available_mb: float
    sys_used_pct: float
    n_workers: int


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
            self._samples: List[SystemSample] = []

        def record(self, rec: StageRecord) -> None:
            self._records.append(rec)

        def record_sample(self, sample: SystemSample) -> None:
            self._samples.append(sample)

        def dump(self) -> List[Dict[str, Any]]:
            return [asdict(r) for r in self._records]

        def dump_samples(self) -> List[Dict[str, Any]]:
            return [asdict(s) for s in self._samples]

        def clear(self) -> None:
            self._records.clear()
            self._samples.clear()

    return StageTimingCollector


def _enumerate_ray_worker_processes() -> List[Any]:
    """Return psutil.Process handles for Ray worker processes on this host.

    Ray worker processes are named ``ray::<actor_class>`` (or ``ray::IDLE``).
    Returns an empty list if psutil is unavailable.
    """
    try:
        import psutil
    except Exception:
        return []
    workers: List[Any] = []
    try:
        for p in psutil.process_iter(["pid", "name"]):
            try:
                name = p.info.get("name") or ""
                if name.startswith("ray::"):
                    workers.append(p)
            except Exception:
                continue
    except Exception:
        return []
    return workers


def _process_pss_bytes(p: Any) -> int:
    """Return PSS in bytes for a psutil.Process, falling back to RSS off-Linux."""
    try:
        return int(p.memory_full_info().pss)
    except (AttributeError, OSError, Exception):
        try:
            return int(p.memory_info().rss)
        except Exception:
            return 0


class _MemorySampler(threading.Thread):
    """Daemon thread that snapshots driver + worker memory every ``interval_s`` seconds.

    Memory uses PSS (proportional set size) on Linux: a page mapped by N
    processes contributes ``1/N`` of its size to each.  Summing PSS across
    processes therefore approximates true physical-memory usage, unlike
    summing RSS which double-counts shared library / mmap pages.
    """

    def __init__(self, collector_handle: Any, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True)
        self._collector = collector_handle
        self._interval = float(interval_s)
        self._stop = threading.Event()
        self._t0 = 0.0
        # Captured eagerly in start_memory_sampler before run() begins.
        self.baseline_sys_used_mb: float = 0.0

    def run(self) -> None:  # pragma: no cover - thread loop
        import time as _time

        try:
            import psutil
        except Exception:
            return
        try:
            driver = psutil.Process()
        except Exception:
            return
        self._t0 = _time.perf_counter()
        while not self._stop.wait(self._interval):
            try:
                drv_mem = _process_pss_bytes(driver)
                workers = _enumerate_ray_worker_processes()
                worker_mem = 0
                live = 0
                for p in workers:
                    v = _process_pss_bytes(p)
                    if v > 0:
                        worker_mem += v
                        live += 1
                vm = psutil.virtual_memory()
                sample = SystemSample(
                    t_rel_s=_time.perf_counter() - self._t0,
                    driver_pss_mb=drv_mem / 1e6,
                    workers_pss_mb=worker_mem / 1e6,
                    workload_pss_mb=(drv_mem + worker_mem) / 1e6,
                    sys_used_mb=float(vm.used) / 1e6,
                    sys_available_mb=vm.available / 1e6,
                    sys_used_pct=float(vm.percent),
                    n_workers=live,
                )
                try:
                    self._collector.record_sample.remote(sample)
                except Exception:
                    pass
            except Exception:
                continue

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout=timeout)


def start_memory_sampler(collector_handle: Any, interval_s: float = 1.0) -> Optional[_MemorySampler]:
    """Start a daemon sampler thread.

    Also captures a baseline of ``psutil.virtual_memory().used`` *before*
    sampling begins so the report can show "memory the run added" rather
    than absolute host usage.
    """
    if collector_handle is None:
        return None
    sampler = _MemorySampler(collector_handle, interval_s=interval_s)
    try:
        import psutil

        sampler.baseline_sys_used_mb = float(psutil.virtual_memory().used) / 1e6
    except Exception:
        sampler.baseline_sys_used_mb = 0.0
    sampler.start()
    return sampler


def stop_memory_sampler(sampler: Optional[_MemorySampler]) -> None:
    """Stop the sampler thread (no-op if ``None``)."""
    if sampler is not None:
        try:
            sampler.stop()
        except Exception:
            pass


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
    memory_samples: Optional[List[Dict[str, Any]]] = None,
    baseline_sys_used_mb: Optional[float] = None,
) -> str:
    """Build a human-readable per-stage timing report from collected records."""
    memory_block = ""
    if not records:
        body = "(no stage timing records were collected)"
    else:
        by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_stage[r["stage"]].append(r)

        cols = ("stage", "batches", "rows_in", "total_s", "pre_ms/b", "proc_ms/b", "post_ms/b", "ms/row")
        widths = (34, 9, 11, 10, 11, 11, 11, 10)
        header = "".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(cols, widths)))
        sep = "-" * len(header)
        lines = [header, sep]

        pipeline_total_ms = 0.0
        ordered_stages = sorted(by_stage.items(), key=lambda kv: -sum(r["total_ms"] for r in kv[1]))
        for stage, recs in ordered_stages:
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

        # Memory section: only emitted when at least one record carries a
        # non-zero rss measurement (psutil was available on the workers).
        has_mem = any(float(r.get("rss_peak_mb") or 0.0) > 0 for r in records)
        if has_mem:
            mcols = ("stage", "peak_rss_mb", "max_rss_mb", "mean_delta_mb", "min_avail_mb")
            mwidths = (34, 14, 14, 16, 14)
            mhead = "".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(mcols, mwidths)))
            msep = "-" * len(mhead)
            mlines = [mhead, msep]
            for stage, recs in ordered_stages:
                peak = max((float(r.get("rss_peak_mb") or 0.0) for r in recs), default=0.0)
                max_after = max((float(r.get("rss_after_mb") or 0.0) for r in recs), default=0.0)
                deltas = [float(r.get("rss_after_mb") or 0.0) - float(r.get("rss_before_mb") or 0.0) for r in recs]
                mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
                min_avail = min(
                    (
                        float(r.get("avail_before_mb") or 0.0)
                        for r in recs
                        if float(r.get("avail_before_mb") or 0.0) > 0
                    ),
                    default=0.0,
                )
                mlines.append(
                    stage[: mwidths[0]].ljust(mwidths[0])
                    + f"{peak:.1f}".rjust(mwidths[1])
                    + f"{max_after:.1f}".rjust(mwidths[2])
                    + f"{mean_delta:+.2f}".rjust(mwidths[3])
                    + f"{min_avail:.0f}".rjust(mwidths[4])
                )
            mlines.append(msep)
            memory_block = "\n".join(mlines)

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
    if memory_block:
        out += ["", "Per-stage memory (host-side, per worker process):", memory_block]
    if memory_samples:
        peak_workload = max((float(s.get("workload_pss_mb") or 0.0) for s in memory_samples), default=0.0)
        peak_driver = max((float(s.get("driver_pss_mb") or 0.0) for s in memory_samples), default=0.0)
        peak_workers = max((float(s.get("workers_pss_mb") or 0.0) for s in memory_samples), default=0.0)
        min_avail = min(
            (
                float(s.get("sys_available_mb") or 0.0)
                for s in memory_samples
                if float(s.get("sys_available_mb") or 0.0) > 0
            ),
            default=0.0,
        )
        max_used_mb = max((float(s.get("sys_used_mb") or 0.0) for s in memory_samples), default=0.0)
        max_used_pct = max((float(s.get("sys_used_pct") or 0.0) for s in memory_samples), default=0.0)
        mean_workers = (
            sum(int(s.get("n_workers") or 0) for s in memory_samples) / len(memory_samples) if memory_samples else 0.0
        )
        # "Memory the run added": host-used at peak minus host-used before
        # the sampler started.  This is the cleanest "what did this run cost"
        # number -- it ignores stale Ray IDLE workers, kernel caches, and
        # anything else that was already resident before ingestion began.
        delta_lines: List[str] = []
        if baseline_sys_used_mb is not None and baseline_sys_used_mb > 0:
            delta = max_used_mb - float(baseline_sys_used_mb)
            delta_lines = [
                f"  baseline host MemUsed (pre-run)          : {baseline_sys_used_mb:8.1f} MB",
                f"  peak host MemUsed (during run)           : {max_used_mb:8.1f} MB",
                f"  delta MemUsed (memory the run added)     : {delta:+8.1f} MB",
            ]
        out += [
            "",
            "Run-level memory (driver-sampled, PSS-based):",
            f"  peak workload PSS (driver + ray workers) : {peak_workload:8.1f} MB",
            f"  peak driver PSS                          : {peak_driver:8.1f} MB",
            f"  peak ray-workers PSS (sum)               : {peak_workers:8.1f} MB",
            *delta_lines,
            f"  host worst-case available                : {min_avail:8.1f} MB",
            f"  host worst-case used                     : {max_used_pct:8.1f} %",
            f"  mean ray-worker count seen               : {mean_workers:8.1f}",
            f"  sample count                             : {len(memory_samples)}",
            "",
            "  Note: PSS attributes shared pages proportionally across the",
            "  processes mapping them; sum-of-PSS ~= true physical memory used.",
            "  'delta MemUsed' is the most defensible 'this run's cost' figure.",
        ]
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
    memory_samples: Optional[List[Dict[str, Any]]] = None,
    baseline_sys_used_mb: Optional[float] = None,
) -> str:
    """Format and emit the report. Returns the report text."""
    node_list = list(node_names) if node_names is not None else None
    samples = list(memory_samples) if memory_samples is not None else None
    text = format_report(
        records,
        ray_stats_text,
        call_index=call_index,
        graph_label=graph_label,
        node_names=node_list,
        memory_samples=samples,
        baseline_sys_used_mb=baseline_sys_used_mb,
    )
    logger.info("\n%s", text)
    out_path = resolve_report_path(call_index, graph_label)
    if out_path is not None:
        try:
            peak_workload_pss_mb = (
                max((float(s.get("workload_pss_mb") or 0.0) for s in samples), default=0.0) if samples else 0.0
            )
            min_sys_available_mb = (
                min(
                    (
                        float(s.get("sys_available_mb") or 0.0)
                        for s in samples
                        if float(s.get("sys_available_mb") or 0.0) > 0
                    ),
                    default=0.0,
                )
                if samples
                else 0.0
            )
            peak_sys_used_mb = (
                max((float(s.get("sys_used_mb") or 0.0) for s in samples), default=0.0) if samples else 0.0
            )
            delta_sys_used_mb = (
                peak_sys_used_mb - float(baseline_sys_used_mb)
                if (baseline_sys_used_mb is not None and baseline_sys_used_mb > 0 and peak_sys_used_mb > 0)
                else None
            )
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "call_index": call_index,
                        "graph_label": graph_label,
                        "node_names": node_list,
                        "run_timestamp": _RUN_TIMESTAMP,
                        "peak_workload_pss_mb": peak_workload_pss_mb,
                        "min_sys_available_mb": min_sys_available_mb,
                        "baseline_sys_used_mb": baseline_sys_used_mb,
                        "peak_sys_used_mb": peak_sys_used_mb,
                        "delta_sys_used_mb": delta_sys_used_mb,
                        "records": records,
                        "memory_samples": samples or [],
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
