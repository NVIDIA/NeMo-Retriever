# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualise a stage-timing JSON dump produced by ``stage_timing.write_report``.

Usage
-----
    python -m nemo_retriever.utils.stage_timing_viz \
        --input /tmp/timing.json --output-dir /tmp/timing_charts

Produces:
    overview.png           - per-stage totals, ms/row, throughput, phase-breakdown
    timeline.png           - per-batch Gantt across all stages
    per_stage/<stage>.png  - latency distribution, time-series, batch-size scatter,
                             phase breakdown for each stage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_PHASE_COLS = ("preprocess_ms", "process_ms", "postprocess_ms")
_PHASE_COLORS = {"preprocess_ms": "#7ec8e3", "process_ms": "#ff7f50", "postprocess_ms": "#9ad19a"}


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def load_payload(path: str) -> dict:
    """Load a timing JSON and return the full payload dict."""
    with open(path) as f:
        return json.load(f)


def load_records(path: str) -> pd.DataFrame:
    payload = load_payload(path) if isinstance(path, str) else path
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not records:
        raise SystemExit(f"No records found in {path!r}")
    df = pd.DataFrame(records)
    # Zero the wallclock so the first batch starts at t=0 on charts.
    df["t_start_s"] = df["wallclock_start"] - df["wallclock_start"].min()
    df["t_end_s"] = df["t_start_s"] + df["total_ms"] / 1000.0
    # Per-batch derived metrics
    df["rows_per_s"] = np.where(df["total_ms"] > 0, df["n_rows_in"] / (df["total_ms"] / 1000.0), 0.0)
    df["ms_per_row"] = np.where(df["n_rows_in"] > 0, df["total_ms"] / df["n_rows_in"], 0.0)
    # Memory deltas (zero when memory tracking was disabled)
    for col in ("rss_before_mb", "rss_after_mb", "rss_peak_mb", "avail_before_mb", "avail_after_mb"):
        if col not in df.columns:
            df[col] = 0.0
    df["rss_delta_mb"] = df["rss_after_mb"] - df["rss_before_mb"]
    # Stable per-stage batch index for time-series plots
    df = df.sort_values(["stage", "t_start_s"]).reset_index(drop=True)
    df["batch_idx"] = df.groupby("stage").cumcount()
    return df


def load_samples(payload_or_path) -> pd.DataFrame:
    """Return a DataFrame of memory samples, or an empty frame if none were collected."""
    payload = load_payload(payload_or_path) if isinstance(payload_or_path, str) else payload_or_path
    samples = payload.get("memory_samples") or []
    if not samples:
        return pd.DataFrame()
    sdf = pd.DataFrame(samples)
    if "t_rel_s" in sdf.columns:
        sdf = sdf.sort_values("t_rel_s").reset_index(drop=True)
    return sdf


def _stage_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stage", sort=False)
    agg = g.agg(
        n_batches=("total_ms", "size"),
        rows_in=("n_rows_in", "sum"),
        total_ms=("total_ms", "sum"),
        mean_pre_ms=("preprocess_ms", "mean"),
        mean_proc_ms=("process_ms", "mean"),
        mean_post_ms=("postprocess_ms", "mean"),
        median_total_ms=("total_ms", "median"),
        p95_total_ms=("total_ms", lambda s: float(np.percentile(s, 95))),
        max_total_ms=("total_ms", "max"),
        peak_rss_mb=("rss_peak_mb", "max"),
        max_rss_after_mb=("rss_after_mb", "max"),
        mean_delta_mb=("rss_delta_mb", "mean"),
        min_avail_mb=("avail_before_mb", "min"),
    )
    agg["ms_per_row"] = np.where(agg["rows_in"] > 0, agg["total_ms"] / agg["rows_in"], 0.0)
    agg["rows_per_s"] = np.where(agg["total_ms"] > 0, agg["rows_in"] / (agg["total_ms"] / 1000.0), 0.0)
    agg = agg.sort_values("total_ms", ascending=False)
    return agg


def _stage_has_memory(agg: pd.DataFrame) -> bool:
    return bool(agg["peak_rss_mb"].max() > 0)


def _stage_color_map(stages):
    """Stable color per stage so the same stage gets the same wedge color across panels."""
    cmap = plt.get_cmap("tab20")
    return {s: cmap(i % cmap.N) for i, s in enumerate(stages)}


def _pie(ax, labels, values, colors, *, title, value_unit, hide_below_pct=2.0):
    """Render a pie with auto-suppressed tiny labels and a value-aware autopct.

    Wedges whose share is below ``hide_below_pct`` percent get their label and
    percent text suppressed in the wedge (they still appear in the legend),
    keeping pies legible when one or two stages dominate.
    """
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    shares = 100.0 * values / total

    def _autopct(pct):
        if pct < hide_below_pct:
            return ""
        return f"{pct:.1f}%"

    wedge_labels = [lbl if shares[i] >= hide_below_pct else "" for i, lbl in enumerate(labels)]
    wedges, _texts, _autotexts = ax.pie(
        values,
        labels=wedge_labels,
        colors=[colors[lbl] for lbl in labels],
        autopct=_autopct,
        startangle=90,
        pctdistance=0.72,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
    )
    ax.set_title(title)
    # Legend shows every stage with its absolute value (so small slices aren't lost)
    legend_labels = [
        f"{lbl}: {v:.2f} {value_unit} ({shares[i]:.1f}%)" for i, (lbl, v) in enumerate(zip(labels, values))
    ]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)


def plot_memory_timeline(sdf: pd.DataFrame, payload: dict, out_path: str) -> bool:
    """Driver-sampled memory timeline: PSS components + host MemUsed (delta).

    Returns ``True`` if a plot was written, ``False`` if there were no samples.
    """
    if sdf.empty or "t_rel_s" not in sdf.columns:
        return False
    fig, ax = plt.subplots(figsize=(15, 6))
    t = sdf["t_rel_s"].values

    # PSS components (left axis, MB)
    if "workload_pss_mb" in sdf.columns:
        ax.plot(t, sdf["workload_pss_mb"], color="#2a4d8f", lw=2.2, label="workload PSS (driver + workers)")
    if "workers_pss_mb" in sdf.columns:
        ax.plot(t, sdf["workers_pss_mb"], color="#5b8fd6", lw=1.4, label="ray workers PSS (sum)")
    if "driver_pss_mb" in sdf.columns:
        ax.plot(t, sdf["driver_pss_mb"], color="#cc6633", lw=1.4, label="driver PSS")

    # Host MemUsed delta (the "what did this run cost" line)
    baseline = payload.get("baseline_sys_used_mb")
    if "sys_used_mb" in sdf.columns and baseline:
        delta = sdf["sys_used_mb"].values - float(baseline)
        ax.plot(t, delta, color="#222222", lw=2.2, linestyle="--", label=f"host MemUsed - baseline ({baseline:.0f} MB)")

    ax.set_xlabel("wallclock seconds (relative to sampler start)")
    ax.set_ylabel("memory (MB)")
    ax.set_title("Run-level memory timeline (driver-sampled, PSS-based)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))

    # Worker count on a twin axis (so memory plot stays readable)
    if "n_workers" in sdf.columns:
        ax2 = ax.twinx()
        ax2.plot(t, sdf["n_workers"], color="#888888", lw=1.0, linestyle=":", label="n ray workers")
        ax2.set_ylabel("n ray workers (idle + active)", color="#666666")
        ax2.tick_params(axis="y", colors="#666666")

    # Annotated peak
    if "workload_pss_mb" in sdf.columns:
        peak_idx = int(np.argmax(sdf["workload_pss_mb"].values))
        peak_t = float(sdf["t_rel_s"].iloc[peak_idx])
        peak_v = float(sdf["workload_pss_mb"].iloc[peak_idx])
        ax.annotate(
            f"peak {peak_v:,.0f} MB",
            xy=(peak_t, peak_v),
            xytext=(10, 14),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#444"),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def plot_memory_overview(agg: pd.DataFrame, payload: dict, out_path: str) -> bool:
    """Per-stage memory overview: peak RSS, mean delta, host headroom, share of peak.

    Returns ``True`` if a plot was written, ``False`` if no memory data exists.
    """
    if not _stage_has_memory(agg):
        return False
    stages = list(agg.index)
    colors = _stage_color_map(stages)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    fig.suptitle("Stage Timing - Memory Overview", fontsize=15, fontweight="bold")

    # 1. Peak RSS per stage (per-worker high-water mark)
    ax = axes[0, 0]
    vals = agg["peak_rss_mb"].values
    bars = ax.bar(range(len(stages)), vals, color=[colors[s] for s in stages])
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=35, ha="right", fontsize=9)
    ax.set_title("Peak per-worker RSS by stage (ru_maxrss)")
    ax.set_ylabel("MB")
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:,.0f}",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)

    # 2. Mean RSS delta per batch (retained allocation)
    ax = axes[0, 1]
    vals = agg["mean_delta_mb"].values
    bcolors = ["#cc4444" if v > 0 else "#4f9d4f" for v in vals]
    bars = ax.bar(range(len(stages)), vals, color=bcolors)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=35, ha="right", fontsize=9)
    ax.set_title("Mean RSS delta per batch (retained per call)")
    ax.set_ylabel("MB per batch (+ retained, - released)")
    ax.axhline(0, color="#888", lw=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:+.2f}",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 3 if v >= 0 else -10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    # 3. Share of peak RSS (pie)
    ax = axes[1, 0]
    vals = agg["peak_rss_mb"].values.astype(float)
    if vals.sum() > 0:
        _pie(ax, stages, vals, colors, title="Share of peak RSS across stages", value_unit="MB")
    else:
        ax.axis("off")
        ax.set_title("Share of peak RSS across stages")
        ax.text(0.5, 0.5, "no data", ha="center", va="center")

    # 4. Run-level summary box
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["Run-level memory summary"]
    def fmt(k, v, unit="MB", prec=1):
        return f"  {k:<32s} {v:>10,.{prec}f} {unit}" if v is not None else f"  {k:<32s} {'-':>10s}"
    base = payload.get("baseline_sys_used_mb")
    peak = payload.get("peak_sys_used_mb")
    delta = payload.get("delta_sys_used_mb")
    lines += [
        fmt("peak workload PSS", payload.get("peak_workload_pss_mb")),
        fmt("baseline host MemUsed (pre-run)", base),
        fmt("peak host MemUsed (during run)", peak),
        fmt("delta MemUsed (run added)", delta) if delta is not None else fmt("delta MemUsed", None),
        fmt("min host available", payload.get("min_sys_available_mb")),
    ]
    lines.append("")
    lines.append("This 'delta MemUsed' is the most defensible")
    lines.append("'this run's cost' number.  PSS de-duplicates")
    lines.append("shared pages so sum-of-PSS ~ true physical RAM.")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=11,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def plot_overview(df: pd.DataFrame, agg: pd.DataFrame, out_path: str) -> None:
    stages = list(agg.index)
    colors = _stage_color_map(stages)
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle("Stage Timing - Overview", fontsize=15, fontweight="bold")

    _pie(
        axes[0, 0],
        stages,
        agg["total_ms"].values / 1000.0,
        colors,
        title="Share of total wall time (worker-side sum)",
        value_unit="s",
    )
    _pie(
        axes[0, 1],
        stages,
        agg["ms_per_row"].values,
        colors,
        title="Per-row cost share (ms/row across stages)",
        value_unit="ms/row",
    )
    _pie(
        axes[1, 0],
        stages,
        agg["rows_in"].values.astype(float),
        colors,
        title="Share of rows processed per stage",
        value_unit="rows",
    )

    # Phase breakdown: single pie of the mean phase split across all stages.
    ax = axes[1, 1]
    phase_labels = ["preprocess", "process", "postprocess"]
    phase_values = np.array(
        [
            float(df["preprocess_ms"].sum()),
            float(df["process_ms"].sum()),
            float(df["postprocess_ms"].sum()),
        ]
    )
    phase_colors = {label: _PHASE_COLORS[col] for label, col in zip(phase_labels, _PHASE_COLS)}
    _pie(
        ax,
        phase_labels,
        phase_values,
        phase_colors,
        title="Where time is spent overall (per phase, all stages)",
        value_unit="ms",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_timeline(df: pd.DataFrame, agg: pd.DataFrame, out_path: str) -> None:
    stages = list(agg.index)
    stage_to_y = {s: i for i, s in enumerate(stages)}
    cmap = plt.get_cmap("tab20")
    colors = {s: cmap(i % cmap.N) for i, s in enumerate(stages)}

    fig, ax = plt.subplots(figsize=(15, max(3.0, 0.5 * len(stages) + 2)))
    for _, row in df.iterrows():
        y = stage_to_y[row["stage"]]
        width_s = max(row["total_ms"] / 1000.0, 1e-4)
        ax.barh(
            y=y,
            width=width_s,
            left=row["t_start_s"],
            height=0.7,
            color=colors[row["stage"]],
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_yticks(list(stage_to_y.values()))
    ax.set_yticklabels(list(stage_to_y.keys()))
    ax.invert_yaxis()
    ax.set_xlabel("wallclock seconds (relative to first batch)")
    ax.set_title("Per-batch timeline (one bar per batch, colored by stage)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_stage_detail(stage: str, sdf: pd.DataFrame, out_path: str) -> None:
    has_mem = "rss_peak_mb" in sdf.columns and float(sdf["rss_peak_mb"].max()) > 0
    nrows = 3 if has_mem else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows))
    fig.suptitle(f"Stage detail - {stage}", fontsize=14, fontweight="bold")

    # 1. Latency time-series (per batch) with phase breakdown stacked
    ax = axes[0, 0]
    x = sdf["batch_idx"].values
    pre = sdf["preprocess_ms"].values
    proc = sdf["process_ms"].values
    post = sdf["postprocess_ms"].values
    ax.bar(x, pre, color=_PHASE_COLORS["preprocess_ms"], label="preprocess", width=1.0)
    ax.bar(x, proc, bottom=pre, color=_PHASE_COLORS["process_ms"], label="process", width=1.0)
    ax.bar(x, post, bottom=pre + proc, color=_PHASE_COLORS["postprocess_ms"], label="postprocess", width=1.0)
    ax.set_title("Per-batch latency over experiment (stacked phases)")
    ax.set_xlabel("batch index (chronological)")
    ax.set_ylabel("ms")
    ax.legend(loc="upper right", fontsize=8)

    # 2. Latency distribution (histogram + percentiles)
    ax = axes[0, 1]
    total = sdf["total_ms"].values
    ax.hist(total, bins=min(30, max(5, len(total) // 2)), color="#7a7a7a", edgecolor="white")
    for q, color, label in [
        (50, "#2266aa", "p50"),
        (95, "#cc6633", "p95"),
        (100, "#aa2222", "max"),
    ]:
        v = float(np.percentile(total, q)) if q < 100 else float(total.max())
        ax.axvline(v, color=color, linestyle="--", linewidth=1.2, label=f"{label} = {v:.1f} ms")
    ax.set_title("Distribution of total per-batch latency")
    ax.set_xlabel("ms")
    ax.set_ylabel("# batches")
    ax.legend(fontsize=8)

    # 3. Batch size vs latency
    ax = axes[1, 0]
    rows = sdf["n_rows_in"].values
    ax.scatter(rows, total, alpha=0.7, color="#33558c", s=24)
    if len(rows) >= 3 and np.ptp(rows) > 0:
        coef = np.polyfit(rows, total, 1)
        xs = np.array([rows.min(), rows.max()])
        ax.plot(
            xs,
            coef[0] * xs + coef[1],
            color="#cc4444",
            linestyle="--",
            label=f"fit: {coef[0]:.2f} ms/row + {coef[1]:.1f} ms",
        )
        ax.legend(fontsize=8)
    ax.set_title("Batch size vs. latency")
    ax.set_xlabel("rows in batch")
    ax.set_ylabel("total ms")

    # 4. Phase share (pie of mean phase time)
    ax = axes[1, 1]
    means = [sdf[c].mean() for c in _PHASE_COLS]
    labels = ["preprocess", "process", "postprocess"]
    colors_l = [_PHASE_COLORS[c] for c in _PHASE_COLS]
    if sum(means) > 0:
        ax.pie(means, labels=labels, colors=colors_l, autopct="%1.1f%%", startangle=90)
        ax.set_title("Where time is spent on average (per phase)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no data", ha="center", va="center")

    # 5 + 6. Memory panels (only when memory data is present)
    if has_mem:
        x = sdf["batch_idx"].values

        # 5. RSS over batches: before / after / peak overlaid
        ax = axes[2, 0]
        ax.plot(x, sdf["rss_before_mb"], color="#5b8fd6", lw=1.2, label="rss before")
        ax.plot(x, sdf["rss_after_mb"], color="#2a4d8f", lw=1.5, label="rss after")
        ax.plot(x, sdf["rss_peak_mb"], color="#cc4444", lw=1.5, linestyle="--", label="rss peak (ru_maxrss)")
        ax.fill_between(x, sdf["rss_before_mb"], sdf["rss_after_mb"], color="#5b8fd6", alpha=0.15)
        ax.set_title("Worker RSS over batches")
        ax.set_xlabel("batch index (chronological)")
        ax.set_ylabel("MB")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)

        # 6. RSS delta per batch (retained memory per call)
        ax = axes[2, 1]
        deltas = sdf["rss_delta_mb"].values
        bar_colors = ["#cc4444" if v > 0 else "#4f9d4f" for v in deltas]
        ax.bar(x, deltas, color=bar_colors, width=1.0)
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_title(f"RSS delta per batch (mean = {float(deltas.mean()):+.2f} MB)")
        ax.set_xlabel("batch index (chronological)")
        ax.set_ylabel("delta MB (+ retained, - released)")
        ax.grid(True, linestyle=":", alpha=0.4)

    # Stats footnote
    rows_in_total = int(sdf["n_rows_in"].sum())
    total_s = float(sdf["total_ms"].sum() / 1000.0)
    ms_row = total_s * 1000.0 / rows_in_total if rows_in_total else 0.0
    footnote = (
        f"{len(sdf)} batches | {rows_in_total} rows in | "
        f"{total_s:.2f} s total | {ms_row:.3f} ms/row | "
        f"mean batch: {sdf['total_ms'].mean():.1f} ms (p95 {np.percentile(total, 95):.1f} ms)"
    )
    if has_mem:
        footnote += (
            f" | peak rss: {float(sdf['rss_peak_mb'].max()):,.0f} MB"
            f" | mean delta: {float(sdf['rss_delta_mb'].mean()):+.2f} MB/batch"
        )
    fig.text(0.5, 0.005, footnote, ha="center", fontsize=9, style="italic")

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", default="/tmp/timing.json", help="Path to timing JSON.")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/tmp/timing_charts",
        help="Directory to write PNGs into (created if missing).",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print the per-stage aggregate table to stdout.",
    )
    args = parser.parse_args(argv)

    payload = load_payload(args.input)
    df = load_records(payload)
    sdf_samples = load_samples(payload)
    agg = _stage_aggregates(df)

    out_dir = Path(args.output_dir)
    (out_dir / "per_stage").mkdir(parents=True, exist_ok=True)

    overview_png = out_dir / "overview.png"
    timeline_png = out_dir / "timeline.png"
    memory_overview_png = out_dir / "memory_overview.png"
    memory_timeline_png = out_dir / "memory_timeline.png"

    plot_overview(df, agg, str(overview_png))
    plot_timeline(df, agg, str(timeline_png))
    wrote_mem_overview = plot_memory_overview(agg, payload, str(memory_overview_png))
    wrote_mem_timeline = plot_memory_timeline(sdf_samples, payload, str(memory_timeline_png))

    per_stage_paths: Dict[str, str] = {}
    for stage in agg.index:
        stage_df = df[df["stage"] == stage].reset_index(drop=True)
        out_path = out_dir / "per_stage" / f"{_safe_name(stage)}.png"
        plot_stage_detail(stage, stage_df, str(out_path))
        per_stage_paths[stage] = str(out_path)

    if args.show_summary:
        print(agg.round(3).to_string())

    print(f"Wrote overview        -> {overview_png}")
    print(f"Wrote timeline        -> {timeline_png}")
    if wrote_mem_overview:
        print(f"Wrote memory overview -> {memory_overview_png}")
    else:
        print("Skipped memory overview (no per-batch memory data in JSON).")
    if wrote_mem_timeline:
        print(f"Wrote memory timeline -> {memory_timeline_png}")
    else:
        print("Skipped memory timeline (no memory_samples in JSON).")
    for stage, p in per_stage_paths.items():
        print(f"Wrote per-stage       -> {p}    ({stage})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
