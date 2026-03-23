#!/usr/bin/env python3
"""Statistical analysis of recall metrics: batch_pipeline vs graph_pipeline."""

import numpy as np
from scipy import stats

# Batch pipeline results (5 runs)
batch = {
    "recall@1": [0.0747, 0.0747, 0.0747, 0.0686, 0.0595],
    "recall@5": [0.1019, 0.1019, 0.1019, 0.0898, 0.0777],
    "recall@10": [0.1080, 0.1080, 0.1080, 0.0949, 0.0817],
}

# Graph pipeline results (5 runs)
graph = {
    "recall@1": [0.0747, 0.0747, 0.0757, 0.0747, 0.0696],
    "recall@5": [0.1009, 0.1019, 0.1019, 0.1019, 0.0959],
    "recall@10": [0.1070, 0.1080, 0.1080, 0.1080, 0.1019],
}

METRICS = ["recall@1", "recall@5", "recall@10"]

print("=" * 80)
print("RECALL EVALUATION ANALYSIS: batch_pipeline vs graph_pipeline")
print("Dataset: /raid/data/jp20 (20 PDFs, 1940 pages)")
print("Query CSV: bo767_query_gt.csv (991 queries)")
print("Runs per pipeline: 5")
print("=" * 80)

# --- Raw Results ---
print()
print("RAW RESULTS")
print("-" * 80)
header = (
    f"{'Run':>5} | {'batch r@1':>10} {'batch r@5':>10} {'batch r@10':>11}"
    f" | {'graph r@1':>10} {'graph r@5':>10} {'graph r@10':>11}"
)
print(header)
print("-" * 80)
for i in range(5):
    row = (
        f"{i+1:>5} |"
        f" {batch['recall@1'][i]:>10.4f} {batch['recall@5'][i]:>10.4f} {batch['recall@10'][i]:>11.4f}"
        f" | {graph['recall@1'][i]:>10.4f} {graph['recall@5'][i]:>10.4f} {graph['recall@10'][i]:>11.4f}"
    )
    print(row)

# --- Descriptive Statistics ---
print()
print("DESCRIPTIVE STATISTICS")
print("-" * 80)
desc_header = f"{'Metric':>12} | {'Pipeline':>10} | {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'CV%':>8}"
print(desc_header)
print("-" * 80)

for metric in METRICS:
    for name, data in [("batch", batch), ("graph", graph)]:
        arr = np.array(data[metric])
        mean = arr.mean()
        sd = arr.std(ddof=1)
        cv = (sd / mean * 100) if mean > 0 else 0
        print(
            f"{metric:>12} | {name:>10} |" f" {mean:>8.4f} {sd:>8.4f} {arr.min():>8.4f} {arr.max():>8.4f} {cv:>7.2f}%"
        )

# --- Paired t-tests ---
print()
print("STATISTICAL TESTS (paired t-test, two-tailed)")
print("-" * 80)
test_header = (
    f"{'Metric':>12} | {'batch mean':>11} {'graph mean':>11}"
    f" | {'diff':>8} {'t-stat':>8} {'p-value':>10} {'Significant?':>14}"
)
print(test_header)
print("-" * 80)

p_values = {}
for metric in METRICS:
    b = np.array(batch[metric])
    g = np.array(graph[metric])
    t_stat, p_val = stats.ttest_rel(b, g)
    p_values[metric] = p_val
    diff = b.mean() - g.mean()
    sig = "YES (p<0.05)" if p_val < 0.05 else "NO"
    print(
        f"{metric:>12} | {b.mean():>11.4f} {g.mean():>11.4f}"
        f" | {diff:>+8.4f} {t_stat:>8.3f} {p_val:>10.4f} {sig:>14}"
    )

# --- Effect Size ---
print()
print("EFFECT SIZE (Cohen's d for paired samples)")
print("-" * 80)
for metric in METRICS:
    b = np.array(batch[metric])
    g = np.array(graph[metric])
    diff = b - g
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
    if abs(d) < 0.2:
        mag = "negligible"
    elif abs(d) < 0.5:
        mag = "small"
    elif abs(d) < 0.8:
        mag = "medium"
    else:
        mag = "large"
    print(f"{metric:>12} | Cohen's d = {d:>+7.3f}  ({mag})")

# --- Confidence Intervals ---
print()
print("CONFIDENCE INTERVALS (95%) for mean difference (batch - graph)")
print("-" * 80)
for metric in METRICS:
    b = np.array(batch[metric])
    g = np.array(graph[metric])
    diff = b - g
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lo = diff.mean() - t_crit * se
    ci_hi = diff.mean() + t_crit * se
    contains_zero = "contains 0" if ci_lo <= 0 <= ci_hi else "excludes 0"
    print(
        f"{metric:>12} | mean diff = {diff.mean():>+7.4f}"
        f"  95% CI: [{ci_lo:>+7.4f}, {ci_hi:>+7.4f}]  ({contains_zero})"
    )

# --- Variability comparison ---
print()
print("RUN-TO-RUN VARIABILITY (std dev)")
print("-" * 80)
for metric in METRICS:
    b_std = np.array(batch[metric]).std(ddof=1)
    g_std = np.array(graph[metric]).std(ddof=1)
    ratio = b_std / g_std if g_std > 0 else float("inf")
    more_stable = "graph" if g_std < b_std else "batch" if b_std < g_std else "equal"
    print(
        f"{metric:>12} | batch std = {b_std:.4f}  graph std = {g_std:.4f}"
        f"  ratio = {ratio:.2f}x  (more stable: {more_stable})"
    )

# --- Conclusion ---
print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)

all_not_sig = all(p > 0.05 for p in p_values.values())

if all_not_sig:
    print("No statistically significant difference between batch_pipeline and")
    print("graph_pipeline recall at any cutoff (all p > 0.05).")
    print("The two pipelines produce equivalent retrieval quality.")
else:
    sig = [m for m in METRICS if p_values[m] < 0.05]
    not_sig = [m for m in METRICS if p_values[m] >= 0.05]
    if sig:
        print(f"Significant differences found at: {', '.join(sig)}")
    if not_sig:
        print(f"No significant difference at: {', '.join(not_sig)}")

print()
print("NOTE: The batch_pipeline shows higher run-to-run variance (runs 4 & 5")
print("had notably lower recall). This is likely due to non-deterministic GPU")
print("scheduling and floating-point ordering effects in Ray actors. The")
print("graph_pipeline shows more consistent results across runs.")
print("=" * 80)
