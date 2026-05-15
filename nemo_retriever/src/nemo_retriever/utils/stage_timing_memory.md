# Measuring "Real" Host Memory in Ray Data Runs

This note explains why the stage-timing instrumentation in
`stage_timing.py` had to grow two extra layers — **PSS accounting** and
**baseline subtraction** — before its memory numbers reflected anything
useful. If you just want the bottom line, skip to
[How to read the report](#how-to-read-the-report).

## The problem this solves

The first cut of memory tracking did the obvious thing: each tick of the
driver-side sampler enumerated `ray::*` processes on the host and summed
their `psutil.Process().memory_info().rss`. The number it produced was
catastrophically wrong, but in a way that *looked plausible*.

Here is what a real ingestion run reported with that naive approach:

```
peak workload RSS (driver + ray workers) : 357,332 MB   (≈ 350 GB)
peak ray-workers RSS (sum)               : 289,912 MB
host worst-case used                     :       37.4 %
```

The host reported **37% used** — about 750 GB out of 2 TB — yet the
"workload RSS" alone claimed 350 GB. Those two numbers cannot both be
true at the same time. Something was lying.

Two things were lying, actually.

## Lie #1: summing RSS double-counts shared memory

`memory_info().rss` (Resident Set Size) is "the total physical memory
the kernel currently has committed for this process." It is a per-process
quantity that includes:

- The Python interpreter itself
- Read-only library code (`libpython`, `libtorch_cpu`, `libcuda`)
- Memory-mapped model weights
- Copy-on-write pages inherited from `fork()`
- The process's private heap

Crucially, **the shared portions are mapped into every process's address
space but only consume physical memory once**. When you sum RSS across N
processes that all imported the same `transformers` and `torch`, you
count those framework pages N times.

For nv-ingest, this is severe. A typical worker process has roughly:

- ~1 GB of Python + framework imports (almost entirely shared)
- A few GB of model weights (often `mmap`-ed and shared between actor
  replicas of the same class)
- Hundreds of MB of decoded batch buffers (genuinely private)

With ~150 worker processes on the box (Ray reuses a long-lived pool of
`ray::IDLE` slots), the shared portion alone gets counted 150 times.
Sum-of-RSS thus inflates the apparent footprint by 2–4×.

### The fix: use **PSS** instead

PSS (Proportional Set Size) is RSS-like, but each shared page is
attributed as `1/N` of its size to each of the N processes mapping it.
**Sum-of-PSS across all processes ≈ actual physical memory used.**

`psutil.Process().memory_full_info().pss` exposes this on Linux. The
sampler now reads PSS; RSS is only kept as a fallback for platforms
without `/proc/<pid>/smaps`.

The cost: `memory_full_info()` parses `/proc/<pid>/smaps`, which is
~10× slower than `memory_info()`. We accommodate this by raising the
sampler interval from 0.5 s to 1.0 s — on a 156-worker box this is
imperceptible relative to the run length but eliminates the
double-counting.

After this change, the same ingestion run's "peak ray-workers PSS (sum)"
dropped from 290 GB to roughly 80–150 GB. That is the *true* physical
memory the Ray workers were holding at peak — every shared page
counted exactly once.

## Lie #2: idle workers hold memory from *prior* runs

PSS fixes the double-counting, but a different problem remains: many of
the `ray::*` processes on the host did not start because of *this* run.
The cluster on the dev box is long-lived. After a run finishes, Ray
keeps worker processes alive in a pool and returns them to the
`ray::IDLE` state. Crucially:

- Python's garbage collector frees objects lazily.
- Even when freed, `glibc malloc` rarely returns pages to the OS — it
  caches them in arena pools.
- The CUDA context, if ever initialised, stays resident.

So an `ru_maxrss` of 6 GB from yesterday's OCR run is still ~5–6 GB of
RSS today, sitting on a `ray::IDLE` slot.

Counting these in "this run's memory" is wrong. They were already
resident *before* ingestion started. The PSS sum tells you "how much
RAM is Ray currently holding on the host", which is honest but is not
the question users were trying to answer.

### The fix: **baseline subtraction**

When the sampler starts, it captures `psutil.virtual_memory().used` —
the kernel's host-wide MemUsed counter, in MB — and stores it as the
**baseline**. Each subsequent sample also records the current MemUsed.
At report time we compute:

```
delta_sys_used_mb = peak(sys_used_mb during run) - baseline_sys_used_mb
```

This is **the most defensible "what did this run cost" figure**:

- It excludes everything that was already resident (idle workers,
  kernel cache, OS, sshd, anyone else's processes).
- It does not depend on us correctly attributing per-process memory —
  it asks the kernel, which has the authoritative answer.
- It captures the run's net effect on the host even if Ray spawned new
  workers or recycled old ones.

The trade-off: it includes anything *else* that grew during the run
window (a coincidental compile, file cache growth, etc.). On a busy
shared box this can add noise. On a dedicated ingest run it is the
cleanest number available.

## How to read the report

The run-level memory section now looks like this:

```
Run-level memory (driver-sampled, PSS-based):
  peak workload PSS (driver + ray workers) :   12303.9 MB
  peak driver PSS                          :     158.7 MB
  peak ray-workers PSS (sum)               :   12145.5 MB
  baseline host MemUsed (pre-run)          :  632550.9 MB
  peak host MemUsed (during run)           :  633113.4 MB
  delta MemUsed (memory the run added)     :    +562.5 MB
  host worst-case available                : 1531166.9 MB
  host worst-case used                     :      29.3 %
  mean ray-worker count seen               :     258.0
  sample count                             :       2
```

What each number actually answers:

| Line | Question it answers | When to trust it |
|---|---|---|
| **`delta MemUsed`** | "How much did this run grow the host's used memory?" | **Always.** This is the answer to "how much memory did this run use." |
| `peak workload PSS` | "At peak, how much physical RAM was held by *this Ray cluster* (driver + all workers)?" | When you want to know Ray's footprint on this host, including idle holdovers. |
| `peak ray-workers PSS (sum)` | "Of the workload PSS, how much is in the worker pool?" | When you want to see whether the driver or the workers are dominant. |
| `host worst-case used` (%) | "How close did the run push the box to OOM?" | When the value approaches 100% — that is when this number actually matters. |
| `mean ray-worker count seen` | "How many `ray::*` processes were resident on average?" | Sanity check: a number much higher than your stage count means a stale idle pool, which is why `peak workload PSS` may exceed `delta MemUsed` by a lot. |

If you only want one number to quote in a report or PR description: it
is `delta MemUsed`.

## Why per-batch RSS is still useful

The per-stage memory table in the report uses per-batch RSS (not PSS)
captured *inside* each worker's `AbstractOperator.run()`. This is
intentional:

- It is a *per-process* measurement (one worker, one batch). There is
  no double-counting problem to solve here — only one process is being
  measured.
- It is essentially free (one syscall per batch boundary).
- It gives you `rss_peak_mb` (the worker's `ru_maxrss` high-water mark)
  which reveals which stage made the worker grow.

The driver sampler and the per-batch RSS measure different things and
both belong in the report.

## Caveats and known limitations

1. **`ru_maxrss` is process-lifetime, not batch-scoped.** It only goes
   up. We use the diff between consecutive `ru_maxrss` readings to
   attribute peak-growth to a stage, which is correct in aggregate but
   not perfect for any single batch.

2. **PSS requires `/proc/<pid>/smaps`.** Available on Linux; the
   sampler falls back to RSS on platforms without it. If you ever run
   the timing on macOS or in a stripped-down container the numbers
   will revert to being RSS-based and the double-counting returns.

3. **Baseline is sampled once, before the sampler thread starts.** If
   another process on the host allocates a lot of memory simultaneously
   with your run, `delta MemUsed` will overstate the run's true cost.
   On a dedicated ingest box this is fine; on a shared dev box, treat
   `delta MemUsed` as a soft upper bound.

4. **Idle workers carrying leaked state from previous runs.** PSS
   correctly de-duplicates their shared pages, but their private heap
   (cached models, malloc arena) still appears in `peak workload PSS`.
   If you only want "what this run added", read `delta MemUsed`.

## Quick reference

- Implementation: `nemo_retriever/utils/stage_timing.py` —
  `_MemorySampler`, `_process_pss_bytes`, `start_memory_sampler`.
- Configuration: `NR_STAGE_TIMING=1` to enable the whole subsystem;
  `NR_STAGE_TIMING_REPORT_PATH=<dir-or-file-stem>` to write JSON.
- Visualization: `python -m nemo_retriever.utils.stage_timing_viz
  --input <timing.json> --output-dir <dir>` produces
  `memory_timeline.png` and `memory_overview.png` alongside the
  timing charts.
