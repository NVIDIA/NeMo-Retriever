# Evaluate on your data

!!! note

    This documentation describes NeMo Retriever Library.


Retrieval and ingestion performance **depend on your documents**, hardware, and pipeline settings. Use the following when measuring quality and throughput on **your** datasets.

## Benchmarking and baselines

Start with [Benchmarking](benchmarking.md) for methodology and baseline expectations. Combine with [Telemetry](telemetry.md) to observe production-like runs.

## Throughput and dataset effects

Read [Throughput is dataset-dependent](throughput-is-dataset-dependent.md) for why raw numbers from generic benchmarks may not match your corpus (layout complexity, file types, image density, and so on).

## Operational tuning

- [Resource scaling modes](scaling-modes.md)
- [Support matrix](support-matrix.md) for supported configurations
- [Troubleshoot](troubleshoot.md) when results or performance diverge from expectations
