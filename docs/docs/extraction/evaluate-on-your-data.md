# Evaluate on your data

Retrieval and ingestion performance **depend on your documents**, hardware, and pipeline settings. Use the following when measuring quality and throughput on **your** datasets.

## Benchmarking and baselines

Use this page as the baseline for methodology and expectations. Combine with [Telemetry](telemetry.md) to observe production-like runs.

## Throughput and dataset effects

Read [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality) for why raw numbers from generic benchmarks may not match your corpus (layout complexity, file types, image density, and so on).

## Operational tuning

- [Ray and distributed ingest](ray-logging.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for supported configurations
- [Troubleshoot](troubleshoot.md) when results or performance diverge from expectations
