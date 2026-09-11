# Evaluate on your data

Retrieval and ingestion performance **depend on your documents**, hardware, and pipeline settings. Use the following when measuring quality and throughput on **your** datasets.

## Benchmarking and baselines { #benchmarking-and-baselines }

Use this page as the baseline for methodology and expectations. Use [Operational tuning](#operational-tuning) below to observe production-like runs.

## Throughput and dataset effects { #throughput-and-dataset-effects }

Read [Throughput is dataset-dependent](../ingest/multimodal-extraction.md#extraction-limitations-and-quality) for why raw numbers from generic benchmarks may not match your corpus (layout complexity, file types, image density, and so on).

## Operational tuning { #operational-tuning }

- [Ray and distributed ingest](../deploy/ray-logging.md)
- [Pre-Requisites & Support Matrix](../get-started/prerequisites-support-matrix.md) for supported configurations
- [Troubleshoot](troubleshoot.md) when results or performance diverge from expectations
