# Performance Guide

This page is a starting point for NeMo Retriever Library performance tuning guidance.

## Scope

Use this guide to document practical recommendations for:

- Extraction throughput and latency tuning
- Task-level settings (for example `extract`, `caption`, and `embed`)
- Deployment-specific tuning for library mode and Kubernetes/Helm
- NIM endpoint sizing and concurrency settings
- Benchmarking methodology and repeatable test setups

## Batch resource sizing

In batch mode, NeMo Retriever Library sizes unspecified Ray actor pools from the CPU and GPU resources that Ray reports as available immediately before the pipeline is submitted. This prevents default extraction, OCR, and embedding pools from reserving more resources than the cluster can schedule.

If you set `BatchTuningParams` worker counts or direct `node_overrides`, those requests must fit the available Ray CPU and GPU budget. The library validates the final plan before submitting work and raises an error when it is infeasible. Reduce `*_workers` or per-node concurrency, or wait for shared-cluster capacity before retrying.

Use the Ray dashboard to verify the available-resource snapshot and the planned worker allocation when you tune throughput.
