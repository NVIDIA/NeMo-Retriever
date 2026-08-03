<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harness Expected Results

This page records dataset facts and observed benchmark results. It is not an
executable baseline or universal pass/fail policy.

Checked-in runfiles enforce only portable integrity facts such as file, page,
and query counts. Quality and performance vary with the Retriever revision,
hardware, runtime, and model configuration. Use those observations for review,
then add an explicit `--require` gate only when the environment and intended
policy justify it.

## Dataset Facts

Registry paths describe the standard internal mount. Use
[`dataset_paths.example.yaml`](../dataset_paths.example.yaml) to map these datasets
on another host.

| Dataset | Benchmark | Standard path | Files | Pages | Queries |
| --- | --- | --- | ---: | ---: | ---: |
| JP20 | `jp20_beir` | `/datasets/nv-ingest/jp20` | 20 | 1,940 | 115 |
| BO20 | — | `/datasets/nv-ingest/bo20` | 20 | — | — |
| BO767 | `bo767_beir` | `/datasets/nv-ingest/bo767` | 767 | 54,730 | 991 |
| FinanceBench | `financebench_beir` | `/datasets/nv-ingest/foundation_rag/financebench` | 369 | 54,057 | 150 |
| BO10K | Not yet defined | `/datasets/nv-ingest/bo10k` | 10,000 | — | — |
| Earnings Consulting | `earnings_beir` | `/datasets/nv-ingest/earnings_consulting_flattened` | 514 | 12,988 | 628 |

The query files are:

| Dataset | Query or qrels file |
| --- | --- |
| JP20 | `data/jp20_query_gt.csv` |
| BO767 | `data/bo767_query_gt.csv` |
| FinanceBench | `data/financebench_train.json` |
| BO10K | `data/digital_corpora_10k_annotations.csv` |
| Earnings Consulting | `data/earnings_consulting_multimodal.csv` |

## JP20 Observations

Recent `jp20_beir` runs on local hardware:

| Metric | Observed |
| --- | ---: |
| Rows processed | 3,154 |
| Ingest seconds | 215–223 |
| Query p50 | 909–915 ms |
| Query p95 | 953–1,003 ms |
| Recall@5 | 0.878–0.887 |
| Recall@10 | 0.930–0.948 |
| nDCG@10 | 0.793–0.802 |

The original observations did not record the GPU SKU or workload-visible GPU
count. Treat their latency and throughput as context only.

## BO767 Observations

Runs on an eight-H100 80GB HBM3 host:

| Configuration | Workload GPUs | Rows | Ingest seconds | Pages/s | Recall@5 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RC26.05 Perflab | Not recorded | 79,221 | 4,036.847 | 13.56 | — | — | — |
| Automatic batch | 1 | 79,229 | 1,594.339 | 34.328 | 0.848638 | 0.897074 | 0.750110 |
| Automatic batch | 8 | 79,230 | 764.946 | 71.548 | 0.849647 | 0.895055 | 0.748583 |
| Legacy worker-capped batch | 8 visible, effectively 1 used | 79,230 | about 2,265 | about 24.16 | 0.850656 | 0.896065 | 0.751507 |

The RC26.05 artifact recorded eight physical GPUs but not the number visible to
the workload. Current runfiles leave worker counts and batch sizes automatic so
the planner can scale to available GPUs. The one- and eight-GPU automatic runs
differed by one output row; keep row counts and quality visible when comparing
throughput.

## ViDoRe V3

The eight public ViDoRe v3 benchmarks use original PDFs with:

- `nvidia/llama-nemotron-embed-vl-1b-v2`
- `text_image` page embeddings
- page-image and infographic extraction
- page-level BEIR document IDs

Integrity facts:

| Dataset | Files | Pages | Queries |
| --- | ---: | ---: | ---: |
| Computer Science | 2 | 1,360 | 1,290 |
| Energy | 41 | 2,225 | 1,848 |
| Finance EN | 6 | 2,942 | 1,854 |
| Finance FR | 5 | 2,384 | 1,920 |
| HR | 14 | 1,110 | 1,908 |
| Industrial | 27 | 5,244 | 1,698 |
| Pharmaceuticals | 52 | 2,313 | 2,184 |
| Physics | 42 | 1,674 | 1,812 |

Observed complete batch runs on an eight-H100 DGX:

| Dataset | Indexed rows | Ingest seconds | Pages/s | Recall@5 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Computer Science | 1,358 | 100.2–123.5 | 11.0–13.6 | 0.599–0.600 | 0.729–0.730 | 0.708–0.709 |
| Energy | 2,211 | 116.7 | 19.1 | 0.575 | 0.674 | 0.581 |
| Finance EN | 2,927 | 149.4 | 19.7 | 0.496 | 0.609 | 0.547 |
| Finance FR | 2,149 | 106.4 | 22.4 | 0.324 | 0.426 | 0.345 |
| HR | 1,091 | 82.6 | 13.4 | 0.452 | 0.574 | 0.530 |
| Industrial | 5,039 | 137.5 | 38.1 | 0.348 | 0.426 | 0.381 |
| Pharmaceuticals | 2,290 | 93.7 | 24.7 | 0.547 | 0.647 | 0.607 |
| Physics | 1,674 | 89.2 | 18.8 | 0.369 | 0.485 | 0.451 |

Computer Science was run twice; the other domains have one complete observation.
The observed all-domain macro-average nDCG@10 was about `0.519`. The observations
predate separate physical and workload-visible GPU counts, so do not infer
scaling behavior from their throughput.

### RC26.05 Recall Comparison

| Dataset | RC26.05 Recall@5 | Observed Recall@5 |
| --- | ---: | ---: |
| Finance EN | 0.499 | 0.496 |
| Industrial | 0.348 | 0.348 |
| Computer Science | 0.600 | 0.599 |
| Pharmaceuticals | 0.549 | 0.547 |
| HR | 0.453 | 0.452 |
| Energy | 0.577 | 0.575 |
| Physics | 0.367 | 0.369 |
| Finance FR | 0.324 | 0.324 |

The observed macro-average Recall@5 was `0.464` across all eight domains.

Finance FR omitted 235 empty-text pages, including 69 judged image-only pages,
because the current dense LanceDB path drops records without text even when an
image embedding exists. Its Recall@5 still matched RC26.05, but preserving those
judged image-only pages remains correctness work.
