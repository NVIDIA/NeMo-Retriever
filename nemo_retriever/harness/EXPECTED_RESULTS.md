<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harness Expected Results

Known dataset facts, benchmark result ranges, and suggested `--require` gates
for `retriever harness`.

This file is documentation, not executable policy. Use these values to choose
explicit `--require` gates for local validation, agent-run ablations, and
nightly jobs. Update it when datasets, benchmark definitions, hardware, or
retrieval behavior intentionally change.

## JP20

Dataset:

- Corpus path: `/datasets/nv-ingest/jp20`
- Query/qrels file: `data/jp20_query_gt.csv`
- Files: `20`
- Pages: `1940`

Benchmarks:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `jp20_smoke` | Cheap ingest/artifact check | `fast-text` | `0` | No BEIR metrics |
| `jp20_beir` | End-to-end retrieval quality | `auto` | `115` | `recall_5 >= 0.85`, `ndcg_10 >= 0.75` |

Suggested smoke command:

```bash
retriever harness run jp20_smoke \
  --require 'files==20' \
  --require 'pages==1940'
```

Suggested full BEIR command:

```bash
retriever harness run jp20_beir \
  --require 'files==20' \
  --require 'pages==1940' \
  --require 'query_count==115' \
  --require 'recall_5>=0.85' \
  --require 'ndcg_10>=0.75'
```

Recent observed `jp20_beir` metrics on local hardware:

- `rows_processed`: `3154`
- `ingest_secs`: about `215` to `223`
- `query_latency_p50_ms`: about `909` to `915`
- `query_latency_p95_ms`: about `953` to `1003`
- `recall_5`: about `0.878` to `0.887`
- `recall_10`: about `0.930` to `0.948`
- `ndcg_10`: about `0.793` to `0.802`

Avoid hard-gating on latency unless the run environment is controlled.

## BO20

Dataset:

- Corpus path: `/datasets/nv-ingest/bo20`
- Files: `20`
- BEIR qrels: not expected

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `bo20_smoke` | Cheap ingest/artifact check | `fast-text` | `0` | No BEIR metrics |

## BO767

Dataset:

- Corpus path: `/datasets/nv-ingest/bo767`
- Query/qrels file: `data/bo767_annotations.csv`
- Files: `767`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `bo767_beir` | End-to-end retrieval quality | `auto` | TBD | TBD |

## FinanceBench

Dataset:

- Corpus path: `/datasets/nv-ingest/financebench`
- Query/qrels file: `data/financebench_train.json`
- Files: `369`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `financebench_beir` | End-to-end retrieval quality | `auto` | TBD | TBD |

## BO10K

Dataset:

- Corpus path: `/datasets/nv-ingest/bo10k`
- Query/qrels file: `data/digital_corpora_10k_annotations.csv`
- Files: `10000`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `bo10k_beir_fast_text` | Large-corpus fast-text validation | `fast-text` | TBD | TBD |

## Earnings Consulting

Dataset:

- Corpus path: `/datasets/nv-ingest/earnings_consulting`
- Files: `514`
- Expected old query/qrels file `data/earnings_consulting_multimodal.csv` is
  absent in this checkout.

No phase-one harness benchmark should depend on this dataset until the qrels
file is restored or replaced.
