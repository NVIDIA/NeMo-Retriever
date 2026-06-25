<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harness Expected Results

Known dataset facts, canonical benchmark result ranges, and suggested
`--require` gates for `retriever harness`.

This file is documentation, not executable policy. Use these values to choose
explicit `--require` gates for local validation, agent-run ablations, and
nightly jobs. Update it when datasets, benchmark definitions, hardware, or
retrieval behavior intentionally change.

Only canonical benchmark expectations belong here. Exploratory runs, fast-text
fallbacks, chunking experiments, and failed attempts should stay in run
artifacts or handoff notes until the team chooses them as canonical benchmark
definitions.

## JP20

Dataset:

- Corpus path: `/datasets/nv-ingest/jp20`
- Query/qrels file: `data/jp20_query_gt.csv`
- Files: `20`
- Pages: `1940`

Benchmarks:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `jp20_beir` | End-to-end retrieval quality | `auto` | `115` | `recall_5 >= 0.85`, `ndcg_10 >= 0.75` |

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

## BO767

Dataset:

- Corpus path: `/datasets/nv-ingest/bo767`
- Query/qrels file: `data/bo767_query_gt.csv`
- Files: `767`
- Pages: `54730`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `bo767_beir` | End-to-end retrieval quality | `auto` | `991` | `recall_5 >= 0.84`, `ndcg_10 >= 0.74` |

Suggested full BEIR command:

```bash
retriever harness run \
  --runfile nemo_retriever/harness/runfiles/bo767_beir.json \
  --require 'files==767' \
  --require 'pages==54730' \
  --require 'query_count==991' \
  --require 'recall_5>=0.84' \
  --require 'ndcg_10>=0.74'
```

Recent observed `bo767_beir` metrics on H100 batch execution:

- `rows_processed`: `79230`
- `ingest_secs`: about `2563`
- `pages_per_sec_ingest`: about `21.35`
- `query_latency_p50_ms`: about `1100`
- `query_latency_p95_ms`: about `1171`
- `recall_5`: about `0.849`
- `recall_10`: about `0.896`
- `ndcg_10`: about `0.750`

The checked-in BO767 runfile includes conservative batch worker and batch-size
overrides matching the observed successful run.

## FinanceBench

Dataset:

- Corpus path: `/datasets/nv-ingest/foundation_rag/financebench`
- Query/qrels file: `data/financebench_train.json`
- Files: `369`
- Pages: `54057`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `financebench_beir` | End-to-end retrieval quality | `auto` | `150` | TBD after canonical run |

## BO10K

Dataset:

- Corpus path: `/datasets/nv-ingest/bo10k`
- Query/qrels file: `data/digital_corpora_10k_annotations.csv`
- Files: `10000`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| TBD | Canonical end-to-end retrieval quality | `auto` | TBD | TBD after canonical benchmark is defined |

## Earnings Consulting

Dataset:

- Corpus path: `/datasets/nv-ingest/earnings_consulting_flattened`
- Query/qrels file: `data/earnings_consulting_multimodal.csv`
- Files: `514`
- Pages: `12988`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `earnings_beir` | End-to-end retrieval quality | `auto` | `628` | TBD after canonical run |

## ViDoRe V3 Finance EN

Dataset:

- Corpus path: `/datasets/nv-ingest/vidore_v3/vidore_v3_finance_en`
- Query/qrels source: HuggingFace dataset `vidore/vidore_v3_finance_en`
- Files: `6`
- Pages: `2942`

Benchmark:

| Benchmark | Purpose | Ingest Profile | Queries | Expected Quality |
|-----------|---------|----------------|---------|------------------|
| `vidore_v3_finance_en_beir` | Canonical end-to-end page-level ViDoRe retrieval quality | `auto` | `1854` | TBD after canonical run |
