<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harness Expected Results

Known dataset facts, canonical benchmark result ranges, and suggested integrity
gates for `retriever harness`.

This file is documentation, not executable policy. Use dataset facts for
explicit `--require` gates. Use quality and performance ranges to judge whether
a result is in the expected ballpark. Update the references when datasets,
benchmark definitions, hardware, or retrieval behavior intentionally change.

Only canonical benchmark expectations belong here. Exploratory runs, fast-text
fallbacks, chunking experiments, and failed attempts should stay in run
artifacts or handoff notes until the team chooses them as canonical benchmark
definitions.

File counts, page counts, and query counts are portable dataset-integrity gates.
Recall and nDCG are reference ranges that help developers and agents identify
results outside the expected ballpark. Do not treat the quality ranges as
universal pass/fail policy across different hardware and runtime profiles.

Performance observations such as ingest seconds, pages/sec, and query latency
are reference points for a specific environment. Treat them as hardware- and
configuration-sensitive unless the GPU SKU/count, CUDA driver, model backend,
vLLM/kernel settings, Ray worker layout, storage path, and dataset mount are
controlled.

The dataset paths below are registry reference paths, not a portable filesystem
contract. Use `harness/dataset_paths.example.yaml` and `run-files
--dataset-paths` to point checked-in runfiles at the current machine.

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
retriever harness run-files \
  --session-name jp20_beir \
  --output-dir /local/path/to/retriever-artifacts/jp20-beir \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --require 'files==20' \
  --require 'pages==1940' \
  --require 'query_count==115' \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

Recent observed `jp20_beir` metrics on local hardware:

- `rows_processed`: `3154`
- `ingest_secs`: about `215` to `223`
- `query_latency_p50_ms`: about `909` to `915`
- `query_latency_p95_ms`: about `953` to `1003`
- `recall_5`: about `0.878` to `0.887`
- `recall_10`: about `0.930` to `0.948`
- `ndcg_10`: about `0.793` to `0.802`

Avoid hard-gating on latency or throughput unless the run environment is
controlled and recorded in the artifacts.

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
retriever harness run-files \
  --session-name bo767_beir \
  --output-dir /local/path/to/retriever-artifacts/bo767-beir \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --require 'files==767' \
  --require 'pages==54730' \
  --require 'query_count==991' \
  nemo_retriever/harness/runfiles/bo767_beir.json
```

Observed `bo767_beir` metrics on an eight-H100 80GB HBM3 host:

| Configuration | Workload GPUs | Rows | Ingest seconds | Pages/s | Recall@5 | Recall@10 | nDCG@10 |
|---------------|---------------|------|----------------|---------|----------|-----------|---------|
| RC26.05 Perflab | Not recorded | 79221 | 4036.847 | 13.56 | Not recorded | Not recorded | Not recorded |
| Automatic batch | 1 | 79229 | 1594.339 | 34.328 | 0.848638 | 0.897074 | 0.750110 |
| Automatic batch | 8 | 79230 | 764.946 | 71.548 | 0.849647 | 0.895055 | 0.748583 |
| Legacy worker-capped batch | 8 visible, effectively 1 used | 79230 | about 2265 | about 24.16 | 0.850656 | 0.896065 | 0.751507 |

The RC26.05 Perflab artifact recorded eight physical GPUs but did not
distinguish the workload-visible GPU count. The checked-in BO767 runfile leaves
worker counts and batch sizes automatic so the batch planner can scale to the
GPUs available to the workload. Do not use
the legacy worker-capped result as evidence of eight-GPU scaling. The automatic
one- and eight-GPU runs differed by one output row; keep quality and row counts
visible when comparing performance results.

### BO767 VL modality sweep

The release-qualification sweep uses
`nvidia/llama-nemotron-embed-vl-1b-v2` with element-level embeddings and a
hybrid LanceDB index. Query mode is `auto`, which detects the vector and FTS
indexes created during ingestion. Both runfiles use a single-GPU co-scheduling
profile: three embed workers, an embed batch size of 256, and 0.1 GPU per embed
actor so the workers can run alongside page-elements and OCR. This profile has
been checked for actor placement and runtime health; it is not yet a proven
performance optimum.

| Runfile | Ingest modality | Index mode | Query mode | Workload GPUs | Expected quality |
|---------|-----------------|------------|------------|---------------|------------------|
| `bo767_vl_text_image_hybrid_beir.json` | `text_image` | `hybrid` | `auto` | 1 | Record after full validation |
| `bo767_vl_image_hybrid_beir.json` | `image` | `hybrid` | `auto` | 1 | Record after full validation |

Both runs require 767 files, 54,730 pages, and 991 queries. Do not add quality
thresholds or observed values until a complete published-RC session succeeds.
Zero recall or nDCG from a child with a nonzero exit, a populated `failure`,
`DeepGEMM backend is not available`, or `Engine core initialization failed` is
a runtime failure rather than an observed quality result. Dense mode remains a
supported ingestion contract and may be used as a diagnostic control, but it
is not part of this VL modality qualification sweep.

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

## ViDoRe V3

The eight public ViDoRe v3 benchmarks use original PDFs for ingest and load
queries and qrels from the corresponding `vidore/<dataset>` Hugging Face
dataset. The canonical configuration uses:

- `nvidia/llama-nemotron-embed-vl-1b-v2`
- `text_image` embedding at page granularity
- page-image and infographic extraction
- page-level BEIR document IDs

Dataset integrity gates:

| Dataset | Files | Pages | Queries |
|---------|------:|------:|--------:|
| `vidore_v3_computer_science` | 2 | 1360 | 1290 |
| `vidore_v3_energy` | 41 | 2225 | 1848 |
| `vidore_v3_finance_en` | 6 | 2942 | 1854 |
| `vidore_v3_finance_fr` | 5 | 2384 | 1920 |
| `vidore_v3_hr` | 14 | 1110 | 1908 |
| `vidore_v3_industrial` | 27 | 5244 | 1698 |
| `vidore_v3_pharmaceuticals` | 52 | 2313 | 2184 |
| `vidore_v3_physics` | 42 | 1674 | 1812 |

Canonical benchmarks:

| Benchmark | Purpose | Ingest Profile | Expected Quality |
|-----------|---------|----------------|------------------|
| `vidore_v3_computer_science_beir` | Computer science page retrieval | `auto` | Observed `ndcg_10` about `0.708` to `0.709` |
| `vidore_v3_energy_beir` | Energy page retrieval | `auto` | Observed `ndcg_10` about `0.581` |
| `vidore_v3_finance_en_beir` | English finance page retrieval | `auto` | Observed `ndcg_10` about `0.547` |
| `vidore_v3_finance_fr_beir` | French finance page retrieval | `auto` | Observed `ndcg_10` about `0.345`; see coverage warning below |
| `vidore_v3_hr_beir` | Human-resources page retrieval | `auto` | Observed `ndcg_10` about `0.530` |
| `vidore_v3_industrial_beir` | Industrial page retrieval | `auto` | Observed `ndcg_10` about `0.381` |
| `vidore_v3_pharmaceuticals_beir` | Pharmaceuticals page retrieval | `auto` | Observed `ndcg_10` about `0.607` |
| `vidore_v3_physics_beir` | Physics page retrieval | `auto` | Observed `ndcg_10` about `0.451` |

Use the checked-in runfiles for executable file, page, and query-count gates.
Add quality and performance observations here only after a complete run on the
canonical default configuration.

Observed metrics from complete batch runs on an eight-H100 DGX with the default
benchmark configuration:

| Dataset | Rows Processed | Indexed Rows | Ingest Seconds | Pages/s | Query p50 ms | Query p95 ms | Recall@5 | Recall@10 | nDCG@10 |
|---------|---------------:|-------------:|---------------:|--------:|-------------:|-------------:|---------:|----------:|--------:|
| Computer Science | 1360 | 1358 | 100.2-123.5 | 11.0-13.6 | 40.3-40.7 | 66.7-66.9 | 0.599-0.600 | 0.729-0.730 | 0.708-0.709 |
| Energy | 2225 | 2211 | 116.7 | 19.1 | 43.3 | 51.4 | 0.575 | 0.674 | 0.581 |
| Finance EN | 2942 | 2927 | 149.4 | 19.7 | 45.1 | 53.7 | 0.496 | 0.609 | 0.547 |
| Finance FR | 2384 | 2149 | 106.4 | 22.4 | 44.8 | 52.0 | 0.324 | 0.426 | 0.345 |
| HR | 1110 | 1091 | 82.6 | 13.4 | 39.7 | 46.3 | 0.452 | 0.574 | 0.530 |
| Industrial | 5244 | 5039 | 137.5 | 38.1 | 51.0 | 61.8 | 0.348 | 0.426 | 0.381 |
| Pharmaceuticals | 2313 | 2290 | 93.7 | 24.7 | 46.0 | 54.9 | 0.547 | 0.647 | 0.607 |
| Physics | 1674 | 1674 | 89.2 | 18.8 | 45.0 | 52.5 | 0.369 | 0.485 | 0.451 |

Computer Science was run twice; the other domains have one complete observation
each. Computer Science quality was stable to within `0.0011` nDCG@10. The
eight-domain macro-average nDCG@10 was about `0.519`, using the mean of the two
Computer Science runs.

The NeMo Retriever 26.05 image-plus-text release baseline reports average
Recall@5 of `0.490` over the English datasets and `0.465` over all datasets.
The default harness run reproduces the domain-level release results to within
`0.0032` absolute Recall@5:

| Dataset | 26.05 Release Recall@5 | Observed Recall@5 | Delta |
|---------|------------------------:|------------------:|------:|
| Finance EN | 0.499 | 0.496 | -0.003 |
| Industrial | 0.348 | 0.348 | +0.000 |
| Computer Science | 0.600 | 0.599 | -0.001 |
| Pharmaceuticals | 0.549 | 0.547 | -0.002 |
| HR | 0.453 | 0.452 | -0.001 |
| Energy | 0.577 | 0.575 | -0.002 |
| Physics | 0.367 | 0.369 | +0.002 |
| Finance FR | 0.324 | 0.324 | +0.000 |

The observed simple macro-average Recall@5 was `0.464` over all eight domains.

The indexed-row audit found that every omitted page had empty corpus text. No
judged pages were omitted for seven of the eight datasets. Finance FR omitted
`235` empty-text pages, including `69` judged image-only pages, because the
current dense LanceDB write path drops records without text even when an image
embedding exists. Its Recall@5 still matches the 26.05 release baseline, which
suggests the release exercised the same behavior, but retaining those judged
image-only pages remains a correctness prerequisite before nightly scheduling.

Do not apply default hard quality gates across machine profiles or GPU SKUs.
Keep the checked-in file, page, and query-count requirements as hard integrity
gates, record quality metrics on every run, and establish comparison ranges
from each nightly environment's own history.
