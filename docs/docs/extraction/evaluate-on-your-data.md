# Evaluate on your data

Retrieval quality and ingest throughput depend on your documents, hardware, and pipeline settings. NeMo Retriever Library is instrumentation for those measurements. Name the Nemotron model and configuration in every result you report. Treat the library as methodology, not as the result.

Use this page to measure quality and throughput on **your** datasets. Do not treat these procedures as a public ranking for the library, and do not treat a full-pipeline score as the quality of one model.

## Separate evaluation jobs { #separate-evaluation-jobs }

Use a method that matches the question you are answering. The following table lists common jobs.

| Evaluation target | Appropriate method |
| --- | --- |
| One embedding, reranking, or parsing model in isolation | Use the model's published evaluation tools and model card. The library does not replace MTEB, ViDoRe, OmniDocBench, or similar tools. |
| Embed retrieval quality on your corpus | Ingest with a named embedder, query with the same embedder, and score against your judgments. |
| Incremental lift from reranking | Hold the embed-only index and queries constant. Compare `rerank=False` against the same run with a named reranker enabled. |
| Parse or extraction quality on your documents | Run the documented extraction method, such as `method="nemotron_parse"`, and inspect the extracted rows. |
| Your production pipeline | Evaluate that pipeline on your data. Library results do not guarantee that your pipeline will match. |

For Python query construction, including `rerank=False`, refer to the [package quick start](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/README.md#run-a-recall-query). For CLI ingest and query flags, refer to the [Retriever CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli). For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

## Hugging Face local evaluation { #hugging-face-local-evaluation }

A local GPU workflow can run documented Nemotron checkpoints from Hugging Face. NIM is supported. It is not required for that path.

Install the `[local]` extra and follow the [package quick start](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever). For tested official local embedding checkpoints, including Nemotron-3 Embed and Llama Nemotron repositories, refer to [Dense Nemotron embedding checkpoints](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#dense-nemotron-embedding-checkpoints). Pass the same `--embed-model-name` (or on-disk path) at ingest and at query time. A local checkpoint is not a Helm NIM unless the support matrix lists a matching chart image.

Do not present Helm NIM defaults as the only evaluation path. The four default Helm NIMs remain the Kubernetes extraction stack. They are a different deployment from local Hugging Face evaluation.

## Record enough detail to rerun { #record-enough-detail-to-rerun }

A partner or teammate should be able to rerun your measurement. Record at least the following:

- Dataset identity and any query or judgment files you used
- Model repository IDs, revisions or tags, and precision
- Library version and the exact ingest, extract, embed, and query settings
- Runtime (local Hugging Face, vLLM, hosted NIM, or self-hosted NIM)
- Hardware
- Quality metrics, kept separate from ingest throughput

Keep quality scores separate from latency, throughput, and multi-node scaling claims. A faster ingest does not make a relevance score more valid.

The public library surfaces do not currently emit a versioned machine-readable result-card schema. Until that contract exists, use the checklist above and keep your own pinned configs.

## What these results do not mean { #what-these-results-do-not-mean }

Library measurements do not do the following:

- Rank NeMo Retriever Library as a product against other retrieval systems
- Transfer to a partner's production pipeline without a new evaluation
- Attribute a full extraction-to-retrieval score to a single model
- Replace the model's own published evaluation tools

Agentic retrieval is an optional query path. Local CLI agentic runs default to an in-process vLLM agent large language model (LLM), not to the embedding or reranking model under test. Attribute retrieval lift to the named Nemotron retrieval model and configuration. Refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).

## Throughput and dataset effects { #throughput-and-dataset-effects }

Refer to [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality) for why raw numbers from generic benchmarks may not match your corpus (layout complexity, file types, image density, and so on). Treat ingest throughput as an operational measurement, not as retrieval quality.

## Operational tuning { #operational-tuning }

- [Ray and distributed ingest](ray-logging.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for supported configurations
- [Troubleshoot](troubleshoot.md) when results or performance diverge from expectations
