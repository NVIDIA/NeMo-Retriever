#!/usr/bin/env python3
"""
Convert retrieval-bench per-query trace files into a FileRetriever JSON
that the QA evaluation harness can consume.

retrieval-bench produces per-query traces like:
    traces/<run_name>/<dataset_dir>/<query_id>.json
        {"query_id": "42", "run": {"doc_0": 0.95, "doc_1": 0.82, ...}, ...}

The QA eval harness expects:
    {"queries": {"query text": {"chunks": ["text1", "text2", ...]}}}

This script loads the same dataset retrieval-bench used, maps doc IDs back
to corpus text, and writes the bridge JSON.

Usage:
    python convert_traces_to_retrieval.py \
        --traces-dir traces \
        --trace-run-name DenseRetrievalPipeline__model \
        --dataset-name vidore/vidore_v3_finance_en \
        --top-k 5 \
        --output retrieval_for_qa_eval.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    """Mirror retrieval-bench's _slugify for trace path resolution."""
    import re

    value = (value or "").strip()
    if not value:
        return "unnamed"
    value = value.replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_{3,}", "__", value).strip("_")
    return value or "unnamed"


def _dataset_trace_dir(dataset_name: str) -> str:
    parts = [p for p in str(dataset_name or "unknown_dataset").split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "bright":
        return _slugify(f"bright__{parts[1]}")
    return _slugify(parts[-1] if parts else "unknown_dataset")


def main():
    parser = argparse.ArgumentParser(description="Convert retrieval-bench traces to QA eval FileRetriever JSON.")
    parser.add_argument(
        "--traces-dir",
        required=True,
        help="Root traces directory (same as --traces-dir passed to retrieval-bench).",
    )
    parser.add_argument(
        "--trace-run-name",
        required=True,
        help="Trace run name (subdirectory under traces-dir). "
        "Shown in retrieval-bench output, e.g. 'DenseRetrievalPipeline__model_name'.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset identifier (same as --dataset-name passed to retrieval-bench), "
        "e.g. 'vidore/vidore_v3_finance_en' or 'bright/biology'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top chunks per query (default: 5).",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument("--language", default=None, help="Language filter (optional).")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the FileRetriever-compatible file.",
    )
    args = parser.parse_args()

    dataset_dir = _dataset_trace_dir(args.dataset_name)
    trace_root = Path(args.traces_dir) / _slugify(args.trace_run_name) / dataset_dir

    if not trace_root.exists():
        logger.error("Trace directory not found: %s", trace_root)
        logger.error(
            "Expected structure: %s/<trace_run_name>/<dataset_dir>/<query_id>.json",
            args.traces_dir,
        )
        sys.exit(1)

    trace_files = sorted(trace_root.glob("*.json"))
    if not trace_files:
        logger.error("No trace JSON files found in %s", trace_root)
        sys.exit(1)
    logger.info("Found %d trace files in %s", len(trace_files), trace_root)

    logger.info("Loading dataset '%s' (split=%s) ...", args.dataset_name, args.split)
    from retrieval_bench.pipeline_evaluation import load_vidore_dataset

    (
        query_ids,
        queries,
        corpus_ids,
        corpus_images,
        corpus_texts,
        qrels,
        query_languages,
        _,
    ) = load_vidore_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        language=args.language,
    )

    qid_to_query = {qid: q for qid, q in zip(query_ids, queries)}
    cid_to_text = {cid: t for cid, t in zip(corpus_ids, corpus_texts)}
    logger.info("Dataset: %d queries, %d corpus documents", len(qid_to_query), len(cid_to_text))

    output_queries: dict[str, dict] = {}
    skipped = 0
    missing_docs = 0

    for trace_file in trace_files:
        try:
            with open(trace_file) as f:
                trace = json.load(f)
        except Exception:
            logger.warning("Failed to parse %s, skipping", trace_file)
            skipped += 1
            continue

        qid = str(trace.get("query_id", trace_file.stem))
        run = trace.get("run")
        if not isinstance(run, dict):
            logger.debug("No 'run' dict in %s, skipping", trace_file.name)
            skipped += 1
            continue

        query_text = qid_to_query.get(qid)
        if query_text is None:
            logger.debug("query_id %s not found in dataset, skipping", qid)
            skipped += 1
            continue

        ranked = sorted(run.items(), key=lambda x: float(x[1]), reverse=True)[: args.top_k]

        chunks = []
        metadata = []
        for doc_id, score in ranked:
            text = cid_to_text.get(doc_id)
            if text is None:
                missing_docs += 1
                continue
            chunks.append(text)
            metadata.append({"doc_id": doc_id, "score": float(score)})

        if chunks:
            output_queries[query_text] = {"chunks": chunks, "metadata": metadata}

    if skipped:
        logger.warning("Skipped %d trace files (parse errors or missing data)", skipped)
    if missing_docs:
        logger.warning("%d doc_id lookups failed (ID not in corpus)", missing_docs)

    output = {"queries": output_queries}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Wrote %d queries to %s (top_k=%d)",
        len(output_queries),
        out_path,
        args.top_k,
    )
    print(f"\nDone. FileRetriever JSON: {out_path}")
    print(f"Queries converted: {len(output_queries)} / {len(trace_files)}")


if __name__ == "__main__":
    main()
