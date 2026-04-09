"""Convert retrieval-bench per-query trace files into a FileRetriever JSON.

Usable as a standalone CLI::

    python convert_traces_to_retrieval.py \\
        --traces-dir /path/to/traces \\
        --trace-run-name DenseRetrievalPipeline__backend \\
        --dataset-name vidore/vidore_v3_finance_en \\
        --output output.json

or as an importable helper via :func:`convert_traces`.
"""

import json
import logging
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


def convert_traces(
    traces_dir: str,
    trace_run_name: str,
    dataset_name: str,
    output: str,
    top_k: int = 5,
    split: str = "test",
    language: str | None = None,
) -> dict:
    """Convert retrieval-bench trace files into a FileRetriever JSON.

    Returns a dict with keys: ``queries_written``, ``traces_found``,
    ``skipped``, ``missing_docs``, ``output_path``.
    """
    dataset_dir = _dataset_trace_dir(dataset_name)
    trace_root = Path(traces_dir) / _slugify(trace_run_name) / dataset_dir

    if not trace_root.exists():
        raise FileNotFoundError(
            f"Trace directory not found: {trace_root}\n"
            f"Expected structure: {traces_dir}/<trace_run_name>/<dataset_dir>/<query_id>.json"
        )

    trace_files = sorted(trace_root.glob("*.json"))
    if not trace_files:
        raise FileNotFoundError(f"No trace JSON files found in {trace_root}")
    logger.info("Found %d trace files in %s", len(trace_files), trace_root)

    logger.info("Loading dataset '%s' (split=%s) ...", dataset_name, split)
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
        dataset_name=dataset_name,
        split=split,
        language=language,
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

        ranked = sorted(run.items(), key=lambda x: float(x[1]), reverse=True)[:top_k]

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

    result = {"queries": output_queries}
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Wrote %d queries to %s (top_k=%d)", len(output_queries), out_path, top_k)
    return {
        "queries_written": len(output_queries),
        "traces_found": len(trace_files),
        "skipped": skipped,
        "missing_docs": missing_docs,
        "output_path": str(out_path),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert retrieval-bench traces to FileRetriever JSON.",
    )
    parser.add_argument("--traces-dir", required=True, help="Root directory containing trace folders.")
    parser.add_argument("--trace-run-name", required=True, help="Name of the trace run (subfolder under traces-dir).")
    parser.add_argument(
        "--dataset-name", required=True, help="HuggingFace dataset ID (e.g. vidore/vidore_v3_finance_en)."
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--top-k", type=int, default=5, help="Max documents per query (default: 5).")
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument("--language", default=None, help="Language filter (default: none).")
    args = parser.parse_args()

    result = convert_traces(
        traces_dir=args.traces_dir,
        trace_run_name=args.trace_run_name,
        dataset_name=args.dataset_name,
        output=args.output,
        top_k=args.top_k,
        split=args.split,
        language=args.language,
    )
    print(f"\nDone: {result['queries_written']} queries written to {result['output_path']}")
    if result["skipped"]:
        print(f"  Skipped: {result['skipped']}")
    if result["missing_docs"]:
        print(f"  Missing docs: {result['missing_docs']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
