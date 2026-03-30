#!/usr/bin/env python3
"""
Export VDB retrieval results for all ground-truth queries to a FileRetriever JSON.

This bridges ingestion (e2e case) and QA evaluation (qa_eval case):
  1. Run e2e to ingest PDFs into VDB.
  2. Run this script to query the VDB for every ground-truth question.
  3. Run qa_eval with qa_retriever=file pointing at the exported JSON.

Separating retrieval from generation/judging means you can:
  - Iterate on LLM configs without re-running expensive VDB queries.
  - Cache retrieval results for reproducibility.
  - Inspect retrieved chunks before spending money on API calls.

Usage:
  uv run python retrieve_and_export.py

Environment variables:
  HOSTNAME          Service hostname (default: localhost)
  VDB_BACKEND       "lancedb" or "milvus" (default: lancedb)
  COLLECTION_NAME   VDB collection / LanceDB table name (default: bo767_multimodal)
  LANCEDB_DIR       LanceDB base directory (default: tools/harness/lancedb)
  TOP_K             Chunks per query (default: 5)
  QA_DATASET        Dataset key or csv: path (default: csv:data/bo767_annotations.csv)
  GROUND_TRUTH_DIR  Directory with the CSV (default: tools/harness/data)
  BATCH_SIZE        Queries per VDB batch call (default: 50)
  OUTPUT_FILE       Where to write the JSON (default: data/test_retrieval/bo767_retrieval.json)
  SPARSE            Enable sparse/hybrid for Milvus (default: false)
  GPU_SEARCH        Use GPU search for Milvus (default: false)
  HYBRID            LanceDB hybrid retrieval (default: false)
"""

import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))


def _bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


def main() -> int:
    hostname = os.environ.get("HOSTNAME", "localhost")
    vdb_backend = os.environ.get("VDB_BACKEND", "lancedb")
    collection_name = os.environ.get("COLLECTION_NAME", "bo767_multimodal")
    lancedb_dir = os.environ.get("LANCEDB_DIR", os.path.join(_HERE, "lancedb"))
    top_k = int(os.environ.get("TOP_K", "5"))
    qa_dataset = os.environ.get("QA_DATASET", "csv:data/bo767_annotations.csv")
    ground_truth_dir = os.environ.get("GROUND_TRUTH_DIR", os.path.join(_HERE, "data"))
    batch_size = int(os.environ.get("BATCH_SIZE", "50"))
    output_file = os.environ.get("OUTPUT_FILE", os.path.join(_HERE, "data", "test_retrieval", "bo767_retrieval.json"))
    sparse = _bool_env("SPARSE")
    gpu_search = _bool_env("GPU_SEARCH")
    hybrid = _bool_env("HYBRID")

    embedding_endpoint = f"http://{hostname}:8012/v1"

    print("=" * 60)
    print("Retrieve & Export")
    print("=" * 60)
    print(f"VDB Backend:    {vdb_backend}")
    print(f"Collection:     {collection_name}")
    print(f"Hostname:       {hostname}")
    print(f"Embedding:      {embedding_endpoint}")
    print(f"Top-K:          {top_k}")
    print(f"Batch size:     {batch_size}")
    print(f"Dataset:        {qa_dataset}")
    print(f"Ground truth:   {ground_truth_dir}")
    print(f"Output:         {output_file}")
    print("=" * 60)

    # Load ground truth queries
    from nv_ingest_harness.utils.qa.ground_truth import get_qa_dataset_loader

    loader = get_qa_dataset_loader(qa_dataset)
    qa_pairs = loader(data_dir=ground_truth_dir)
    queries = [pair["query"] for pair in qa_pairs]
    print(f"\nLoaded {len(queries)} queries from '{qa_dataset}'")

    # Detect embedding model
    from nv_ingest_harness.utils.interact import embed_info

    model_name, _ = embed_info()
    print(f"Embedding model: {model_name}")

    # Build retrieval function
    table_path = None
    if vdb_backend == "lancedb":
        table_path = str(Path(lancedb_dir) / collection_name)
        if not os.path.exists(table_path):
            print(f"ERROR: LanceDB table not found at {table_path}", file=sys.stderr)
            print("Run the e2e ingestion case first:", file=sys.stderr)
            print("  uv run nv-ingest-harness-run --case=e2e --dataset=bo767", file=sys.stderr)
            return 1

    from nv_ingest_harness.utils.recall import get_retrieval_func

    retrieval_func = get_retrieval_func(
        vdb_backend=vdb_backend,
        table_path=table_path,
        table_name=collection_name,
        hybrid=hybrid,
    )

    # Batch retrieval
    all_results: dict[str, dict] = {}
    total = len(queries)
    t0 = time.monotonic()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_queries = queries[batch_start:batch_end]

        if vdb_backend == "lancedb":
            raw = retrieval_func(
                batch_queries,
                embedding_endpoint=embedding_endpoint,
                model_name=model_name,
                top_k=top_k,
            )
        else:
            raw = retrieval_func(
                batch_queries,
                collection_name,
                hybrid=sparse,
                embedding_endpoint=embedding_endpoint,
                model_name=model_name,
                top_k=top_k,
                gpu_search=gpu_search,
            )

        for i, query in enumerate(batch_queries):
            hits = raw[i] if i < len(raw) else []
            chunks = []
            metadata = []

            for hit in hits:
                entity = hit.get("entity", {})
                text = entity.get("text", "")
                chunks.append(text)

                source = entity.get("source", {})
                content_meta = entity.get("content_metadata", {})
                metadata.append({
                    "source_id": source.get("source_id", ""),
                    "page_number": content_meta.get("page_number", ""),
                    "distance": hit.get("distance"),
                })

            all_results[query] = {
                "chunks": chunks,
                "metadata": metadata,
            }

        elapsed = time.monotonic() - t0
        print(f"  Progress: {batch_end}/{total} queries retrieved ({elapsed:.1f}s)")

    total_elapsed = time.monotonic() - t0

    # Stats
    empty_count = sum(1 for r in all_results.values() if not r["chunks"])
    avg_chunks = sum(len(r["chunks"]) for r in all_results.values()) / max(len(all_results), 1)

    print(f"\nRetrieval complete in {total_elapsed:.1f}s")
    print(f"  Queries:       {len(all_results)}")
    print(f"  Avg chunks:    {avg_chunks:.1f}")
    print(f"  Empty results: {empty_count}")

    # Write output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output = {
        "metadata": {
            "vdb_backend": vdb_backend,
            "collection_name": collection_name,
            "top_k": top_k,
            "embedding_model": model_name,
            "query_count": len(all_results),
            "dataset": qa_dataset,
            "elapsed_s": round(total_elapsed, 1),
        },
        "queries": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
