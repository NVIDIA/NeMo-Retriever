"""Ingest bo767 PDFs into LanceDB using NeMo Retriever Library (no containers).

Usage (single-file smoke test):
    python ingest_bo767.py --smoke-test

Usage (full dataset):
    python ingest_bo767.py

The LanceDB table is written to ./lancedb/nv-ingest by default.
"""

import argparse
import glob
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Ingest bo767 via nemo-retriever library")
    parser.add_argument("--dataset-dir",
                        default=os.environ.get("DATASET_DIR", ""),
                        help="Path to bo767 PDF directory (or set DATASET_DIR env var)")
    parser.add_argument("--lancedb-uri", default="lancedb",
                        help="LanceDB URI (directory)")
    parser.add_argument("--lancedb-table", default="nv-ingest",
                        help="LanceDB table name (must match export_retrieval_nemo.py LANCEDB_TABLE)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Ingest only the first file as a quick check")
    args = parser.parse_args()

    if not args.dataset_dir:
        parser.error("--dataset-dir is required (or set DATASET_DIR env var)")

    pdf_pattern = os.path.join(args.dataset_dir, "*.pdf")
    all_pdfs = sorted(glob.glob(pdf_pattern))
    if not all_pdfs:
        print(f"No PDFs found at {pdf_pattern}", file=sys.stderr)
        sys.exit(1)

    if args.smoke_test:
        documents = [all_pdfs[0]]
        print(f"Smoke test: ingesting 1 file: {documents[0]}")
    else:
        documents = all_pdfs
        print(f"Ingesting {len(documents)} PDFs from {args.dataset_dir}")

    from nemo_retriever import create_ingestor

    ingestor = create_ingestor(run_mode="batch")
    ingestor = (
        ingestor.files(documents)
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
        )
        .embed()
        .vdb_upload(
            lancedb_uri=args.lancedb_uri,
            lancedb_table=args.lancedb_table,
        )
    )

    print(f"LanceDB: {args.lancedb_uri}/{args.lancedb_table}")
    t0 = time.time()
    ray_dataset = ingestor.ingest()
    elapsed = time.time() - t0

    chunks = ray_dataset.get_dataset().take_all()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"First chunk preview: {chunks[0].get('text', '')[:200]}...")

    summary = {
        "documents": len(documents),
        "chunks": len(chunks),
        "elapsed_s": round(elapsed, 2),
        "lancedb_uri": args.lancedb_uri,
        "lancedb_table": args.lancedb_table,
    }
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
