"""Extract bo767 PDFs to Parquet (no embed, no VDB upload).

Runs only the extraction stage of the NeMo Retriever pipeline and saves
the full records (including table/chart/infographic columns) to Parquet.
These records are consumed by build_page_markdown_index.py to reconstruct
full-page markdown for the QA eval pipeline.

Usage:
    python extract_bo767_parquet.py

Env vars:
    DATASET_DIR    Path to bo767 PDF directory (required)
    OUTPUT_DIR     Where to write Parquet files (default: data/bo767_extracted)
"""

import argparse
import glob
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Extract bo767 PDFs to Parquet")
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR", ""),
        help="Path to bo767 PDF directory (or set DATASET_DIR env var)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "OUTPUT_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bo767_extracted"),
        ),
        help="Where to write Parquet files",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Extract only the first file as a quick check",
    )
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
        print(f"Smoke test: extracting 1 file: {documents[0]}")
    else:
        documents = all_pdfs
        print(f"Extracting {len(documents)} PDFs from {args.dataset_dir}")

    print(f"Output dir: {args.output_dir}")

    from nemo_retriever import create_ingestor

    ingestor = create_ingestor(run_mode="batch")
    ingestor = ingestor.files(documents).extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )

    print("Running extraction (no embed, no vdb_upload) ...")
    t0 = time.time()
    ingestor.ingest()
    elapsed_ingest = time.time() - t0
    print(f"Extraction done in {elapsed_ingest:.1f}s")

    print(f"Saving intermediate results to {args.output_dir} ...")
    t1 = time.time()
    ingestor.save_intermediate_results(args.output_dir)
    elapsed_save = time.time() - t1
    print(f"Parquet saved in {elapsed_save:.1f}s")

    summary = {
        "documents": len(documents),
        "output_dir": args.output_dir,
        "extract_elapsed_s": round(elapsed_ingest, 2),
        "save_elapsed_s": round(elapsed_save, 2),
    }
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
