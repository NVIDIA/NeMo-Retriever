"""Build a page-level markdown index from extracted Parquet files.

Loads extraction results (Parquet files from the ingestion pipeline),
groups records by document, renders each page via to_markdown_by_page,
and writes a JSON index mapping source_id -> page_number -> markdown.

Usage:
    python build_page_markdown_index.py --parquet-dir data/bo767_extracted
"""

import argparse
import json
import os
import sys
import time


def main() -> int:
    _here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Build page-level markdown index from Parquet")
    parser.add_argument(
        "--parquet-dir",
        default=os.environ.get("PARQUET_DIR", os.path.join(_here, "data", "bo767_extracted")),
        help="Directory containing Parquet files (default: data/bo767_extracted)",
    )
    parser.add_argument(
        "--output-file",
        default=os.environ.get("OUTPUT_FILE", os.path.join(_here, "data", "bo767_page_markdown.json")),
        help="Where to write the JSON index (default: data/bo767_page_markdown.json)",
    )
    args = parser.parse_args()

    from nemo_retriever.io.markdown import build_page_index

    print(f"Parquet dir:  {args.parquet_dir}")
    print(f"Output file:  {args.output_file}")

    t0 = time.monotonic()
    index, failures = build_page_index(parquet_dir=args.parquet_dir)
    elapsed = time.monotonic() - t0

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    total_pages = sum(len(pages) for pages in index.values())
    size_mb = os.path.getsize(args.output_file) / 1024 / 1024
    print(f"\nBuilt index in {elapsed:.1f}s")
    print(f"  Documents: {len(index)}")
    print(f"  Pages:     {total_pages}")
    print(f"  File size: {size_mb:.1f} MB")
    if failures:
        print(f"  Failures:  {len(failures)} documents failed rendering")
        for source_id, error in list(failures.items())[:10]:
            print(f"    {source_id}: {error}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
