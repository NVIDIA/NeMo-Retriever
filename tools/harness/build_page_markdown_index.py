"""Build a page-level markdown index from extracted Parquet files.

Loads extraction results saved by extract_bo767_parquet.py, groups records
by (source document, page number), renders each page via to_markdown_by_page,
and writes a JSON index mapping source_id -> page_number -> markdown.

Usage:
    python build_page_markdown_index.py

Env vars:
    PARQUET_DIR    Directory containing Parquet files (default: data/bo767_extracted)
    OUTPUT_FILE    Where to write the JSON index (default: data/bo767_page_markdown.json)
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    parquet_dir = os.environ.get(
        "PARQUET_DIR", os.path.join(_HERE, "data", "bo767_extracted"),
    )
    output_file = os.environ.get(
        "OUTPUT_FILE", os.path.join(_HERE, "data", "bo767_page_markdown.json"),
    )

    print("=" * 60)
    print("Build Page Markdown Index")
    print("=" * 60)
    print(f"Parquet dir:  {parquet_dir}")
    print(f"Output file:  {output_file}")

    if not os.path.isdir(parquet_dir):
        print(f"ERROR: Parquet directory not found: {parquet_dir}", file=sys.stderr)
        return 1

    import pandas as pd
    from nemo_retriever.io.markdown import to_markdown_by_page

    parquet_files = sorted(Path(parquet_dir).rglob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No .parquet files found in {parquet_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(parquet_files)} Parquet file(s)")

    t0 = time.monotonic()
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} records in {time.monotonic() - t0:.1f}s")
    print(f"Columns: {list(df.columns)}")

    path_col = "path" if "path" in df.columns else "source_id"
    if path_col not in df.columns:
        print(f"ERROR: Neither 'path' nor 'source_id' found in columns", file=sys.stderr)
        return 1

    def _ndarray_to_list(record: dict) -> dict:
        """Pandas reads Parquet list columns as numpy arrays.
        to_markdown_by_page checks isinstance(items, list), so convert them."""
        for key in ("table", "chart", "infographic", "tables", "charts", "infographics"):
            val = record.get(key)
            if isinstance(val, np.ndarray):
                record[key] = val.tolist()
        return record

    docs_grouped = defaultdict(list)
    for _, row in df.iterrows():
        source = str(row.get(path_col, ""))
        if source:
            docs_grouped[source].append(_ndarray_to_list(row.to_dict()))

    print(f"Grouped into {len(docs_grouped)} documents")

    t1 = time.monotonic()
    index: dict[str, dict[str, str]] = {}
    total_pages = 0

    for source_id, records in docs_grouped.items():
        try:
            pages = to_markdown_by_page(records)
        except Exception as exc:
            print(f"  WARNING: Failed to render {source_id}: {exc}")
            continue

        page_map: dict[str, str] = {}
        for page_number, markdown in pages.items():
            page_map[str(page_number)] = markdown
            total_pages += 1

        index[source_id] = page_map

    elapsed_render = time.monotonic() - t1
    print(f"Rendered {total_pages} pages from {len(index)} documents in {elapsed_render:.1f}s")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\nIndex written to {output_file} ({size_mb:.1f} MB)")
    print(f"  Documents: {len(index)}")
    print(f"  Pages:     {total_pages}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
