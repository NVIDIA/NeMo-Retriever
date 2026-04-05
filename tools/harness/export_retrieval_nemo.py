"""Export retrieval results from NeMo Retriever LanceDB to FileRetriever JSON.

Uses nemo_retriever.export to query the LanceDB table populated by
graph_pipeline.py, producing a JSON file compatible with the QA eval
pipeline's FileRetriever.

Full-page markdown mode (optional):
    Set PAGE_MARKDOWN_INDEX to a JSON file produced by build_page_markdown_index.py.
    When set, vector search hits are expanded to full-page markdown -- multiple
    hits from the same page are deduplicated into a single chunk containing the
    complete page rendered via to_markdown_by_page.

Usage:
    python export_retrieval_nemo.py

The output JSON lands at data/test_retrieval/bo767_retrieval.json by default.
"""

import csv
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_EMBEDDER = "nvidia/llama-nemotron-embed-1b-v2"


def _load_queries(csv_path: str) -> list[dict]:
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("query", "").strip()
            answer = row.get("answer", "").strip()
            if query and answer:
                pairs.append(row)
    return pairs


def main() -> int:
    lancedb_uri = os.environ.get("LANCEDB_URI", os.path.join(_HERE, "lancedb"))
    lancedb_table = os.environ.get("LANCEDB_TABLE", "nv-ingest")
    top_k = int(os.environ.get("TOP_K", "5"))
    embedder = os.environ.get("EMBEDDER", DEFAULT_EMBEDDER)
    _repo_root = os.path.normpath(os.path.join(_HERE, "..", ".."))
    csv_path = os.environ.get(
        "QA_CSV",
        os.path.join(_repo_root, "data", "bo767_annotations.csv"),
    )
    output_file = os.environ.get(
        "OUTPUT_FILE",
        os.path.join(_HERE, "data", "test_retrieval", "bo767_retrieval.json"),
    )
    page_markdown_index_path = os.environ.get("PAGE_MARKDOWN_INDEX", "")

    use_fullpage = bool(page_markdown_index_path)
    page_index: dict[str, dict[str, str]] | None = None

    print("=" * 60)
    print("NeMo Retriever -> FileRetriever JSON Export")
    print("=" * 60)
    print(f"LanceDB URI:    {lancedb_uri}")
    print(f"LanceDB Table:  {lancedb_table}")
    print(f"Top-K:          {top_k}")
    print(f"Embedder:       {embedder}")
    print(f"QA CSV:         {csv_path}")
    print(f"Output:         {output_file}")
    print(f"Full-page mode: {'ON (' + page_markdown_index_path + ')' if use_fullpage else 'OFF'}")
    print("=" * 60)

    if use_fullpage:
        print("\nLoading page markdown index ...")
        with open(page_markdown_index_path, encoding="utf-8") as f:
            page_index = json.load(f)
        total_pages = sum(len(pages) for pages in page_index.values())
        print(f"  {len(page_index)} documents, {total_pages} pages")

    qa_pairs = _load_queries(csv_path)
    print(f"\nLoaded {len(qa_pairs)} queries")

    from nemo_retriever.export import export_retrieval_json

    t0 = time.monotonic()
    output = export_retrieval_json(
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        queries=qa_pairs,
        output_path=output_file,
        top_k=top_k,
        embedder=embedder,
        page_index=page_index,
    )
    total_elapsed = time.monotonic() - t0

    all_results = output.get("queries", {})
    empty_count = sum(1 for r in all_results.values() if not r["chunks"])
    avg_chunks = sum(len(r["chunks"]) for r in all_results.values()) / max(len(all_results), 1)

    print(f"\nRetrieval complete in {total_elapsed:.1f}s")
    print(f"  Queries:       {len(all_results)}")
    print(f"  Avg chunks:    {avg_chunks:.1f}")
    print(f"  Empty results: {empty_count}")
    if use_fullpage:
        print(f"  Page index misses: {output['metadata'].get('page_index_misses', 0)}")

    print(f"\nExported to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
