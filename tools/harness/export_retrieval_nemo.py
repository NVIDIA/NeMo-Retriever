"""Export retrieval results from NeMo Retriever LanceDB to FileRetriever JSON.

Uses nemo_retriever.retriever.Retriever to query the LanceDB table populated
by ingest_bo767.py, producing a JSON file compatible with the QA eval
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
BATCH_SIZE = 50


def load_queries(csv_path: str) -> list[dict]:
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("query", "").strip()
            answer = row.get("answer", "").strip()
            if query and answer:
                pairs.append(row)
    return pairs


def parse_json_field(raw):
    """Safely parse a field that may be a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _load_page_index(path: str) -> dict[str, dict[str, str]]:
    """Load the page markdown index JSON: {source_id: {page_number_str: markdown}}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _expand_hits_to_pages(
    hits: list[dict],
    page_index: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict], int]:
    """Deduplicate hits by (source_id, page_number) and look up full-page markdown.

    Returns (chunks, metadata, miss_count) where miss_count is the number of
    (source_id, page) pairs that had no entry in the page index.
    """
    seen: dict[tuple[str, int], float] = {}
    ordered_pages: list[tuple[str, int]] = []

    for hit in hits:
        source = parse_json_field(hit.get("source", "{}"))
        meta = parse_json_field(hit.get("metadata", "{}"))
        source_id = source.get("source_id", "")
        page_number = meta.get("page_number", hit.get("page_number", -1))
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = -1
        distance = hit.get("_distance")

        key = (source_id, page_number)
        if key not in seen:
            seen[key] = distance
            ordered_pages.append(key)
        elif distance is not None and (seen[key] is None or distance < seen[key]):
            seen[key] = distance

    chunks = []
    metadata = []
    miss_count = 0
    for source_id, page_number in ordered_pages:
        page_str = str(page_number)
        doc_pages = page_index.get(source_id, {})
        md = doc_pages.get(page_str)
        if md is None:
            miss_count += 1
            continue
        chunks.append(md)
        metadata.append({
            "source_id": source_id,
            "page_number": page_number,
            "distance": seen[(source_id, page_number)],
        })

    return chunks, metadata, miss_count


def _validate_page_index_keys(
    first_batch_hits: list[list[dict]],
    page_index: dict[str, dict[str, str]],
) -> None:
    """Sample source_ids from LanceDB hits and verify they exist in the page index.

    Aborts with an actionable error if none of the sampled keys match, which
    indicates a source_id convention mismatch between ingest and extract.
    """
    sampled: set[str] = set()
    for hits in first_batch_hits:
        for hit in hits:
            source = parse_json_field(hit.get("source", "{}"))
            sid = source.get("source_id", "")
            if sid:
                sampled.add(sid)
            if len(sampled) >= 5:
                break
        if len(sampled) >= 5:
            break

    if not sampled:
        return

    matched = sum(1 for sid in sampled if sid in page_index)
    if matched == 0:
        sample_list = list(sampled)[:3]
        index_sample = list(page_index.keys())[:3]
        print(
            f"ERROR: None of {len(sampled)} sampled LanceDB source_ids found in "
            f"page index.\n"
            f"  LanceDB samples:  {sample_list}\n"
            f"  Index samples:    {index_sample}\n"
            f"  This usually means ingest and extract used different path "
            f"conventions for source_id.\n"
            f"  Re-run extract_bo767_parquet.py with the same --dataset-dir "
            f"used for ingest_bo767.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Page index key check: {matched}/{len(sampled)} sampled source_ids found")


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
    page_index: dict[str, dict[str, str]] = {}

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
        page_index = _load_page_index(page_markdown_index_path)
        total_pages = sum(len(pages) for pages in page_index.values())
        print(f"  {len(page_index)} documents, {total_pages} pages")

    qa_pairs = load_queries(csv_path)
    query_strings = [p["query"] for p in qa_pairs]
    print(f"\nLoaded {len(query_strings)} queries")

    from nemo_retriever.retriever import Retriever

    retriever = Retriever(
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedder=embedder,
        top_k=top_k,
        reranker=False,
    )

    all_results: dict[str, dict] = {}
    total_page_misses = 0
    t0 = time.monotonic()
    validated_index = False

    for batch_start in range(0, len(query_strings), BATCH_SIZE):
        batch = query_strings[batch_start : batch_start + BATCH_SIZE]
        batch_hits = retriever.queries(batch)

        if use_fullpage and not validated_index:
            _validate_page_index_keys(batch_hits, page_index)
            validated_index = True

        for query, hits in zip(batch, batch_hits):
            if use_fullpage:
                chunks, metadata, misses = _expand_hits_to_pages(hits, page_index)
                total_page_misses += misses
            else:
                chunks = []
                metadata = []
                for hit in hits:
                    chunks.append(hit.get("text", ""))

                    source = parse_json_field(hit.get("source", "{}"))
                    meta = parse_json_field(hit.get("metadata", "{}"))

                    metadata.append({
                        "source_id": source.get("source_id", ""),
                        "page_number": meta.get("page_number", hit.get("page_number", "")),
                        "distance": hit.get("_distance"),
                    })

            all_results[query] = {"chunks": chunks, "metadata": metadata}

        elapsed = time.monotonic() - t0
        done = min(batch_start + BATCH_SIZE, len(query_strings))
        print(f"  Progress: {done}/{len(query_strings)} queries ({elapsed:.1f}s)")

    total_elapsed = time.monotonic() - t0

    empty_count = sum(1 for r in all_results.values() if not r["chunks"])
    avg_chunks = sum(len(r["chunks"]) for r in all_results.values()) / max(len(all_results), 1)

    print(f"\nRetrieval complete in {total_elapsed:.1f}s")
    print(f"  Queries:       {len(all_results)}")
    print(f"  Avg chunks:    {avg_chunks:.1f}")
    print(f"  Empty results: {empty_count}")
    if use_fullpage:
        print(f"  Page index misses: {total_page_misses}")

    chunk_mode = "full-page markdown" if use_fullpage else "sub-page chunks"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    meta = {
        "vdb_backend": "lancedb",
        "collection_name": lancedb_table,
        "top_k": top_k,
        "embedding_model": embedder,
        "chunk_mode": chunk_mode,
        "query_count": len(all_results),
        "elapsed_s": round(total_elapsed, 1),
    }
    if use_fullpage:
        meta["page_index_misses"] = total_page_misses
    output = {
        "metadata": meta,
        "queries": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
