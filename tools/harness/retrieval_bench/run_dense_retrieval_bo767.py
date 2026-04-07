#!/usr/bin/env python3
"""
Run retrieval-bench dense retrieval on the bo767 corpus and output
a FileRetriever JSON compatible with the QA evaluation harness.

This enables apples-to-apples comparison between:
  - LanceDB retrieval (existing bo767_retrieval_fullpage.json)
  - Dense retrieval with llama-nv-embed-reasoning-3b (this script)

Usage:
    python run_dense_retrieval_bo767.py \
        --backend llama-nv-embed-reasoning-3b \
        --top-k 5 \
        --output data/test_retrieval/bo767_retrieval_dense.json
"""

import argparse
import csv
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def run_dense_retrieval(
    output: str,
    backend: str = "llama-nv-embed-reasoning-3b",
    top_k: int = 5,
    retriever_top_k: int = 100,
    markdown_index: str | None = None,
    csv_path: str | None = None,
) -> dict:
    """Run dense retrieval on the bo767 corpus and write FileRetriever JSON.

    Returns a dict with keys: ``queries_written``, ``corpus_pages``,
    ``documents``, ``backend``, ``top_k``, ``index_time_s``,
    ``retrieval_time_s``, ``output_path``.
    """
    if markdown_index is None:
        markdown_index = os.path.join(_HERE, "..", "data", "bo767_page_markdown.json")
    if csv_path is None:
        csv_path = os.path.join(_HERE, "..", "..", "..", "data", "bo767_annotations.csv")

    with open(markdown_index) as f:
        md_index = json.load(f)

    corpus_ids = []
    corpus_texts = []
    for doc_path, pages in md_index.items():
        doc_name = os.path.basename(doc_path)
        for page_num, text in pages.items():
            corpus_ids.append(f"{doc_name}:{page_num}")
            corpus_texts.append(text)

    csv_path = os.path.normpath(csv_path)
    queries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("query", "").strip()
            if q:
                queries.append(q)
    query_ids = [str(i) for i in range(len(queries))]

    from retrieval_bench.pipelines.dense import DenseRetrievalPipeline

    pipeline = DenseRetrievalPipeline(backend=backend, top_k=retriever_top_k)
    corpus_images = [None] * len(corpus_ids)

    t0 = time.time()
    pipeline.index(
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        dataset_name="bo767",
    )
    index_time = time.time() - t0

    t0 = time.time()
    result = pipeline.search(query_ids=query_ids, queries=queries)
    if isinstance(result, tuple):
        run_dict, _ = result
    else:
        run_dict = result
    retrieval_time = time.time() - t0

    cid_to_text = {cid: t for cid, t in zip(corpus_ids, corpus_texts)}

    output_queries = {}
    for qid, query_text in zip(query_ids, queries):
        scores = run_dict.get(qid, {})
        ranked = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)[:top_k]

        chunks = []
        metadata = []
        for doc_id, score in ranked:
            text = cid_to_text.get(doc_id, "")
            if text:
                chunks.append(text)
                metadata.append({"doc_id": doc_id, "score": float(score)})

        output_queries[query_text] = {"chunks": chunks, "metadata": metadata}

    file_output = {"queries": output_queries}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(file_output, f, indent=2)

    return {
        "queries_written": len(output_queries),
        "corpus_pages": len(corpus_ids),
        "documents": len(md_index),
        "backend": backend,
        "top_k": top_k,
        "index_time_s": round(index_time, 1),
        "retrieval_time_s": round(retrieval_time, 1),
        "output_path": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval on bo767 corpus.")
    parser.add_argument(
        "--backend",
        default="llama-nv-embed-reasoning-3b",
        help="retrieval-bench backend (default: llama-nv-embed-reasoning-3b)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks per query in output (default: 5)")
    parser.add_argument(
        "--retriever-top-k",
        type=int,
        default=100,
        help="Top-k passed to the dense retriever (default: 100, same as retrieval-bench)",
    )
    parser.add_argument(
        "--markdown-index",
        default=os.path.join(_HERE, "..", "data", "bo767_page_markdown.json"),
        help="Path to bo767_page_markdown.json",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(_HERE, "..", "..", "..", "data", "bo767_annotations.csv"),
        help="Path to annotations CSV with query column",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output FileRetriever JSON path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dense Retrieval on bo767 corpus")
    print("=" * 60)
    print(f"Backend:        {args.backend}")
    print(f"Markdown index: {args.markdown_index}")
    print(f"CSV:            {args.csv}")
    print(f"Top-K:          {args.top_k}")
    print(f"Retriever top-K:{args.retriever_top_k}")
    print("=" * 60)

    stats = run_dense_retrieval(
        output=args.output,
        backend=args.backend,
        top_k=args.top_k,
        retriever_top_k=args.retriever_top_k,
        markdown_index=args.markdown_index,
        csv_path=args.csv,
    )

    print(f"\nWrote {stats['queries_written']} queries to {stats['output_path']}")
    print(f"Documents: {stats['documents']}, Pages: {stats['corpus_pages']}")
    print(f"Index: {stats['index_time_s']}s, Retrieval: {stats['retrieval_time_s']}s")
    print("Done.")


if __name__ == "__main__":
    main()
