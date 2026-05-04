#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throughput + recall evaluation for a deployed nemo-retriever service.

Phase 1 — **Ingest**: walks an input directory, hands every file to
``create_ingestor(run_mode="service")``, and reports pages-per-second.

Phase 2 — **Recall** (optional): loads a ground-truth CSV, sends each
query to ``POST /v1/query``, optionally reranks via ``POST /v1/rerank``,
and reports Recall@1, Recall@5, and Recall@10.

The ground-truth CSV must have columns ``query``, ``pdf``, and ``page``.
Hits are considered relevant when the source filename contains the
ground-truth ``pdf`` value and the page number matches.

Examples::

    # Ingest only
    python throughput_test.py --input-dir ./pdfs

    # Ingest + recall
    python throughput_test.py --input-dir ./pdfs \\
        --ground-truth ./data/bo767_annotations.csv

    # Recall with reranking
    python throughput_test.py --input-dir ./pdfs \\
        --ground-truth ./data/bo767_annotations.csv --rerank

    # Skip ingest (already done), just run recall
    python throughput_test.py --skip-ingest \\
        --ground-truth ./data/bo767_annotations.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from nemo_retriever import create_ingestor

DEFAULT_URL = "http://10.86.5.28:30670/"
DEFAULT_GROUND_TRUTH = "./data/bo767_annotations.csv"
PROGRESS_INTERVAL_S = 2.0


# ======================================================================
# Ingest helpers
# ======================================================================


_INGESTABLE_EXTENSIONS = {
    ".pdf", ".txt", ".html", ".htm", ".doc", ".docx",
    ".ppt", ".pptx", ".xls", ".xlsx", ".md", ".rst",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
}


def _iter_input_files(input_dir: Path) -> list[str]:
    """Return ingestable files under ``input_dir`` (recursive), sorted.

    Excludes non-document files (CSVs, JSONs, images, etc.) that would
    be sent to the service but cannot be meaningfully processed.
    """
    return [
        str(p)
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in _INGESTABLE_EXTENSIONS
    ]


def _print_progress(counts: Counter, files_total: int, elapsed: float) -> None:
    pages = counts.get("page_result", 0)
    failed = counts.get("page_failed", 0)
    started = counts.get("job_started", 0)
    completed = counts.get("job_complete", 0)
    pps = pages / elapsed if elapsed > 0 else 0.0
    print(
        f"[{elapsed:6.1f}s] jobs {completed}/{started}/{files_total} " f"pages={pages} failed={failed} PPS={pps:6.2f}",
        file=sys.stderr,
        flush=True,
    )


async def run_ingest(args: argparse.Namespace) -> int:
    """Run the ingest phase. Returns 0 on success."""
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}", file=sys.stderr)
        return 2

    files = _iter_input_files(input_dir)
    if not files:
        print(f"No files found under {input_dir}", file=sys.stderr)
        return 1

    base_url = args.url.rstrip("/")
    print(f"Target:           {base_url}", file=sys.stderr)
    print(f"Files:            {len(files)} under {input_dir}", file=sys.stderr)
    print(f"Max concurrency:  {args.max_concurrency}", file=sys.stderr)

    ingestor = create_ingestor(
        run_mode="service",
        base_url=base_url,
        documents=files,
        api_key=args.api_key or None,
        max_concurrency=args.max_concurrency,
    )

    counts: Counter = Counter()
    stream_errors: list[str] = []
    overflow_seen = False

    t0 = time.monotonic()
    last_progress = t0

    async for event in ingestor.aingest_stream():
        etype = event.get("event", "message")
        counts[etype] += 1

        if etype == "stream_error":
            err = event.get("error", "<unknown>")
            stream_errors.append(err)
            print(f"!! stream_error: {err}", file=sys.stderr, flush=True)
        elif etype == "stream_overflow":
            overflow_seen = True
            print(
                "!! stream_overflow: server lost events (in-memory bus full); "
                "throughput numbers will be a lower bound",
                file=sys.stderr,
                flush=True,
            )
        elif etype == "page_failed":
            print(
                f"!! page_failed: {event.get('source_file')} p{event.get('page_number')} " f"-- {event.get('error')}",
                file=sys.stderr,
                flush=True,
            )

        now = time.monotonic()
        if now - last_progress >= PROGRESS_INTERVAL_S:
            _print_progress(counts, len(files), now - t0)
            last_progress = now

    elapsed = time.monotonic() - t0
    pages = counts.get("page_result", 0)
    failed = counts.get("page_failed", 0)
    pps = pages / elapsed if elapsed > 0 else 0.0

    print()
    print("=" * 60)
    print("INGEST RESULTS")
    print("=" * 60)
    print(f"Files:    {len(files)}")
    print(f"Pages:    {pages}  (failed {failed})")
    print(f"Elapsed:  {elapsed:.2f}s")
    print(f"PPS:      {pps:.2f} pages/sec")
    print()
    print("Event counts:")
    for name in sorted(counts):
        print(f"  {name:<20} {counts[name]}")
    if stream_errors:
        print()
        print(f"Stream errors ({len(stream_errors)}):")
        for err in stream_errors:
            print(f"  - {err}")
    if overflow_seen:
        print()
        print("NOTE: server emitted stream_overflow; some events were dropped.")

    return 0 if (pages > 0 and not stream_errors) else 1


# ======================================================================
# Recall helpers
# ======================================================================


def _load_ground_truth(csv_path: Path) -> list[dict[str, str]]:
    """Load ground-truth CSV and return list of row dicts."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    required = {"query", "pdf", "page"}
    if rows and not required.issubset(rows[0].keys()):
        missing = required - set(rows[0].keys())
        raise ValueError(f"Ground-truth CSV is missing required columns: {missing}. " f"Found: {list(rows[0].keys())}")
    return rows


def _hit_matches(hit: dict[str, Any], gt_pdf: str, gt_page: int) -> bool:
    """Check if a query/rerank hit matches the ground-truth PDF + page.

    The ground-truth annotations CSV uses 0-indexed page numbers while the
    pipeline stores 1-indexed page numbers (page_idx + 1).  We accept a
    match if the hit page equals gt_page (legacy) OR gt_page + 1 (correct
    offset), and also check the composite pdf_page field.
    """
    source = str(hit.get("source", ""))
    pdf_basename = str(hit.get("pdf_basename", ""))
    pdf_page = str(hit.get("pdf_page", ""))
    hit_page = int(hit.get("page_number", -1))

    pdf_id = gt_pdf.strip()
    source_match = pdf_id in source or pdf_id in pdf_basename
    page_match = (
        hit_page == gt_page
        or hit_page == gt_page + 1
        or pdf_page == f"{pdf_id}_{gt_page + 1}"
    )

    return source_match and page_match


async def _query_service(
    client: httpx.AsyncClient,
    base_url: str,
    query_text: str,
    top_k: int,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Call POST /v1/query and return the list of hits."""
    payload: dict[str, Any] = {"query": query_text, "top_k": top_k}
    resp = await client.post(f"{base_url}/v1/query", json=payload, headers=headers)
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results", [])
    if results:
        return results[0].get("hits", [])
    return []


async def _rerank_service(
    client: httpx.AsyncClient,
    base_url: str,
    query_text: str,
    hits: list[dict[str, Any]],
    top_n: int,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Call POST /v1/rerank and return reranked hits."""
    passages = []
    for h in hits:
        p = dict(h)
        if "text" not in p:
            p["text"] = ""
        passages.append(p)

    payload: dict[str, Any] = {
        "query": query_text,
        "passages": passages,
        "top_n": top_n,
    }
    resp = await client.post(f"{base_url}/v1/rerank", json=payload, headers=headers)
    resp.raise_for_status()
    body = resp.json()
    return body.get("results", [])


def _compute_recall_at_k(
    results: list[dict[str, Any]],
    gt_pdf: str,
    gt_page: int,
    k: int,
) -> bool:
    """Return True if the ground-truth document appears in the top-k results."""
    for hit in results[:k]:
        if _hit_matches(hit, gt_pdf, gt_page):
            return True
    return False


async def run_recall(args: argparse.Namespace) -> int:
    """Run the recall evaluation phase. Returns 0 on success."""
    gt_path = Path(args.ground_truth).expanduser().resolve()
    if not gt_path.is_file():
        print(f"Ground-truth file not found: {gt_path}", file=sys.stderr)
        return 2

    rows = _load_ground_truth(gt_path)
    if not rows:
        print(f"Ground-truth file is empty: {gt_path}", file=sys.stderr)
        return 1

    base_url = args.url.rstrip("/")
    use_rerank = args.rerank
    top_k = args.top_k
    query_batch_size = args.query_batch_size

    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("RECALL EVALUATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Ground truth:     {gt_path} ({len(rows)} queries)", file=sys.stderr)
    print(f"Target:           {base_url}", file=sys.stderr)
    print(f"Top-K retrieval:  {top_k}", file=sys.stderr)
    print(f"Reranking:        {'ON' if use_rerank else 'OFF'}", file=sys.stderr)
    print(f"Batch size:       {query_batch_size}", file=sys.stderr)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    recall_at: dict[int, int] = {1: 0, 5: 0, 10: 0}
    total_evaluated = 0
    errors = 0
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=120.0) as client:
        sem = asyncio.Semaphore(query_batch_size)

        async def _evaluate_one(row: dict[str, str]) -> None:
            nonlocal total_evaluated, errors
            query_text = row["query"].strip()
            gt_pdf = row["pdf"].strip()
            try:
                gt_page = int(row["page"])
            except (ValueError, KeyError):
                return

            async with sem:
                try:
                    hits = await _query_service(client, base_url, query_text, top_k, headers)

                    if use_rerank and hits:
                        hits = await _rerank_service(
                            client,
                            base_url,
                            query_text,
                            hits,
                            top_k,
                            headers,
                        )

                    for k in recall_at:
                        if _compute_recall_at_k(hits, gt_pdf, gt_page, k):
                            recall_at[k] += 1

                    total_evaluated += 1
                except Exception as exc:
                    errors += 1
                    if errors <= 5:
                        print(
                            f"!! query error: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    elif errors == 6:
                        print(
                            "!! suppressing further query errors …",
                            file=sys.stderr,
                            flush=True,
                        )

        tasks = [asyncio.create_task(_evaluate_one(row)) for row in rows]

        done_count = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done_count += 1
            if done_count % 50 == 0 or done_count == len(tasks):
                elapsed = time.monotonic() - t0
                print(
                    f"  [{elapsed:6.1f}s] {done_count}/{len(rows)} queries evaluated",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed = time.monotonic() - t0
    qps = total_evaluated / elapsed if elapsed > 0 else 0.0

    print()
    print("=" * 60)
    print("RECALL RESULTS")
    print("=" * 60)
    print(f"Ground-truth:   {gt_path.name}")
    print(f"Queries:        {total_evaluated} evaluated, {errors} errors, {len(rows)} total")
    print(f"Reranking:      {'ON' if use_rerank else 'OFF'}")
    print(f"Elapsed:        {elapsed:.2f}s ({qps:.1f} queries/sec)")
    print()

    if total_evaluated > 0:
        r1 = recall_at[1] / total_evaluated * 100
        r5 = recall_at[5] / total_evaluated * 100
        r10 = recall_at[10] / total_evaluated * 100
        print(f"  Recall@1:     {recall_at[1]:>5}/{total_evaluated}  ({r1:6.2f}%)")
        print(f"  Recall@5:     {recall_at[5]:>5}/{total_evaluated}  ({r5:6.2f}%)")
        print(f"  Recall@10:    {recall_at[10]:>5}/{total_evaluated}  ({r10:6.2f}%)")
    else:
        print("  (no queries were successfully evaluated)")

    print()
    return 0 if total_evaluated > 0 else 1


# ======================================================================
# Main
# ======================================================================


async def run(args: argparse.Namespace) -> int:
    rc = 0

    if not args.skip_ingest:
        rc = await run_ingest(args)
        if rc != 0 and not args.ground_truth:
            return rc

    if args.ground_truth:
        recall_rc = await run_recall(args)
        rc = rc or recall_rc

    if not args.ground_truth and args.skip_ingest:
        print(
            "Nothing to do: --skip-ingest was set and no --ground-truth was provided.",
            file=sys.stderr,
        )
        return 2

    return rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Directory of files to ingest (walked recursively). Required unless --skip-ingest.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Service base URL (default: {DEFAULT_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional bearer token forwarded to the service.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help=(
            "Maximum concurrent page uploads (default: 4).  Lower this "
            "(e.g. 2) if you see httpx.ReadError from a NodePort path."
        ),
    )

    # Recall evaluation
    parser.add_argument(
        "--ground-truth",
        default="",
        metavar="CSV",
        help=("Path to a ground-truth CSV with columns: query, pdf, page.  " f"Default: {DEFAULT_GROUND_TRUTH}"),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Rerank query results via POST /v1/rerank before computing recall.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=8,
        help="Max concurrent query requests during recall evaluation (default: 8).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        default=False,
        help="Skip the ingest phase (assume data is already loaded).",
    )

    args = parser.parse_args()

    if not args.skip_ingest and not args.input_dir:
        parser.error("--input-dir is required unless --skip-ingest is set.")

    return args


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
