# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recall@k A/B: dense-only vs hybrid retrieval over a labeled manifest.

Manifest: JSONL, one object per line:
  {"query": "...", "source": "<source_id substring>", "gold_pages": [1, 4]}

A hit counts as relevant when its source_id contains the gold ``source`` and its
page_number is in ``gold_pages``. Self-contained page-match scoring (does not
depend on recall.core internals).

Iteration cost: retrieval is the expensive step, so each (query, leg) is
retrieved exactly once at the largest k and sliced for every smaller k. Set
``EMBED_INVOKE_URL`` (printed by ``retriever serve-models``) to reuse a warm
embedding server instead of cold-loading the model per query."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nemo_retriever.adapters.cli.sdk_workflow import query_documents


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_relevant(hit: dict, source: str, gold_pages: set[int]) -> bool:
    src = str(hit.get("source_id") or hit.get("source") or "")
    page = hit.get("page_number")
    return source in src and page in gold_pages


def ranked_relevance(
    row: dict, *, k_max: int, lancedb_uri: str, table_name: str, hybrid: bool, embed_invoke_url: str | None
) -> list[bool]:
    """Retrieve once at k_max and return the per-rank relevance flags for one query."""
    hits = query_documents(
        row["query"],
        top_k=k_max,
        candidate_k=max(k_max * 4, 40),
        hybrid=hybrid,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        embed_invoke_url=embed_invoke_url,
    )
    gold = set(int(p) for p in row["gold_pages"])
    return [is_relevant(h, row["source"], gold) for h in hits]


def recall_at_k(per_query_rel: list[list[bool]], k: int) -> float:
    if not per_query_rel:
        return 0.0
    return sum(1 for rel in per_query_rel if any(rel[:k])) / len(per_query_rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--lancedb-uri", required=True)
    ap.add_argument("--table-name", required=True)
    ap.add_argument("--ks", default="1,3,5,10")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    ks = sorted(int(x) for x in args.ks.split(","))
    k_max = ks[-1]
    embed_invoke_url = os.environ.get("EMBED_INVOKE_URL") or None

    # Retrieve once per (query, leg) at k_max; slice for every smaller k.
    legs = {}
    for hybrid in (False, True):
        legs[hybrid] = [
            ranked_relevance(
                row,
                k_max=k_max,
                lancedb_uri=args.lancedb_uri,
                table_name=args.table_name,
                hybrid=hybrid,
                embed_invoke_url=embed_invoke_url,
            )
            for row in rows
        ]

    print(f"n_queries={len(rows)}  embed={'warm' if embed_invoke_url else 'local cold-load'}")
    print(f"{'k':>4} {'dense':>8} {'hybrid':>8} {'delta':>8}")
    for k in ks:
        d = recall_at_k(legs[False], k)
        h = recall_at_k(legs[True], k)
        print(f"{k:>4} {d:>8.3f} {h:>8.3f} {h - d:>+8.3f}")


if __name__ == "__main__":
    main()
