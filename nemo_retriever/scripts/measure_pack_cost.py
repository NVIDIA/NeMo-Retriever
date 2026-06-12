# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-cost A/B: today's full-chunk retrieve payload vs the spans-first evidence pack.

For each query it retrieves the same top-k hits once, then shapes them two ways:
  - FAT  : list of ``_evidence_item(h)`` — the current ``retrieve()`` payload, full chunk text.
  - PACK : ``build_evidence_pack(hits, query)`` — minimal spans + fidelity + citation + handle + confidence.
Both are JSON-serialized and token-counted (tiktoken cl100k_base — a proxy; the
*ratio* is what matters and is ~tokenizer-invariant). This measures the recall
layer's cost lever — how much smaller the payload that crosses the agent boundary
gets — independent of corpus size or recall.

Manifest: JSONL with at least {"query": "..."} per line (gold fields ignored here).
Set EMBED_INVOKE_URL to reuse a warm embedder; else the local embedder loads once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tiktoken

from nemo_retriever.adapters.cli.sdk_workflow import _evidence_item, query_documents
from nemo_retriever.evidence import build_evidence_pack

_ENC = tiktoken.get_encoding("cl100k_base")


def toks(obj) -> int:
    return len(_ENC.encode(json.dumps(obj, ensure_ascii=False, default=str)))


def text_toks(strings) -> int:
    return sum(len(_ENC.encode(s or "")) for s in strings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--lancedb-uri", required=True)
    ap.add_argument("--table-name", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-sentences", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    embed_invoke_url = os.environ.get("EMBED_INVOKE_URL") or None

    fat_tot = pack_tot = full_text_tot = span_text_tot = 0
    n = 0
    print(f"{'query':<42} {'fat_tok':>8} {'pack_tok':>9} {'reduce':>7}")
    for row in rows:
        q = row["query"]
        hits = query_documents(
            q, top_k=args.top_k, candidate_k=max(args.top_k * 4, 40),
            hybrid=True, lancedb_uri=args.lancedb_uri, table_name=args.table_name,
            embed_invoke_url=embed_invoke_url,
        )
        if not hits:
            continue
        fat = [_evidence_item(h) for h in hits]
        pack = build_evidence_pack(hits, q, max_sentences=args.max_sentences)
        ft, pt = toks(fat), toks(pack)
        full_t = text_toks(str(h.get("text", "")) for h in hits)
        span_t = text_toks(e["span"] for e in pack["evidence"])
        fat_tot += ft; pack_tot += pt; full_text_tot += full_t; span_text_tot += span_t; n += 1
        red = 1 - pt / ft if ft else 0.0
        print(f"{q[:42]:<42} {ft:>8} {pt:>9} {red:>6.0%}")

    if not n:
        print("no hits for any query")
        return
    print("-" * 70)
    print(f"queries={n}  top_k={args.top_k}  max_sentences={args.max_sentences}")
    print(f"avg FAT  payload/query : {fat_tot / n:8.0f} tok   (full chunk text: {full_text_tot / n:.0f} tok)")
    print(f"avg PACK payload/query : {pack_tot / n:8.0f} tok   (spans only:      {span_text_tot / n:.0f} tok)")
    print(f"payload reduction      : {1 - pack_tot / fat_tot:6.1%}   (text-only: {1 - span_text_tot / full_text_tot:.1%})")


if __name__ == "__main__":
    main()
