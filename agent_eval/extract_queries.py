#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script A: extract an answer-free queries file from an eval manifest.

Reads a manifest like
``/raid/retriever-sdg-v3/runs/.../agent_scenario_manifest.json`` and writes a
``queries.json`` containing ONLY ``{query_id, prompt, domain}`` per query — no
gold answer, no relevant pages, no ``category``/``scoring_mode`` eval-intent
labels (a ``refusal`` label, for instance, would tell the agent the question is
a trap). Gold is recovered later by the report, which re-reads this same
manifest and joins on ``query_id``.

Selection: an entry is included iff it has a ``primary_eval_id`` and
``prompt_export_status == "exported"`` and a candidate prompt — i.e. the same
set of queries the existing skill_eval harness actually runs.

Usage:
    python extract_queries.py --manifest MANIFEST.json --out queries.json
        [--categories query,extract] [--domains vidore_v3_hr] [--limit N]

Pure stdlib; no nemo_retriever import.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script or as a module.
try:
    from .schema import Query, QuerySet
except ImportError:  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import Query, QuerySet


def _candidate_prompt(entry: dict) -> str | None:
    """Return the agent-facing prompt for an entry, or None if absent.

    Handles both manifest shapes: the corpus-level manifest exposes the final
    exported prompt at top-level ``prompt``; the scenario manifest nests it under
    ``scenario_prompt_candidates.candidates[]``. Falls back to ``original_query``.
    """
    # 1. exported top-level prompt (corpus-level manifest)
    top = entry.get("prompt")
    if isinstance(top, str) and top.strip():
        return top.strip()
    # 2. any ``*_prompt_candidates`` block (scenario_/corpus_level_/…)
    for key, val in entry.items():
        if key.endswith("_prompt_candidates") and isinstance(val, dict):
            for c in val.get("candidates") or []:
                prompt = (c or {}).get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()
    # 3. fall back to the raw query
    oq = entry.get("original_query")
    return oq.strip() if isinstance(oq, str) and oq.strip() else None


def _is_usable(entry: dict) -> bool:
    return bool(entry.get("primary_eval_id")) and entry.get("prompt_export_status") == "exported"


def extract(
    manifest_path: Path,
    *,
    categories: list[str] | None = None,
    domains: list[str] | None = None,
    limit: int | None = None,
) -> QuerySet:
    raw = json.loads(manifest_path.read_text())
    entries = raw if isinstance(raw, list) else (raw.get("entries") or list(raw.values()))

    cat_set = {c.strip() for c in categories} if categories else None
    dom_set = {d.strip() for d in domains} if domains else None

    queries: list[Query] = []
    seen: set[str] = set()
    skipped_no_prompt = 0
    for entry in entries:
        if not isinstance(entry, dict) or not _is_usable(entry):
            continue
        if cat_set is not None and entry.get("category") not in cat_set:
            continue
        if dom_set is not None and entry.get("domain") not in dom_set:
            continue
        qid = str(entry["primary_eval_id"])
        if qid in seen:
            continue
        prompt = _candidate_prompt(entry)
        if not prompt:
            skipped_no_prompt += 1
            continue
        domain = str(entry.get("domain") or "")
        queries.append(Query(query_id=qid, prompt=prompt, domain=domain))
        seen.add(qid)
        if limit is not None and len(queries) >= limit:
            break

    if skipped_no_prompt:
        print(f"  note: skipped {skipped_no_prompt} usable entries with no candidate prompt", file=sys.stderr)

    return QuerySet(
        source_manifest=str(manifest_path.resolve()),
        extracted_at=QuerySet.now_iso(),
        count=len(queries),
        queries=queries,
        category_filter=sorted(cat_set) if cat_set else None,
        domain_filter=sorted(dom_set) if dom_set else None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="Path to the eval manifest JSON.")
    ap.add_argument("--out", type=Path, default=Path("queries.json"), help="Output queries.json path.")
    ap.add_argument("--categories", default=None, help="Comma-separated category filter (e.g. query,extract).")
    ap.add_argument("--domains", default=None, help="Comma-separated domain filter.")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of queries (for smoke tests).")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        ap.error(f"manifest not found: {args.manifest}")

    categories = args.categories.split(",") if args.categories else None
    domains = args.domains.split(",") if args.domains else None

    qs = extract(args.manifest, categories=categories, domains=domains, limit=args.limit)
    qs.save(args.out)

    from collections import Counter

    by_domain = Counter(q.domain for q in qs.queries)
    print(f"Wrote {qs.count} queries to {args.out}")
    print(f"  source: {qs.source_manifest}")
    if qs.category_filter:
        print(f"  category filter: {qs.category_filter}")
    print(f"  by domain: {dict(by_domain)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
