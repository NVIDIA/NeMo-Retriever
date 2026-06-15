#!/usr/bin/env python3
"""Phase 1 (functional): extract the domain-less `functional_corpus_variants` prompts.

These behavioral tests have no `domain` and reference their corpus by path in the
prompt (`test-data/...`). This writes an answer-free `functional_queries.json` with
`{query_id, prompt, functional_type, corpus_refs}` — the pass/fail criteria
(`expected_behavior`/`validation_signal`/`tests`/`contract`) stay in the manifest for the
evaluator. `functional_type` is task-routing metadata (the prompt already states the
task), used by the runner to mount corpora and by the evaluator to pick a grader; it is
not injected into the agent prompt.

By default excludes `stateful_orchestration` (multi-turn + injected interrupt — handled
by a separate driver). Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Paths the prompts reference, e.g. `test-data/financebench/pdfs/*_2022_10K.pdf`.
_PATH_RE = re.compile(r"test-data/[A-Za-z0-9_][\w\-./*]*")


def _corpus_refs(prompt: str) -> list[str]:
    """Distinct corpus DIRECTORIES referenced by the prompt (glob/file tails stripped)."""
    refs: list[str] = []
    for raw in _PATH_RE.findall(prompt or ""):
        p = raw.rstrip("/.,);`'\"")
        # if the tail is a glob or a file, mount its parent directory
        tail = p.rsplit("/", 1)[-1]
        if "*" in tail or re.search(r"\.[A-Za-z0-9]{2,4}$", tail):
            p = p.rsplit("/", 1)[0]
        if p and p not in refs:
            refs.append(p)
    # Fallback: prompts like i12 ("the full corpus at test-data") name the bare root with no
    # parseable subpath, so _PATH_RE finds nothing. Mount the whole test-data tree.
    if not refs and re.search(r"\btest-data\b", prompt or ""):
        refs.append("test-data")
    return refs


def extract(manifest_path: Path, *, include_stateful: bool = False, limit: int | None = None):
    raw = json.loads(manifest_path.read_text())
    entries = raw if isinstance(raw, list) else (raw.get("entries") or list(raw.values()))
    out = []
    excluded = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        if not e.get("primary_eval_id") or e.get("prompt_export_status") != "exported":
            continue
        if e.get("domain") or e.get("batch4_track") != "functional_corpus_variants":
            continue
        ft = e.get("functional_type")
        if ft == "stateful_orchestration" and not include_stateful:
            excluded += 1
            continue
        prompt = e.get("prompt") or ""
        if not prompt.strip():
            continue
        out.append(
            {
                "query_id": str(e["primary_eval_id"]),
                "prompt": prompt.strip(),
                "functional_type": ft,
                "corpus_refs": _corpus_refs(prompt),
            }
        )
        if limit and len(out) >= limit:
            break
    return out, excluded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("functional_queries.json"))
    ap.add_argument(
        "--include-stateful",
        action="store_true",
        help="Include stateful_orchestration tests (need the multi-turn driver).",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if not args.manifest.exists():
        ap.error(f"manifest not found: {args.manifest}")

    queries, excluded = extract(args.manifest, include_stateful=args.include_stateful, limit=args.limit)
    doc = {
        "source_manifest": str(args.manifest.resolve()),
        "track": "functional_corpus_variants",
        "count": len(queries),
        "excluded_stateful": excluded,
        "queries": queries,
    }
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    from collections import Counter

    print(f"Wrote {len(queries)} functional queries to {args.out}  (excluded stateful: {excluded})")
    print("  by functional_type:", dict(Counter(q["functional_type"] for q in queries)))
    corpora = Counter(r for q in queries for r in q["corpus_refs"])
    print("  corpus_refs (top):", dict(corpora.most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
