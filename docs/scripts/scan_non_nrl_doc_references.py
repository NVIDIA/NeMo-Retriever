#!/usr/bin/env python3
"""
Scan NRL doc sources for legacy naming and non-NRL references (manual review).

This does not fail the build; it lists candidates for editorial cleanup.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"

# Lines matching these patterns are reported (case-sensitive variants handled separately).
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("nv-ingest (substring)", re.compile(r"nv-ingest", re.IGNORECASE)),
    ("NV-Ingest", re.compile(r"NV-Ingest")),
    ("NVIDIA Ingest (rename note)", re.compile(r"NVIDIA Ingest")),
    ("github.com/.../nv-ingest", re.compile(r"github\.com/[^/\s]+/nv-ingest")),
    ("nvcr.io/.../nv-ingest", re.compile(r"nvcr\.io/[^\s)]+nv-ingest")),
    ("nv_ingest (Python package)", re.compile(r"\bnv_ingest\b")),
    ("NV_INGEST_", re.compile(r"NV_INGEST_")),
    ("releasenotes-nv-ingest filename", re.compile(r"releasenotes-nv-ingest")),
]


def main() -> int:
    if not DOCS_ROOT.is_dir():
        print(f"Missing docs tree: {DOCS_ROOT}", file=sys.stderr)
        return 1

    md_files = sorted(DOCS_ROOT.rglob("*.md"))
    print("Non-NRL / legacy reference scan (informational)")
    print(f"Root: {DOCS_ROOT}")
    print(f"Files scanned: {len(md_files)}")
    print()

    total = 0
    for path in md_files:
        rel = path.relative_to(DOCS_ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"{rel}: read error: {e}", file=sys.stderr)
            continue
        lines = text.splitlines()
        file_hits: list[str] = []
        for i, line in enumerate(lines, start=1):
            for label, pat in PATTERNS:
                if pat.search(line):
                    snippet = line.strip()
                    if len(snippet) > 160:
                        snippet = snippet[:157] + "..."
                    file_hits.append(f"    L{i} [{label}] {snippet}")
                    break
        if file_hits:
            total += len(file_hits)
            print(f"{rel}:")
            print("\n".join(file_hits))
            print()

    print(f"Total flagged lines (may overlap patterns): {total}")
    print()
    print(
        "Notes: Some hits are expected (historical rename notes, CLI compatibility, "
        "environment variables, or container image names). Use this list for editorial review only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
