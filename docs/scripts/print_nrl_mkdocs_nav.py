#!/usr/bin/env python3
"""Print the navigation tree from mkdocs.nrl-github-pages.yml for pre-deploy review."""

from __future__ import annotations

import sys
from pathlib import Path

from mkdocs.utils import yaml_load

CONFIG = Path(__file__).resolve().parents[1] / "mkdocs.nrl-github-pages.yml"


def _walk(node, indent: int = 0) -> None:
    pad = "  " * indent
    if isinstance(node, str):
        print(f"{pad}- {node}")
        return
    if isinstance(node, list):
        for item in node:
            _walk(item, indent)
        return
    if isinstance(node, dict):
        for key, val in node.items():
            if isinstance(val, str):
                print(f"{pad}- {key}: {val}")
            else:
                print(f"{pad}- {key}")
                _walk(val, indent + 1)
        return
    print(f"{pad}- {node!r}")


def main() -> int:
    if not CONFIG.is_file():
        print(f"Missing config: {CONFIG}", file=sys.stderr)
        return 1
    data = yaml_load(CONFIG.read_text(encoding="utf-8"))
    nav = data.get("nav")
    print(f"Config: {CONFIG.name}")
    print("Navigation hierarchy (NRL GitHub Pages):")
    print()
    _walk(nav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
