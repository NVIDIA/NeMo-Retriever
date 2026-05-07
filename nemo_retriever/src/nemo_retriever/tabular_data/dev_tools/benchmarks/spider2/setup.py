#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time setup script: clone Spider2-lite and load its databases into DuckDB.

Each database in spider2-lite becomes a DuckDB schema, so you can query:

    conn.execute("SELECT * FROM Airlines.flights LIMIT 5")

Run once per machine (from the repo root):

    python3 nemo_retriever/tabular-dev-tools/benchmarks/spider2/setup.py

Optional flags:

    python3 nemo_retriever/tabular-dev-tools/benchmarks/spider2/setup.py \\
        --spider2-dir ~/my_spider2 --db ./my.duckdb --overwrite

After this completes, query via DuckDB:

    from duckdb import DuckDB  # run from tabular-dev-tools/
    conn = DuckDB("./spider2.duckdb")
    rows = conn.execute("SELECT * FROM Airlines.flights LIMIT 5")
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from loader import load_spider2_lite


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SPIDER2_DIR = Path.home() / "spider2"
DEFAULT_DB_PATH = Path("spider2.duckdb")
SPIDER2_REPO_URL = "https://github.com/xlang-ai/Spider2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone_spider2(target_dir: Path) -> None:
    if target_dir.exists():
        print(f"[skip] Spider2 already present at {target_dir}")
        return
    print(f"[git ] Cloning {SPIDER2_REPO_URL} → {target_dir} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", SPIDER2_REPO_URL, str(target_dir)],
        check=False,
    )
    if result.returncode != 0:
        print(
            f"\n[error] git clone failed (exit {result.returncode}).\n"
            "Make sure git is installed and you have network access.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print("[git ] Clone complete.")


def _load_data(spider2_lite_dir: Path, db_path: Path, overwrite: bool) -> dict:
    if not spider2_lite_dir.exists():
        print(
            f"\n[error] spider2-lite directory not found: {spider2_lite_dir}\n"
            "Expected layout:\n"
            "  <spider2_dir>/spider2-lite/resource/databases/sqlite/<DbName>/<table>.json\n"
            "Check --spider2-dir points at the repo root.",
            file=sys.stderr,
        )
        sys.exit(1)

    action = "Overwriting" if overwrite else "Loading (skipping existing schemas)"
    print(f"\n[ddb ] {action} data from {spider2_lite_dir}")
    print(f"[ddb ] Database → {db_path}\n")

    summary = load_spider2_lite(db_path, spider2_lite_dir, overwrite=overwrite)

    print(f"  Databases found : {summary['databases_found']}")
    print(f"  Loaded          : {summary['loaded']}")
    print(f"  Skipped         : {summary['skipped']}")
    print(f"  Failed          : {summary['failed']}")

    if summary["schemas"]:
        print("\nSchemas loaded into DuckDB:")
        for s in sorted(summary["schemas"]):
            print(f"  ✓ {s}")

    if summary["failures"]:
        print("\n[warn] Some databases could not be loaded:")
        for f in summary["failures"]:
            print(f"  ✗ {f['database']} → {f['error']}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone Spider2 and load spider2-lite databases into DuckDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spider2-dir",
        type=Path,
        default=DEFAULT_SPIDER2_DIR,
        help="Root of the Spider2 repository.",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB database file to create or update.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Drop and recreate schemas that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    spider2_dir: Path = args.spider2_dir.expanduser().resolve()
    spider2_lite_dir: Path = spider2_dir / "spider2-lite"
    db_path: Path = args.db_path.expanduser().resolve()

    print("=" * 60)
    print("  Spider2-lite × DuckDB  — one-time setup")
    print("=" * 60)
    print(f"  Spider2 repo     : {spider2_dir}")
    print(f"  spider2-lite dir : {spider2_lite_dir}")
    print(f"  DuckDB file      : {db_path}")
    print(f"  Overwrite        : {args.overwrite}")
    print("=" * 60 + "\n")

    if not spider2_dir.exists():
        _clone_spider2(spider2_dir)

    _load_data(spider2_lite_dir, db_path, overwrite=args.overwrite)
    print(f"Setup complete. Database written to: {db_path}")


if __name__ == "__main__":
    main()
