#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time setup script: download BIRD and load its SQLite databases into DuckDB.

Each database in BIRD becomes a DuckDB schema with full data, so you can query:

    conn.execute("SELECT * FROM california_schools.schools LIMIT 5")

Run once per machine (from the repo root):

    python3 nemo_retriever/tabular-dev-tools/benchmarks/bird/setup.py

Optional flags:

    python3 nemo_retriever/tabular-dev-tools/benchmarks/bird/setup.py \\
        --split dev \\
        --bird-dir ~/my_bird \\
        --db ./my.duckdb \\
        --overwrite

After this completes, query via DuckDB:

    import duckdb
    conn = duckdb.connect("./bird.duckdb")
    rows = conn.execute("SELECT * FROM california_schools.schools LIMIT 5").fetchall()
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path
from loader import load_bird


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BIRD_DIR = Path.home() / "bird"
DEFAULT_DB_PATH = Path("bird.duckdb")

DOWNLOAD_URLS = {
    "mini-dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip",
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)
    else:
        mb = downloaded / 1_048_576
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


def _download_bird(split: str, target_dir: Path) -> Path:
    """Download and extract the BIRD zip for *split* into *target_dir*.

    Returns the path to the extracted data root (the folder containing
    ``dev_databases/``).
    """
    url = DOWNLOAD_URLS[split]
    zip_path = target_dir / f"bird_{split}.zip"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[http] Downloading {url}")
    print(f"[http] → {zip_path}")
    urllib.request.urlretrieve(url, zip_path, reporthook=_show_progress)
    print()  # newline after progress

    print(f"[zip ] Extracting {zip_path} → {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    print("[zip ] Extraction complete.")

    zip_path.unlink()  # remove zip to save disk space

    # Find the folder that contains dev_databases/.
    for candidate in sorted(target_dir.rglob("dev_databases")):
        if candidate.is_dir():
            return candidate.parent

    print(
        "\n[error] Could not locate 'dev_databases/' after extraction.\n" f"Check the contents of {target_dir}.",
        file=sys.stderr,
    )
    sys.exit(1)


def _find_or_download(split: str, bird_dir: Path) -> Path:
    """Return the directory that contains ``dev_databases/``, downloading if needed."""
    # 1. Already fully extracted — dev_databases/ exists somewhere under bird_dir.
    if bird_dir.exists():
        for candidate in sorted(bird_dir.rglob("dev_databases")):
            if candidate.is_dir():
                data_root = candidate.parent
                print(f"[skip] BIRD data already present at {data_root}")
                return data_root

    # 2. Zip was downloaded but not yet extracted.
    zip_path = bird_dir / f"bird_{split}.zip"
    if zip_path.exists():
        print(f"[skip] Zip already downloaded at {zip_path}, skipping download.")
        bird_dir.mkdir(parents=True, exist_ok=True)
        print(f"[zip ] Extracting {zip_path} → {bird_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(bird_dir)
        print("[zip ] Extraction complete.")
        zip_path.unlink()
        for candidate in sorted(bird_dir.rglob("dev_databases")):
            if candidate.is_dir():
                return candidate.parent
        print("\n[error] Could not locate 'dev_databases/' after extraction.", file=sys.stderr)
        sys.exit(1)

    # 3. Nothing found — download from scratch.
    print(f"[info] BIRD data not found at {bird_dir}. Starting download...")
    return _download_bird(split, bird_dir)


def _load_data(data_root: Path, db_path: Path, overwrite: bool) -> dict:
    action = "Overwriting" if overwrite else "Loading (skipping existing schemas)"
    print(f"\n[ddb ] {action} data from {data_root}")
    print(f"[ddb ] Database → {db_path}\n")

    summary = load_bird(db_path, data_root, overwrite=overwrite)

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
        description="Download BIRD and load its SQLite databases into DuckDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=list(DOWNLOAD_URLS),
        default="mini-dev",
        help="Which BIRD split to download if not already present.",
    )
    parser.add_argument(
        "--bird-dir",
        type=Path,
        default=DEFAULT_BIRD_DIR,
        help="Directory where BIRD data is (or will be) stored.",
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

    bird_dir: Path = args.bird_dir.expanduser().resolve()
    db_path: Path = args.db_path.expanduser().resolve()

    print("=" * 60)
    print("  BIRD × DuckDB  — one-time setup")
    print("=" * 60)
    print(f"  Split            : {args.split}")
    print(f"  BIRD dir         : {bird_dir}")
    print(f"  DuckDB file      : {db_path}")
    print(f"  Overwrite        : {args.overwrite}")
    print("=" * 60 + "\n")

    data_root = _find_or_download(args.split, bird_dir)
    _load_data(data_root, db_path, overwrite=args.overwrite)
    print(f"\nSetup complete. Database written to: {db_path}")


if __name__ == "__main__":
    main()
