#!/usr/bin/env python3
"""Regenerate every tracked uv.lock and exit non-zero if any file changed (pre-commit / CI)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    root_r = _git("rev-parse", "--show-toplevel")
    if root_r.returncode != 0:
        print(root_r.stderr or root_r.stdout, file=sys.stderr, end="")
        return root_r.returncode or 1
    repo_root = Path(root_r.stdout.strip()).resolve()

    ls_r = _git("-C", str(repo_root), "ls-files", "*/uv.lock", "uv.lock")
    if ls_r.returncode != 0:
        print(ls_r.stderr or ls_r.stdout, file=sys.stderr, end="")
        return ls_r.returncode or 1

    lockfiles = [p for p in ls_r.stdout.splitlines() if p.strip()]
    if not lockfiles:
        return 0

    changed = 0
    for lockfile in lockfiles:
        subdir = str(Path(lockfile).parent)
        if subdir == ".":
            subdir = ""
        workdir = repo_root / subdir if subdir else repo_root

        print(f"uv lock: {subdir or '.'}")
        lock_r = subprocess.run(
            [sys.executable, "-m", "uv", "lock", "--quiet"],
            cwd=workdir,
            check=False,
        )
        if lock_r.returncode != 0:
            return lock_r.returncode

        diff_r = _git("-C", str(repo_root), "diff", "--quiet", lockfile)
        if diff_r.returncode == 1:
            print(
                f"  ERROR: {lockfile} is out of date -- stage the regenerated file and re-commit.",
                file=sys.stderr,
            )
            changed = 1
        elif diff_r.returncode != 0:
            print(diff_r.stderr or diff_r.stdout, file=sys.stderr, end="")
            return diff_r.returncode or 1

    return changed


if __name__ == "__main__":
    raise SystemExit(main())
