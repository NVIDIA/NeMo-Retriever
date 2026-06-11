"""Git provenance helper shared by the runners (record commit at eval time) and the report
generators (display it). Pure stdlib + the `git` CLI; degrades gracefully if git is absent."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit(repo_hint: str | Path | None = None) -> dict:
    """Return ``{commit, short, branch, dirty}`` for the repo containing this code (or
    ``repo_hint``). ``commit`` is the full HEAD sha; ``dirty`` is True if the working tree has
    uncommitted changes. All fields degrade to ``"unknown"``/False if git can't be queried."""
    repo = Path(repo_hint) if repo_hint else Path(__file__).resolve().parent

    def _run(args: list[str]) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    commit = _run(["rev-parse", "HEAD"]) or "unknown"
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    dirty = bool(_run(["status", "--porcelain"]))
    return {
        "commit": commit,
        "short": commit[:12] if commit != "unknown" else "unknown",
        "branch": branch,
        "dirty": dirty,
    }


def format_commit(info: dict | None) -> str:
    """One-line human form: ``<short> on <branch> (dirty)``."""
    if not info:
        return "unknown"
    d = " (dirty — uncommitted changes)" if info.get("dirty") else ""
    return f"`{info.get('short', 'unknown')}` on `{info.get('branch', '?')}`{d}"
