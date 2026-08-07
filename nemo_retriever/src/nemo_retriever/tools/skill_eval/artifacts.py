# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact helpers owned by the skill evaluation tool."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def now_timestr() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def last_commit() -> str:
    """Return the source revision when running from a Git checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    commit = result.stdout.strip().lower()
    return commit if result.returncode == 0 and commit else "unknown"


def create_session_dir(prefix: str, base_dir: str | None = None) -> Path:
    root = Path(base_dir).expanduser().resolve() if base_dir else DEFAULT_ARTIFACTS_ROOT
    session_dir = root / f"{prefix}_{now_timestr()}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def write_session_summary(
    session_dir: Path,
    run_results: list[dict[str, Any]],
    *,
    session_type: str,
    config_path: str,
    run_commit: str | None = None,
) -> Path:
    payload = {
        "session_type": session_type,
        "timestamp": now_timestr(),
        "run_commit": run_commit or last_commit(),
        "latest_commit": last_commit(),
        "config_path": config_path,
        "all_passed": all(bool(item.get("success")) for item in run_results),
        "results": run_results,
    }
    out_path = session_dir / "session_summary.json"
    temporary_path = out_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary_path.replace(out_path)
    return out_path
