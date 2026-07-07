# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_retriever_skill_static_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    doctor = repo_root / "skills" / "nemo-retriever" / "scripts" / "doctor.py"
    contract = json.loads((repo_root / "skills" / "nemo-retriever" / "contract" / "cli-contract.json").read_text())
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", "")))

    result = subprocess.run(
        [sys.executable, str(doctor), "--static-only"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Retriever skill contract {contract['contract_version']}" in result.stdout
    assert "[FAIL]" not in result.stdout
