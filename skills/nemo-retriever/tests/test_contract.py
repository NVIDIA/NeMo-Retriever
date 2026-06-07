"""Contract test: the installed retriever engine must satisfy the skill contract.

Runs the live doctor probe (ingest + retrieve + schema check). Requires the retriever
venv with GPU access; skips cleanly if the CLI is absent.
"""

import shutil
import subprocess
import sys
import os

import pytest

DOCTOR = os.path.join(os.path.dirname(__file__), "..", "scripts", "doctor.py")


@pytest.mark.skipif(shutil.which("retriever") is None, reason="retriever CLI not installed")
def test_engine_satisfies_contract():
    proc = subprocess.run([sys.executable, DOCTOR], capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, f"doctor reported contract violations:\n{proc.stdout}\n{proc.stderr}"
