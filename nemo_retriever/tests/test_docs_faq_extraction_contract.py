# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documentation contracts for NVBug 6610405 (extraction FAQ).

The FAQ must describe the Python extraction-only result as a pandas.DataFrame
and must not recommend ``retriever ingest`` as an extraction-only surface.
The public CLI plan includes embedding and LanceDB upload.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

import nemo_retriever.ingest.execution as ingest_execution
import nemo_retriever.ingest.plan as ingest_plan
from nemo_retriever.ingestor import create_ingestor

RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAQ_PATH = _REPO_ROOT / "docs" / "docs" / "extraction" / "faq.md"
_STALE_FAQ_CLAIMS = (
    "CLI or Python APIs to perform extraction only",
    "list object with one entry",
)
_FAQ_AVAILABLE = pytest.mark.skipif(not _FAQ_PATH.exists(), reason="Published FAQ is not in this checkout")


@_FAQ_AVAILABLE
def test_faq_describes_python_dataframe_not_cli_extraction_only() -> None:
    text = _FAQ_PATH.read_text(encoding="utf-8")
    for claim in _STALE_FAQ_CLAIMS:
        assert claim not in text
    assert "pandas.DataFrame" in text
    assert "retriever ingest" in text
    assert "LanceDB" in text


@_FAQ_AVAILABLE
def test_faq_python_examples_are_valid_syntax() -> None:
    text = _FAQ_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks
    for code in blocks:
        ast.parse(code)


def test_extraction_only_python_graph_returns_dataframe(tmp_path: Path) -> None:
    document = tmp_path / "README.md"
    document.write_text("# Heading\n\nBody text\n", encoding="utf-8")

    result = create_ingestor(run_mode="inprocess").files([str(document)]).extract().ingest(show_progress=False)

    assert isinstance(result, pd.DataFrame)
    assert not isinstance(result, list)
    assert len(result) >= 1
    for column in ("text", "content", "path", "page_number", "metadata"):
        assert column in result.columns


def test_public_ingest_cli_dry_run_plan_includes_embed_and_vdb_upload(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "README.md"
    document.write_text("# Heading\n\nBody text\n", encoding="utf-8")

    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called for --dry-run")

    monkeypatch.setattr(ingest_execution, "create_ingestor", fail_create_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--dry-run"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["vdb_upload"] is not None
    assert payload["vdb_upload"]["vdb_op"] == "lancedb"
    vdb_kwargs = payload["vdb_upload"]["vdb_kwargs"]
    assert vdb_kwargs["uri"]
    assert vdb_kwargs["table_name"]

    plan = ingest_plan.resolve_ingest_plan(
        ingest_plan.IngestPlanRequest(
            source=ingest_plan.IngestSourceOptions(documents=[str(document)]),
        )
    )
    assert plan.sparse is False
    assert plan.vdb_params is not None
    assert plan.vdb_params.vdb_op == "lancedb"
