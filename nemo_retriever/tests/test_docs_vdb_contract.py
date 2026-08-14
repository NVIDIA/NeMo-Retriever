# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documentation contracts for NVBug 6610357 (VDB page).

The VDB page must describe public CLI upload and the lancedb default.
Omitting ``VdbUploadParams.vdb_op`` selects ``lancedb``, and the public
``retriever ingest`` plan includes ``vdb_upload``.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import nemo_retriever.ingest.execution as ingest_execution
import nemo_retriever.ingest.plan as ingest_plan
from nemo_retriever.common.params import VdbUploadParams
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VDBS_PATH = _REPO_ROOT / "docs" / "docs" / "extraction" / "vdbs.md"
_STALE_VDB_CLAIMS = (
    "data upload is not supported through the",
    'defaults the string argument to `"milvus"`',
    "omitting `vdb_op` does not select LanceDB",
)
_VDBS_AVAILABLE = pytest.mark.skipif(not _VDBS_PATH.exists(), reason="Published VDB page is not in this checkout")


@_VDBS_AVAILABLE
def test_vdbs_page_describes_cli_upload_and_lancedb_default() -> None:
    text = _VDBS_PATH.read_text(encoding="utf-8")
    for claim in _STALE_VDB_CLAIMS:
        assert claim not in text
    assert "retriever ingest" in text
    assert "service-configured storage" in text
    assert 'defaults to `"lancedb"`' in text


@_VDBS_AVAILABLE
def test_vdbs_python_examples_are_valid_syntax() -> None:
    text = _VDBS_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks
    for code in blocks:
        ast.parse(code)


def test_vdb_upload_params_default_is_lancedb() -> None:
    assert VdbUploadParams().vdb_op == "lancedb"


def test_graph_ingestor_omitted_vdb_op_selects_lancedb() -> None:
    ingestor = GraphIngestor(run_mode="inprocess")
    ingestor.vdb_upload()
    assert ingestor._vdb_upload_params is not None
    assert ingestor._vdb_upload_params.vdb_op == "lancedb"


def test_public_ingest_cli_dry_run_plan_includes_lancedb_vdb_upload(monkeypatch, tmp_path: Path) -> None:
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
