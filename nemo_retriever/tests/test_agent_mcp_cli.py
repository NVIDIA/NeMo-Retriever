# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from nemo_retriever.adapters.cli.main import app


def test_agent_mcp_command_is_registered() -> None:
    result = CliRunner().invoke(app, ["agent-mcp", "--help"])

    assert result.exit_code == 0
    assert "Start the NeMo Retriever agent MCP server" in result.output


def test_agent_mcp_help_runs_via_package_module(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "nemo_retriever", "agent-mcp", "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Start the NeMo Retriever agent MCP server" in result.stdout


def test_agent_mcp_start_builds_app_and_runs_uvicorn(tmp_path: Path) -> None:
    application = object()
    data_root = tmp_path / "mcp"

    with (
        patch("nemo_retriever.agent_mcp.cli.build_asgi_app", return_value=application) as build_app,
        patch("nemo_retriever.agent_mcp.cli.uvicorn.run") as run,
    ):
        result = CliRunner().invoke(
            app,
            [
                "agent-mcp",
                "start",
                "--data-root",
                str(data_root),
                "--allowed-root",
                str(tmp_path),
                "--host",
                "127.0.0.1",
                "--port",
                "8099",
            ],
        )

    assert result.exit_code == 0
    build_app.assert_called_once_with(data_root=data_root, allowed_roots=[tmp_path])
    run.assert_called_once_with(application, host="127.0.0.1", port=8099, log_level="info")
