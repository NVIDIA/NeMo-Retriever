# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_retriever.tools.skill_eval.runner import (
    _codex_exit_ok,
    _retriever_in_command,
    _retriever_segment_kind,
)


@pytest.mark.parametrize(
    "cmd, expected",
    [
        ("retriever query 'x'", "bare"),
        ("./retriever query", "rel"),
        ("./.bin/retriever query", "rel"),
        ("./retriever/bin/retriever query", "rel"),
        ("/raid/nemo_retriever/.venv/bin/retriever query", "abs"),
        ("TOKENIZERS=1 ./.bin/retriever query", "rel"),
        ("uv run retriever query", "uv"),
        ("python -m nemo_retriever query", "pythonm"),
        ("cat retriever.log", None),
        ("python read_lance.py", None),
        ("", None),
    ],
)
def test_segment_kind(cmd: str, expected: str | None) -> None:
    assert _retriever_segment_kind(cmd.strip()) == expected


@pytest.mark.parametrize(
    "cmd, c1_expected",
    [
        # In c1_base only the real installed CLI (absolute path / uv / python -m)
        # counts; the PATH deny shim and workdir-bundled binaries do not.
        ("retriever query", False),
        ("./retriever query", False),
        ("./.bin/retriever query", False),
        ("./retriever/bin/retriever query", False),
        ("/raid/nemo_retriever/.venv/bin/retriever query", True),
        ("uv run retriever query", True),
        ("python -m nemo_retriever query", True),
        ("./.bin/retriever query | tee /tmp/hits.json", False),
    ],
)
def test_retriever_in_command_condition_aware(cmd: str, c1_expected: bool) -> None:
    assert _retriever_in_command(cmd, "c1_base") is c1_expected
    # Every retriever invocation form counts when the CLI is provided (c2/c3).
    assert _retriever_in_command(cmd, "c2_retriever") is True
    assert _retriever_in_command(cmd, "c3_retriever_skill") is True


def test_retriever_in_command_ignores_non_invocations() -> None:
    for cond in ("c1_base", "c2_retriever"):
        assert _retriever_in_command("cat retriever.log", cond) is False
        assert _retriever_in_command("ls ./retriever", cond) is False
        assert _retriever_in_command("", cond) is False


@pytest.mark.parametrize(
    "output, expected",
    [
        ("Process exited with code 0\nOutput:\n...", True),
        ("Wall time: 0.1\nexited with code 0", True),
        ("Process exited with code 127", False),
        ("Process exited with code 1", False),
        ("still running, no exit line yet", False),
        ("", False),
    ],
)
def test_codex_exit_ok(output: str, expected: bool) -> None:
    assert _codex_exit_ok(output) is expected
