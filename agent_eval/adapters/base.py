# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent adapter interface — agnostic over claude/codex.

Pure stdlib. An adapter knows how to: build the CLI command for a single-turn
query, run it as a subprocess, parse the final answer + usage, locate the raw
session log the CLI wrote, and render a compact human-readable trace from it.
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentRunResult:
    final_text: str
    session_id: str | None
    exit_status: int
    duration_ms: int
    tokens: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    stdout: str = ""
    stderr: str = ""
    raw_log_path: str | None = None
    timed_out: bool = False
    error: str = ""


class AgentAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def build_command(
        self, *, model: str, workdir: Path, session_id: str, profile: str, budget_usd: float
    ) -> list[str]: ...

    @abstractmethod
    def parse_output(self, stdout: str) -> tuple[str, str | None, dict[str, int], float | None]:
        """Return (final_text, session_id, tokens, cost_usd) from CLI stdout."""

    @abstractmethod
    def locate_session_log(self, session_id: str | None, workdir: Path) -> Path | None: ...

    @abstractmethod
    def compact_trace(self, log_path: Path) -> str: ...

    def run(
        self,
        prompt: str,
        *,
        model: str,
        workdir: Path,
        session_id: str,
        profile: str,
        budget_usd: float,
        timeout_s: int,
        env: dict[str, str],
    ) -> AgentRunResult:
        cmd = self.build_command(
            model=model, workdir=workdir, session_id=session_id, profile=profile, budget_usd=budget_usd
        )
        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=str(workdir),
                env=env,
                timeout=timeout_s,
                check=False,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            code = 124
            timed_out = True
        dur_ms = int((time.monotonic() - t0) * 1000)

        final_text, parsed_sid, tokens, cost = "", None, {}, None
        if not timed_out:
            try:
                final_text, parsed_sid, tokens, cost = self.parse_output(stdout)
            except Exception as exc:  # noqa: BLE001 - defensive parse
                stderr += f"\n[adapter parse error] {exc}"
        sid = parsed_sid or session_id
        log_path = self.locate_session_log(sid, workdir)
        return AgentRunResult(
            final_text=final_text,
            session_id=sid,
            exit_status=code,
            duration_ms=dur_ms,
            tokens=tokens,
            cost_usd=cost,
            stdout=stdout,
            stderr=stderr,
            raw_log_path=str(log_path) if log_path else None,
            timed_out=timed_out,
        )
