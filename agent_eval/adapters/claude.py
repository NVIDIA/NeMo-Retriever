# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claude Code CLI adapter for agent_eval.

Single-turn, non-interactive: ``claude --print --output-format json``. Each query
runs in its own fresh session (``--session-id``); we don't resume. Mirrors the
flags the existing skill_eval harness used so behavior is comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .base import AgentAdapter
except ImportError:  # pragma: no cover
    from base import AgentAdapter


class ClaudeAdapter(AgentAdapter):
    name = "claude"

    def build_command(self, *, model, workdir, session_id, profile, budget_usd):
        cmd = [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--model",
            model,
            "--add-dir",
            str(workdir),
            "--permission-mode",
            "bypassPermissions",
            "--setting-sources",
            "project",
            "--session-id",
            session_id,
        ]
        if budget_usd:
            cmd += ["--max-budget-usd", str(budget_usd)]
        # baseline blocks the skill/retriever via the project settings.json deny
        # list; also forbid slash commands so /nemo-retriever can't fire.
        if profile == "baseline":
            cmd.append("--disable-slash-commands")
        return cmd

    def parse_output(self, stdout):
        """Parse the single JSON envelope claude --output-format json emits."""
        stdout = (stdout or "").strip()
        if not stdout:
            return "", None, {}, None
        env = json.loads(stdout)
        final = env.get("result") or env.get("text") or ""
        sid = env.get("session_id")
        usage = env.get("usage") or {}
        tokens = {
            "input": int(usage.get("input_tokens", 0) or 0),
            "output": int(usage.get("output_tokens", 0) or 0),
            "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
            "cache_creation": int(usage.get("cache_creation_input_tokens", 0) or 0),
        }
        cost = env.get("total_cost_usd")
        cost = float(cost) if isinstance(cost, (int, float)) else None
        return str(final), sid, tokens, cost

    def locate_session_log(self, session_id, workdir):
        if not session_id:
            return None
        proj = Path.home() / ".claude" / "projects"
        if not proj.exists():
            return None
        matches = list(proj.glob(f"**/{session_id}.jsonl"))
        return matches[0] if matches else None

    def compact_trace(self, log_path):
        lines: list[str] = []
        try:
            events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            return f"(could not read session log: {exc})\n"
        turn = 0
        for ev in events:
            if ev.get("type") == "assistant":
                msg = ev.get("message", {}) or {}
                for b in msg.get("content") or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text" and b.get("text", "").strip():
                        lines.append(f"  assistant: {b['text'].strip()[:400]}")
                    elif b.get("type") == "tool_use":
                        inp = b.get("input") or {}
                        summary = inp.get("command") or inp.get("file_path") or json.dumps(inp)[:200]
                        lines.append(f"  tool_use {b.get('name')}: {str(summary)[:240]}")
            elif ev.get("type") == "user":
                msg = ev.get("message", {}) or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    turn += 1
                    lines.append(f"\n[Turn {turn}] user: {content.strip()[:300]}")
        return "\n".join(lines) + "\n"
