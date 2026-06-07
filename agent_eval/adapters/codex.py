# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex CLI adapter for agent_eval.

Single-turn, non-interactive: ``codex exec --json``. Codex streams JSONL events
on stdout (``thread.started`` carries the session/thread id, ``item.completed``
with ``agent_message`` the final text, ``turn.completed`` the token usage) and
assigns its own thread id (we don't pass one). The captured stdout *is* the
event log, so we persist it directly rather than hunting ~/.codex/sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .base import AgentAdapter
except ImportError:  # pragma: no cover
    from base import AgentAdapter


class CodexAdapter(AgentAdapter):
    name = "codex"

    def build_command(self, *, model, workdir, session_id, profile, budget_usd):
        # Codex has no project settings.json deny / slash mechanism; in the baseline
        # profile the PATH shim (env_for) blocks bare `retriever`. No skill is copied.
        return [
            "codex",
            "exec",
            "--json",
            "--model",
            model,
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            str(workdir),
            "--add-dir",
            str(workdir),
            "-",
        ]

    def parse_output(self, stdout):
        final_text, session_id, tokens, cost = "", None, {}, None
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get("type")
            if t == "thread.started":
                session_id = ev.get("thread_id")
            elif t == "item.completed":
                item = ev.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    final_text = item["text"]  # keep the last agent message
            elif t == "turn.completed":
                u = ev.get("usage") or {}
                tokens = {
                    "input": int(u.get("input_tokens", 0) or 0),
                    "output": int(u.get("output_tokens", 0) or 0),
                    "cache_read": int(u.get("cached_input_tokens", 0) or 0),
                    "cache_creation": 0,
                }
        return str(final_text), session_id, tokens, cost

    def locate_session_log(self, session_id, workdir):
        # Best-effort; the runner persists captured stdout as the event log instead.
        if session_id:
            base = Path.home() / ".codex" / "sessions"
            if base.exists():
                hits = list(base.glob(f"**/*{session_id}*.jsonl"))
                if hits:
                    return hits[0]
        return None

    def compact_trace(self, log_path):
        lines: list[str] = []
        try:
            events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            return f"(could not read codex log: {exc})\n"
        for ev in events:
            t = ev.get("type")
            item = ev.get("item") or {}
            it = item.get("type")
            if t == "item.completed" and it == "agent_message" and item.get("text"):
                lines.append(f"  assistant: {item['text'][:400]}")
            elif t == "item.completed" and it in ("command_execution", "function_call"):
                cmd = item.get("command") or item.get("arguments") or ""
                lines.append(f"  exec: {str(cmd)[:240]}")
            elif t == "turn.started":
                lines.append("\n[turn]")
        return "\n".join(lines) + "\n"
