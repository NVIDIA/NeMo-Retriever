# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-generated tool-use summaries via the ``claude`` CLI.

Reads a compact trace of one agent session (setup turn + N query turns) and
asks a strong Anthropic model to narrate what the agent did: which tools it
called, in what order, what strategy it took, and where it improvised.

Shells out to ``claude --print`` so it reuses Claude Code's existing auth. Each
call runs in a neutral temp cwd with ``--setting-sources user`` so project-level
skills or settings do not leak into the summarization session.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)

_SUMMARIZER_PROMPT_TEMPLATE = """\
You are summarizing the tool-use trace of a coding agent that just ran an
information-retrieval benchmark over a corpus of PDFs.

Produce a concise markdown narrative with these sections:

**Overall strategy** - one or two sentences. What approach did the agent take?
Did it build an index, fall back to grep/pdftotext, use a skill?

**Tool-use breakdown** - bulleted list of tool names with counts and one or two
representative invocations each. Keep inputs short.

**Notable patterns** - retries, dead ends, fallback chains, suspicious behavior.
Skip this section if nothing stands out.

**Per-question variation** - only include if the agent's approach changed
between query turns. Otherwise omit.

Be terse. Aim for under 250 words total. Do not editorialize about whether the
strategy was good or bad; just describe what happened.

---

Condition: {condition}
Domain: {domain}

Trace:
{trace}
"""

_DEFAULT_MODEL = "claude-opus-4-7"


class TraceSummarizer:
    """Per-session tool-use narrator backed by the ``claude`` CLI."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        timeout: float = 120.0,
    ):
        self.model = model
        self._timeout = timeout

    @classmethod
    def from_kwargs(cls, **kwargs) -> "TraceSummarizer":
        return cls(**kwargs)

    def summarize(self, condition: str, domain: str, trace: str) -> str:
        """Return a markdown narrative of ``trace``. Empty string on failure."""
        if not trace.strip():
            return ""

        prompt = _SUMMARIZER_PROMPT_TEMPLATE.format(condition=condition, domain=domain, trace=trace)
        cmd = [
            "claude",
            "--print",
            "--model",
            self.model,
            "--setting-sources",
            "user",
        ]
        with tempfile.TemporaryDirectory(prefix="skill_eval_summarize_") as tmpdir:
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=tmpdir,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                logger.warning("trace summarizer timed out after %ss", self._timeout)
                return ""
            except FileNotFoundError:
                logger.warning("trace summarizer: `claude` CLI not on PATH")
                return ""

        if proc.returncode != 0:
            logger.warning(
                "trace summarizer exited %d: %s",
                proc.returncode,
                (proc.stderr or "")[:300],
            )
            return ""
        return (proc.stdout or "").strip()
