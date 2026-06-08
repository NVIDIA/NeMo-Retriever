# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trial runner: build sandboxed workdirs, spawn an agent CLI, parse outputs."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from nemo_retriever.tools.skill_eval.dataset import DatasetEntry

logger = logging.getLogger(__name__)

BASE_CONDITION = "c1_base"
CONDITIONS = ("c1_base", "c2_retriever", "c3_retriever_skill")
SUPPORTED_AGENTS = ("claude", "codex")
DEFAULT_AGENT_MODELS = {
    "claude": "claude-opus-4-7",
    "codex": "gpt-5.5",
}


@functools.lru_cache(maxsize=8)
def _load_prompt_template(name: str) -> str:
    return Path(str(pkg_files("nemo_retriever.skill_eval").joinpath(f"prompts/{name}"))).read_text(encoding="utf-8")


@dataclass
class TrialResult:
    trial_id: str
    condition: str
    entry_id: int
    query_id: str
    status: str
    extraction_method: str
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    total_cost_usd: float
    model_id: str
    session_id: str
    agent: str = "claude"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0
    final_answer: str = ""
    ranked_retrieved: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retriever_first_use_turn: int | None = None
    retriever_attempted: bool = False
    retriever_succeeded: bool = False
    skill_fired: bool | None = None
    is_setup: bool = False
    domain: str = ""
    judge_score: int | None = None
    judge_reasoning: str = ""
    judge_error: str = ""
    tool_use_summary: str = ""
    cost_available: bool = True
    execution_mode: str = "linear_session"
    raw_log_path: str = ""
    compact_trace_path: str = ""


@dataclass
class ConditionRun:
    """Scratch paths and trial results for one agent/condition/domain run."""

    workdir: Path
    workdirs: list[Path]
    results: list[TrialResult]
    result_workdirs: dict[str, Path]
    execution_mode: str = "linear_session"


def _remap_pdf_paths(text: str, prefixes: tuple[str, ...]) -> str:
    """Rewrite caller-supplied path prefixes in *text* to ``./pdfs/``.

    Some agent-eval manifests' paraphrased prompts hard-code dataset-source
    paths in the user-facing text. Each trial workdir symlinks the domain's
    PDFs to ``./pdfs/``, so the agent only needs the basename.
    """
    for prefix in prefixes:
        text = text.replace(prefix, "./pdfs")
    return text


def _render_prompt(entry: DatasetEntry, condition: str, testdata_prefixes: tuple[str, ...] = ()) -> str:
    tpl_name = "trial_user_slash.j2" if condition == "c3_retriever_skill" else "trial_user_nl.j2"
    text = _load_prompt_template(tpl_name)
    return text.replace(
        "{{ paraphrased_prompt }}", _remap_pdf_paths(entry.paraphrased_prompt, testdata_prefixes)
    ).replace("{{ original_query }}", entry.original_query)


def _render_setup_prompt(condition: str, domain_label: str = "PDFs") -> str:
    tpl_name = "setup_slash.j2" if condition == "c3_retriever_skill" else "setup_nl.j2"
    text = _load_prompt_template(tpl_name)
    return text.replace("{{ domain_label }}", domain_label)


_ISOLATED_QUERY_NOTE = (
    "The setup turn has already completed in this isolated copy of the setup workdir. "
    "Use the artifacts already present here. If helpful, read ./setup_context.md before answering."
)


def _render_isolated_query_prompt(
    entry: DatasetEntry,
    condition: str,
    testdata_prefixes: tuple[str, ...] = (),
) -> str:
    """Render a query prompt for a fresh session branched from setup artifacts."""
    base = _render_prompt(entry, condition, testdata_prefixes)
    if base.startswith("/") and "\n" in base:
        first, rest = base.split("\n", 1)
        return f"{first}\n\n{_ISOLATED_QUERY_NOTE}\n\n{rest}"
    return f"{_ISOLATED_QUERY_NOTE}\n\n{base}"


def _write_setup_context(workdir: Path, *, agent: str, condition: str) -> Path:
    """Write a durable handoff for isolated query sessions."""
    artifact_names: list[str] = []
    ignored = {".claude", ".codex", ".bin", ".hf_empty", "pdfs", "setup_context.md"}
    for child in sorted(workdir.iterdir(), key=lambda p: p.name):
        name = child.name
        if name in ignored or name == "output.json" or name.startswith("output_e"):
            continue
        suffix = "/" if child.is_dir() else ""
        artifact_names.append(f"- `{name}{suffix}`")

    lines = [
        "# skill_eval setup context",
        "",
        "This file was generated by the skill_eval harness after the setup turn completed.",
        "Future query sessions start from an isolated copy of this workdir.",
        "",
        f"- Agent: `{agent}`",
        f"- Condition: `{condition}`",
        "- PDFs are available under `./pdfs/`.",
    ]
    if condition == BASE_CONDITION:
        lines.append("- The `retriever` CLI is intentionally blocked for this baseline condition.")
    else:
        lines.append("- The `retriever` CLI and nemo-retriever skill are available for this condition.")
    lines += ["", "## Top-level setup artifacts"]
    lines.extend(artifact_names or ["- No additional top-level artifacts were detected."])
    out = workdir / "setup_context.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _ignore_runtime_outputs(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "output.json" or name.startswith("output_e")}


def _copy_setup_workdir_for_query(setup_workdir: Path, entry: DatasetEntry) -> Path:
    """Clone setup artifacts into a per-query scratch workdir."""
    dest = setup_workdir.parent / f"{setup_workdir.name}_query_e{entry.entry_id}_{uuid.uuid4().hex[:8]}"
    shutil.copytree(setup_workdir, dest, symlinks=True, ignore=_ignore_runtime_outputs)
    return dest


def _build_pdf_symlinks(pdf_source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(pdf_source.glob("*.pdf")):
        target = dest / pdf.name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(pdf.resolve())


def _write_shim(shim_dir: Path, name: str) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / name
    shim.write_text(
        "#!/usr/bin/env bash\n" f"echo 'skill_eval shim: {name} not available in this trial' >&2\n" "exit 127\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _copy_skill(skill_source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "SKILL.md").exists():
        return
    shutil.copy2(skill_source / "SKILL.md", dest / "SKILL.md")
    ref_src = skill_source / "references"
    if ref_src.is_dir():
        shutil.copytree(ref_src, dest / "references", dirs_exist_ok=True)


# Bash patterns that route the agent into the nemo_retriever library, regardless
# of whether it tries the CLI, a Python module invocation, or a direct full-path
# call into the project's venv. Used in c1's project-level settings.json. The
# patterns are intentionally written as substring globs (Claude Code semantics)
# so they catch the command line as the agent assembled it.
_C1_BASH_DENY_PATTERNS: tuple[str, ...] = (
    "Bash(retriever:*)",
    "Bash(*nemo_retriever*)",
    "Bash(*nemo-retriever*)",
    "Bash(python*-m*nemo_retriever*)",
    "Bash(uv*run*retriever*)",
    "Bash(*/bin/retriever*)",
    # Hide the HuggingFace model cache so the agent can't enumerate the
    # NVIDIA stack that c2/c3 populated. HF env vars (HF_HOME etc.) are
    # redirected in _env_for; these deny patterns close the `ls`/`find`
    # side channel that bypasses env-var resolution.
    "Bash(*huggingface*)",
    "Bash(*.cache/huggingface*)",
)


def _c1_settings_json() -> str:
    """Project-level settings for the c1_base Claude trial.

    ``--permission-mode bypassPermissions`` auto-approves tool calls that aren't
    explicitly denied; these deny patterns catch every reasonable path into the
    nemo_retriever library so Claude has to fall back on CPU-only primitives.
    """
    return json.dumps({"permissions": {"deny": list(_C1_BASH_DENY_PATTERNS)}}, indent=2) + "\n"


def _build_condition_workdir(
    agent: str,
    condition: str,
    root: Path,
    pdf_source: Path,
    skill_source: Path,
    domain: str = "",
) -> Path:
    """Build one workdir per agent/condition/domain session.

    Workdir contents:
      - pdfs/ symlink farm into the source PDF folder
      - .claude/ sandbox (settings + per-condition skill copy for Claude)
      - .codex/ skill copy for Codex skill-aware installations
      - .bin/retriever shim (c1 only) so the retriever CLI is unavailable on PATH
    """
    domain_seg = f"_{domain}" if domain else ""
    workdir = root / f"{agent}_{condition}{domain_seg}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    _build_pdf_symlinks(pdf_source, workdir / "pdfs")

    if agent == "claude":
        (workdir / ".claude").mkdir(parents=True, exist_ok=True)
        settings_text = _c1_settings_json() if condition == "c1_base" else "{}\n"
        (workdir / ".claude" / "settings.json").write_text(settings_text, encoding="utf-8")

    if condition in ("c2_retriever", "c3_retriever_skill"):
        if agent == "claude":
            _copy_skill(skill_source, workdir / ".claude" / "skills" / "nemo-retriever")
        elif agent == "codex":
            _copy_skill(skill_source, workdir / ".codex" / "skills" / "nemo-retriever")

    if condition == "c1_base":
        _write_shim(workdir / ".bin", "retriever")
        # Empty HuggingFace cache redirect; env vars are wired up in _env_for.
        (workdir / ".hf_empty").mkdir(parents=True, exist_ok=True)
    return workdir


def cleanup_condition_workdir(workdir: Path) -> None:
    """Remove a condition's scratch workdir after results have been persisted."""
    if not workdir.exists():
        return
    shutil.rmtree(workdir, ignore_errors=True)
    logger.info("cleaned up workdir %s", workdir)


def _env_for(condition: str, workdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    if condition == "c1_base":
        env["PATH"] = f"{workdir / '.bin'}{os.pathsep}{env.get('PATH', '')}"
        # Point HuggingFace cache env vars at an empty workdir-local dir so
        # any HF Python tooling the agent invokes sees no cached models.
        hf_empty = str(workdir / ".hf_empty")
        env["HF_HOME"] = hf_empty
        env["HF_HUB_CACHE"] = hf_empty
        env["TRANSFORMERS_CACHE"] = hf_empty
    return env


def _build_claude_command(
    condition: str,
    model: str,
    budget_usd: float,
    session_uuid: str,
    workdir: Path,
    *,
    resume: bool = False,
) -> list[str]:
    """Build the ``claude --print`` command.

    First turn uses ``--session-id``; subsequent turns use ``--resume``. We
    deliberately keep session persistence enabled because this benchmark is
    multi-turn.
    """
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
        "--max-budget-usd",
        str(budget_usd),
        "--setting-sources",
        "project",
    ]
    # c2/c3 run fully ungated. c1 omits the dangerous skip flag so the
    # project-level deny rules are consulted.
    if condition != "c1_base":
        cmd.append("--allow-dangerously-skip-permissions")
    if resume:
        cmd.extend(["--resume", session_uuid])
    else:
        cmd.extend(["--session-id", session_uuid])
    # Only c1 disables skills entirely. c2 has the skill loaded but uses an NL
    # prompt; c3 explicitly invokes via slash.
    if condition == "c1_base":
        cmd.append("--disable-slash-commands")
    return cmd


def _build_codex_command(
    model: str,
    session_uuid: str,
    workdir: Path,
    *,
    resume: bool = False,
) -> list[str]:
    """Build a non-interactive Codex command.

    Codex assigns the first session id itself; subsequent turns resume the id
    parsed from the setup turn's JSONL events.
    """
    common = [
        "--json",
        "--model",
        model,
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    if resume:
        return ["codex", "exec", "resume", *common, session_uuid, "-"]
    return [
        "codex",
        "exec",
        *common,
        "--cd",
        str(workdir),
        "--add-dir",
        str(workdir),
        "-",
    ]


def _build_command(
    *,
    agent: str,
    condition: str,
    model: str,
    budget_usd: float,
    session_uuid: str,
    workdir: Path,
    resume: bool = False,
) -> list[str]:
    if agent == "claude":
        return _build_claude_command(condition, model, budget_usd, session_uuid, workdir, resume=resume)
    if agent == "codex":
        return _build_codex_command(model, session_uuid, workdir, resume=resume)
    raise ValueError(f"unsupported agent: {agent}")


def _parse_envelope(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        logger.warning("could not parse claude envelope (first 200 chars): %r", raw[:200])
        return {}


def _parse_jsonl_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def _codex_session_id(events: list[dict[str, Any]], fallback: str) -> str:
    for ev in events:
        if ev.get("type") != "session_meta":
            continue
        payload = ev.get("payload") or {}
        if isinstance(payload, dict) and payload.get("id"):
            return str(payload["id"])
    return fallback


def _codex_has_error(events: list[dict[str, Any]]) -> bool:
    for ev in events:
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if payload.get("type") in {"error", "task_failed", "turn_aborted"}:
            return True
    return False


def _populate_claude_tokens(result: TrialResult, envelope: dict[str, Any]) -> None:
    usage = envelope.get("usage") or {}
    result.input_tokens = int(usage.get("input_tokens") or 0)
    result.output_tokens = int(usage.get("output_tokens") or 0)
    result.cache_read_input_tokens = int(usage.get("cache_read_input_tokens") or 0)
    result.cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    cache_detail = usage.get("cache_creation") or {}
    result.ephemeral_5m_input_tokens = int(cache_detail.get("ephemeral_5m_input_tokens") or 0)
    result.ephemeral_1h_input_tokens = int(cache_detail.get("ephemeral_1h_input_tokens") or 0)


_CODEX_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
)


def _extract_codex_total_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Return the most recent cumulative ``total_token_usage`` from codex events.

    Each ``token_count`` event carries running session-wide counters; we want the
    last one so deltas between two snapshots equal one turn's true work.
    """
    for ev in reversed(events):
        if ev.get("type") != "event_msg":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info") or {}
        if not isinstance(info, dict):
            continue
        usage = info.get("total_token_usage") or {}
        if not isinstance(usage, dict):
            continue
        return {k: int(usage.get(k) or 0) for k in _CODEX_USAGE_FIELDS}
    return {k: 0 for k in _CODEX_USAGE_FIELDS}


def _populate_codex_tokens(
    result: TrialResult,
    current_totals: dict[str, int],
    prior_totals: dict[str, int],
) -> None:
    """Set per-turn token fields as the delta of cumulative ``total_token_usage``.

    Codex's resumed-session log is append-only across all turns, and each
    ``token_count`` event reports cumulative counters, so per-turn cost is the
    difference between snapshots taken before and after the subprocess call.
    ``output_tokens`` here folds in ``reasoning_output_tokens`` so the column
    reflects everything the model emitted, matching Claude's accounting.
    """

    def d(key: str) -> int:
        return max(0, current_totals.get(key, 0) - prior_totals.get(key, 0))

    result.input_tokens = d("input_tokens")
    result.output_tokens = d("output_tokens") + d("reasoning_output_tokens")
    result.cache_read_input_tokens = d("cached_input_tokens")
    result.cache_creation_input_tokens = 0


def _parse_output_json(workdir: Path) -> tuple[str, list[dict[str, Any]], str, list[str]]:
    out_path = workdir / "output.json"
    errors: list[str] = []
    if not out_path.exists():
        return "", [], "missing", ["output.json not written"]
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return "", [], "invalid_json", [f"invalid JSON: {e}"]
    if not isinstance(payload, dict):
        return "", [], "invalid_json", [f"top-level must be an object, got {type(payload).__name__}"]
    for required in ("final_answer", "ranked_retrieved"):
        if required not in payload:
            errors.append(f"missing required key '{required}'")
    ranked = payload.get("ranked_retrieved") or []
    if not isinstance(ranked, list):
        errors.append(f"ranked_retrieved must be a list, got {type(ranked).__name__}")
        ranked = []
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(ranked, start=1):
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        page = item.get("page_number")
        if doc_id is None or page is None:
            continue
        cleaned.append({"doc_id": str(doc_id), "page_number": int(page), "rank": int(item.get("rank") or i)})
    return str(payload.get("final_answer") or ""), cleaned, ("ok" if not errors else "schema_warning"), errors


def _extract_model_id(envelope: dict[str, Any], fallback: str) -> str:
    model_usage = envelope.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        return next(iter(model_usage.keys()))
    return str(envelope.get("model") or fallback)


def _extract_claude_error_detail(envelope: dict[str, Any]) -> str:
    for key in ("error", "message", "result"):
        value = envelope.get(key)
        if value:
            return str(value)

    content = envelope.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        if parts:
            return " ".join(parts)
    return ""


_PIPELINE_SEP = re.compile(r"(?:;|&&|\|\||\||\n|\$\(|`)")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPER_CMDS = {"sudo", "time", "nice", "nohup", "exec", "env", "command", "builtin"}


# In c1_base the retriever CLI is intentionally denied: a bare ``retriever``
# resolves to the PATH deny shim, and any workdir-relative path (e.g.
# ``./.bin/retriever`` or a bundled ``./retriever/bin/retriever`` venv leaked by
# skill setup) is a stub/leak rather than the real CLI. Only an absolute
# installed-CLI path -- or ``uv run`` / ``python -m`` which resolve the
# installed package -- counts as genuine retriever usage there. In c2/c3 the
# CLI is provided, so every invocation form counts.
_REAL_RETRIEVER_KINDS = frozenset({"abs", "uv", "pythonm"})


def _retriever_segment_kind(seg: str) -> str | None:
    """Classify the executable head of one pipeline segment.

    Returns ``"bare"`` (``retriever`` resolved via PATH), ``"rel"`` (a
    workdir-relative path such as ``./retriever`` or ``./.bin/retriever``),
    ``"abs"`` (an absolute path such as ``/raid/.../bin/retriever``), ``"uv"``
    (``uv run retriever``), ``"pythonm"`` (``python -m nemo_retriever``), or
    ``None`` when the segment does not invoke the retriever CLI. Deliberately
    ignores cases where ``retriever`` appears only as a path argument or prose.
    """
    while seg:
        first = seg.split(None, 1)
        if not first:
            break
        head = first[0]
        rest = first[1] if len(first) > 1 else ""
        if _ENV_ASSIGN.match(head) or head in _WRAPPER_CMDS:
            seg = rest
            continue
        break
    if not seg:
        return None

    tokens = seg.split()
    head = tokens[0]
    if head == "retriever":
        return "bare"
    if head == "./retriever":
        return "rel"
    if head.endswith("/retriever") and "/" in head[: -len("/retriever") + 1]:
        return "abs" if head.startswith("/") else "rel"
    if len(tokens) >= 3 and tokens[0] == "uv" and tokens[1] == "run" and tokens[2] == "retriever":
        return "uv"
    if (
        len(tokens) >= 3
        and tokens[0].startswith("python")
        and tokens[1] == "-m"
        and tokens[2].startswith("nemo_retriever")
    ):
        return "pythonm"
    return None


def _retriever_in_command(cmd: str, condition: str = "") -> bool:
    """Return whether this shell command invokes the retriever CLI as a command.

    Condition-aware: under ``c1_base`` only the real installed CLI counts (see
    :data:`_REAL_RETRIEVER_KINDS`); under every other condition any retriever
    invocation form counts.
    """
    if not cmd:
        return False

    for segment in _PIPELINE_SEP.split(cmd):
        kind = _retriever_segment_kind(segment.strip())
        if kind is None:
            continue
        if condition == BASE_CONDITION and kind not in _REAL_RETRIEVER_KINDS:
            continue
        return True
    return False


_CODEX_EXIT_RE = re.compile(r"exited with code (\d+)")


def _codex_exit_ok(output: str) -> bool:
    """Return whether a codex ``exec_command`` output reports a clean exit.

    Codex wraps command output with a ``Process exited with code N`` line. A
    missing exit line (interrupted, killed, or still-running command) is treated
    as a failure so blocked/hung retriever calls do not count as succeeded.
    """
    if not output:
        return False
    match = _CODEX_EXIT_RE.search(output)
    if match is None:
        return False
    return match.group(1) == "0"


def _claude_session_log_path(workdir: Path, session_uuid: str) -> Path:
    """Return Claude Code's per-session JSONL transcript path."""
    slug = str(workdir).replace("/", "-").replace("_", "-")
    if not slug.startswith("-"):
        slug = "-" + slug
    return Path.home() / ".claude" / "projects" / slug / f"{session_uuid}.jsonl"


def _codex_session_log_path(session_uuid: str) -> Path | None:
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return None
    matches = sorted(
        sessions_root.glob(f"**/*{session_uuid}.jsonl"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return matches[0] if matches else None


def _codex_session_meta_from_log(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    ev = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "session_meta":
                    continue
                payload = ev.get("payload") or {}
                return payload if isinstance(payload, dict) else {}
    except OSError:
        return {}
    return {}


def _codex_session_log_for_workdir(workdir: Path) -> Path | None:
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return None
    workdir_str = str(workdir)
    matches = sorted(
        sessions_root.glob("**/rollout-*.jsonl"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for path in matches:
        meta = _codex_session_meta_from_log(path)
        if str(meta.get("cwd") or "") == workdir_str:
            return path
    return None


def _read_jsonl_events(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        return _parse_jsonl_events(path.read_text(encoding="utf-8"))
    except OSError:
        return []


_TRACE_TOOL_INPUT_CAP = 200
_TRACE_FINAL_TEXT_CAP = 400


def _truncate(s: str, cap: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= cap else s[: cap - 1] + "..."


def _format_tool_input(name: str, inp: dict[str, Any]) -> str:
    """Render a Claude tool_use input dict to a single short line."""
    if name == "Bash":
        cmd = str(inp.get("command", ""))
        return f"Bash: {_truncate(cmd, _TRACE_TOOL_INPUT_CAP)}"
    if name == "Read":
        path = str(inp.get("file_path", ""))
        offset = inp.get("offset")
        limit = inp.get("limit")
        tail = f" offset={offset} limit={limit}" if offset is not None or limit is not None else ""
        return f"Read: {path}{tail}"
    if name == "Grep":
        pat = str(inp.get("pattern", ""))
        path = str(inp.get("path", ""))
        return f"Grep: pattern={_truncate(pat, 80)} path={path}"
    if name == "Glob":
        return f"Glob: {inp.get('pattern', '')}"
    if name in ("Edit", "Write"):
        return f"{name}: {inp.get('file_path', '')}"
    parts = [f"{k}={_truncate(str(v), 80)}" for k, v in inp.items()]
    return f"{name}: " + " ".join(parts) if parts else name


def _extract_claude_compact_trace(
    workdir: Path,
    session_uuid: str,
    *,
    first_turn_label: str | None = None,
) -> str | None:
    """Walk a Claude Code JSONL transcript and emit a turn-organized trace."""
    log_path = _claude_session_log_path(workdir, session_uuid)
    if not log_path.exists():
        return None

    turn_idx = 0
    lines_out: list[str] = []
    current_assistant_text: list[str] = []
    try:
        with log_path.open(encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    ev = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                msg = ev.get("message") or {}
                role = msg.get("role") or ev.get("type")
                content = msg.get("content")

                if role == "user":
                    if current_assistant_text:
                        joined = " ".join(current_assistant_text).strip()
                        if joined:
                            lines_out.append(f"  assistant: {_truncate(joined, _TRACE_FINAL_TEXT_CAP)}")
                        current_assistant_text = []
                    turn_idx += 1
                    user_text = ""
                    if isinstance(content, str):
                        user_text = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_text = str(item.get("text", ""))
                                break
                    label = (
                        first_turn_label
                        if turn_idx == 1 and first_turn_label
                        else ("setup" if turn_idx == 1 else f"query {turn_idx - 1}")
                    )
                    lines_out.append("")
                    lines_out.append(f"[Turn {turn_idx} - {label}]")
                    if user_text:
                        lines_out.append(f"  user: {_truncate(user_text, _TRACE_FINAL_TEXT_CAP)}")
                elif role == "assistant" and isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        itype = item.get("type")
                        if itype == "tool_use":
                            name = str(item.get("name", "?"))
                            inp = item.get("input") or {}
                            if isinstance(inp, dict):
                                lines_out.append(f"  tool_use {_format_tool_input(name, inp)}")
                            else:
                                lines_out.append(f"  tool_use {name}")
                        elif itype == "text":
                            text = str(item.get("text", "")).strip()
                            if text:
                                current_assistant_text.append(text)
    except OSError:
        return None

    if current_assistant_text:
        joined = " ".join(current_assistant_text).strip()
        if joined:
            lines_out.append(f"  assistant: {_truncate(joined, _TRACE_FINAL_TEXT_CAP)}")

    trace = "\n".join(lines_out).strip()
    return trace or None


def _string_from_content_items(content: Any, *, input_text: bool = True) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    wanted = "input_text" if input_text else "output_text"
    fallback = "text"
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {wanted, fallback}:
            out.append(str(item.get("text") or ""))
    return " ".join(x for x in out if x).strip()


def _codex_tool_arguments(payload: dict[str, Any]) -> Any:
    args = payload.get("arguments") or ""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def _codex_tool_command(payload: dict[str, Any]) -> str:
    args = _codex_tool_arguments(payload)
    if isinstance(args, dict):
        for key in ("cmd", "command"):
            value = args.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(args, sort_keys=False)
    return str(args)


def _format_codex_tool_input(payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "?")
    args = _codex_tool_arguments(payload)
    if not isinstance(args, str):
        args = json.dumps(args, sort_keys=False)
    return f"{name}: {_truncate(args, _TRACE_TOOL_INPUT_CAP)}"


def _extract_codex_compact_trace(
    session_uuid: str,
    *,
    first_turn_label: str | None = None,
) -> str | None:
    log_path = _codex_session_log_path(session_uuid)
    events = _read_jsonl_events(log_path)
    if not events:
        return None

    turn_idx = 0
    lines_out: list[str] = []
    for ev in events:
        etype = ev.get("type")
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        if etype == "event_msg" and payload.get("type") == "user_message":
            turn_idx += 1
            label = (
                first_turn_label
                if turn_idx == 1 and first_turn_label
                else ("setup" if turn_idx == 1 else f"query {turn_idx - 1}")
            )
            lines_out.append("")
            lines_out.append(f"[Turn {turn_idx} - {label}]")
            text = str(payload.get("message") or "")
            if text:
                lines_out.append(f"  user: {_truncate(text, _TRACE_FINAL_TEXT_CAP)}")
        elif etype == "event_msg" and payload.get("type") == "agent_message":
            text = str(payload.get("message") or "")
            if text:
                lines_out.append(f"  assistant: {_truncate(text, _TRACE_FINAL_TEXT_CAP)}")
        elif etype == "response_item":
            ptype = payload.get("type")
            if ptype == "function_call":
                lines_out.append(f"  tool_use {_format_codex_tool_input(payload)}")
            elif ptype == "message" and payload.get("role") == "assistant":
                text = _string_from_content_items(payload.get("content"), input_text=False)
                if text:
                    lines_out.append(f"  assistant: {_truncate(text, _TRACE_FINAL_TEXT_CAP)}")

    trace = "\n".join(lines_out).strip()
    return trace or None


def extract_compact_trace(
    agent: str,
    workdir: Path,
    session_uuid: str,
    *,
    first_turn_label: str | None = None,
) -> str | None:
    if agent == "claude":
        return _extract_claude_compact_trace(workdir, session_uuid, first_turn_label=first_turn_label)
    if agent == "codex":
        return _extract_codex_compact_trace(session_uuid, first_turn_label=first_turn_label)
    return None


def _scan_claude_transcript_for_signals(
    envelope: dict[str, Any],
    workdir: Path | None,
    session_uuid: str | None,
    condition: str,
) -> tuple[int | None, bool, bool]:
    """Return ``(first_use_turn, attempted, succeeded)`` for a Claude session."""
    if workdir is not None and session_uuid:
        log_path = _claude_session_log_path(workdir, session_uuid)
        if log_path.exists():
            try:
                matched_tool_ids: set[str] = set()
                error_by_id: dict[str, bool] = {}
                with log_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = ev.get("message") or {}
                        content = msg.get("content")
                        if not isinstance(content, list):
                            continue
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            itype = item.get("type")
                            if itype == "tool_use" and item.get("name") == "Bash":
                                cmd = (item.get("input") or {}).get("command") or ""
                                tool_id = item.get("id")
                                if isinstance(tool_id, str) and _retriever_in_command(cmd, condition):
                                    matched_tool_ids.add(tool_id)
                            elif itype == "tool_result":
                                tool_id = item.get("tool_use_id")
                                if isinstance(tool_id, str):
                                    error_by_id[tool_id] = bool(item.get("is_error"))
                attempted = bool(matched_tool_ids)
                # Succeeded only when a matched call has a recorded non-error
                # result; a missing result (interrupted/killed) counts as failed.
                succeeded = any(not error_by_id.get(tid, True) for tid in matched_tool_ids)
                return (1 if attempted else None), attempted, succeeded
            except OSError:
                pass

    text = str(envelope.get("result") or "")
    used = "retriever " in text or "\nretriever\n" in text
    return (1 if used else None), used, False


def _scan_codex_transcript_for_signals(
    session_uuid: str,
    fallback_events: list[dict[str, Any]],
    condition: str,
) -> tuple[int | None, bool, bool]:
    """Return ``(first_use_turn, attempted, succeeded)`` for a codex session."""
    log_events = _read_jsonl_events(_codex_session_log_path(session_uuid))
    events = log_events or fallback_events

    # Map call_id -> output text so each matched call can be checked for a clean
    # exit; outputs may appear before or after their function_call in the log.
    outputs_by_call: dict[str, str] = {}
    has_function_calls = False
    for ev in events:
        if ev.get("type") != "response_item":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")
        if ptype == "function_call":
            has_function_calls = True
        elif ptype == "function_call_output":
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                outputs_by_call[call_id] = str(payload.get("output") or "")

    attempted = False
    succeeded = False
    for ev in events:
        if ev.get("type") != "response_item":
            continue
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("type") != "function_call":
            continue
        if not _retriever_in_command(_codex_tool_command(payload), condition):
            continue
        attempted = True
        call_id = payload.get("call_id")
        if isinstance(call_id, str) and _codex_exit_ok(outputs_by_call.get(call_id, "")):
            succeeded = True

    # Trust the structured verdict whenever tool calls were recorded. Only when
    # there are no tool calls at all do we fall back to a weak prose heuristic
    # (no reliable success signal there, so succeeded stays False).
    if attempted or has_function_calls:
        return (1 if attempted else None), attempted, succeeded

    text_parts: list[str] = []
    for ev in events:
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if ev.get("type") == "event_msg" and payload.get("type") == "agent_message":
            text_parts.append(str(payload.get("message") or ""))
        elif ev.get("type") == "response_item" and payload.get("type") == "message":
            text_parts.append(_string_from_content_items(payload.get("content"), input_text=False))
    text = "\n".join(text_parts)
    used = "retriever " in text or "\nretriever\n" in text
    return (1 if used else None), used, False


def _scan_transcript_for_signals(
    *,
    agent: str,
    condition: str,
    envelope: dict[str, Any],
    codex_events: list[dict[str, Any]],
    workdir: Path | None = None,
    session_uuid: str | None = None,
) -> tuple[int | None, bool, bool]:
    """Detect retriever CLI usage, returning ``(first_use, attempted, succeeded)``."""
    if agent == "claude":
        return _scan_claude_transcript_for_signals(envelope, workdir, session_uuid, condition)
    if agent == "codex" and session_uuid:
        return _scan_codex_transcript_for_signals(session_uuid, codex_events, condition)
    return None, False, False


def _run_one_turn(
    *,
    agent: str,
    condition: str,
    prompt: str,
    trial_id: str,
    entry_id: int,
    query_id: str,
    domain: str,
    is_setup: bool,
    turn_idx: int,
    workdir: Path,
    session_uuid: str,
    cmd: list[str],
    env: dict[str, str],
    timeout_s: int,
    model: str,
) -> TrialResult:
    """Execute one turn. Query turns expect the agent to write ``./output.json``."""
    out_path = workdir / "output.json"
    if out_path.exists():
        out_path.unlink()

    domain_tag = f"[{domain}] " if domain else ""
    label = "setup" if is_setup else f"entry_id={entry_id}, query_id={query_id}"
    logger.info("turn %d for %s/%s %s(%s)", turn_idx + 1, agent, condition, domain_tag, label)

    prior_codex_usage: dict[str, int] = {k: 0 for k in _CODEX_USAGE_FIELDS}
    if agent == "codex":
        prior_log = _codex_session_log_path(session_uuid)
        if prior_log is not None:
            prior_codex_usage = _extract_codex_total_usage(_read_jsonl_events(prior_log))

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(workdir),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TrialResult(
            trial_id=trial_id,
            condition=condition,
            entry_id=entry_id,
            query_id=query_id,
            status="timeout",
            extraction_method="none",
            duration_ms=int((time.monotonic() - t0) * 1000),
            duration_api_ms=0,
            num_turns=turn_idx + 1,
            total_cost_usd=0.0,
            model_id=model,
            session_id=session_uuid,
            agent=agent,
            errors=[f"turn exceeded {timeout_s}s wall timeout"],
            is_setup=is_setup,
            domain=domain,
            cost_available=(agent == "claude"),
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    envelope: dict[str, Any] = {}
    codex_events: list[dict[str, Any]] = []
    token_events: list[dict[str, Any]] = []
    if agent == "claude":
        envelope = _parse_envelope(proc.stdout)
        agent_error = bool(envelope.get("is_error", False))
        duration_ms = int(envelope.get("duration_ms") or elapsed_ms)
        duration_api_ms = int(envelope.get("duration_api_ms") or 0)
        total_cost_usd = float(envelope.get("total_cost_usd") or 0.0)
        model_id = _extract_model_id(envelope, fallback=model)
        actual_session_id = str(envelope.get("session_id") or session_uuid)
    else:
        codex_events = _parse_jsonl_events(proc.stdout)
        agent_error = _codex_has_error(codex_events)
        duration_ms = elapsed_ms
        duration_api_ms = 0
        total_cost_usd = 0.0
        model_id = model
        log_path = _codex_session_log_path(session_uuid)
        if log_path is None:
            log_path = _codex_session_log_for_workdir(workdir)
        token_events = _read_jsonl_events(log_path) if log_path is not None else codex_events
        actual_session_id = _codex_session_id(
            token_events,
            fallback=_codex_session_id(codex_events, fallback=session_uuid),
        )

    stderr = proc.stderr.strip()
    result = TrialResult(
        trial_id=trial_id,
        condition=condition,
        entry_id=entry_id,
        query_id=query_id,
        status="ok" if proc.returncode == 0 and not agent_error else "error",
        extraction_method="n/a" if is_setup else "output_json",
        duration_ms=duration_ms,
        duration_api_ms=duration_api_ms,
        num_turns=turn_idx + 1,
        total_cost_usd=total_cost_usd,
        model_id=model_id,
        session_id=actual_session_id,
        agent=agent,
        is_setup=is_setup,
        domain=domain,
        cost_available=(agent == "claude"),
    )
    if agent == "claude":
        _populate_claude_tokens(result, envelope)
    else:
        current_codex_usage = _extract_codex_total_usage(token_events or codex_events)
        _populate_codex_tokens(result, current_codex_usage, prior_codex_usage)
    if proc.returncode != 0:
        result.errors.append(f"non-zero exit {proc.returncode}")
    if agent == "claude" and envelope.get("is_error"):
        result.errors.append(f"envelope is_error: {envelope.get('subtype') or '?'}")
        detail = _extract_claude_error_detail(envelope)
        if detail:
            result.errors.append(f"claude error: {detail[:500]}")
    if agent == "codex" and agent_error:
        result.errors.append("codex event stream reported an error")
    if stderr:
        result.errors.append(f"stderr: {stderr[:500]}")

    if not is_setup:
        answer, ranked, extract_status, extract_errors = _parse_output_json(workdir)
        if extract_status in ("missing", "invalid_json"):
            result.extraction_method = extract_status
            if result.status == "ok":
                result.status = "extraction_failed"
        elif extract_status == "schema_warning":
            result.extraction_method = "schema_warning"
        result.final_answer = answer
        result.ranked_retrieved = ranked
        result.errors.extend(extract_errors)
        if out_path.exists():
            out_path.rename(workdir / f"output_e{entry_id}.json")

    first_use, attempted, succeeded = _scan_transcript_for_signals(
        agent=agent,
        condition=condition,
        envelope=envelope,
        codex_events=codex_events,
        workdir=workdir,
        session_uuid=actual_session_id,
    )
    result.retriever_first_use_turn = first_use
    result.retriever_attempted = attempted
    result.retriever_succeeded = succeeded
    # c1 has the skill unavailable; leave skill_fired=None to distinguish from
    # "loaded but didn't fire".
    if condition in ("c2_retriever", "c3_retriever_skill"):
        result.skill_fired = attempted and (first_use is not None) and first_use <= 2
    return result


UNSCORABLE_JUDGE_ERRORS: frozenset[str] = frozenset({"no_ground_truth", "empty_candidate"})


def _apply_judge(judge: Any, entry: DatasetEntry, result: TrialResult) -> None:
    """Score ``result.final_answer`` against ``entry.ground_truth_answer``.

    Missing ground truth and empty candidates are recorded as terminal
    ``judge_error`` values so ``rescore`` can skip intrinsically unscorable
    trials instead of retrying them forever.
    """
    if judge is None:
        return
    if not entry.ground_truth_answer:
        result.judge_error = "no_ground_truth"
        return
    if not result.final_answer:
        result.judge_error = "empty_candidate"
        return
    try:
        verdict = judge.judge(
            query=entry.original_query,
            reference=entry.ground_truth_answer,
            candidate=result.final_answer,
        )
    except Exception as exc:
        result.judge_error = f"judge_invocation_error: {exc}"
        logger.warning("LLMJudge raised for entry_id=%s: %s", result.entry_id, exc, exc_info=True)
        return
    result.judge_score = verdict.score
    result.judge_reasoning = verdict.reasoning or ""
    if verdict.error:
        result.judge_error = verdict.error


def run_condition(
    *,
    agent: str,
    condition: str,
    entries: list[DatasetEntry],
    workdir_root: Path,
    pdf_source: Path,
    skill_source: Path,
    model: str,
    budget_usd: float,
    timeout_s: int,
    domain: str = "",
    domain_label: str = "PDFs",
    judge: Any = None,
    testdata_prefixes: tuple[str, ...] = (),
    query_parallelism: int = 1,
) -> ConditionRun:
    """Run one agent condition: setup once, then query linearly or in isolated parallel sessions."""
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"unsupported agent: {agent}")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    query_parallelism = max(1, int(query_parallelism or 1))
    execution_mode = "parallel_isolated" if query_parallelism > 1 else "linear_session"

    workdir = _build_condition_workdir(agent, condition, workdir_root, pdf_source, skill_source, domain=domain)
    session_uuid = str(uuid.uuid4())
    env = _env_for(condition, workdir)
    logger.info(
        "starting session for %s/%s/%s: workdir=%s session_id=%s mode=%s query_parallelism=%d",
        agent,
        condition,
        domain or "default",
        workdir,
        session_uuid,
        execution_mode,
        query_parallelism,
    )

    results: list[TrialResult] = []
    workdirs: list[Path] = [workdir]
    result_workdirs: dict[str, Path] = {}

    setup_trial_id = f"{agent}_{condition}_{domain or 'default'}_setup_t1"
    setup_cmd = _build_command(
        agent=agent,
        condition=condition,
        model=model,
        budget_usd=budget_usd,
        session_uuid=session_uuid,
        workdir=workdir,
        resume=False,
    )
    setup_result = _run_one_turn(
        agent=agent,
        condition=condition,
        prompt=_render_setup_prompt(condition, domain_label),
        trial_id=setup_trial_id,
        entry_id=0,
        query_id="",
        domain=domain,
        is_setup=True,
        turn_idx=0,
        workdir=workdir,
        session_uuid=session_uuid,
        cmd=setup_cmd,
        env=env,
        timeout_s=timeout_s,
        model=model,
    )
    setup_result.execution_mode = execution_mode
    results.append(setup_result)
    result_workdirs[setup_result.trial_id] = workdir

    run = ConditionRun(
        workdir=workdir,
        workdirs=workdirs,
        results=results,
        result_workdirs=result_workdirs,
        execution_mode=execution_mode,
    )

    if setup_result.status != "ok":
        logger.warning(
            "setup turn failed for %s/%s/%s; skipping %d query turns",
            agent,
            condition,
            domain or "default",
            len(entries),
        )
        return run

    _write_setup_context(workdir, agent=agent, condition=condition)
    session_uuid = setup_result.session_id or session_uuid
    entries_by_id = {e.entry_id: e for e in entries}

    if query_parallelism <= 1:
        resume_cmd = _build_command(
            agent=agent,
            condition=condition,
            model=model,
            budget_usd=budget_usd,
            session_uuid=session_uuid,
            workdir=workdir,
            resume=True,
        )
        for i, entry in enumerate(entries):
            turn_idx = i + 1
            result = _run_one_turn(
                agent=agent,
                condition=condition,
                prompt=_render_prompt(entry, condition, testdata_prefixes),
                trial_id=f"{agent}_{condition}_{domain or 'default'}_e{entry.entry_id}_t{turn_idx + 1}",
                entry_id=entry.entry_id,
                query_id=entry.query_id,
                domain=domain,
                is_setup=False,
                turn_idx=turn_idx,
                workdir=workdir,
                session_uuid=session_uuid,
                cmd=resume_cmd,
                env=env,
                timeout_s=timeout_s,
                model=model,
            )
            result.execution_mode = execution_mode
            _apply_judge(judge, entry, result)
            results.append(result)
            result_workdirs[result.trial_id] = workdir
        return run

    query_contexts: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        turn_idx = i + 1
        query_workdir = _copy_setup_workdir_for_query(workdir, entry)
        workdirs.append(query_workdir)
        query_session_uuid = str(uuid.uuid4())
        query_contexts.append(
            {
                "entry": entry,
                "turn_idx": turn_idx,
                "workdir": query_workdir,
                "session_uuid": query_session_uuid,
                "cmd": _build_command(
                    agent=agent,
                    condition=condition,
                    model=model,
                    budget_usd=budget_usd,
                    session_uuid=query_session_uuid,
                    workdir=query_workdir,
                    resume=False,
                ),
                "env": _env_for(condition, query_workdir),
                "prompt": _render_isolated_query_prompt(entry, condition, testdata_prefixes),
                "trial_id": f"{agent}_{condition}_{domain or 'default'}_e{entry.entry_id}_t{turn_idx + 1}",
            }
        )

    def execute_query(ctx: dict[str, Any]) -> TrialResult:
        entry = ctx["entry"]
        result = _run_one_turn(
            agent=agent,
            condition=condition,
            prompt=ctx["prompt"],
            trial_id=ctx["trial_id"],
            entry_id=entry.entry_id,
            query_id=entry.query_id,
            domain=domain,
            is_setup=False,
            turn_idx=ctx["turn_idx"],
            workdir=ctx["workdir"],
            session_uuid=ctx["session_uuid"],
            cmd=ctx["cmd"],
            env=ctx["env"],
            timeout_s=timeout_s,
            model=model,
        )
        result.execution_mode = execution_mode
        return result

    query_results: list[TrialResult] = []
    if query_contexts:
        workers = min(query_parallelism, len(query_contexts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ctx = {executor.submit(execute_query, ctx): ctx for ctx in query_contexts}
            for future in as_completed(future_to_ctx):
                ctx = future_to_ctx[future]
                entry = ctx["entry"]
                try:
                    result = future.result()
                except Exception as exc:  # defensive: subprocess timeouts are handled in _run_one_turn
                    result = TrialResult(
                        trial_id=ctx["trial_id"],
                        condition=condition,
                        entry_id=entry.entry_id,
                        query_id=entry.query_id,
                        status="error",
                        extraction_method="none",
                        duration_ms=0,
                        duration_api_ms=0,
                        num_turns=int(ctx["turn_idx"]) + 1,
                        total_cost_usd=0.0,
                        model_id=model,
                        session_id=str(ctx["session_uuid"]),
                        agent=agent,
                        errors=[f"isolated query worker failed: {exc}"],
                        is_setup=False,
                        domain=domain,
                        cost_available=(agent == "claude"),
                        execution_mode=execution_mode,
                    )
                query_results.append(result)
                result_workdirs[result.trial_id] = ctx["workdir"]

    entry_order = {entry.entry_id: i for i, entry in enumerate(entries)}
    query_results.sort(key=lambda r: entry_order.get(r.entry_id, len(entry_order)))
    for result in query_results:
        entry = entries_by_id.get(result.entry_id)
        if entry is not None:
            _apply_judge(judge, entry, result)
    results.extend(query_results)
    return run


def save_trial(result: TrialResult, session_dir: Path) -> Path:
    parts = [session_dir, "trials", result.agent, result.condition]
    if result.domain:
        parts.append(result.domain)
    out = Path(*[str(p) for p in parts]) / f"{result.trial_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return out


def save_compact_trace(result: TrialResult, session_dir: Path, trace: str, *, suffix: str = "") -> Path:
    parts = [session_dir, "trials", result.agent, result.condition]
    if result.domain:
        parts.append(result.domain)
    traces_dir = Path(*[str(p) for p in parts]) / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    stem = result.trial_id if not suffix else f"{result.trial_id}_{suffix}"
    out = traces_dir / f"{stem}.md"
    out.write_text(trace.rstrip() + "\n", encoding="utf-8")
    return out


def archive_session_log(
    *,
    session_dir: Path,
    agent: str,
    condition: str,
    domain: str,
    session_uuid: str,
    workdir: Path,
) -> Path | None:
    """Copy the agent's rollout log into the artifact dir so it survives ``cleanup_condition_workdir``.

    Without this, the per-trial JSONs are the only persistent record of the run —
    you cannot retroactively recompute token deltas, tool-use signals, or anything
    else that requires the raw event stream.
    """
    if agent == "claude":
        src = _claude_session_log_path(workdir, session_uuid)
    elif agent == "codex":
        src = _codex_session_log_path(session_uuid)
    else:
        return None
    if src is None or not src.exists():
        return None
    parts = [session_dir, "trials", agent, condition]
    if domain:
        parts.append(domain)
    logs_dir = Path(*[str(p) for p in parts]) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    dest = logs_dir / src.name
    shutil.copy2(src, dest)
    return dest
