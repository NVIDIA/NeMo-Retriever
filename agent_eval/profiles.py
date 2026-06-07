# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile/environment construction for agent_eval runs.

A *profile* is the with/without-nemo_retriever axis:

* ``baseline`` — no skill, retriever blocked: a PATH shim ``./.bin/retriever``
  that exits 127, a project ``settings.json`` deny-list (Claude), and an empty
  HF cache. The agent must answer using only native tools over ``./pdfs/``.
* ``skill`` — the nemo-retriever skill is copied in, the real ``retriever`` CLI
  is on PATH, and a prebuilt ``./lancedb`` index is mounted.

Per (profile, domain) we build one *base* workdir once (and, for ``skill``,
ingest the corpus once). Each query then gets a lightweight *query* workdir that
symlinks the shared corpus/index, so parallel queries don't re-copy or re-ingest.

Pure stdlib + subprocess to the ``retriever`` CLI (no nemo_retriever import).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

BASELINE = "baseline"
SKILL = "skill"
PROFILES = (BASELINE, SKILL)

# settings.json deny-list applied in the baseline profile (Claude reads it via
# --setting-sources project). Mirrors the skill_eval c1_base shim.
_BASELINE_DENY = {
    "permissions": {
        "deny": [
            "Bash(retriever:*)",
            "Bash(*nemo_retriever*)",
            "Bash(*nemo-retriever*)",
            "Bash(python*-m*nemo_retriever*)",
            "Bash(uv*run*retriever*)",
            "Bash(*/bin/retriever*)",
            "Bash(*huggingface*)",
            "Bash(*.cache/huggingface*)",
        ]
    }
}

_RETRIEVER_SHIM = "#!/usr/bin/env bash\necho 'retriever: command not found (baseline profile)' >&2\nexit 127\n"


def mount_corpus_refs(workdir: Path, corpus_refs: list[str], data_root: Path) -> int:
    """Symlink each corpus_ref (a relative path like ``test-data/financebench/pdfs``)
    into ``workdir`` at the SAME relative path, pointing at ``data_root/<ref>``, so a
    prompt's literal ``test-data/...`` paths resolve. Returns the number mounted."""
    n = 0
    for ref in corpus_refs:
        ref = ref.strip().lstrip("/")
        src = (data_root / ref).resolve()
        if not src.exists():
            continue
        dst = workdir / ref
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        n += 1
    return n


def has_index(lancedb_dir: Path) -> bool:
    """True if a LanceDB table exists, regardless of table name (agents may name
    it nv-ingest, nemo-retriever, etc.)."""
    return lancedb_dir.exists() and any(lancedb_dir.glob("*.lance"))


def _symlink_corpus(corpus_dir: Path, dest_pdfs: Path) -> int:
    dest_pdfs.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted(corpus_dir.glob("*.pdf")):
        link = dest_pdfs / pdf.name
        if not link.exists():
            link.symlink_to(pdf.resolve())
            n += 1
    return n


def _copy_skill(skill_src: Path, dest: Path) -> None:
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    if (skill_src / "SKILL.md").exists():
        shutil.copy2(skill_src / "SKILL.md", dest / "SKILL.md")
    refs = skill_src / "references"
    if refs.exists():
        shutil.copytree(refs, dest / "references", dirs_exist_ok=True)
    scripts = skill_src / "scripts"
    if scripts.exists():
        shutil.copytree(scripts, dest / "scripts", dirs_exist_ok=True)


def build_base_workdir(
    *,
    run_dir: Path,
    domain: str,
    profile: str,
    corpus_dir: Path,
    agent: str,
    skill_src: Path | None,
    retriever_bin: Path | None,
    embed_model: str,
    dry_run: bool = False,
    skip_setup: bool = False,
    prebuilt_index: Path | None = None,
    agent_setup: bool = True,
) -> dict:
    """Create (or locate) the shared per-domain base workdir. Returns info dict.

    For the skill profile the index is built by an AGENT setup turn
    (``agent_setup=True``) so its tokens are measured; the orchestrator runs that
    turn when ``index == "agent-setup-pending"``. ``agent_setup=False`` falls back
    to a direct ``retriever ingest`` subprocess (no LLM tokens)."""
    base = run_dir / "_setup" / f"{profile}_{domain}"
    base.mkdir(parents=True, exist_ok=True)
    n_pdfs = _symlink_corpus(corpus_dir, base / "pdfs")

    info = {"base_dir": str(base), "domain": domain, "profile": profile, "n_pdfs": n_pdfs}

    if profile == SKILL:
        # Skill copy lives under the agent-specific skills dir.
        skills_root = ".claude" if agent == "claude" else ".codex"
        if skill_src is not None:
            _copy_skill(skill_src, base / skills_root / "skills" / "nemo-retriever")
        # Index: reuse prebuilt, skip, or ingest once.
        lancedb = base / "lancedb"
        if prebuilt_index is not None and prebuilt_index.exists():
            if not lancedb.exists():
                lancedb.symlink_to(prebuilt_index.resolve())
            info["index"] = f"prebuilt -> {prebuilt_index}"
        elif has_index(lancedb):
            info["index"] = "reused existing"
        elif skip_setup or dry_run:
            info["index"] = "SKIPPED (no ingest)"
        elif agent_setup:
            # The orchestrator runs an agent setup turn to build this.
            info["index"] = "agent-setup-pending"
        else:
            info["index"] = _ingest(base, retriever_bin, embed_model)
    elif profile == BASELINE:
        # PATH shim + deny settings + empty HF cache.
        binshim = base / ".bin" / "retriever"
        binshim.parent.mkdir(parents=True, exist_ok=True)
        binshim.write_text(_RETRIEVER_SHIM)
        binshim.chmod(0o755)
        (base / ".hf_empty").mkdir(exist_ok=True)
        if agent == "claude":
            settings = base / ".claude" / "settings.json"
            settings.parent.mkdir(parents=True, exist_ok=True)
            settings.write_text(json.dumps(_BASELINE_DENY, indent=2))
        info["index"] = "n/a (baseline)"

    return info


def _ingest(base: Path, retriever_bin: Path | None, embed_model: str) -> str:
    if retriever_bin is None or not Path(retriever_bin).exists():
        return "ERROR: retriever binary not found; cannot ingest"
    cmd = [
        str(retriever_bin),
        "ingest",
        "./pdfs/",
        "--embed-model-name",
        embed_model,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(base), capture_output=True, text=True)
    dur = time.monotonic() - t0
    if proc.returncode != 0 or not has_index(base / "lancedb"):
        tail = (proc.stderr or proc.stdout or "")[-300:]
        return f"INGEST FAILED (rc={proc.returncode}, {dur:.0f}s): {tail}"
    return f"ingested ({dur:.0f}s)"


def build_query_workdir(*, base_dir: Path, query_dir: Path, profile: str, agent: str) -> Path:
    """Create a per-query workdir that symlinks the shared base contents."""
    wd = query_dir / "workdir"
    wd.mkdir(parents=True, exist_ok=True)
    # Shared, read-only-ish artifacts: symlink to the base.
    for name in ("pdfs", "lancedb"):
        src = base_dir / name
        dst = wd / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    # Skill / settings / shim: symlink the whole agent dotdir tree.
    skills_root = ".claude" if agent == "claude" else ".codex"
    src_dot = base_dir / skills_root
    if src_dot.exists() and not (wd / skills_root).exists():
        (wd / skills_root).symlink_to(src_dot.resolve())
    if profile == BASELINE:
        for name in (".bin", ".hf_empty"):
            src = base_dir / name
            if src.exists() and not (wd / name).exists():
                (wd / name).symlink_to(src.resolve())
    return wd


def env_for(profile: str, workdir: Path, base_env: dict | None = None, gpu: int | None = None) -> dict:
    env = dict(base_env if base_env is not None else os.environ)
    if profile == BASELINE:
        env["PATH"] = f"{workdir / '.bin'}{os.pathsep}{env.get('PATH', '')}"
        hf = str((workdir / ".hf_empty").resolve())
        env["HF_HOME"] = hf
        env["HF_HUB_CACHE"] = hf
        env["TRANSFORMERS_CACHE"] = hf
    # Pin this query's retriever (and any model it loads) to one GPU so parallel
    # query processes spread across devices instead of all stacking on GPU 0.
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env
