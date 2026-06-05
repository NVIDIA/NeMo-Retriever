#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script B: run an agent over an answer-free queries.json and save artifacts.

Standalone — imports only stdlib + the local agent_eval modules. The agent CLIs
(``claude`` / ``codex``) and, for the ``skill`` profile, the ``retriever`` CLI
are invoked as subprocesses; this module never imports ``nemo_retriever``, so it
can be copied out and run in a clean environment for the baseline.

For each query it builds an isolated workdir (sharing a per-domain corpus/index),
runs the agent for a single turn, and writes a per-question subfolder:

    <save-root>/<run_id>/<query_id>/
        output.json        # {final_answer, selected_chunks[...]}  (normalized)
        agent_output.raw    # raw agent ./output.json as the agent wrote it
        agent_log.jsonl     # full session log (actions + thought process)
        trace.md            # compact human-readable trace
        meta.json           # agent/model/profile/timing/tokens/cost/status

Usage:
    python run_agent_eval.py --queries queries.json --agent claude \
        --model claude-opus-4-7 --profile skill --save-root ./runs
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from schema import QuerySet, Query  # noqa: E402
import profiles  # noqa: E402
import gitinfo  # noqa: E402
from prompt import render_prompt, render_setup_prompt, parse_output_json  # noqa: E402
from adapters.claude import ClaudeAdapter  # noqa: E402
from adapters.codex import CodexAdapter  # noqa: E402

ADAPTERS = {"claude": ClaudeAdapter, "codex": CodexAdapter}

# Page-number convention the agent emits, by profile. The skill agent copies the
# retriever's raw 1-indexed page_number verbatim; baseline native-reads 0-indexed.
# The report normalizes to the 0-indexed gold via this base.
PAGE_INDEX_BASE = {profiles.SKILL: 1, profiles.BASELINE: 0}


@dataclass
class TrialMeta:
    query_id: str
    domain: str
    agent: str
    model: str
    profile: str
    status: str  # ok | no_output | timeout | error
    exit_status: int | None = None
    duration_ms: int | None = None
    session_id: str | None = None
    tokens: dict = field(default_factory=dict)
    cost_usd: float | None = None
    num_selected_chunks: int = 0
    final_answer_len: int = 0
    error: str = ""


def _corpus_dir(corpus_root: Path, domain: str) -> Path:
    return corpus_root / domain


def _detect_gpus() -> int:
    """Number of visible GPUs (honors CUDA_VISIBLE_DEVICES), else nvidia-smi count."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip() != ""]) or 1
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        n = len([ln for ln in out.stdout.splitlines() if ln.strip().startswith("GPU ")])
        return n or 1
    except Exception:  # noqa: BLE001
        return 1


def save_root_leak_issues(save_root: Path, profile: str) -> list[tuple[str, str]]:
    """Detect ancestors of --save-root that would un-blind a baseline run.

    Claude Code discovers project skills / settings / CLAUDE.md by walking UP the
    directory tree from the agent's cwd. Because each per-query workdir lives
    under save_root, any nemo-retriever skill or project context in save_root OR
    an ancestor leaks into the (supposedly blind) baseline. This is the real
    "location" hazard — not where the script is launched from.

    Returns a list of (severity, message); severity is "error" (blocks baseline
    unless overridden) or "warn".
    """
    issues: list[tuple[str, str]] = []
    resolved = save_root.resolve()
    home = Path.home().resolve()
    for a in (resolved, *resolved.parents):
        skill = a / ".claude" / "skills" / "nemo-retriever"
        if skill.exists():  # follows the symlink farm
            sev = "error" if profile == profiles.BASELINE else "warn"
            issues.append((sev, f"discoverable nemo-retriever skill under an ancestor: {skill}"))
        for sname in ("settings.json", "settings.local.json"):
            s = a / ".claude" / sname
            if s.exists() and a != home:
                issues.append(("warn", f"project settings under an ancestor may alter tool permissions: {s}"))
        md = a / "CLAUDE.md"
        if md.exists() and a != home:
            issues.append(("warn", f"CLAUDE.md under an ancestor may inject project context: {md}"))
    return issues


def _run_setup(
    domain: str,
    base_dir: Path,
    *,
    adapter,
    model: str,
    profile: str,
    budget_usd: float,
    timeout_s: int,
    gpu: int | None = None,
) -> dict:
    """Run an agent-driven setup turn (build the index) in the base workdir."""
    prompt = render_setup_prompt()
    (base_dir / "setup_prompt.txt").write_text(prompt)
    sid = str(uuid.uuid4())
    env = profiles.env_for(profile, base_dir, gpu=gpu)
    res = adapter.run(
        prompt,
        model=model,
        workdir=base_dir,
        session_id=sid,
        profile=profile,
        budget_usd=budget_usd,
        timeout_s=timeout_s,
        env=env,
    )
    index_built = profiles.has_index(base_dir / "lancedb")
    status = "timeout" if res.timed_out else ("ok" if index_built else "no_index")
    if res.raw_log_path and Path(res.raw_log_path).exists():
        shutil.copy2(res.raw_log_path, base_dir / "setup_log.jsonl")
        try:
            (base_dir / "setup_trace.md").write_text(adapter.compact_trace(Path(res.raw_log_path)))
        except Exception:  # noqa: BLE001
            pass
    if res.stderr.strip():
        (base_dir / "setup_stderr.txt").write_text(res.stderr[-20000:])
    meta = {
        "domain": domain,
        "phase": "setup",
        "agent": adapter.name,
        "model": model,
        "profile": profile,
        "status": status,
        "index_built": index_built,
        "exit_status": res.exit_status,
        "duration_ms": res.duration_ms,
        "session_id": res.session_id,
        "tokens": res.tokens,
        "cost_usd": res.cost_usd,
    }
    (base_dir / "setup_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _run_one(
    query: Query,
    *,
    adapter,
    model: str,
    profile: str,
    run_dir: Path,
    base_dirs: dict[str, Path],
    top_k: int,
    budget_usd: float,
    timeout_s: int,
    dry_run: bool,
    gpu: int | None = None,
) -> TrialMeta:
    qdir = run_dir / query.query_id.replace("/", "_").replace(":", "_")
    qdir.mkdir(parents=True, exist_ok=True)
    base_dir = base_dirs[query.domain]
    wd = profiles.build_query_workdir(base_dir=base_dir, query_dir=qdir, profile=profile, agent=adapter.name)
    rendered = render_prompt(query.query_id, query.prompt, top_k=top_k, page_index_base=PAGE_INDEX_BASE.get(profile, 0))
    (qdir / "prompt.txt").write_text(rendered)
    session_id = str(uuid.uuid4())
    env = profiles.env_for(profile, wd, gpu=gpu)

    if dry_run:
        cmd = adapter.build_command(
            model=model, workdir=wd, session_id=session_id, profile=profile, budget_usd=budget_usd
        )
        (qdir / "meta.json").write_text(json.dumps({"dry_run": True, "cmd": cmd, "workdir": str(wd)}, indent=2))
        return TrialMeta(
            query_id=query.query_id,
            domain=query.domain,
            agent=adapter.name,
            model=model,
            profile=profile,
            status="dry_run",
        )

    res = adapter.run(
        rendered,
        model=model,
        workdir=wd,
        session_id=session_id,
        profile=profile,
        budget_usd=budget_usd,
        timeout_s=timeout_s,
        env=env,
    )

    # Persist the raw agent output and the normalized version.
    raw_out = wd / "output.json"
    if raw_out.exists():
        shutil.copy2(raw_out, qdir / "agent_output.raw.json")
    normalized = parse_output_json(wd)
    status = "ok"
    if res.timed_out:
        status = "timeout"
    elif normalized is None:
        status = "no_output"
    elif res.exit_status not in (0, None):
        status = "ok"  # agent may exit nonzero yet still write output

    out_payload = {
        "query_id": query.query_id,
        "final_answer": (normalized or {}).get("final_answer", res.final_text or ""),
        "selected_chunks": (normalized or {}).get("selected_chunks", []),
        "schema_version": 1,
    }
    (qdir / "output.json").write_text(json.dumps(out_payload, indent=2, ensure_ascii=False))

    # Archive the full agent session log + compact trace. Claude writes a session
    # JSONL we copy; codex's event stream is its stdout, so fall back to that.
    log_dst = qdir / "agent_log.jsonl"
    if res.raw_log_path and Path(res.raw_log_path).exists():
        shutil.copy2(res.raw_log_path, log_dst)
    elif res.stdout.strip():
        log_dst.write_text(res.stdout)
    if log_dst.exists():
        try:
            (qdir / "trace.md").write_text(adapter.compact_trace(log_dst))
        except Exception:  # noqa: BLE001
            pass
    if res.stderr.strip():
        (qdir / "stderr.txt").write_text(res.stderr[-20000:])

    meta = TrialMeta(
        query_id=query.query_id,
        domain=query.domain,
        agent=adapter.name,
        model=model,
        profile=profile,
        status=status,
        exit_status=res.exit_status,
        duration_ms=res.duration_ms,
        session_id=res.session_id,
        tokens=res.tokens,
        cost_usd=res.cost_usd,
        num_selected_chunks=len(out_payload["selected_chunks"]),
        final_answer_len=len(out_payload["final_answer"]),
        error=res.error,
    )
    (qdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--agent", default="claude", choices=sorted(ADAPTERS))
    ap.add_argument("--model", default="claude-opus-4-7", help="Editable model id.")
    ap.add_argument("--profile", default=profiles.SKILL, choices=profiles.PROFILES)
    ap.add_argument(
        "--corpus-root",
        type=Path,
        default=Path(os.environ["AGENT_EVAL_CORPUS_ROOT"]) if "AGENT_EVAL_CORPUS_ROOT" in os.environ else None,
        help="Domain corpora live at <corpus-root>/<domain>/*.pdf (or set AGENT_EVAL_CORPUS_ROOT)",
    )
    ap.add_argument(
        "--skill-src",
        type=Path,
        default=Path(os.environ["AGENT_EVAL_SKILL_SRC"]) if "AGENT_EVAL_SKILL_SRC" in os.environ else None,
        help="Path to the nemo-retriever skill source (or set AGENT_EVAL_SKILL_SRC)",
    )
    ap.add_argument(
        "--retriever-bin",
        type=Path,
        default=Path(os.environ["AGENT_EVAL_RETRIEVER_BIN"]) if "AGENT_EVAL_RETRIEVER_BIN" in os.environ else None,
        help="Path to the retriever binary (or set AGENT_EVAL_RETRIEVER_BIN)",
    )
    ap.add_argument("--embed-model", default="nvidia/llama-nemotron-embed-1b-v2")
    ap.add_argument(
        "--prebuilt-index",
        type=Path,
        default=None,
        help="Reuse an existing lancedb dir (skill profile) instead of ingesting.",
    )
    ap.add_argument(
        "--prebuilt-index-root",
        type=Path,
        default=None,
        help="Reuse per-domain indexes from a prior run dir: resolves "
        "<root>/_setup/skill_<domain>/lancedb for each domain.",
    )
    ap.add_argument("--save-root", type=Path, default=Path("./agent_eval_runs"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="GPUs to round-robin query processes across (skill profile). 0 = auto-detect.",
    )
    ap.add_argument(
        "--gpu-list",
        default=None,
        help="Explicit physical GPU ids to use (comma-separated, e.g. '0,1,2,3'). Overrides "
        "--gpus; lets two concurrent runs use disjoint GPU halves. Pins setup ingests too.",
    )
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--domains", default=None, help="Comma-separated domain filter.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--allow-unsafe-save-root",
        action="store_true",
        help="Proceed even if --save-root would let the baseline discover the skill/project context.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Build workdirs + print commands; no agent calls.")
    ap.add_argument("--skip-setup", action="store_true", help="Skill profile: don't ingest (assume index present).")
    ap.add_argument(
        "--subprocess-ingest",
        action="store_true",
        help="Skill: build the index via a direct `retriever ingest` subprocess instead of an "
        "agent setup turn (no setup LLM tokens).",
    )
    args = ap.parse_args(argv)

    qs = QuerySet.load(args.queries)
    queries = qs.queries
    if args.domains:
        keep = {d.strip() for d in args.domains.split(",")}
        queries = [q for q in queries if q.domain in keep]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("no queries after filtering", file=sys.stderr)
        return 1

    # Safety guard: a save-root whose ancestors expose the skill/project context
    # would silently un-blind a baseline run (Claude walks up from the workdir).
    issues = save_root_leak_issues(args.save_root, args.profile)
    errors = [m for sev, m in issues if sev == "error"]
    warns = [m for sev, m in issues if sev == "warn"]
    for m in warns:
        print(f"  WARN (save-root): {m}", file=sys.stderr)
    if errors:
        print("\nUNSAFE --save-root for a baseline run:", file=sys.stderr)
        for m in errors:
            print(f"  - {m}", file=sys.stderr)
        print(
            "  The baseline agent would discover nemo-retriever via the directory tree.\n"
            "  Fix: choose a --save-root OUTSIDE that tree (e.g. /tmp/agent_eval_runs).\n"
            "  Or pass --allow-unsafe-save-root to proceed anyway (not recommended for baseline).",
            file=sys.stderr,
        )
        if not args.allow_unsafe_save_root:
            return 2
        print("  proceeding despite unsafe save-root (--allow-unsafe-save-root).", file=sys.stderr)

    adapter = ADAPTERS[args.agent]()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    run_id = f"agenteval_{args.agent}_{args.profile}_{ts}"
    run_dir = args.save_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    domains = sorted({q.domain for q in queries})
    print(
        f"Run: {run_id}\n  agent={args.agent} model={args.model} profile={args.profile} "
        f"queries={len(queries)} domains={domains}"
    )

    # GPU pool for the skill profile: explicit --gpu-list (disjoint halves for
    # concurrent runs) else 0..num_gpus-1. Used to pin both setup ingests and
    # query processes so they spread instead of stacking on GPU 0.
    if args.gpu_list:
        gpu_pool = [int(x) for x in args.gpu_list.split(",") if x.strip() != ""]
    else:
        ng = args.gpus if args.gpus > 0 else _detect_gpus()
        gpu_pool = list(range(ng))
    pin = args.profile == profiles.SKILL and len(gpu_pool) >= 1
    if pin:
        print(f"  GPU pinning (skill): pool={gpu_pool}")

    # Per-domain base workdir + (skill) one-time index build. By default the index
    # is built by an agent SETUP TURN (measured tokens); --subprocess-ingest uses
    # a direct `retriever ingest` instead.
    base_dirs: dict[str, Path] = {}
    setup_info: dict[str, Any] = {}
    setup_metas: list[dict] = []
    for di, dom in enumerate(domains):
        cdir = _corpus_dir(args.corpus_root, dom)
        if not cdir.exists():
            print(f"  !! corpus dir missing for {dom}: {cdir}", file=sys.stderr)
        pidx = args.prebuilt_index
        if args.prebuilt_index_root is not None:
            cand = args.prebuilt_index_root / "_setup" / f"skill_{dom}" / "lancedb"
            if cand.exists():
                pidx = cand
            else:
                print(f"  !! no prebuilt index for {dom} under {args.prebuilt_index_root}", file=sys.stderr)
        info = profiles.build_base_workdir(
            run_dir=run_dir,
            domain=dom,
            profile=args.profile,
            corpus_dir=cdir,
            agent=args.agent,
            skill_src=args.skill_src,
            retriever_bin=args.retriever_bin,
            embed_model=args.embed_model,
            dry_run=args.dry_run,
            skip_setup=args.skip_setup,
            prebuilt_index=pidx,
            agent_setup=not args.subprocess_ingest,
        )
        base = Path(info["base_dir"])
        base_dirs[dom] = base
        setup_info[dom] = info
        if info.get("index") == "agent-setup-pending" and not args.dry_run:
            print(f"  setup[{dom}]: running agent setup turn (build index) ...")
            sm = _run_setup(
                dom,
                base,
                adapter=adapter,
                model=args.model,
                profile=args.profile,
                budget_usd=args.budget_usd,
                timeout_s=args.timeout,
                gpu=(gpu_pool[di % len(gpu_pool)] if pin else None),
            )
            setup_metas.append(sm)
            info["index"] = f"agent-setup status={sm['status']} index_built={sm['index_built']}"
        print(f"  setup[{dom}]: pdfs={info['n_pdfs']} index={info.get('index')}")
    (run_dir / "setup_metas.json").write_text(json.dumps(setup_metas, indent=2))

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "agent": args.agent,
                "model": args.model,
                "profile": args.profile,
                "code_commit": gitinfo.git_commit(),
                "source_manifest": qs.source_manifest,
                "queries_file": str(args.queries.resolve()),
                "corpus_root": str(args.corpus_root),
                "top_k": args.top_k,
                "timeout_s": args.timeout,
                "budget_usd": args.budget_usd,
                "embed_model": args.embed_model,
                "num_queries": len(queries),
                "domains": domains,
                "setup": setup_info,
                "setup_metas": setup_metas,
                "subprocess_ingest": args.subprocess_ingest,
                "dry_run": args.dry_run,
                "page_index_base": PAGE_INDEX_BASE.get(args.profile, 0),
                "parallelism": args.parallelism,
                "num_gpus": (args.gpus if args.gpus > 0 else _detect_gpus()),
            },
            indent=2,
        )
    )

    # Run queries (parallel), pinning each to a GPU from the pool (round-robin).
    metas: list[TrialMeta] = []
    workers = max(1, args.parallelism)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _run_one,
                q,
                adapter=adapter,
                model=args.model,
                profile=args.profile,
                run_dir=run_dir,
                base_dirs=base_dirs,
                top_k=args.top_k,
                budget_usd=args.budget_usd,
                timeout_s=args.timeout,
                dry_run=args.dry_run,
                gpu=(gpu_pool[i % len(gpu_pool)] if pin else None),
            ): q
            for i, q in enumerate(queries)
        }
        for i, fut in enumerate(as_completed(futs), 1):
            q = futs[fut]
            try:
                m = fut.result()
                metas.append(m)
                print(
                    f"  [{i}/{len(queries)}] {m.query_id} -> {m.status} "
                    f"(chunks={m.num_selected_chunks}, {m.duration_ms or 0}ms)"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(queries)}] {q.query_id} -> EXCEPTION {exc}", file=sys.stderr)
                metas.append(
                    TrialMeta(
                        query_id=q.query_id,
                        domain=q.domain,
                        agent=args.agent,
                        model=args.model,
                        profile=args.profile,
                        status="error",
                        error=str(exc),
                    )
                )

    (run_dir / "run_metas.json").write_text(json.dumps([asdict(m) for m in metas], indent=2))
    print(f"\nDone. status: {dict(Counter(m.status for m in metas))}\n  -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
