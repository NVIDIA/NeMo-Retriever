#!/usr/bin/env python3
"""Phase 2 (functional): run the functional_corpus_variants prompts under the skill profile.

Mounts each prompt's referenced corpus at its literal relative path
(``test-data/...`` -> ``<corpus-data-root>/test-data/...``), runs the agent for one turn,
and captures the final message + workdir artifacts (incl. whether a LanceDB table was
built) for pass/fail grading (eval_functional.py).

Index handling:
* retrieval_answer    -> reuse ONE shared index per corpus-group (built once); the agent
                        just queries (avoids re-ingesting financebench's 368 PDFs per query).
* ingest / ingest_plus_answer -> NO prebuilt index; the task IS to ingest, so the agent
                        builds it in-turn (that's what we grade).

Skill profile only (these tests assume the retriever pipeline). No nemo_retriever import.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import profiles  # noqa: E402
import gitinfo  # noqa: E402
from prompt import render_functional_prompt  # noqa: E402
from adapters.claude import ClaudeAdapter  # noqa: E402
from adapters.codex import CodexAdapter  # noqa: E402

ADAPTERS = {"claude": ClaudeAdapter, "codex": CodexAdapter}
RETRIEVAL = "retrieval_answer"
INGEST_TYPES = {"ingest", "ingest_plus_answer"}


@dataclass
class FuncMeta:
    query_id: str
    functional_type: str
    agent: str
    model: str
    status: str  # ok | timeout | error
    exit_status: int | None = None
    duration_ms: int | None = None
    session_id: str | None = None
    tokens: dict = field(default_factory=dict)
    cost_usd: float | None = None
    response_len: int = 0
    lancedb_built: bool = False
    lancedb_tables: list = field(default_factory=list)
    corpus_refs: list = field(default_factory=list)
    error: str = ""


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace("*", "x")


def _lancedb_state(wd: Path) -> tuple[bool, list]:
    ld = wd / "lancedb"
    tables = [p.name for p in ld.glob("*.lance")] if ld.exists() else []
    return (bool(tables), tables)


def _build_shared_index(
    idx_dir: Path, corpus_refs: list, data_root: Path, retriever_bin: Path, embed_model: str, gpu: int | None
) -> str:
    """Mount the corpus into idx_dir and `retriever ingest` it once (shared index)."""
    idx_dir.mkdir(parents=True, exist_ok=True)
    profiles.mount_corpus_refs(idx_dir, corpus_refs, data_root)
    if profiles.has_index(idx_dir / "lancedb"):
        return "reused"
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.monotonic()
    for ref in corpus_refs:
        cmd = [str(retriever_bin), "ingest", ref, "--embed-model-name", embed_model]
        subprocess.run(cmd, cwd=str(idx_dir), capture_output=True, text=True, env=env)
    ok = profiles.has_index(idx_dir / "lancedb")
    return f"{'built' if ok else 'FAILED'} ({time.monotonic()-t0:.0f}s)"


def _run_one(
    q: dict,
    *,
    adapter,
    model,
    run_dir: Path,
    data_root: Path,
    skill_src: Path,
    shared_index: dict,
    budget_usd: float,
    timeout_s: int,
    gpu,
    dry_run: bool,
) -> FuncMeta:
    qid = q["query_id"]
    ft = q.get("functional_type")
    refs = q.get("corpus_refs") or []
    qdir = run_dir / _slug(qid)
    wd = qdir / "workdir"
    wd.mkdir(parents=True, exist_ok=True)
    profiles.mount_corpus_refs(wd, refs, data_root)
    profiles._copy_skill(skill_src, wd / ".claude" / "skills" / "nemo-retriever")
    if adapter.name == "codex":
        profiles._copy_skill(skill_src, wd / ".codex" / "skills" / "nemo-retriever")
    # retrieval_answer reuses a shared prebuilt index
    if ft == RETRIEVAL:
        base = shared_index.get(tuple(sorted(refs)))
        if base and (base / "lancedb").exists() and not (wd / "lancedb").exists():
            (wd / "lancedb").symlink_to((base / "lancedb").resolve())
    rendered = render_functional_prompt(q["prompt"])
    (qdir / "prompt.txt").write_text(rendered)
    sid = str(uuid.uuid4())
    env = profiles.env_for(profiles.SKILL, wd, gpu=gpu)

    if dry_run:
        cmd = adapter.build_command(
            model=model, workdir=wd, session_id=sid, profile=profiles.SKILL, budget_usd=budget_usd
        )
        (qdir / "meta.json").write_text(
            json.dumps({"dry_run": True, "functional_type": ft, "corpus_refs": refs, "cmd": cmd}, indent=2)
        )
        return FuncMeta(qid, ft, adapter.name, model, "dry_run", corpus_refs=refs)

    res = adapter.run(
        rendered,
        model=model,
        workdir=wd,
        session_id=sid,
        profile=profiles.SKILL,
        budget_usd=budget_usd,
        timeout_s=timeout_s,
        env=env,
    )
    (qdir / "response.txt").write_text(res.final_text or "")
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
    built, tables = _lancedb_state(wd)
    status = "timeout" if res.timed_out else "ok"
    meta = FuncMeta(
        qid,
        ft,
        adapter.name,
        model,
        status,
        res.exit_status,
        res.duration_ms,
        res.session_id,
        res.tokens,
        res.cost_usd,
        len(res.final_text or ""),
        built,
        tables,
        refs,
        res.error,
    )
    (qdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--agent", default="claude", choices=sorted(ADAPTERS))
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--corpus-data-root", type=Path, default=Path("/raid/retriever-sdg-v3"))
    ap.add_argument("--skill-src", type=Path, default=Path("/raid/nemo_retriever/skills/nemo-retriever"))
    ap.add_argument("--retriever-bin", type=Path, default=Path("/raid/nemo_retriever/.venv/bin/retriever"))
    ap.add_argument("--embed-model", default="nvidia/llama-nemotron-embed-1b-v2")
    ap.add_argument("--save-root", type=Path, default=Path("./agent_eval_functional"))
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--gpu-list", default=None)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--types", default=None, help="Comma filter on functional_type.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    doc = json.loads(args.queries.read_text())
    queries = doc.get("queries", [])
    if args.types:
        keep = {t.strip() for t in args.types.split(",")}
        queries = [q for q in queries if q.get("functional_type") in keep]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("no queries after filtering", file=sys.stderr)
        return 1

    adapter = ADAPTERS[args.agent]()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    run_id = f"agenteval_func_{args.agent}_{ts}"
    run_dir = args.save_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu_list:
        gpu_pool = [int(x) for x in args.gpu_list.split(",") if x.strip()]
    else:
        ng = args.gpus if args.gpus > 0 else 1
        gpu_pool = list(range(ng))

    from collections import Counter

    print(
        f"Run: {run_id}\n  agent={args.agent} model={args.model} queries={len(queries)} "
        f"types={dict(Counter(q.get('functional_type') for q in queries))} gpus={gpu_pool}"
    )

    # Shared indexes for retrieval_answer corpus-groups (built once).
    shared_index: dict = {}
    groups = sorted(
        {tuple(sorted(q.get("corpus_refs") or [])) for q in queries if q.get("functional_type") == RETRIEVAL}
    )
    for gi, grp in enumerate(groups):
        idx_dir = run_dir / "_index" / _slug("__".join(grp))[:80]
        if args.dry_run:
            shared_index[grp] = idx_dir
            print(f"  [dry] shared index for {grp} -> {idx_dir}")
            continue
        print(f"  building shared index for {grp} ...")
        st = _build_shared_index(
            idx_dir,
            list(grp),
            args.corpus_data_root,
            args.retriever_bin,
            args.embed_model,
            gpu_pool[gi % len(gpu_pool)],
        )
        shared_index[grp] = idx_dir
        print(f"    {st}")

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "agent": args.agent,
                "model": args.model,
                "profile": "skill-functional",
                "source": doc.get("source_manifest"),
                "queries_file": str(args.queries.resolve()),
                "corpus_data_root": str(args.corpus_data_root),
                "num_queries": len(queries),
                "gpu_pool": gpu_pool,
                "dry_run": args.dry_run,
                "code_commit": gitinfo.git_commit(),
            },
            indent=2,
        )
    )

    metas: list[FuncMeta] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as ex:
        futs = {
            ex.submit(
                _run_one,
                q,
                adapter=adapter,
                model=args.model,
                run_dir=run_dir,
                data_root=args.corpus_data_root,
                skill_src=args.skill_src,
                shared_index=shared_index,
                budget_usd=args.budget_usd,
                timeout_s=args.timeout,
                gpu=gpu_pool[i % len(gpu_pool)],
                dry_run=args.dry_run,
            ): q
            for i, q in enumerate(queries)
        }
        for i, fut in enumerate(as_completed(futs), 1):
            q = futs[fut]
            try:
                m = fut.result()
                metas.append(m)
                print(
                    f"  [{i}/{len(queries)}] {m.query_id} [{m.functional_type}] -> {m.status} "
                    f"(lancedb={m.lancedb_built}, {m.duration_ms or 0}ms)"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(queries)}] {q['query_id']} -> EXCEPTION {exc}", file=sys.stderr)
                metas.append(
                    FuncMeta(q["query_id"], q.get("functional_type"), args.agent, args.model, "error", error=str(exc))
                )
    (run_dir / "run_metas.json").write_text(json.dumps([asdict(m) for m in metas], indent=2))
    print(f"\nDone. status: {dict(Counter(m.status for m in metas))}\n  -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
