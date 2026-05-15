# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`retriever skill-eval run` benchmark."""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from nemo_retriever.harness.artifacts import create_session_dir
from nemo_retriever.harness.config import REPO_ROOT
from nemo_retriever.skill_eval.dataset import DatasetEntry, load_config, load_eval_manifest
from nemo_retriever.skill_eval.report import overall_recall, write_summary
from nemo_retriever.skill_eval.runner import (
    CONDITIONS,
    cleanup_condition_workdir,
    run_condition,
    save_trial,
)

DEFAULT_ORDER = ("c1_base", "c2_retriever", "c3_retriever_skill")

app = typer.Typer(help="Benchmark Claude with vs. without the /nemo-retriever skill on a folder of PDFs.")
logger = logging.getLogger(__name__)


def _resolve_pdf_source(
    cfg: dict,
    domain: str,
) -> Path:
    pdf_dirs = cfg.get("pdf_dirs")
    if isinstance(pdf_dirs, dict):
        if domain not in pdf_dirs:
            raise typer.BadParameter(
                f"config 'pdf_dirs' is missing an entry for domain '{domain}'. "
                f"Known domains: {sorted(pdf_dirs.keys())}"
            )
        return Path(str(pdf_dirs[domain])).expanduser().resolve()
    if cfg.get("pdf_dir"):
        return Path(str(cfg["pdf_dir"])).expanduser().resolve()
    raise typer.BadParameter("config must define either 'pdf_dirs' (per-domain map) or 'pdf_dir'.")


def _build_judge(cfg: dict) -> Optional[Any]:
    """Construct an ``LLMJudge`` from ``cfg['judge']`` or return ``None``.

    Skips silently (with a console note) when the API key env var is unset, so
    runs work end-to-end without network access. Import is deferred so the
    ``litellm`` extra isn't required when judging is disabled.
    """
    judge_cfg = cfg.get("judge") or {}
    if not judge_cfg.get("enabled", True):
        typer.echo("Judge disabled by config (judge.enabled=false).")
        return None
    api_key_env = str(judge_cfg.get("api_key_env", "NVIDIA_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        typer.echo(f"Judge disabled: ${api_key_env} is not set in the environment.")
        return None
    try:
        from nemo_retriever.llm.clients.judge import LLMJudge
    except ImportError as exc:
        typer.echo(f"Judge disabled: failed to import LLMJudge ({exc}). Install nemo-retriever[llm].")
        return None
    judge = LLMJudge.from_kwargs(
        model=str(judge_cfg.get("model", "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")),
        api_base=judge_cfg.get("api_base"),
        api_key=api_key,
    )
    typer.echo(f"Judge enabled: model={judge.model}")
    return judge


def _resolve_domain_label(entries: list[DatasetEntry], cfg: dict, domain: str) -> str:
    """Pick a human-readable label for the setup prompt.

    Prefers the manifest-provided ``domain_label`` carried on the entry. Falls
    back to an optional ``domain_labels`` map in the config, then to ``"PDFs"``.
    """
    for e in entries:
        if e.domain == domain and e.domain_label:
            return e.domain_label
    labels = cfg.get("domain_labels")
    if isinstance(labels, dict) and domain in labels:
        return str(labels[domain])
    return "PDFs"


@app.command("run")
def run_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to a skill_eval.yaml; defaults to the packaged config (copy and edit it).",
    ),
    eval_manifest: Optional[Path] = typer.Option(
        None,
        "--eval-manifest",
        help="Path to an agent-eval manifest (JSON list). Overrides config.eval_manifest_path.",
    ),
    conditions: str = typer.Option(
        ",".join(DEFAULT_ORDER),
        "--conditions",
        help=(
            "Comma-separated conditions in execution order. Each (condition, domain) workdir is deleted after it runs, "
            "so only one LanceDB is on disk at a time."
        ),
    ),
    domains: Optional[str] = typer.Option(
        None,
        "--domains",
        help="Optional comma-separated list of domains to include. Defaults to all domains present in the dataset.",
    ),
    artifacts_root: Optional[Path] = typer.Option(
        None, "--artifacts-root", help="Override the artifact root; defaults to <repo>/nemo_retriever/artifacts/"
    ),
) -> None:
    """Run the benchmark across the dataset's domains × selected conditions, sequentially."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if shutil.which("claude") is None:
        typer.echo("Error: `claude` CLI is not on PATH; install Claude Code first.", err=True)
        raise typer.Exit(code=2)

    cfg = load_config(config)
    selected = [c.strip() for c in conditions.split(",") if c.strip()]
    for c in selected:
        if c not in CONDITIONS:
            typer.echo(f"Error: unknown condition '{c}'. Choose from {CONDITIONS}.", err=True)
            raise typer.Exit(code=2)

    manifest_path = eval_manifest or cfg.get("eval_manifest_path")
    if not manifest_path:
        typer.echo("Error: config is missing 'eval_manifest_path' and --eval-manifest was not provided.", err=True)
        raise typer.Exit(code=2)
    entries = load_eval_manifest(Path(str(manifest_path)).expanduser().resolve())
    typer.echo(f"Loaded {len(entries)} dataset entries.")

    by_domain: dict[str, list[DatasetEntry]] = defaultdict(list)
    for e in entries:
        by_domain[e.domain].append(e)

    if domains:
        wanted = {d.strip() for d in domains.split(",") if d.strip()}
        unknown = wanted - set(by_domain)
        if unknown:
            typer.echo(
                f"Error: --domains references unknown domains {sorted(unknown)}. " f"Available: {sorted(by_domain)}",
                err=True,
            )
            raise typer.Exit(code=2)
        by_domain = {d: by_domain[d] for d in wanted}

    domain_order = sorted(by_domain.keys())
    typer.echo(f"Domains in this run: {domain_order} ({sum(len(v) for v in by_domain.values())} entries total)")

    skill_source = Path(
        str(cfg.get("skill_source_dir") or REPO_ROOT / ".claude" / "skills" / "nemo-retriever")
    ).expanduser()
    workdir_root = Path(str(cfg.get("per_trial_workdir_root", "/tmp/skill_eval"))).expanduser()
    workdir_root.mkdir(parents=True, exist_ok=True)
    model = str(cfg.get("agent_model", "claude-opus-4-7"))
    budget = float(cfg.get("per_trial_budget_usd", 5.0))
    timeout = int(cfg.get("per_trial_timeout_s", 600))
    testdata_prefixes_raw = cfg.get("testdata_prefixes") or []
    if not isinstance(testdata_prefixes_raw, list):
        typer.echo("Error: config 'testdata_prefixes' must be a list of strings.", err=True)
        raise typer.Exit(code=2)
    testdata_prefixes = tuple(str(p) for p in testdata_prefixes_raw)

    judge = _build_judge(cfg)

    base_dir = str(artifacts_root) if artifacts_root else None
    session_dir = create_session_dir("skilleval", base_dir=base_dir)
    typer.echo(f"Session dir: {session_dir}")

    (session_dir / "config.yaml").write_text(yaml.safe_dump(cfg, default_flow_style=False), encoding="utf-8")

    # Results are keyed (condition, domain) so the report can break out per-domain numbers.
    results_by_key: dict[tuple[str, str], list] = {}
    for cond in selected:
        for domain in domain_order:
            domain_entries = by_domain[domain]
            pdf_source = _resolve_pdf_source(cfg, domain)
            if not pdf_source.is_dir():
                typer.echo(
                    f"Error: PDF directory '{pdf_source}' for domain '{domain}' does not exist or is not a directory. "
                    f"Check the 'pdf_dirs' (or 'pdf_dir') setting in your config.",
                    err=True,
                )
                raise typer.Exit(code=2)
            domain_label = _resolve_domain_label(domain_entries, cfg, domain)
            typer.echo(
                f"Starting session for {cond}/{domain} — setup + {len(domain_entries)} query turns "
                f"(pdfs={pdf_source})"
            )
            workdir, results = run_condition(
                condition=cond,
                entries=domain_entries,
                workdir_root=workdir_root,
                pdf_source=pdf_source,
                skill_source=skill_source,
                model=model,
                budget_usd=budget,
                timeout_s=timeout,
                domain=domain,
                domain_label=domain_label,
                judge=judge,
                testdata_prefixes=testdata_prefixes,
            )
            for r in results:
                save_trial(r, session_dir)
                kind = "setup" if r.is_setup else f"entry_id={r.entry_id} query_id={r.query_id}"
                judge_str = "" if r.is_setup or r.judge_score is None else f" judge={r.judge_score}"
                typer.echo(
                    f"  turn {r.num_turns} [{domain}] {kind}: status={r.status} "
                    f"tokens(in/out/cache_r)={r.input_tokens}/{r.output_tokens}/{r.cache_read_input_tokens} "
                    f"cost=${r.total_cost_usd:.3f} retrieved={len(r.ranked_retrieved)}{judge_str}"
                )
            results_by_key[(cond, domain)] = results

            entries_by_id = {e.entry_id: e for e in domain_entries}
            scores = overall_recall(results, entries_by_id)
            typer.echo(
                f"\nRecall for {cond}/{domain}: "
                f"recall@1={scores['recall_1']:.3f}  "
                f"recall@5={scores['recall_5']:.3f}  "
                f"recall@10={scores['recall_10']:.3f}"
            )

            cleanup_condition_workdir(workdir)
            typer.echo(f"Cleaned up workdir for {cond}/{domain}\n")

    if judge is not None:
        typer.echo("\nLLM-as-judge scores (mean over query turns, 0-5 scale):")
        for cond in selected:
            scored: list[int] = []
            errored = 0
            for domain in domain_order:
                for r in results_by_key.get((cond, domain), []):
                    if r.is_setup:
                        continue
                    if r.judge_score is not None:
                        scored.append(int(r.judge_score))
                    elif r.judge_error:
                        errored += 1
            if scored:
                mean_score = sum(scored) / len(scored)
                typer.echo(f"  {cond}: mean={mean_score:.2f}  n={len(scored)}  errors={errored}")
            else:
                typer.echo(f"  {cond}: no scores  errors={errored} (check judge config / litellm install)")

    json_path, md_path = write_summary(
        session_dir=session_dir,
        results_by_key=results_by_key,
        entries=entries,
        config=cfg,
        config_path=str(config) if config else "<packaged default>",
    )
    typer.echo(f"\nWrote {json_path}")
    typer.echo(f"Wrote {md_path}")
    typer.echo("\nDone.")
