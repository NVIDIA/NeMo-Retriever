# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process scheduler for cron-based and GitHub-push triggered harness runs."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from nemo_retriever.harness import history

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None

GITHUB_POLL_SECS = int(os.environ.get("RETRIEVER_HARNESS_GITHUB_POLL_SECS", "60"))
GITHUB_TOKEN = os.environ.get("RETRIEVER_HARNESS_GITHUB_TOKEN", "")


_round_robin_index: int = 0


def match_runner(
    *,
    min_gpu_count: int | None = None,
    gpu_type_pattern: str | None = None,
    min_cpu_count: int | None = None,
    min_memory_gb: float | None = None,
    preferred_runner_id: int | None = None,
    preferred_runner_ids: list[int] | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any] | None:
    """Find the best matching online runner for the given resource requirements.

    When multiple candidates match, jobs are distributed across them in
    round-robin order so that a batch of jobs doesn't all land on the same
    machine.

    ``preferred_runner_ids`` (list) takes precedence over the legacy scalar
    ``preferred_runner_id``.  When a list is provided the search is limited
    to those runners (round-robin among the online ones).
    """
    global _round_robin_index
    runners = history.get_runners()
    online = [r for r in runners if r.get("status") == "online"]

    allowed_runner_ids: set[int] | None = None
    if dataset_name:
        ids = history.get_runner_ids_for_dataset_name(dataset_name)
        if ids is not None:
            allowed_runner_ids = set(ids)

    pref_set: set[int] | None = None
    if preferred_runner_ids:
        pref_set = set(preferred_runner_ids)
    elif preferred_runner_id:
        pref_set = {preferred_runner_id}

    if pref_set:
        pref_candidates = []
        for r in online:
            if r["id"] not in pref_set:
                continue
            if allowed_runner_ids is not None and r["id"] not in allowed_runner_ids:
                continue
            pref_candidates.append(r)
        if pref_candidates:
            pref_candidates.sort(key=lambda r: r.get("id", 0))
            chosen = pref_candidates[_round_robin_index % len(pref_candidates)]
            _round_robin_index += 1
            return chosen

    candidates = []
    for r in online:
        if allowed_runner_ids is not None and r["id"] not in allowed_runner_ids:
            continue
        if min_gpu_count and (r.get("gpu_count") or 0) < min_gpu_count:
            continue
        if gpu_type_pattern and gpu_type_pattern.lower() not in (r.get("gpu_type") or "").lower():
            continue
        if min_cpu_count and (r.get("cpu_count") or 0) < min_cpu_count:
            continue
        if min_memory_gb and (r.get("memory_gb") or 0) < min_memory_gb:
            continue
        candidates.append(r)

    if not candidates:
        return None

    candidates.sort(key=lambda r: r.get("id", 0))
    chosen = candidates[_round_robin_index % len(candidates)]
    _round_robin_index += 1
    return chosen


def _resolve_dataset_config(dataset_name: str) -> tuple[str | None, dict[str, Any] | None]:
    """Look up the filesystem path and full config overrides for a dataset."""
    managed = history.get_dataset_by_name(dataset_name)
    if managed and managed.get("path"):
        overrides: dict[str, Any] = {"dataset_dir": managed["path"]}
        if managed.get("query_csv"):
            overrides["query_csv"] = managed["query_csv"]
        if managed.get("input_type"):
            overrides["input_type"] = managed["input_type"]
        if managed.get("recall_required") is not None:
            overrides["recall_required"] = managed["recall_required"]
        if managed.get("recall_match_mode"):
            overrides["recall_match_mode"] = managed["recall_match_mode"]
        if managed.get("recall_adapter"):
            overrides["recall_adapter"] = managed["recall_adapter"]
        return managed["path"], overrides

    return None, None


def _resolve_preset_overrides(preset_name: str | None) -> dict[str, Any]:
    """Look up a managed preset and return its config + overrides merged together."""
    if not preset_name:
        return {}
    managed = history.get_preset_by_name(preset_name)
    if not managed:
        return {}
    result: dict[str, Any] = {}
    cfg = managed.get("config")
    if isinstance(cfg, dict):
        result.update(cfg)
    ovr = managed.get("overrides")
    if isinstance(ovr, dict):
        result.update(ovr)
    return result


MAX_PENDING_PER_SCHEDULE = 3


def _resolve_run_code_ref_sha() -> tuple[str | None, str | None]:
    """Resolve the portal ``run_code_ref`` setting to a concrete commit SHA.

    Returns ``(sha, ref)`` so that every job in a batch can be pinned to
    the same commit even when the symbolic ref (e.g. ``nvidia/main``)
    advances between job executions.
    """
    ref = history.get_portal_setting("run_code_ref")
    if not ref:
        return None, None
    try:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        if "/" in ref and not ref.startswith("origin/"):
            remote = ref.split("/")[0]
            subprocess.run(
                ["git", "fetch", remote, "--prune"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=env,
            )
        else:
            subprocess.run(
                ["git", "fetch", "--all", "--prune"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=env,
            )
        result = subprocess.run(
            ["git", "rev-parse", ref],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env=env,
        )
        return result.stdout.strip(), ref
    except Exception as exc:
        logger.warning("Failed to resolve run_code_ref '%s' to SHA: %s", ref, exc)
        return ref, ref


def _enforce_backlog_limit(schedule_id: int) -> None:
    """Cancel the oldest pending jobs for a schedule if the backlog exceeds the limit."""
    pending = history.get_pending_jobs_for_schedule(schedule_id)
    if len(pending) <= MAX_PENDING_PER_SCHEDULE:
        return
    to_cancel = pending[: len(pending) - MAX_PENDING_PER_SCHEDULE]
    for job in to_cancel:
        history.cancel_job(job["id"], reason="Cancelled — schedule backlog exceeded limit")
        logger.info(
            "Cancelled backlogged job %s for schedule #%s (keeping newest %d)",
            job["id"],
            schedule_id,
            MAX_PENDING_PER_SCHEDULE,
        )


def _dispatch_schedule(
    schedule: dict[str, Any],
    trigger_source: str = "scheduled",
    git_commit: str | None = None,
    git_ref: str | None = None,
) -> dict[str, Any] | None:
    """Create pending job(s) for a schedule.

    When the schedule references a ``preset_matrix``, all (dataset, preset)
    combinations from the matrix are dispatched as individual jobs.  Otherwise
    a single job is created for the schedule's dataset + preset.

    If a preferred runner or resource requirements are specified, attempt to
    match a runner now and assign the job directly.  Otherwise (or if no
    matching runner is currently online) the job is created as *pending* with
    ``assigned_runner_id = None`` so that any available runner can pick it up
    on its next heartbeat / work-poll.

    After creating the job(s), enforces a backlog limit of
    ``MAX_PENDING_PER_SCHEDULE`` pending jobs per schedule — the oldest
    pending jobs are cancelled if the limit is exceeded.
    """
    if not git_commit:
        git_commit, git_ref = _resolve_run_code_ref_sha()

    matrix_name = schedule.get("preset_matrix")
    if matrix_name:
        return _dispatch_schedule_matrix(schedule, matrix_name, trigger_source, git_commit, git_ref)

    runner = match_runner(
        min_gpu_count=schedule.get("min_gpu_count"),
        gpu_type_pattern=schedule.get("gpu_type_pattern"),
        min_cpu_count=schedule.get("min_cpu_count"),
        min_memory_gb=schedule.get("min_memory_gb"),
        preferred_runner_id=schedule.get("preferred_runner_id"),
        preferred_runner_ids=schedule.get("preferred_runner_ids"),
        dataset_name=schedule.get("dataset"),
    )

    dataset_path, dataset_overrides = _resolve_dataset_config(schedule.get("dataset", ""))
    preset_overrides = _resolve_preset_overrides(schedule.get("preset"))
    merged_overrides = {**(dataset_overrides or {}), **preset_overrides}

    job_data: dict[str, Any] = {
        "schedule_id": schedule["id"],
        "trigger_source": trigger_source,
        "dataset": schedule["dataset"],
        "dataset_path": dataset_path,
        "dataset_overrides": merged_overrides if merged_overrides else None,
        "preset": schedule.get("preset"),
        "config": schedule.get("config"),
        "assigned_runner_id": runner["id"] if runner else None,
        "git_commit": git_commit,
        "git_ref": git_ref,
        "tags": schedule.get("tags") or [],
    }

    job = history.create_job(job_data)
    history.mark_schedule_triggered(schedule["id"])

    if runner:
        logger.info(
            "Dispatched job %s for schedule '%s' -> runner #%s",
            job["id"],
            schedule.get("name"),
            runner["id"],
        )
    else:
        logger.info(
            "Queued job %s for schedule '%s' (no runner matched yet — will be picked up on next heartbeat)",
            job["id"],
            schedule.get("name"),
        )

    _enforce_backlog_limit(schedule["id"])

    return job


def _dispatch_schedule_matrix(
    schedule: dict[str, Any],
    matrix_name: str,
    trigger_source: str,
    git_commit: str | None,
    git_ref: str | None,
) -> dict[str, Any] | None:
    """Expand a preset matrix into individual jobs for every (dataset, preset) pair."""
    matrix = history.get_preset_matrix_by_name(matrix_name)
    if not matrix:
        logger.warning(
            "Schedule '%s' references unknown preset_matrix '%s' — skipping", schedule.get("name"), matrix_name
        )
        return None

    dataset_names: list[str] = matrix.get("dataset_names") or []
    preset_names: list[str] = matrix.get("preset_names") or []
    if not dataset_names or not preset_names:
        logger.warning("Preset matrix '%s' is empty — skipping dispatch", matrix_name)
        return None

    schedule_tags = schedule.get("tags") or []
    matrix_tags = matrix.get("tags") or []
    merged_tags = list(dict.fromkeys(schedule_tags + matrix_tags))
    first_job = None

    effective_preferred_runner = matrix.get("preferred_runner_id") or schedule.get("preferred_runner_id")
    effective_preferred_runners = schedule.get("preferred_runner_ids") or []
    effective_gpu_type = matrix.get("gpu_type_filter") or schedule.get("gpu_type_pattern")

    for ds_name in dataset_names:
        dataset_path, dataset_overrides = _resolve_dataset_config(ds_name)

        for pr_name in preset_names:
            runner = match_runner(
                min_gpu_count=schedule.get("min_gpu_count"),
                gpu_type_pattern=effective_gpu_type,
                min_cpu_count=schedule.get("min_cpu_count"),
                min_memory_gb=schedule.get("min_memory_gb"),
                preferred_runner_id=effective_preferred_runner,
                preferred_runner_ids=effective_preferred_runners or None,
                dataset_name=ds_name,
            )
            preset_overrides = _resolve_preset_overrides(pr_name)
            merged_overrides = {**(dataset_overrides or {}), **preset_overrides}

            job = history.create_job(
                {
                    "schedule_id": schedule["id"],
                    "trigger_source": trigger_source,
                    "dataset": ds_name,
                    "dataset_path": dataset_path,
                    "dataset_overrides": merged_overrides if merged_overrides else None,
                    "preset": pr_name,
                    "config": schedule.get("config"),
                    "assigned_runner_id": runner["id"] if runner else None,
                    "git_commit": git_commit,
                    "git_ref": git_ref,
                    "tags": merged_tags,
                }
            )
            if first_job is None:
                first_job = job
            logger.info(
                "Matrix dispatch: job %s for schedule '%s' (dataset=%s, preset=%s)",
                job["id"],
                schedule.get("name"),
                ds_name,
                pr_name,
            )

    history.mark_schedule_triggered(schedule["id"])
    _enforce_backlog_limit(schedule["id"])
    return first_job


def _on_schedule_fire(schedule_id: int) -> None:
    """Callback invoked by APScheduler when a cron trigger fires."""
    schedule = history.get_schedule_by_id(schedule_id)
    if not schedule or not schedule.get("enabled"):
        return
    _dispatch_schedule(schedule, trigger_source="scheduled")


def _parse_cron_expression(expr: str) -> dict[str, str]:
    """Parse a 5-field cron expression into APScheduler CronTrigger kwargs."""
    parts = expr.strip().split()
    if len(parts) < 5:
        parts.extend(["*"] * (5 - len(parts)))
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


def _github_poll() -> None:
    """Poll GitHub for new commits on branches tracked by github_push schedules."""
    schedules = history.get_enabled_schedules(trigger_type="github_push")
    for sched in schedules:
        repo = sched.get("github_repo")
        branch = sched.get("github_branch") or "main"
        if not repo:
            continue

        try:
            sha = _git_ls_remote(repo, branch)
        except Exception as exc:
            logger.debug("git ls-remote failed for %s: %s", repo, exc)
            continue

        if not sha:
            continue

        last_sha = sched.get("github_last_sha")
        if sha == last_sha:
            continue

        logger.info("New commit on %s/%s: %s (was %s)", repo, branch, sha[:12], (last_sha or "none")[:12])
        history.update_schedule(sched["id"], {"github_last_sha": sha})
        _dispatch_schedule(sched, trigger_source="github_push", git_commit=sha, git_ref=branch)


def _git_ls_remote(repo: str, branch: str) -> str | None:
    """Get the latest commit SHA for a branch using git ls-remote."""
    if not repo.startswith(("https://", "git@")):
        repo = f"https://github.com/{repo}.git"

    env = dict(os.environ)
    if GITHUB_TOKEN:
        env["GIT_ASKPASS"] = "echo"
        env["GIT_TERMINAL_PROMPT"] = "0"
        if repo.startswith("https://"):
            repo = repo.replace("https://", f"https://{GITHUB_TOKEN}@", 1)

    try:
        result = subprocess.run(
            ["git", "ls-remote", repo, f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 1:
            return parts[0]
    return None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def start_scheduler() -> None:
    """Start the background scheduler and load all enabled cron schedules."""
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler(daemon=True)

    schedules = history.get_enabled_schedules(trigger_type="cron")
    for sched in schedules:
        _add_cron_job(sched)

    _scheduler.add_job(
        _github_poll,
        trigger=IntervalTrigger(seconds=GITHUB_POLL_SECS),
        id="__github_poller__",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Scheduler started with %d cron schedule(s), GitHub polling every %ds", len(schedules), GITHUB_POLL_SECS
    )


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")


def _add_cron_job(schedule: dict[str, Any]) -> None:
    """Register an APScheduler cron job for a schedule."""
    if _scheduler is None:
        return
    cron_expr = schedule.get("cron_expression")
    if not cron_expr:
        return
    try:
        cron_kwargs = _parse_cron_expression(cron_expr)
        _scheduler.add_job(
            _on_schedule_fire,
            trigger=CronTrigger(**cron_kwargs),
            args=[schedule["id"]],
            id=f"schedule_{schedule['id']}",
            replace_existing=True,
        )
    except Exception as exc:
        logger.error("Failed to add cron job for schedule %s: %s", schedule["id"], exc)


def sync_schedule(schedule_id: int) -> None:
    """Called when a schedule is created/updated/deleted — sync the APScheduler job."""
    if _scheduler is None:
        return

    job_id = f"schedule_{schedule_id}"

    schedule = history.get_schedule_by_id(schedule_id)
    if schedule is None or not schedule.get("enabled"):
        try:
            _scheduler.remove_job(job_id)
        except Exception:
            pass
        return

    if schedule.get("trigger_type") == "cron" and schedule.get("cron_expression"):
        _add_cron_job(schedule)
    else:
        try:
            _scheduler.remove_job(job_id)
        except Exception:
            pass


def trigger_schedule_now(schedule_id: int) -> dict[str, Any] | None:
    """Manually fire a schedule immediately, bypassing the cron timer."""
    schedule = history.get_schedule_by_id(schedule_id)
    if not schedule:
        return None
    trigger_source = "scheduled" if schedule.get("trigger_type") == "cron" else "github_push"
    return _dispatch_schedule(schedule, trigger_source=trigger_source)


def handle_github_webhook(repo: str, branch: str, commit_sha: str) -> list[dict[str, Any]]:
    """Process a GitHub push webhook event and dispatch matching schedules."""
    schedules = history.get_enabled_schedules(trigger_type="github_push")
    dispatched = []
    for sched in schedules:
        sched_repo = (sched.get("github_repo") or "").rstrip("/").lower()
        target_branch = sched.get("github_branch") or "main"

        repo_match = repo.rstrip("/").lower().endswith(sched_repo) or sched_repo.endswith(repo.rstrip("/").lower())
        if not repo_match:
            normalized_repo = repo.rstrip("/").lower().replace("https://github.com/", "").replace(".git", "")
            normalized_sched = sched_repo.replace("https://github.com/", "").replace(".git", "")
            repo_match = normalized_repo == normalized_sched

        if not repo_match or branch != target_branch:
            continue

        history.update_schedule(sched["id"], {"github_last_sha": commit_sha})
        job = _dispatch_schedule(sched, trigger_source="github_push", git_commit=commit_sha, git_ref=branch)
        if job:
            dispatched.append(job)

    return dispatched
