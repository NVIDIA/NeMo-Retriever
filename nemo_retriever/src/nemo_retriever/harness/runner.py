# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner agent that registers with a harness portal manager and sends heartbeats."""

from __future__ import annotations

import json as json_module
import logging
import signal
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

import typer

logger = logging.getLogger(__name__)


def _http_json(url: str, data: dict[str, Any] | None, method: str, timeout: int = 10) -> dict[str, Any]:
    body = json_module.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json_module.loads(resp.read().decode("utf-8"))


def _post_json(url: str, data: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    return _http_json(url, data, "POST", timeout)


def _put_json(url: str, data: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    return _http_json(url, data, "PUT", timeout)


def _get_json(url: str, timeout: int = 10) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json_module.loads(resp.read().decode("utf-8"))


def _execute_job_on_runner(base_url: str, job: dict[str, Any]) -> None:
    """Claim a job, execute it locally, and report results back."""
    job_id = job["id"]
    try:
        _post_json(f"{base_url}/api/jobs/{job_id}/claim", {})
    except Exception as exc:
        logger.warning("Failed to claim job %s: %s", job_id, exc)
        return

    dataset_value = job.get("dataset_path") or job["dataset"]
    overrides = job.get("dataset_overrides") or {}
    logger.info(
        "Executing job %s (dataset=%s, path=%s, preset=%s)",
        job_id,
        job.get("dataset"),
        dataset_value,
        job.get("preset"),
    )
    try:
        from nemo_retriever.harness.run import _run_entry

        result = _run_entry(
            run_name=None,
            config_file=job.get("config"),
            session_dir=None,
            dataset=dataset_value,
            preset=job.get("preset"),
            sweep_overrides=overrides if overrides else None,
            tags=job.get("tags"),
        )
        success = bool(result.get("success"))
        _post_json(f"{base_url}/api/jobs/{job_id}/complete", {"success": success, "result": result})
        logger.info("Job %s completed (success=%s)", job_id, success)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job_id, exc, tb)
        try:
            _post_json(f"{base_url}/api/jobs/{job_id}/complete", {"success": False, "error": f"{exc}\n\n{tb}"})
        except Exception:
            pass


def runner_start_command(
    name: str | None = typer.Option(None, "--name", help="Runner name. Defaults to hostname."),
    manager_url: str | None = typer.Option(None, "--manager-url", help="Portal URL to register this runner with."),
    heartbeat_interval: int = typer.Option(30, "--heartbeat-interval", help="Heartbeat interval in seconds."),
    tag: list[str] = typer.Option([], "--tag", help="Runner tags. Repeatable."),
) -> None:
    """Start a harness runner and optionally register with a portal manager."""
    from nemo_retriever.harness.run import _collect_run_metadata

    meta = _collect_run_metadata()
    runner_name = name or meta.get("host", "unknown")

    typer.echo(f"Runner: {runner_name}")
    typer.echo(f"  Hostname : {meta.get('host')}")
    typer.echo(f"  CPU      : {meta.get('cpu_count') or 'N/A'} cores")
    typer.echo(f"  Memory   : {meta.get('memory_gb') or 'N/A'} GB")
    typer.echo(f"  GPU      : {meta.get('gpu_type') or 'N/A'} (x{meta.get('gpu_count') or 0})")
    typer.echo(f"  Python   : {meta.get('python_version')}")

    runner_id: int | None = None
    base_url: str | None = None

    if manager_url:
        base_url = manager_url.rstrip("/")
        payload: dict[str, Any] = {
            "name": runner_name,
            "hostname": meta.get("host"),
            "gpu_type": meta.get("gpu_type"),
            "gpu_count": meta.get("gpu_count"),
            "cpu_count": meta.get("cpu_count"),
            "memory_gb": meta.get("memory_gb"),
            "status": "online",
            "tags": tag or [],
            "metadata": {
                "cuda_driver": meta.get("cuda_driver"),
                "ray_version": meta.get("ray_version"),
                "python_version": meta.get("python_version"),
            },
        }
        typer.echo(f"\nRegistering with {base_url} ...")
        try:
            result = _post_json(f"{base_url}/api/runners", payload)
            runner_id = result.get("id")
            typer.echo(f"Registered as runner #{runner_id}")
        except Exception as exc:
            typer.echo(f"Warning: Failed to register — {exc}", err=True)
            typer.echo("Runner will continue in standalone mode.", err=True)
    else:
        typer.echo("\nNo --manager-url provided; running in standalone mode.")

    typer.echo(f"\nRunner is active (heartbeat every {heartbeat_interval}s). Press Ctrl+C to stop.\n")

    stop = False
    active_job_thread: threading.Thread | None = None

    def _handle_signal(sig: int, frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop:
            time.sleep(heartbeat_interval)
            if base_url and runner_id:
                heartbeat_job = None
                try:
                    hb_resp = _post_json(f"{base_url}/api/runners/{runner_id}/heartbeat", {})
                    if hb_resp and hb_resp.get("job"):
                        heartbeat_job = hb_resp["job"]
                except Exception:
                    pass

                if active_job_thread is None or not active_job_thread.is_alive():
                    active_job_thread = None
                    work = heartbeat_job
                    if not work:
                        try:
                            work = _get_json(f"{base_url}/api/runners/{runner_id}/work")
                        except urllib.error.HTTPError:
                            work = None
                        except Exception as exc:
                            logger.debug("Work poll error: %s", exc)
                            work = None
                    if work and work.get("id"):
                        active_job_thread = threading.Thread(
                            target=_execute_job_on_runner,
                            args=(base_url, work),
                            daemon=True,
                        )
                        active_job_thread.start()
    finally:
        if base_url and runner_id:
            typer.echo("\nDeregistering runner...")
            try:
                _put_json(f"{base_url}/api/runners/{runner_id}", {"status": "offline"})
            except Exception:
                pass
        typer.echo("Runner stopped.")
