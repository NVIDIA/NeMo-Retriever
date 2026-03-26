# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
from importlib import metadata
import json
import os
import pty
import re
import select
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from nemo_retriever.harness.artifacts import (
    create_run_artifact_dir,
    create_session_dir,
    last_commit,
    now_timestr,
    write_json,
    write_session_summary,
)
from nemo_retriever.harness.config import (
    DEFAULT_NIGHTLY_CONFIG_PATH,
    HarnessConfig,
    TUNING_FIELDS,
    load_harness_config,
    load_nightly_config,
)
from nemo_retriever.harness.parsers import StreamMetrics
from nemo_retriever.harness.recall_adapters import prepare_recall_query_file
from nemo_retriever.utils.input_files import resolve_input_files

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _collect_gpu_metadata() -> tuple[int | None, str | None, str | None]:
    """Return ``(gpu_count, cuda_driver_version, gpu_name)``."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None, None

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    combined_output = f"{result.stdout}\n{result.stderr}"
    if "No devices were found" in combined_output:
        return 0, None, None
    if result.returncode != 0:
        return None, None, None
    if not output_lines:
        return 0, None, None

    parts = [p.strip() for p in output_lines[0].split(",", 1)]
    gpu_name = parts[0] if parts else None
    driver = parts[1].strip() if len(parts) > 1 else None
    return len(output_lines), driver, gpu_name


def _collect_run_metadata() -> dict[str, Any]:
    try:
        host = socket.gethostname().strip() or "unknown"
    except OSError:
        host = "unknown"

    version_info = getattr(sys, "version_info", None)
    if version_info is None:
        python_version = "unknown"
    else:
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    try:
        ray_version = metadata.version("ray")
    except metadata.PackageNotFoundError:
        ray_version = "unknown"

    gpu_count, cuda_driver, gpu_type = _collect_gpu_metadata()

    try:
        import psutil

        memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        memory_gb = None

    ray_dashboard_url: str | None = None
    try:
        import ray

        if ray.is_initialized():
            ctx = ray.get_runtime_context()
            ray_dashboard_url = getattr(ctx, "dashboard_url", None) or None
            if not ray_dashboard_url:
                ray_dashboard_url = os.environ.get("RAY_DASHBOARD_URL") or None
    except Exception:
        pass

    return {
        "host": host,
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
        "cuda_driver": cuda_driver,
        "ray_version": ray_version,
        "python_version": python_version,
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "ray_dashboard_url": ray_dashboard_url,
    }


def _get_routable_ip() -> str:
    """Return this machine's routable IP address (not 127.0.0.1).

    Opens a UDP socket to a public DNS address (no data is sent) so the OS
    selects the correct outbound interface.  Falls back to hostname resolution.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _resolve_localhost(host: str) -> str:
    """Replace loopback addresses with the machine's routable IP."""
    if host.lower() in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        return _get_routable_ip()
    return host


def _derive_ray_dashboard_url(ray_address: str) -> str | None:
    """Best-effort derivation of the Ray dashboard URL from a cluster address.

    Ray dashboard defaults to port 8265 on the head node.  We attempt to
    extract the hostname from common address formats (``ray://host:port``,
    ``host:port``, or ``auto``) and build the dashboard URL.
    """
    addr = ray_address.strip()
    if addr.lower() == "auto":
        return f"http://{_get_routable_ip()}:8265"
    addr = re.sub(r"^ray://", "", addr, flags=re.IGNORECASE)
    host = addr.split(":")[0] if ":" in addr else addr
    if not host:
        return None
    host = _resolve_localhost(host)
    return f"http://{host}:8265"


def _normalize_tags(tags: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw in tags or []:
        tag = str(raw).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)

    return normalized


def _normalize_recall_metric_key(key: str) -> str:
    metric = str(key).strip().lower()
    return metric.replace("@", "_").replace("-", "_")


def _safe_pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(path))
        try:
            try:
                count = int(len(doc))
            except Exception:
                count = int(doc.get_page_count())  # type: ignore[attr-defined]
        finally:
            try:
                doc.close()
            except Exception:
                pass
        return max(count, 0)
    except Exception:
        return None


def _resolve_summary_metrics(
    cfg: HarnessConfig,
    metrics_payload: dict[str, Any],
    runtime_summary: dict[str, Any] | None,
    subprocess_elapsed_secs: float | None = None,
) -> dict[str, Any]:
    summary_metrics: dict[str, Any] = {
        "pages": metrics_payload.get("pages"),
        "files": metrics_payload.get("files"),
        "ingest_secs": metrics_payload.get("ingest_secs"),
        "pages_per_sec_ingest": metrics_payload.get("pages_per_sec_ingest"),
        "recall_5": metrics_payload.get("recall_5"),
        "ndcg_10": metrics_payload.get("ndcg_10"),
    }

    if summary_metrics["pages"] is None and isinstance(runtime_summary, dict):
        runtime_pages = runtime_summary.get("num_pages")
        if runtime_pages is None:
            runtime_pages = runtime_summary.get("input_pages")
        if runtime_pages is not None:
            try:
                summary_metrics["pages"] = int(runtime_pages)
            except (TypeError, ValueError):
                summary_metrics["pages"] = None

    # Fallback: count input files from the dataset directory.
    if summary_metrics["files"] is None:
        try:
            input_files = resolve_input_files(Path(cfg.dataset_dir), cfg.input_type)
            if input_files:
                summary_metrics["files"] = len(input_files)
        except Exception:
            pass

    if summary_metrics["pages"] is None and cfg.input_type == "pdf":
        total_pages = 0
        counted_any = False
        for path in resolve_input_files(Path(cfg.dataset_dir), cfg.input_type):
            page_count = _safe_pdf_page_count(path)
            if page_count is None:
                continue
            counted_any = True
            total_pages += page_count
        if counted_any:
            summary_metrics["pages"] = total_pages

    # Use subprocess wall-clock time as fallback when the stream parser
    # couldn't extract the ingest time (e.g. print_run_summary was skipped).
    if summary_metrics["ingest_secs"] is None and subprocess_elapsed_secs is not None and subprocess_elapsed_secs > 0:
        summary_metrics["ingest_secs"] = subprocess_elapsed_secs

    if summary_metrics["pages_per_sec_ingest"] is None:
        pages = summary_metrics.get("pages")
        ingest_secs = summary_metrics.get("ingest_secs")
        if pages is not None and ingest_secs not in {None, 0, 0.0}:
            try:
                summary_metrics["pages_per_sec_ingest"] = round(float(pages) / float(ingest_secs), 2)
            except (TypeError, ValueError, ZeroDivisionError):
                summary_metrics["pages_per_sec_ingest"] = None

    return summary_metrics


def _resolve_lancedb_uri(cfg: HarnessConfig, artifact_dir: Path) -> str:
    raw = str(cfg.lancedb_uri or "lancedb")
    if raw == "lancedb":
        return str((artifact_dir / "lancedb").resolve())
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return str(p)


def _build_command(
    cfg: HarnessConfig, artifact_dir: Path, run_id: str
) -> tuple[list[str], Path, Path, Path | None, Path]:
    runtime_dir = artifact_dir / "runtime_metrics"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if cfg.write_detection_file:
        detection_summary_file = artifact_dir / "detection_summary.json"
    else:
        detection_summary_file = runtime_dir / ".detection_summary.json"
    metrics_output_file = runtime_dir / f"{run_id}.metrics.json"
    effective_query_csv: Path | None = None

    cmd = [
        sys.executable,
        "-m",
        "nemo_retriever.examples.batch_pipeline",
        str(Path(cfg.dataset_dir).resolve()),
        "--input-type",
        cfg.input_type,
        "--evaluation-mode",
        cfg.evaluation_mode,
        "--pdf-extract-tasks",
        str(cfg.pdf_extract_workers),
        "--pdf-extract-cpus-per-task",
        str(cfg.pdf_extract_num_cpus),
        "--pdf-extract-batch-size",
        str(cfg.pdf_extract_batch_size),
        "--pdf-split-batch-size",
        str(cfg.pdf_split_batch_size),
        "--page-elements-batch-size",
        str(cfg.page_elements_batch_size),
        "--page-elements-actors",
        str(cfg.page_elements_workers),
        "--ocr-actors",
        str(cfg.ocr_workers),
        "--ocr-batch-size",
        str(cfg.ocr_batch_size),
        "--embed-actors",
        str(cfg.embed_workers),
        "--embed-batch-size",
        str(cfg.embed_batch_size),
        "--page-elements-cpus-per-actor",
        str(cfg.page_elements_cpus_per_actor),
        "--ocr-cpus-per-actor",
        str(cfg.ocr_cpus_per_actor),
        "--embed-cpus-per-actor",
        str(cfg.embed_cpus_per_actor),
        "--page-elements-gpus-per-actor",
        str(cfg.gpu_page_elements),
        "--ocr-gpus-per-actor",
        str(cfg.gpu_ocr),
        "--embed-gpus-per-actor",
        str(cfg.gpu_embed),
        "--embed-model-name",
        cfg.embed_model_name,
        "--embed-modality",
        cfg.embed_modality,
        "--embed-granularity",
        cfg.embed_granularity,
        "--runtime-metrics-dir",
        str(runtime_dir),
        "--runtime-metrics-prefix",
        run_id,
        "--detection-summary-file",
        str(detection_summary_file),
        "--metrics-output-file",
        str(metrics_output_file),
        "--lancedb-uri",
        _resolve_lancedb_uri(cfg, artifact_dir),
    ]

    if cfg.evaluation_mode == "beir":
        cmd += [
            "--beir-loader",
            str(cfg.beir_loader),
            "--beir-dataset-name",
            str(cfg.beir_dataset_name or cfg.dataset_label),
            "--beir-split",
            cfg.beir_split,
            "--beir-doc-id-field",
            cfg.beir_doc_id_field,
        ]
        if cfg.beir_query_language:
            cmd += ["--beir-query-language", cfg.beir_query_language]
        for k in cfg.beir_ks:
            cmd += ["--beir-k", str(int(k))]
    else:
        effective_query_csv = prepare_recall_query_file(
            query_csv=Path(cfg.query_csv) if cfg.query_csv else None,
            recall_adapter=cfg.recall_adapter,
            output_dir=runtime_dir,
        )
        cmd += [
            "--query-csv",
            str(effective_query_csv),
            "--recall-match-mode",
            cfg.recall_match_mode,
            "--no-recall-details",
        ]

    cmd += ["--extract-page-as-image" if cfg.extract_page_as_image else "--no-extract-page-as-image"]
    if cfg.extract_infographics:
        cmd += ["--extract-infographics"]
    if cfg.embed_modality:
        cmd += ["--structured-elements-modality", cfg.embed_modality]
    if cfg.ray_address:
        cmd += ["--ray-address", cfg.ray_address]
    if cfg.hybrid:
        cmd += ["--hybrid"]

    return cmd, runtime_dir, detection_summary_file, effective_query_csv, metrics_output_file


def _evaluate_run_outcome(
    process_rc: int,
    evaluation_mode: str,
    recall_required: bool,
    recall_metrics: dict[str, float],
    evaluation_metrics: dict[str, float] | None = None,
) -> tuple[int, str, bool]:
    if process_rc != 0:
        reason = f"subprocess_exit_{process_rc}"
        return process_rc, reason, False
    if evaluation_mode == "beir" and not (evaluation_metrics or {}):
        return 97, "missing_beir_metrics", False
    if evaluation_mode == "recall" and recall_required and not recall_metrics and not (evaluation_metrics or {}):
        return 98, "missing_recall_metrics", False
    return 0, "", True


_FAIL_SEPARATOR = "\u2500" * 72


def _print_failure_report(
    result: dict[str, Any],
    command_text: str,
    artifact_dir: Path,
    tail_lines: list[str],
) -> None:
    """Pretty-print a detailed failure report so the root cause is easy to find."""
    reason = result.get("failure_reason") or "unknown"
    rc = result.get("return_code", "?")
    cfg = result.get("test_config", {})
    meta = result.get("run_metadata", {})

    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    lines: list[str] = []
    lines.append("")
    lines.append(f"{RED}{BOLD}{_FAIL_SEPARATOR}{RESET}")
    lines.append(f"{RED}{BOLD}  RUN FAILED  {RESET}")
    lines.append(f"{RED}{BOLD}{_FAIL_SEPARATOR}{RESET}")
    lines.append("")

    lines.append(f"  {BOLD}Failure Reason :{RESET}  {RED}{reason}{RESET}")
    lines.append(f"  {BOLD}Return Code    :{RESET}  {rc}")
    lines.append("")

    lines.append(f"  {CYAN}{BOLD}Test Configuration{RESET}")
    lines.append(f"  {DIM}{'-' * 40}{RESET}")
    lines.append(f"  Dataset        :  {cfg.get('dataset_label', '\u2014')}")
    lines.append(f"  Dataset Dir    :  {cfg.get('dataset_dir', '\u2014')}")
    lines.append(f"  Preset         :  {cfg.get('preset', '\u2014')}")
    lines.append(f"  Input Type     :  {cfg.get('input_type', '\u2014')}")
    lines.append(f"  Recall Required:  {cfg.get('recall_required', False)}")
    lines.append(f"  Hybrid         :  {cfg.get('hybrid', False)}")
    lines.append(f"  Embed Model    :  {cfg.get('embed_model_name', '\u2014')}")
    lines.append("")

    lines.append(f"  {CYAN}{BOLD}Host Information{RESET}")
    lines.append(f"  {DIM}{'-' * 40}{RESET}")
    lines.append(f"  Hostname       :  {meta.get('host', '\u2014')}")
    lines.append(f"  GPU            :  {meta.get('gpu_type', '\u2014')} (x{meta.get('gpu_count', '?')})")
    lines.append(f"  CUDA Driver    :  {meta.get('cuda_driver', '\u2014')}")
    lines.append(f"  Python         :  {meta.get('python_version', '\u2014')}")
    lines.append(f"  CPU / Memory   :  {meta.get('cpu_count', '?')} cores / {meta.get('memory_gb', '?')} GB")
    lines.append("")

    lines.append(f"  {CYAN}{BOLD}Artifacts{RESET}")
    lines.append(f"  {DIM}{'-' * 40}{RESET}")
    lines.append(f"  Artifact Dir   :  {artifact_dir.resolve()}")
    lines.append(f"  Results JSON   :  {artifact_dir.resolve() / 'results.json'}")
    lines.append(f"  Command File   :  {artifact_dir.resolve() / 'command.txt'}")
    lines.append("")

    lines.append(f"  {CYAN}{BOLD}Command{RESET}")
    lines.append(f"  {DIM}{'-' * 40}{RESET}")
    # Wrap long commands for readability
    if len(command_text) > 120:
        lines.append(f"  {DIM}{command_text[:120]}...{RESET}")
        lines.append(f"  {DIM}(full command in {artifact_dir.resolve() / 'command.txt'}){RESET}")
    else:
        lines.append(f"  {DIM}{command_text}{RESET}")
    lines.append("")

    if tail_lines:
        lines.append(f"  {YELLOW}{BOLD}Last {len(tail_lines)} Lines of Output{RESET}")
        lines.append(f"  {DIM}{'-' * 40}{RESET}")
        for tl in tail_lines:
            lines.append(f"  {DIM}|{RESET} {tl}")
        lines.append("")

    lines.append(f"{RED}{BOLD}{_FAIL_SEPARATOR}{RESET}")
    lines.append("")

    typer.echo("\n".join(lines), err=True)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _consume_parseable_output(metrics: StreamMetrics, parse_buffer: str) -> str:
    while "\n" in parse_buffer:
        line, parse_buffer = parse_buffer.split("\n", 1)
        cleaned = ANSI_ESCAPE_RE.sub("", line)
        metrics.consume(cleaned + "\n")
    return parse_buffer


def _run_subprocess_with_tty(cmd: list[str], metrics: StreamMetrics) -> int:
    """
    Run command in a pseudo-terminal so Ray renders rich progress.

    We still parse lines from the PTY stream to extract benchmark metrics.
    """
    master_fd, slave_fd = pty.openpty()
    parse_buffer = ""
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=None,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    try:
        while True:
            read_fds, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd not in read_fds:
                if proc.poll() is not None:
                    break
                continue

            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                # PTY EOF on Linux often appears as EIO.
                if exc.errno == errno.EIO:
                    break
                raise

            if not chunk:
                break

            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()

            parse_buffer += text.replace("\r", "\n")
            parse_buffer = _consume_parseable_output(metrics, parse_buffer)

        if parse_buffer:
            cleaned_tail = ANSI_ESCAPE_RE.sub("", parse_buffer)
            metrics.consume(cleaned_tail)

        return proc.wait()
    finally:
        os.close(master_fd)


def _run_single(
    cfg: HarnessConfig,
    artifact_dir: Path,
    run_id: str,
    tags: list[str] | None = None,
    skip_local_history: bool = False,
) -> dict[str, Any]:
    cmd, runtime_dir, detection_summary_file, effective_query_csv, metrics_output_file = _build_command(
        cfg, artifact_dir, run_id
    )

    lancedb_path = Path(_resolve_lancedb_uri(cfg, artifact_dir))
    if lancedb_path.is_dir():
        typer.echo(f"Removing stale LanceDB directory: {lancedb_path}")
        shutil.rmtree(lancedb_path)
    lancedb_path.mkdir(parents=True, exist_ok=True)

    command_text = " ".join(shlex.quote(token) for token in cmd)
    (artifact_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    typer.echo(f"\n=== Running {run_id} ===")
    typer.echo(command_text)

    import time as _time

    metrics = StreamMetrics()
    _wall_start = _time.perf_counter()
    process_rc = _run_subprocess_with_tty(cmd, metrics)
    subprocess_elapsed_secs = round(_time.perf_counter() - _wall_start, 2)
    run_metadata = _collect_run_metadata()

    ray_addr = cfg.ray_address
    if ray_addr and ray_addr.lower() != "local":
        run_metadata["ray_cluster_mode"] = "existing"
        if not run_metadata.get("ray_dashboard_url"):
            dashboard = _derive_ray_dashboard_url(ray_addr)
            if dashboard:
                run_metadata["ray_dashboard_url"] = dashboard
    else:
        run_metadata["ray_cluster_mode"] = "local"

    runtime_summary_path = runtime_dir / f"{run_id}.runtime.summary.json"
    runtime_summary = _read_json_if_exists(runtime_summary_path)
    detection_summary = _read_json_if_exists(detection_summary_file)
    if not cfg.write_detection_file and detection_summary_file.exists():
        detection_summary_file.unlink()

    # ----- Metrics: prefer the structured JSON written by batch_pipeline -----
    pipeline_metrics = _read_json_if_exists(metrics_output_file)

    if isinstance(pipeline_metrics, dict) and (
        pipeline_metrics.get("recall_metrics") or pipeline_metrics.get("evaluation_metrics")
    ):
        recall_raw: dict[str, float] = pipeline_metrics.get("recall_metrics") or {}
        eval_raw: dict[str, float] = pipeline_metrics.get("evaluation_metrics") or {}
    else:
        recall_raw = dict(metrics.recall_metrics)
        eval_raw = dict(metrics.evaluation_metrics)

    recall_metrics_normalized: dict[str, float] = {}
    for key, val in recall_raw.items():
        recall_metrics_normalized[_normalize_recall_metric_key(key)] = val
    evaluation_metrics_normalized: dict[str, float] = {}
    for key, val in eval_raw.items():
        evaluation_metrics_normalized[_normalize_recall_metric_key(key)] = val

    effective_rc, failure_reason, success = _evaluate_run_outcome(
        process_rc=process_rc,
        evaluation_mode=cfg.evaluation_mode,
        recall_required=bool(cfg.recall_required),
        recall_metrics=recall_raw,
        evaluation_metrics=eval_raw,
    )

    pm_files = metrics.files
    pm_pages = metrics.pages
    pm_ingest_secs = metrics.ingest_secs
    pm_pps = metrics.pages_per_sec_ingest
    if isinstance(pipeline_metrics, dict):
        if pm_files is None and pipeline_metrics.get("files") is not None:
            pm_files = int(pipeline_metrics["files"])
        if pm_pages is None and pipeline_metrics.get("pages") is not None:
            pm_pages = int(pipeline_metrics["pages"])
        if pm_ingest_secs is None and pipeline_metrics.get("ingest_secs") is not None:
            pm_ingest_secs = float(pipeline_metrics["ingest_secs"])
        if pm_pps is None and pipeline_metrics.get("pages_per_sec_ingest") is not None:
            pm_pps = float(pipeline_metrics["pages_per_sec_ingest"])

    metrics_payload = {
        "files": pm_files,
        "pages": pm_pages,
        "ingest_secs": pm_ingest_secs,
        "pages_per_sec_ingest": pm_pps,
        **recall_metrics_normalized,
        **evaluation_metrics_normalized,
    }
    summary_metrics = _resolve_summary_metrics(cfg, metrics_payload, runtime_summary, subprocess_elapsed_secs)

    result_payload: dict[str, Any] = {
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "success": success,
        "return_code": effective_rc,
        "failure_reason": failure_reason or None,
        "test_config": {
            "dataset_label": cfg.dataset_label,
            "dataset_dir": cfg.dataset_dir,
            "preset": cfg.preset,
            "query_csv": cfg.query_csv,
            "effective_query_csv": str(effective_query_csv) if effective_query_csv is not None else None,
            "input_type": cfg.input_type,
            "recall_required": cfg.recall_required,
            "recall_match_mode": cfg.recall_match_mode,
            "recall_adapter": cfg.recall_adapter,
            "evaluation_mode": cfg.evaluation_mode,
            "beir_loader": cfg.beir_loader,
            "beir_dataset_name": cfg.beir_dataset_name,
            "beir_split": cfg.beir_split,
            "beir_query_language": cfg.beir_query_language,
            "beir_doc_id_field": cfg.beir_doc_id_field,
            "beir_ks": list(cfg.beir_ks),
            "ray_address": cfg.ray_address,
            "hybrid": cfg.hybrid,
            "embed_model_name": cfg.embed_model_name,
            "embed_modality": cfg.embed_modality,
            "embed_granularity": cfg.embed_granularity,
            "extract_page_as_image": cfg.extract_page_as_image,
            "extract_infographics": cfg.extract_infographics,
            "write_detection_file": cfg.write_detection_file,
            "lancedb_uri": _resolve_lancedb_uri(cfg, artifact_dir),
            "tuning": {field: getattr(cfg, field) for field in sorted(TUNING_FIELDS)},
        },
        "metrics": {
            "files": pm_files,
            "pages": pm_pages,
            "ingest_secs": pm_ingest_secs,
            "pages_per_sec_ingest": pm_pps,
            "rows_processed": metrics.rows_processed,
            "rows_per_sec_ingest": metrics.rows_per_sec_ingest,
            **recall_metrics_normalized,
            **evaluation_metrics_normalized,
        },
        "summary_metrics": summary_metrics,
        "run_metadata": run_metadata,
        "runtime_summary": runtime_summary,
        "detection_summary": detection_summary,
        "artifacts": {
            "command_file": str((artifact_dir / "command.txt").resolve()),
            "runtime_metrics_dir": str(runtime_dir.resolve()),
        },
    }
    if cfg.write_detection_file:
        result_payload["artifacts"]["detection_summary_file"] = str(detection_summary_file.resolve())
    if tags:
        result_payload["tags"] = list(tags)

    write_json(artifact_dir / "results.json", result_payload)

    if not skip_local_history:
        try:
            from nemo_retriever.harness.history import record_run as _record_history

            _record_history(result_payload, artifact_dir)
        except Exception:
            pass

    if failure_reason:
        _print_failure_report(result_payload, command_text, artifact_dir, metrics.tail_lines)

    return result_payload


_GRAPH_RUNNER_SCRIPT = """\
import json, sys, os, traceback, time, shutil

graph_code_file = sys.argv[1]
result_file = sys.argv[2]
ray_address = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "__none__" else None
lancedb_uri = os.environ.get("NEMO_LANCEDB_URI", "")
lancedb_table = os.environ.get("NEMO_LANCEDB_TABLE", "nv-ingest")

if lancedb_uri:
    ldb_path = os.path.abspath(lancedb_uri)
    if os.path.isdir(ldb_path):
        shutil.rmtree(ldb_path)
        print(f"Cleared stale LanceDB directory: {ldb_path}")
    os.makedirs(ldb_path, exist_ok=True)

def _root_cause(exc):
    seen = set()
    while exc.__cause__ is not None and id(exc.__cause__) not in seen:
        seen.add(id(exc))
        exc = exc.__cause__
    return exc

try:
    import ray

    effective_ray = ray_address or os.environ.get("RAY_ADDRESS") or "auto"
    ray.init(address=effective_ray, ignore_reinit_error=True)
    print(f"Ray initialized: {effective_ray}")

    with open(graph_code_file) as f:
        code = f.read()

    ns = {"__name__": "__graph_runner__", "__file__": graph_code_file}

    wall_start = time.perf_counter()
    exec(compile(code, graph_code_file, "exec"), ns)

    result_ds = ns.get("result")
    graph = ns.get("graph")

    lancedb_written = False
    row_count = 0

    if result_ds is not None:
        elapsed = round(time.perf_counter() - wall_start, 2)
        try:
            import ray.data as _rd
            if isinstance(result_ds, _rd.Dataset):
                row_count = result_ds.count()
                print(f"Pipeline complete: {row_count} rows in {elapsed}s")

                if lancedb_uri:
                    try:
                        import lancedb as _ldb
                        import pyarrow as _pa
                        db = _ldb.connect(lancedb_uri)
                        arrow_ds = result_ds.to_arrow_refs()
                        table = _pa.concat_tables([ray.get(ref) for ref in arrow_ds], promote_options="permissive")
                        col_names = [f.name for f in table.schema]
                        if "vector" in col_names:
                            db.create_table(lancedb_table, table, mode="overwrite")
                            lancedb_written = True
                            print(f"LanceDB: wrote {table.num_rows} rows to {lancedb_uri}/{lancedb_table}")
                        else:
                            print(f"LanceDB: skipping write (no 'vector' column in output; columns: {col_names})")
                    except Exception as ldb_err:
                        print(f"LanceDB write failed (non-fatal): {ldb_err}")
            else:
                print(f"Pipeline complete in {elapsed}s (result type: {type(result_ds).__name__})")
        except Exception:
            elapsed = round(time.perf_counter() - wall_start, 2)
            print(f"Pipeline complete in {elapsed}s")

        result = {
            "success": True, "return_code": 0, "rows": row_count,
            "elapsed_secs": elapsed, "lancedb_written": lancedb_written,
        }
    elif graph is not None:
        outputs = graph.execute(None)
        elapsed = round(time.perf_counter() - wall_start, 2)
        print(f"Graph.execute complete: {len(outputs)} output(s) in {elapsed}s")
        result = {
            "success": True, "return_code": 0, "outputs": len(outputs),
            "elapsed_secs": elapsed, "lancedb_written": False,
        }
    else:
        raise RuntimeError("Generated code did not produce a 'result' (Ray Data) or 'graph' variable")

except Exception as exc:
    full_tb = traceback.format_exc()
    print(full_tb, file=sys.stderr)
    print(full_tb)
    root = _root_cause(exc)
    root_msg = f"{type(root).__name__}: {root}"
    failure_lines = [root_msg]
    if len(full_tb) <= 4000:
        failure_lines.append(full_tb)
    else:
        failure_lines.append(full_tb[-4000:])
    result = {
        "success": False, "failure_reason": root_msg,
        "error_detail": "\\n".join(failure_lines), "return_code": 1,
    }

with open(result_file, "w") as f:
    json.dump(result, f)
"""


def _run_graph_pipeline(
    cfg: HarnessConfig,
    graph_code: str,
    artifact_dir: Path,
    run_id: str,
    tags: list[str] | None = None,
    skip_local_history: bool = False,
) -> dict[str, Any]:
    """Execute a Designer graph pipeline and collect metrics, including recall/BEIR evaluation."""
    import time as _time

    runtime_dir = artifact_dir / "runtime_metrics"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    code_file = artifact_dir / "graph_pipeline.py"
    runner_file = artifact_dir / "graph_runner.py"
    result_file = runtime_dir / f"{run_id}.graph_result.json"
    code_file.write_text(graph_code, encoding="utf-8")
    runner_file.write_text(_GRAPH_RUNNER_SCRIPT, encoding="utf-8")

    lancedb_uri = _resolve_lancedb_uri(cfg, artifact_dir)
    lancedb_path = Path(lancedb_uri)
    if lancedb_path.is_dir():
        shutil.rmtree(lancedb_path)
    lancedb_path.mkdir(parents=True, exist_ok=True)
    lancedb_table = "nv-ingest"

    ray_addr = cfg.ray_address or "__none__"
    cmd = [sys.executable, str(runner_file), str(code_file), str(result_file), ray_addr]

    command_text = " ".join(shlex.quote(token) for token in cmd)
    (artifact_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    typer.echo(f"\n=== Running graph pipeline: {run_id} ===")
    typer.echo(command_text)

    env = os.environ.copy()
    env["NEMO_LANCEDB_URI"] = lancedb_uri
    env["NEMO_LANCEDB_TABLE"] = lancedb_table

    metrics = StreamMetrics()
    _wall_start = _time.perf_counter()

    master_fd, slave_fd = pty.openpty()
    parse_buffer = ""
    try:
        proc = subprocess.Popen(cmd, stdin=None, stdout=slave_fd, stderr=slave_fd, close_fds=True, env=env)
    finally:
        os.close(slave_fd)

    try:
        while True:
            read_fds, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd not in read_fds:
                if proc.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            parse_buffer += text.replace("\r", "\n")
            parse_buffer = _consume_parseable_output(metrics, parse_buffer)
        if parse_buffer:
            cleaned_tail = ANSI_ESCAPE_RE.sub("", parse_buffer)
            metrics.consume(cleaned_tail)
        process_rc = proc.wait()
    finally:
        os.close(master_fd)

    subprocess_elapsed_secs = round(_time.perf_counter() - _wall_start, 2)
    run_metadata = _collect_run_metadata()

    if ray_addr and ray_addr not in ("__none__", "local"):
        run_metadata["ray_cluster_mode"] = "existing"
        if not run_metadata.get("ray_dashboard_url"):
            dashboard = _derive_ray_dashboard_url(ray_addr)
            if dashboard:
                run_metadata["ray_dashboard_url"] = dashboard
    else:
        run_metadata["ray_cluster_mode"] = "local"

    graph_result = _read_json_if_exists(result_file) or {}
    rows = graph_result.get("rows", 0)
    elapsed_secs = graph_result.get("elapsed_secs", subprocess_elapsed_secs)
    lancedb_written = graph_result.get("lancedb_written", False)

    recall_raw: dict[str, float] = {}
    eval_raw: dict[str, float] = {}
    effective_query_csv_path: Path | None = None

    can_evaluate = lancedb_written and Path(lancedb_uri).is_dir()

    if cfg.query_csv:
        effective_query_csv_path = prepare_recall_query_file(
            query_csv=Path(cfg.query_csv),
            recall_adapter=cfg.recall_adapter,
            output_dir=runtime_dir,
        )

    if can_evaluate and cfg.evaluation_mode == "recall" and effective_query_csv_path:
        typer.echo("\n=== Running recall evaluation ===")
        try:
            from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

            recall_cfg = RecallConfig(
                lancedb_uri=lancedb_uri,
                lancedb_table=lancedb_table,
                embedding_model=cfg.embed_model_name,
                ks=list(cfg.beir_ks) if cfg.beir_ks else [1, 3, 5, 10],
                match_mode=cfg.recall_match_mode or "pdf_page",
                hybrid=cfg.hybrid,
            )
            _df_query, _gold, _raw_hits, _retrieved_keys, recall_raw = retrieve_and_score(
                effective_query_csv_path, cfg=recall_cfg,
            )
            typer.echo(f"Recall results: {recall_raw}")
        except Exception as recall_exc:
            typer.echo(f"Recall evaluation failed (non-fatal): {recall_exc}", err=True)

    elif can_evaluate and cfg.evaluation_mode == "beir" and cfg.beir_loader:
        typer.echo("\n=== Running BEIR evaluation ===")
        try:
            from nemo_retriever.recall.beir import BeirConfig, evaluate_lancedb_beir

            beir_cfg = BeirConfig(
                lancedb_uri=lancedb_uri,
                lancedb_table=lancedb_table,
                embedding_model=cfg.embed_model_name,
                loader=cfg.beir_loader,
                dataset_name=cfg.beir_dataset_name or cfg.dataset_label,
                split=cfg.beir_split,
                query_language=cfg.beir_query_language,
                doc_id_field=cfg.beir_doc_id_field,
                ks=list(cfg.beir_ks) if cfg.beir_ks else [1, 3, 5, 10],
                hybrid=cfg.hybrid,
            )
            _dataset, _hits, _per_query, eval_raw = evaluate_lancedb_beir(beir_cfg)
            typer.echo(f"BEIR results: {eval_raw}")
        except Exception as beir_exc:
            typer.echo(f"BEIR evaluation failed (non-fatal): {beir_exc}", err=True)

    recall_metrics_normalized: dict[str, float] = {
        _normalize_recall_metric_key(k): v for k, v in recall_raw.items()
    }
    evaluation_metrics_normalized: dict[str, float] = {
        _normalize_recall_metric_key(k): v for k, v in eval_raw.items()
    }

    effective_rc, failure_reason, success = _evaluate_run_outcome(
        process_rc=process_rc,
        evaluation_mode=cfg.evaluation_mode,
        recall_required=bool(cfg.recall_required) if can_evaluate else False,
        recall_metrics=recall_raw,
        evaluation_metrics=eval_raw,
    )

    if not success and graph_result.get("failure_reason"):
        failure_reason = graph_result["failure_reason"]
    if not success and graph_result.get("return_code"):
        effective_rc = graph_result["return_code"]

    pm_files = metrics.files
    pm_pages = rows or metrics.pages
    pm_ingest_secs = elapsed_secs or metrics.ingest_secs
    pm_pps = None
    if pm_pages and pm_ingest_secs and pm_ingest_secs > 0:
        pm_pps = round(pm_pages / pm_ingest_secs, 2)

    metrics_payload = {
        "files": pm_files,
        "pages": pm_pages,
        "ingest_secs": pm_ingest_secs,
        "pages_per_sec_ingest": pm_pps,
        "rows_processed": rows,
        "rows_per_sec_ingest": round(rows / elapsed_secs, 2) if rows and elapsed_secs else None,
        **recall_metrics_normalized,
        **evaluation_metrics_normalized,
    }
    summary_metrics = _resolve_summary_metrics(cfg, metrics_payload, None, subprocess_elapsed_secs)

    result_payload: dict[str, Any] = {
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "success": success,
        "return_code": effective_rc,
        "failure_reason": failure_reason or None,
        "error_detail": graph_result.get("error_detail"),
        "test_config": {
            "dataset_label": cfg.dataset_label,
            "dataset_dir": cfg.dataset_dir,
            "preset": cfg.preset,
            "query_csv": cfg.query_csv,
            "effective_query_csv": str(effective_query_csv_path) if effective_query_csv_path else None,
            "input_type": "graph",
            "recall_required": cfg.recall_required if can_evaluate else False,
            "recall_match_mode": cfg.recall_match_mode,
            "recall_adapter": cfg.recall_adapter,
            "evaluation_mode": cfg.evaluation_mode,
            "beir_loader": cfg.beir_loader,
            "beir_dataset_name": cfg.beir_dataset_name,
            "beir_split": cfg.beir_split,
            "beir_query_language": cfg.beir_query_language,
            "beir_doc_id_field": cfg.beir_doc_id_field,
            "beir_ks": list(cfg.beir_ks),
            "ray_address": cfg.ray_address,
            "hybrid": cfg.hybrid,
            "embed_model_name": cfg.embed_model_name,
            "lancedb_uri": lancedb_uri,
            "lancedb_written": lancedb_written,
            "graph_pipeline": True,
        },
        "metrics": metrics_payload,
        "summary_metrics": summary_metrics,
        "run_metadata": run_metadata,
        "runtime_summary": None,
        "detection_summary": None,
        "artifacts": {
            "command_file": str((artifact_dir / "command.txt").resolve()),
            "runtime_metrics_dir": str(runtime_dir.resolve()),
            "graph_code_file": str(code_file.resolve()),
        },
    }
    if tags:
        result_payload["tags"] = list(tags)

    write_json(artifact_dir / "results.json", result_payload)

    if not skip_local_history:
        try:
            from nemo_retriever.harness.history import record_run as _record_history

            _record_history(result_payload, artifact_dir)
        except Exception:
            pass

    if failure_reason:
        _print_failure_report(result_payload, command_text, artifact_dir, metrics.tail_lines)

    return result_payload


def _run_entry(
    *,
    run_name: str | None,
    config_file: str | None,
    session_dir: Path | None,
    dataset: str | None,
    preset: str | None,
    sweep_overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    recall_required: bool | None = None,
    tags: list[str] | None = None,
    skip_local_history: bool = False,
    graph_code: str | None = None,
) -> dict[str, Any]:
    cfg = load_harness_config(
        config_file=config_file,
        dataset=dataset,
        preset=preset,
        sweep_overrides=sweep_overrides,
        cli_overrides=cli_overrides,
        cli_recall_required=recall_required,
    )

    if session_dir is None:
        artifact_dir = create_run_artifact_dir(cfg.dataset_label, run_name=run_name, base_dir=cfg.artifacts_dir)
    else:
        resolved_run_name = run_name or cfg.dataset_label
        artifact_dir = session_dir / resolved_run_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_name = run_name or cfg.dataset_label
    normalized_tags = _normalize_tags(tags)

    if graph_code:
        result = _run_graph_pipeline(
            cfg, graph_code, artifact_dir,
            run_id=resolved_run_name, tags=normalized_tags,
            skip_local_history=skip_local_history,
        )
    else:
        result = _run_single(
            cfg, artifact_dir, run_id=resolved_run_name, tags=normalized_tags, skip_local_history=skip_local_history
        )
    return result


def execute_runs(
    *,
    runs: list[dict[str, Any]],
    config_file: str | None,
    session_prefix: str,
    preset_override: str | None,
    base_artifacts_dir: str | None = None,
    tags: list[str] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    session_dir = create_session_dir(session_prefix, base_dir=base_artifacts_dir)
    run_results: list[dict[str, Any]] = []

    for idx, run in enumerate(runs):
        run_name = str(run.get("name") or f"run_{idx + 1:03d}")
        run_result = _run_entry(
            run_name=run_name,
            config_file=config_file,
            session_dir=session_dir,
            dataset=run.get("dataset"),
            preset=run.get("preset") if preset_override is None else preset_override,
            sweep_overrides=run.get("overrides") if isinstance(run.get("overrides"), dict) else run,
            recall_required=run.get("recall_required"),
            tags=tags,
        )
        run_results.append(run_result)

    return session_dir, run_results


def run_command(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name from config or direct path."),
    preset: str | None = typer.Option(None, "--preset", help="Preset override."),
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    run_name: str | None = typer.Option(None, "--run-name", help="Optional run name label."),
    override: list[str] = typer.Option([], "--override", help="Override values with KEY=VALUE."),
    tag: list[str] = typer.Option([], "--tag", help="Run tag to persist in harness artifacts. Repeatable."),
    recall_required: bool | None = typer.Option(
        None, "--recall-required/--no-recall-required", help="Override recall-required gate for this run."
    ),
) -> None:
    result = _run_entry(
        run_name=run_name,
        config_file=config,
        session_dir=None,
        dataset=dataset,
        preset=preset,
        cli_overrides=override,
        recall_required=recall_required,
        tags=tag,
    )
    artifact_display = (result.get("artifacts") or {}).get("runtime_metrics_dir", "N/A")
    typer.echo(
        f"\nResult: {'PASS' if result['success'] else 'FAIL'} | "
        f"return_code={result['return_code']} | artifacts={artifact_display}"
    )
    raise typer.Exit(code=0 if result["success"] else 1)


def sweep_command(
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    runs_config: str = typer.Option(str(DEFAULT_NIGHTLY_CONFIG_PATH), "--runs-config", help="Path to sweep runs YAML."),
    preset: str | None = typer.Option(None, "--preset", help="Force preset for all sweep runs."),
    session_prefix: str = typer.Option("sweep", "--session-prefix", help="Session directory prefix."),
    tag: list[str] = typer.Option([], "--tag", help="Session tag to persist on each run. Repeatable."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print run plan without executing."),
) -> None:
    normalized_tags = _normalize_tags(tag)
    sweep_cfg = load_nightly_config(runs_config)
    runs = sweep_cfg["runs"]
    resolved_preset = preset or sweep_cfg.get("preset")
    if dry_run:
        typer.echo("Sweep dry run:")
        for idx, run in enumerate(runs):
            tag_text = f" tags={normalized_tags}" if normalized_tags else ""
            run_preset = run.get("preset") if run.get("preset") is not None else resolved_preset
            plan_line = (
                f"  {idx + 1:03d}: name={run.get('name')} "
                f"dataset={run.get('dataset')} preset={run_preset}{tag_text}"
            )
            typer.echo(plan_line)
        raise typer.Exit(code=0)

    session_dir, run_results = execute_runs(
        runs=runs,
        config_file=config,
        session_prefix=session_prefix,
        preset_override=resolved_preset,
        tags=normalized_tags,
    )
    summary_path = write_session_summary(
        session_dir,
        run_results,
        session_type="sweep",
        config_path=str(Path(runs_config).expanduser().resolve()),
    )

    typer.echo(f"\nSweep session: {session_dir}")
    typer.echo(f"Session summary: {summary_path}")
    failed = [r for r in run_results if not r["success"]]
    raise typer.Exit(code=0 if not failed else 1)
