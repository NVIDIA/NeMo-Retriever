from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_retriever.harness.json_io import read_json_object

DEFAULT_USERNAME = "nemo_retriever Harness"
DEFAULT_ICON_EMOJI = ":satellite:"
_BLANK_ROW = [
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
]
METRIC_LABELS = {
    "gpu_sku": "physical GPU SKU",
    "gpu_count": "physical GPU count",
    "workload_gpu_count": "GPUs available to workload",
    "files": "files",
    "pages_per_sec_ingest": "pages/s",
    "ingest_secs": "ingest_s",
    "pages": "pages",
    "rows_processed": "rows",
    "query_count": "queries",
    "query_latency_p50_ms": "query p50 ms",
    "query_latency_p95_ms": "query p95 ms",
    "ndcg_10": "ndcg@10",
    "recall_5": "recall@5",
    "recall_10": "recall@10",
}
DEFAULT_SLACK_METRIC_KEYS = (
    "files",
    "pages",
    "ingest_secs",
    "pages_per_sec_ingest",
    "query_count",
    "ndcg_10",
    "recall_5",
    "recall_10",
)
MAX_SLACK_TABLE_ROWS = 100
VIDORE_V3_REPORT_DATASETS = {
    "vidore_v3_finance_en": ("finance_en", True),
    "vidore_v3_industrial": ("industrial", True),
    "vidore_v3_computer_science": ("computer_science", True),
    "vidore_v3_pharmaceuticals": ("pharmaceuticals", True),
    "vidore_v3_hr": ("hr", True),
    "vidore_v3_energy": ("energy", True),
    "vidore_v3_physics": ("physics", True),
    "vidore_v3_finance_fr": ("finance_fr", False),
}


@dataclass
class HarnessRunReport:
    run_name: str
    dataset: str
    preset: str | None
    success: bool
    return_code: int | None
    failure_reason: str | None
    artifact_dir: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)
    latest_commit: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessSessionReport:
    session_name: str
    session_dir: Path
    session_type: str
    timestamp: str | None
    latest_commit: str | None
    all_passed: bool
    dry_run: bool
    results: list[HarnessRunReport]


def _normalize_metrics(raw_metrics: Any) -> dict[str, Any]:
    if not isinstance(raw_metrics, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in raw_metrics.items():
        metric_key = str(key).strip()
        if metric_key.startswith("recall_recall_"):
            metric_key = "recall_" + metric_key.removeprefix("recall_recall_")
        normalized[metric_key] = value
    return normalized


def _load_results_payload(artifact_dir: Path | None) -> dict[str, Any]:
    if artifact_dir is None:
        return {}
    results_path = artifact_dir / "results.json"
    if not results_path.exists():
        return {}
    return read_json_object(results_path)


def _load_environment_payload(artifact_dir: Path | None) -> dict[str, Any]:
    if artifact_dir is None:
        return {}
    environment_path = artifact_dir / "environment.json"
    if not environment_path.exists():
        return {}
    return read_json_object(environment_path)


def _load_preferred_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary_metrics = payload.get("summary_metrics")
    if isinstance(summary_metrics, dict) and summary_metrics:
        return _normalize_metrics(summary_metrics)
    return _normalize_metrics(payload.get("metrics", {}))


def _failure_reason(payload: dict[str, Any]) -> str | None:
    failure = payload.get("failure")
    if not isinstance(failure, dict):
        return None
    for key in ("message", "failure_reason"):
        value = failure.get(key)
        if value:
            return str(value)
    return None


def _resolved_dataset_name(payload: dict[str, Any]) -> str | None:
    resolved = payload.get("resolved_benchmark")
    if not isinstance(resolved, dict):
        return None
    dataset = resolved.get("dataset")
    if not isinstance(dataset, dict):
        return None
    value = dataset.get("name")
    return str(value) if value else None


def _run_metadata(results_payload: dict[str, Any], environment_payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_run_metadata = results_payload.get("run_metadata")
    if isinstance(raw_run_metadata, dict):
        metadata.update(raw_run_metadata)
    metadata.update(environment_payload)
    if "python" in metadata and "python_version" not in metadata:
        metadata["python_version"] = metadata["python"]
    return metadata


def _normalize_run_report(summary_entry: dict[str, Any], *, session_dir: Path) -> HarnessRunReport:
    artifact_dir_str = summary_entry.get("artifact_dir")
    artifact_dir = Path(artifact_dir_str).expanduser() if artifact_dir_str else None
    if artifact_dir is not None:
        if not artifact_dir.is_absolute():
            artifact_dir = session_dir / artifact_dir
        artifact_dir = artifact_dir.resolve()
    results_payload = _load_results_payload(artifact_dir)
    environment_payload = _load_environment_payload(artifact_dir)
    test_config = results_payload.get("test_config", {})
    run_metadata = _run_metadata(results_payload, environment_payload)

    metrics = summary_entry.get("metrics") or summary_entry.get("summary_metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        metrics = _load_preferred_metrics(results_payload)
    if not isinstance(metrics, dict):
        metrics = {}

    if not isinstance(test_config, dict):
        test_config = {}
    if not isinstance(run_metadata, dict):
        run_metadata = {}

    return_code = summary_entry.get("return_code", summary_entry.get("exit_code", results_payload.get("exit_code")))
    success = summary_entry.get("success", results_payload.get("success", False))
    latest_commit = (
        results_payload.get("latest_commit")
        or environment_payload.get("git_sha")
        or environment_payload.get("git_commit")
    )

    return HarnessRunReport(
        run_name=str(
            summary_entry.get("run_name")
            or summary_entry.get("benchmark")
            or results_payload.get("benchmark")
            or (artifact_dir.name if artifact_dir else "unknown_run")
        ),
        dataset=str(
            summary_entry.get("dataset")
            or results_payload.get("dataset")
            or _resolved_dataset_name(results_payload)
            or test_config.get("dataset_label")
            or "unknown_dataset"
        ),
        preset=(
            str(summary_entry.get("preset") or test_config.get("preset"))
            if (summary_entry.get("preset") or test_config.get("preset"))
            else None
        ),
        success=bool(success),
        return_code=int(return_code) if return_code is not None else None,
        failure_reason=(
            str(summary_entry.get("failure_reason"))
            if summary_entry.get("failure_reason")
            else _failure_reason(results_payload)
        ),
        artifact_dir=artifact_dir,
        metrics=_normalize_metrics(metrics),
        latest_commit=str(latest_commit) if latest_commit else None,
        run_metadata=run_metadata,
    )


def load_session_report(session_summary_path: Path) -> HarnessSessionReport:
    resolved_summary_path = Path(session_summary_path).expanduser().resolve()
    if resolved_summary_path.is_dir():
        resolved_summary_path = resolved_summary_path / "session_summary.json"

    payload = read_json_object(resolved_summary_path)
    raw_results = payload.get("runs", payload.get("results", []))
    if not isinstance(raw_results, list):
        raise ValueError(f"'results' or 'runs' must be a list in {resolved_summary_path}")
    if not raw_results:
        raise ValueError(f"Session contains no runs in {resolved_summary_path}")
    results: list[HarnessRunReport] = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            raise ValueError(f"Session run at index {index} must be an object in {resolved_summary_path}")
        results.append(_normalize_run_report(item, session_dir=resolved_summary_path.parent))
    latest_commit = (
        payload.get("run_commit")
        or payload.get("latest_commit")
        or next((run.latest_commit for run in results if run.latest_commit), None)
    )
    all_passed = payload.get("all_passed", payload.get("success", False))

    return HarnessSessionReport(
        session_name=str(payload.get("session_name") or payload.get("runset") or resolved_summary_path.parent.name),
        session_dir=resolved_summary_path.parent,
        session_type=str(payload.get("session_type") or "session"),
        timestamp=str(payload.get("timestamp")) if payload.get("timestamp") else None,
        latest_commit=str(latest_commit) if latest_commit else None,
        all_passed=bool(all_passed),
        dry_run=bool(payload.get("dry_run", False)),
        results=results,
    )


def load_replay_report(replay_paths: list[Path]) -> HarnessSessionReport:
    if not replay_paths:
        raise ValueError("At least one replay path is required")

    resolved_paths = [Path(path).expanduser().resolve() for path in replay_paths]
    session_dirs = [path for path in resolved_paths if path.is_dir() and (path / "session_summary.json").exists()]
    if session_dirs:
        if len(resolved_paths) != 1:
            raise ValueError("Replay accepts either one session directory or one or more run directories")
        return load_session_report(session_dirs[0] / "session_summary.json")

    run_reports: list[HarnessRunReport] = []
    latest_commit: str | None = None
    dry_run = True

    for path in resolved_paths:
        results_path = path / "results.json" if path.is_dir() else path
        if results_path.name != "results.json":
            raise ValueError(f"Replay path must be a run directory, session directory, or results.json file: {path}")
        payload = read_json_object(results_path)
        artifact_dir = results_path.parent
        environment_payload = _load_environment_payload(artifact_dir)
        test_config = payload.get("test_config", {})
        if not isinstance(test_config, dict):
            test_config = {}
        run_metadata = _run_metadata(payload, environment_payload)

        payload_commit = payload.get("latest_commit") or environment_payload.get("git_sha")
        return_code = payload.get("return_code", payload.get("exit_code"))
        latest_commit = latest_commit or (str(payload_commit) if payload_commit else None)
        dry_run = dry_run and bool(payload.get("dry_run", False))
        run_reports.append(
            HarnessRunReport(
                run_name=str(payload.get("benchmark") or artifact_dir.name),
                dataset=str(
                    payload.get("dataset")
                    or _resolved_dataset_name(payload)
                    or test_config.get("dataset_label")
                    or artifact_dir.name
                ),
                preset=str(test_config.get("preset")) if test_config.get("preset") else None,
                success=bool(payload.get("success")),
                return_code=int(return_code) if return_code is not None else None,
                failure_reason=_failure_reason(payload),
                artifact_dir=artifact_dir,
                metrics=_load_preferred_metrics(payload),
                latest_commit=str(payload_commit) if payload_commit else None,
                run_metadata=run_metadata,
            )
        )

    session_dir = resolved_paths[0].parent if resolved_paths else Path.cwd()
    return HarnessSessionReport(
        session_name="artifact_replay",
        session_dir=session_dir,
        session_type="artifact_replay",
        timestamp=None,
        latest_commit=latest_commit,
        all_passed=all(run.success for run in run_reports),
        dry_run=dry_run,
        results=run_reports,
    )


def _format_metric_value(metric_name: str, value: Any) -> str:
    if value is None:
        return "N/A"

    if metric_name.endswith("_secs") or metric_name.endswith("_time_s") or metric_name == "ingest_secs":
        seconds = float(value)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        formatted = f"{seconds:.2f}s"
        if hours > 0 or minutes > 0:
            formatted += " ("
            if hours > 0:
                formatted += f"{hours}h : "
            formatted += f"{minutes:02}m : {secs:05.2f}s)"
        return formatted
    if metric_name.endswith("_per_sec_ingest"):
        return f"{float(value):.2f}"
    if metric_name.startswith(("recall_", "ndcg_")):
        return f"{float(value):.3f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _format_metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name)


def _select_metric_names(metrics: dict[str, Any], metric_keys: list[str]) -> list[str]:
    metric_names: list[str] = []
    seen: set[str] = set()

    for key in metric_keys:
        if key in metrics and metrics[key] is not None:
            metric_names.append(key)
            seen.add(key)

    return metric_names


def _table_cell(value: str, *, bold: bool = False) -> dict[str, Any]:
    text_element: dict[str, Any] = {"type": "text", "text": value}
    if bold:
        text_element["style"] = {"bold": True}
    return {
        "type": "rich_text",
        "elements": [{"type": "rich_text_section", "elements": [text_element]}],
    }


def _two_column_row(left: str, right: str) -> list[dict[str, Any]]:
    return [_table_cell(left), _table_cell(right)]


def _two_column_row_bold(left: str, right: str) -> list[dict[str, Any]]:
    return [_table_cell(left, bold=True), _table_cell(right)]


def _three_column_row(left: str, middle: str, right: str, *, bold: bool = False) -> list[dict[str, Any]]:
    return [
        _table_cell(left, bold=bold),
        _table_cell(middle, bold=bold),
        _table_cell(right, bold=bold),
    ]


def _vidore_v3_runs(results: list[HarnessRunReport]) -> list[HarnessRunReport]:
    runs_by_dataset: dict[str, list[HarnessRunReport]] = {}
    for run in results:
        if run.dataset in VIDORE_V3_REPORT_DATASETS:
            runs_by_dataset.setdefault(run.dataset, []).append(run)
    return [run for dataset in VIDORE_V3_REPORT_DATASETS for run in runs_by_dataset.get(dataset, [])]


def _is_complete_vidore_v3_suite(runs: list[HarnessRunReport]) -> bool:
    return len(runs) == len(VIDORE_V3_REPORT_DATASETS) and {run.dataset for run in runs} == set(
        VIDORE_V3_REPORT_DATASETS
    )


def _numeric_metric(run: HarnessRunReport, metric_name: str) -> float | None:
    value = run.metrics.get(metric_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _macro_average(runs: list[HarnessRunReport], metric_name: str) -> float | None:
    if any(not run.success for run in runs):
        return None
    values = [_numeric_metric(run, metric_name) for run in runs]
    if any(value is None for value in values):
        return None
    numeric_values = [value for value in values if value is not None]
    return sum(numeric_values) / len(numeric_values)


def _format_accuracy_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _vidore_v3_accuracy_rows(runs: list[HarnessRunReport]) -> list[list[dict[str, Any]]]:
    rows = [_three_column_row("DATASET", "RECALL@5", "NDCG@10", bold=True)]
    if _is_complete_vidore_v3_suite(runs):
        english_runs = [run for run in runs if VIDORE_V3_REPORT_DATASETS[run.dataset][1]]
        for label, selected_runs in (("Avg (English)", english_runs), ("Avg (all)", runs)):
            rows.append(
                _three_column_row(
                    label,
                    _format_accuracy_value(_macro_average(selected_runs, "recall_5")),
                    _format_accuracy_value(_macro_average(selected_runs, "ndcg_10")),
                    bold=True,
                )
            )
    for run in runs:
        label = VIDORE_V3_REPORT_DATASETS[run.dataset][0]
        rows.append(
            _three_column_row(
                label,
                _format_accuracy_value(_numeric_metric(run, "recall_5")),
                _format_accuracy_value(_numeric_metric(run, "ndcg_10")),
            )
        )
    return rows


def _vidore_v3_status(runs: list[HarnessRunReport], *, dry_run: bool) -> str:
    if dry_run:
        return f"DRY RUN ({len(runs)} planned)"
    passed_count = sum(1 for run in runs if run.success)
    if passed_count == len(runs):
        return f"PASS ({passed_count}/{len(runs)})"
    return f"FAIL ({passed_count}/{len(runs)} passed)"


def _vidore_v3_performance(runs: list[HarnessRunReport]) -> tuple[float | None, float | None]:
    pages = [_numeric_metric(run, "pages") for run in runs]
    ingest_secs = [_numeric_metric(run, "ingest_secs") for run in runs]
    if any(value is None for value in pages) or any(value is None for value in ingest_secs):
        return None, None

    total_pages = sum(value for value in pages if value is not None)
    total_ingest_secs = sum(value for value in ingest_secs if value is not None)
    pages_per_sec = total_pages / total_ingest_secs if total_ingest_secs > 0 else None
    return total_ingest_secs, pages_per_sec


def _gpu_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _run_display_label(run: HarnessRunReport, repeated_datasets: set[str]) -> str:
    if run.dataset not in repeated_datasets:
        return run.dataset
    workload_gpu_count = _gpu_count(run.run_metadata.get("workload_gpu_count"))
    if workload_gpu_count is None:
        return run.run_name
    gpu_label = "GPU" if workload_gpu_count == 1 else "GPUs"
    return f"{run.dataset} ({workload_gpu_count} workload {gpu_label})"


def build_slack_payload(report: HarnessSessionReport, slack_config: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [str(key) for key in slack_config.get("metric_keys", [])]
    post_artifact_paths = bool(slack_config.get("post_artifact_paths", False))
    vidore_v3_runs = _vidore_v3_runs(report.results)
    vidore_v3_datasets = {run.dataset for run in vidore_v3_runs}
    detailed_runs = [run for run in report.results if run.dataset not in vidore_v3_datasets]
    repeated_datasets = {
        run.dataset for run in detailed_runs if sum(item.dataset == run.dataset for item in detailed_runs) > 1
    }
    passed_count = sum(1 for run in report.results if run.success)
    total_count = len(report.results)
    if report.dry_run:
        overall_status = f"DRY RUN ({total_count} planned)"
    elif report.all_passed:
        overall_status = f"PASS ({passed_count}/{total_count})"
    else:
        overall_status = f"FAIL ({passed_count}/{total_count} passed)"
    rows: list[list[dict[str, Any]]] = []
    rows.append(_two_column_row_bold("OVERALL STATUS", overall_status))
    for run in detailed_runs:
        run_status = "DRY RUN" if report.dry_run else "PASS" if run.success else "FAIL"
        rows.append(_two_column_row_bold(f"-    {_run_display_label(run, repeated_datasets)}", run_status))
    if vidore_v3_runs:
        rows.append(
            _two_column_row_bold(
                "-    ViDoRe v3",
                _vidore_v3_status(vidore_v3_runs, dry_run=report.dry_run),
            )
        )

    rows.append(_BLANK_ROW)
    rows.append(_two_column_row_bold("ENVIRONMENT", " "))
    rows.append(_two_column_row("-    session", report.session_name))
    if post_artifact_paths:
        rows.append(_two_column_row("-    session_dir", str(report.session_dir)))
    if report.latest_commit:
        rows.append(_two_column_row("-    git_commit", report.latest_commit))
    for key in [
        "host",
        "gpu_sku",
        "gpu_count",
        "workload_gpu_count",
        "cuda_driver",
        "ray_version",
        "python_version",
    ]:
        metadata_values = list(
            dict.fromkeys(run.run_metadata.get(key) for run in report.results if run.run_metadata.get(key) is not None)
        )
        if not metadata_values:
            continue
        formatted_value = ", ".join(_format_metric_value(key, value) for value in metadata_values)
        rows.append(_two_column_row(f"-    {_format_metric_label(key)}", formatted_value))

    rows.append(_BLANK_ROW)
    rows.append(_two_column_row_bold("RESULTS", " "))
    for run in detailed_runs:
        run_status = "DRY RUN" if report.dry_run else "PASS" if run.success else "FAIL"
        rows.append(_two_column_row_bold(_run_display_label(run, repeated_datasets), run_status))
        if not run.success and run.return_code is not None:
            rows.append(_two_column_row("-    return_code", str(run.return_code)))
        for metric_name in _select_metric_names(run.metrics, metric_keys):
            rows.append(
                _two_column_row(
                    f"-    {_format_metric_label(metric_name)}",
                    _format_metric_value(metric_name, run.metrics[metric_name]),
                )
            )
        if run.failure_reason:
            rows.append(_two_column_row("-    failure_reason", run.failure_reason))
        if post_artifact_paths and run.artifact_dir is not None:
            rows.append(_two_column_row("-    artifact_dir", str(run.artifact_dir)))
        rows.append(_BLANK_ROW)

    if vidore_v3_runs:
        total_ingest_secs, pages_per_sec = _vidore_v3_performance(vidore_v3_runs)
        rows.append(
            _two_column_row_bold(
                "ViDoRe v3",
                _vidore_v3_status(vidore_v3_runs, dry_run=report.dry_run),
            )
        )
        rows.append(
            _two_column_row(
                "-    total ingest time",
                _format_metric_value("ingest_secs", total_ingest_secs),
            )
        )
        rows.append(
            _two_column_row(
                "-    aggregate pages/s",
                _format_metric_value("pages_per_sec_ingest", pages_per_sec),
            )
        )
        for run in vidore_v3_runs:
            if run.success:
                continue
            label = VIDORE_V3_REPORT_DATASETS[run.dataset][0]
            failure = run.failure_reason or (
                f"return code {run.return_code}" if run.return_code is not None else "run failed"
            )
            rows.append(_two_column_row(f"-    {label} failure", failure))
        rows.append(_BLANK_ROW)

    if report.results:
        rows.pop(-1)

    if len(rows) > MAX_SLACK_TABLE_ROWS:
        omitted_count = len(rows) - MAX_SLACK_TABLE_ROWS + 1
        rows = rows[: MAX_SLACK_TABLE_ROWS - 1]
        rows.append(_two_column_row_bold("TRUNCATED", f"{omitted_count} rows omitted; inspect session artifacts"))

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": str(slack_config.get("title") or "nemo_retriever Harness Report")},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"Session: `{report.session_name}`"},
        },
        {"type": "divider"},
        {"type": "table", "rows": rows},
    ]
    if vidore_v3_runs:
        blocks.extend(
            [
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*ViDoRe v3 accuracy*",
                    },
                },
                {"type": "table", "rows": _vidore_v3_accuracy_rows(vidore_v3_runs)},
            ]
        )

    return {
        "username": DEFAULT_USERNAME,
        "icon_emoji": DEFAULT_ICON_EMOJI,
        "blocks": blocks,
    }


def post_slack_payload(payload: dict[str, Any], webhook_url: str) -> None:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("requests is required for Slack posting") from exc

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Slack post failed because the webhook request could not be completed") from exc
    if not response.ok:
        raise RuntimeError(f"Slack post failed with status={response.status_code}")


def resolve_slack_webhook_url(webhook_url: str | None = None) -> str:
    effective_webhook = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not effective_webhook:
        raise RuntimeError("SLACK_WEBHOOK_URL is not set")
    return effective_webhook


def post_report_to_slack(
    report: HarnessSessionReport,
    slack_config: dict[str, Any],
    *,
    webhook_url: str | None = None,
) -> dict[str, Any]:
    payload = build_slack_payload(report, slack_config)
    post_slack_payload(payload, resolve_slack_webhook_url(webhook_url))
    return payload
