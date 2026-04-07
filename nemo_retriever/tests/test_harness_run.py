import json
from pathlib import Path

from typer.testing import CliRunner

from nemo_retriever.harness.cli import app as harness_app
from nemo_retriever.harness.artifacts import create_run_artifact_dir
from nemo_retriever.harness.config import HarnessConfig
from nemo_retriever.harness import run as harness_run
from nemo_retriever.harness.run import _build_command, _evaluate_run_outcome, _normalize_recall_metric_key

RUNNER = CliRunner()


def test_evaluate_run_outcome_passes_when_process_succeeds_and_recall_present() -> None:
    rc, reason, success = _evaluate_run_outcome(0, "recall", True, {"recall@5": 0.9})
    assert rc == 0
    assert reason == ""
    assert success is True


def test_evaluate_run_outcome_fails_when_recall_required_and_missing() -> None:
    rc, reason, success = _evaluate_run_outcome(0, "recall", True, {})
    assert rc == 98
    assert reason == "missing_recall_metrics"
    assert success is False


def test_evaluate_run_outcome_fails_when_beir_metrics_missing() -> None:
    rc, reason, success = _evaluate_run_outcome(0, "beir", False, {}, {})
    assert rc == 97
    assert reason == "missing_beir_metrics"
    assert success is False


def test_evaluate_run_outcome_uses_subprocess_error_code() -> None:
    rc, reason, success = _evaluate_run_outcome(2, "recall", True, {"recall@5": 0.9})
    assert rc == 2
    assert reason == "subprocess_exit_2"
    assert success is False


def test_create_run_artifact_dir_defaults_to_dataset_label(tmp_path: Path) -> None:
    out = create_run_artifact_dir("jp20", run_name=None, base_dir=str(tmp_path))
    assert out.name.startswith("jp20_")


def test_build_command_uses_hidden_detection_file_by_default(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
        write_detection_file=False,
    )
    cmd, runtime_dir, detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")
    assert "--detection-summary-file" in cmd
    assert "--evaluation-mode" in cmd
    assert cmd[cmd.index("--evaluation-mode") + 1] == "recall"
    assert "--recall-match-mode" in cmd
    assert "pdf_page" in cmd
    assert "--pdf-extract-tasks" in cmd
    assert "--pdf-extract-cpus-per-task" in cmd
    assert "--page-elements-actors" in cmd
    assert "--ocr-actors" in cmd
    assert "--embed-actors" in cmd
    assert "--page-elements-gpus-per-actor" in cmd
    assert "--ocr-gpus-per-actor" in cmd
    assert "--embed-gpus-per-actor" in cmd
    assert "--embed-modality" in cmd
    assert "text" in cmd
    assert "--embed-granularity" in cmd
    assert "element" in cmd
    assert "--extract-page-as-image" in cmd
    assert "--no-extract-page-as-image" not in cmd
    assert detection_file.parent == runtime_dir
    assert detection_file.name == ".detection_summary.json"
    assert effective_query_csv == query_csv


def test_build_command_supports_beir_evaluation_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="vidore_v3_computer_science",
        preset="single_gpu",
        evaluation_mode="beir",
        beir_loader="vidore_hf",
        beir_dataset_name="vidore_v3_computer_science",
        query_csv=None,
        recall_required=False,
    )

    cmd, _runtime_dir, _detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")
    assert "--evaluation-mode" in cmd
    assert cmd[cmd.index("--evaluation-mode") + 1] == "beir"
    assert "--beir-loader" in cmd
    assert cmd[cmd.index("--beir-loader") + 1] == "vidore_hf"
    assert "--beir-dataset-name" in cmd
    assert "--beir-k" in cmd
    assert "--query-csv" not in cmd
    assert "--recall-match-mode" not in cmd
    assert effective_query_csv is None


def test_build_command_uses_top_level_detection_file_when_enabled(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
        write_detection_file=True,
    )
    cmd, runtime_dir, detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")
    assert "--detection-summary-file" in cmd
    assert detection_file.parent == tmp_path
    assert detection_file.name == "detection_summary.json"
    assert effective_query_csv == query_csv


def test_build_command_supports_multimodal_embedding_and_infographics(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="earnings",
        preset="single_gpu",
        query_csv=str(query_csv),
        embed_modality="text_image",
        embed_granularity="element",
        extract_page_as_image=False,
        extract_infographics=True,
    )
    cmd, _runtime_dir, _detection_file, _effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")

    assert "--no-extract-page-as-image" in cmd
    assert "--extract-page-as-image" not in cmd
    assert "--extract-infographics" in cmd
    assert "--embed-modality" in cmd
    assert cmd[cmd.index("--embed-modality") + 1] == "text_image"
    assert "--embed-granularity" in cmd
    assert cmd[cmd.index("--embed-granularity") + 1] == "element"
    assert "--structured-elements-modality" in cmd
    assert cmd[cmd.index("--structured-elements-modality") + 1] == "text_image"


def test_build_command_applies_page_plus_one_adapter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf,page\nq,doc_name.pdf,0\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="earnings",
        preset="single_gpu",
        query_csv=str(query_csv),
        recall_adapter="page_plus_one",
    )
    cmd, runtime_dir, _detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")

    assert effective_query_csv.parent == runtime_dir
    assert effective_query_csv.name == "query_adapter.page_plus_one.csv"
    assert "--query-csv" in cmd
    assert str(effective_query_csv) in cmd
    csv_contents = effective_query_csv.read_text(encoding="utf-8")
    assert "query,pdf_page" in csv_contents
    assert "q,doc_name_1" in csv_contents
    assert "--segment-audio" not in cmd
    assert "--no-segment-audio" not in cmd
    assert "--audio-split-type" not in cmd
    assert "--audio-split-interval" not in cmd


def test_build_command_passes_audio_recall_options(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text(
        "query,expected_media_id,expected_start_time,expected_end_time\nq,clip,1.0,2.0\n",
        encoding="utf-8",
    )

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="audio_retrieval",
        preset="single_gpu",
        query_csv=str(query_csv),
        input_type="audio",
        segment_audio=True,
        audio_split_type="time",
        audio_split_interval=45,
        recall_match_mode="audio_segment",
        audio_match_tolerance_secs=3.25,
    )
    cmd, _runtime_dir, _detection_file, _effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")

    assert "--input-type" in cmd
    assert cmd[cmd.index("--input-type") + 1] == "audio"
    assert "--recall-match-mode" in cmd
    assert cmd[cmd.index("--recall-match-mode") + 1] == "audio_segment"
    assert "--audio-match-tolerance-secs" in cmd
    assert cmd[cmd.index("--audio-match-tolerance-secs") + 1] == "3.25"
    assert "--segment-audio" in cmd
    assert "--no-segment-audio" not in cmd
    assert "--audio-split-type" in cmd
    assert cmd[cmd.index("--audio-split-type") + 1] == "time"
    assert "--audio-split-interval" in cmd
    assert cmd[cmd.index("--audio-split-interval") + 1] == "45"


def test_build_command_omits_tuning_flags_when_use_heuristics(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
        use_heuristics=True,
    )
    cmd, _runtime_dir, _detection_file, _effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")

    assert "--pdf-extract-tasks" not in cmd
    assert "--pdf-extract-cpus-per-task" not in cmd
    assert "--pdf-extract-batch-size" not in cmd
    assert "--pdf-split-batch-size" not in cmd
    assert "--page-elements-batch-size" not in cmd
    assert "--page-elements-actors" not in cmd
    assert "--ocr-actors" not in cmd
    assert "--ocr-batch-size" not in cmd
    assert "--embed-actors" not in cmd
    assert "--embed-batch-size" not in cmd
    assert "--page-elements-cpus-per-actor" not in cmd
    assert "--ocr-cpus-per-actor" not in cmd
    assert "--embed-cpus-per-actor" not in cmd
    assert "--page-elements-gpus-per-actor" not in cmd
    assert "--ocr-gpus-per-actor" not in cmd
    assert "--embed-gpus-per-actor" not in cmd
    # non-tuning flags still present
    assert "--embed-model-name" in cmd
    assert "--evaluation-mode" in cmd


def test_normalize_recall_metric_key_removes_duplicate_prefix() -> None:
    assert _normalize_recall_metric_key("recall@1") == "recall_1"
    assert _normalize_recall_metric_key("recall@10") == "recall_10"


def test_run_single_writes_tags_to_results_json(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime_metrics"
    runtime_dir.mkdir()

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
    )

    monkeypatch.setattr(
        harness_run,
        "_build_command",
        lambda *_args, **_kwargs: (["python", "-V"], runtime_dir, runtime_dir / ".detection_summary.json", query_csv),
    )

    def _fake_run_subprocess(_cmd: list[str], metrics) -> int:
        metrics.files = 20
        metrics.pages = 100
        metrics.ingest_secs = 10.0
        metrics.pages_per_sec_ingest = 10.0
        metrics.recall_metrics = {"recall@1": 0.5, "recall@5": 0.8}
        return 0

    monkeypatch.setattr(harness_run, "_run_subprocess_with_tty", _fake_run_subprocess)
    monkeypatch.setattr(harness_run, "last_commit", lambda: "abc123")
    monkeypatch.setattr(harness_run, "now_timestr", lambda: "20260305_000000_UTC")

    captured: dict[str, dict] = {}

    def _fake_write_json(_path: Path, payload: dict) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(harness_run, "write_json", _fake_write_json)

    harness_run._run_single(cfg, tmp_path, run_id="r1", tags=["nightly", "candidate"])
    assert captured["payload"]["tags"] == ["nightly", "candidate"]
    assert captured["payload"]["metrics"]["recall_1"] == 0.5
    assert captured["payload"]["metrics"]["recall_5"] == 0.8


def test_run_entry_session_artifact_dir_uses_run_name(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
    )
    monkeypatch.setattr(harness_run, "load_harness_config", lambda **_: cfg)

    def _fake_run_single(_cfg: HarnessConfig, _artifact_dir: Path, run_id: str, tags: list[str] | None = None) -> dict:
        assert tags == []
        return {
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "summary_metrics": {"pages": 0, "ingest_secs": 1.0, "pages_per_sec_ingest": 0.0, "recall_5": None},
        }

    monkeypatch.setattr(harness_run, "_run_single", _fake_run_single)

    result = harness_run._run_entry(
        run_name="jp20_single",
        config_file=None,
        session_dir=tmp_path,
        dataset="jp20",
        preset="single_gpu",
    )

    assert Path(result["artifact_dir"]).name == "jp20_single"


def test_run_entry_returns_tags(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
    )
    monkeypatch.setattr(harness_run, "load_harness_config", lambda **_: cfg)

    def _fake_run_single(_cfg: HarnessConfig, _artifact_dir: Path, run_id: str, tags: list[str] | None = None) -> dict:
        assert run_id == "jp20_single"
        assert tags == ["nightly", "candidate"]
        return {
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "summary_metrics": {"pages": 0, "ingest_secs": 1.0, "pages_per_sec_ingest": 0.0, "recall_5": None},
        }

    monkeypatch.setattr(harness_run, "_run_single", _fake_run_single)

    result = harness_run._run_entry(
        run_name="jp20_single",
        config_file=None,
        session_dir=tmp_path,
        dataset="jp20",
        preset="single_gpu",
        tags=["nightly", "candidate"],
    )

    assert result["tags"] == ["nightly", "candidate"]


def test_execute_runs_does_not_write_sweep_results_file(monkeypatch, tmp_path: Path) -> None:
    session_dir = tmp_path / "nightly_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(harness_run, "create_session_dir", lambda *_args, **_kwargs: session_dir)

    def _fake_run_entry(**_kwargs) -> dict:
        return {
            "run_name": "jp20_single",
            "dataset": "jp20",
            "preset": "single_gpu",
            "artifact_dir": str((session_dir / "jp20_single").resolve()),
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "metrics": {"files": 20, "pages": 3181},
        }

    monkeypatch.setattr(harness_run, "_run_entry", _fake_run_entry)

    harness_run.execute_runs(
        runs=[{"name": "jp20_single", "dataset": "jp20", "preset": "single_gpu"}],
        config_file=None,
        session_prefix="nightly",
        preset_override=None,
    )

    assert not (session_dir / "sweep_results.json").exists()


def test_sweep_command_uses_top_level_preset_from_runs_config(monkeypatch, tmp_path: Path) -> None:
    runs_path = tmp_path / "vidore_sweep.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "preset: dgx_8gpu",
                "runs:",
                "  - name: vidore_v3_hr_dgx_8gpu",
                "    dataset: vidore_v3_hr",
            ]
        ),
        encoding="utf-8",
    )
    session_dir = tmp_path / "sweep_session"
    session_dir.mkdir()
    summary_path = session_dir / "session_summary.json"

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        harness_run,
        "execute_runs",
        lambda **kwargs: (
            captured.update(kwargs)
            or (
                session_dir,
                [
                    {
                        "run_name": "vidore_v3_hr_dgx_8gpu",
                        "dataset": "vidore_v3_hr",
                        "preset": "dgx_8gpu",
                        "artifact_dir": str((session_dir / "vidore_v3_hr_dgx_8gpu").resolve()),
                        "success": True,
                        "return_code": 0,
                        "failure_reason": None,
                        "metrics": {"ndcg_10": 0.4, "recall_5": 0.3},
                    }
                ],
            )
        ),
    )
    monkeypatch.setattr(harness_run, "write_session_summary", lambda *_args, **_kwargs: summary_path)

    result = RUNNER.invoke(harness_app, ["sweep", "--runs-config", str(runs_path)])

    assert result.exit_code == 0
    assert captured["preset_override"] == "dgx_8gpu"


def test_sweep_command_dry_run_prints_resolved_top_level_preset(tmp_path: Path) -> None:
    runs_path = tmp_path / "vidore_sweep.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "preset: dgx_8gpu",
                "runs:",
                "  - name: vidore_v3_hr_dgx_8gpu",
                "    dataset: vidore_v3_hr",
            ]
        ),
        encoding="utf-8",
    )

    result = RUNNER.invoke(harness_app, ["sweep", "--runs-config", str(runs_path), "--dry-run"])

    assert result.exit_code == 0
    assert "preset=dgx_8gpu" in result.output


def test_collect_run_metadata_falls_back_without_gpu_or_ray(monkeypatch) -> None:
    def _raise_package_not_found(_name: str) -> str:
        raise harness_run.metadata.PackageNotFoundError()

    monkeypatch.setattr(harness_run.socket, "gethostname", lambda: "")
    monkeypatch.setattr(harness_run.metadata, "version", _raise_package_not_found)
    monkeypatch.setattr(harness_run, "_collect_gpu_metadata", lambda: (None, None))
    monkeypatch.setattr(harness_run.sys, "version_info", None)

    assert harness_run._collect_run_metadata() == {
        "host": "unknown",
        "gpu_count": None,
        "cuda_driver": None,
        "ray_version": "unknown",
        "python_version": "unknown",
    }


def test_run_single_writes_results_with_run_metadata(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run_artifacts"
    artifact_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    runtime_dir = artifact_dir / "runtime_metrics"
    runtime_dir.mkdir()
    detection_file = artifact_dir / "detection_summary.json"
    detection_file.write_text(json.dumps({"total_detections": 7}), encoding="utf-8")
    runtime_summary_file = runtime_dir / "jp20_single.runtime.summary.json"
    runtime_summary_file.write_text(json.dumps({"elapsed_secs": 12.5}), encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
        write_detection_file=True,
    )

    monkeypatch.setattr(
        harness_run,
        "_build_command",
        lambda _cfg, _artifact_dir, _run_id: (
            ["python", "-m", "nemo_retriever.examples.batch_pipeline", str(dataset_dir)],
            runtime_dir,
            detection_file,
            query_csv,
        ),
    )

    def _fake_run_subprocess(_cmd: list[str], metrics) -> int:
        metrics.files = None
        metrics.pages = None
        metrics.ingest_secs = 12.5
        metrics.pages_per_sec_ingest = None
        metrics.rows_processed = 3181
        metrics.rows_per_sec_ingest = 254.48
        metrics.recall_metrics = {"recall@5": 0.9}
        return 0

    monkeypatch.setattr(harness_run, "_run_subprocess_with_tty", _fake_run_subprocess)
    monkeypatch.setattr(harness_run, "now_timestr", lambda: "20260305_120000_UTC")
    monkeypatch.setattr(harness_run, "last_commit", lambda: "abc1234")
    monkeypatch.setattr(
        harness_run,
        "_collect_run_metadata",
        lambda: {
            "host": "builder-01",
            "gpu_count": 2,
            "cuda_driver": "550.54.15",
            "ray_version": "2.49.0",
            "python_version": "3.12.4",
        },
    )

    result = harness_run._run_single(cfg, artifact_dir, run_id="jp20_single")
    payload = json.loads((artifact_dir / "results.json").read_text(encoding="utf-8"))

    expected = {
        "timestamp": "20260305_120000_UTC",
        "latest_commit": "abc1234",
        "success": True,
        "return_code": 0,
        "failure_reason": None,
        "test_config": {
            "dataset_label": "jp20",
            "dataset_dir": str(dataset_dir),
            "preset": "single_gpu",
            "query_csv": str(query_csv),
            "effective_query_csv": str(query_csv),
            "input_type": cfg.input_type,
            "recall_required": cfg.recall_required,
            "recall_match_mode": cfg.recall_match_mode,
            "recall_adapter": cfg.recall_adapter,
            "audio_match_tolerance_secs": cfg.audio_match_tolerance_secs,
            "segment_audio": cfg.segment_audio,
            "audio_split_type": cfg.audio_split_type,
            "audio_split_interval": cfg.audio_split_interval,
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
            "write_detection_file": True,
            "use_heuristics": cfg.use_heuristics,
            "lancedb_uri": str((artifact_dir / "lancedb").resolve()),
            "tuning": {field: getattr(cfg, field) for field in sorted(harness_run.TUNING_FIELDS)},
        },
        "metrics": {
            "files": None,
            "pages": None,
            "ingest_secs": 12.5,
            "pages_per_sec_ingest": None,
            "rows_processed": 3181,
            "rows_per_sec_ingest": 254.48,
            "recall_5": 0.9,
        },
        "summary_metrics": {
            "pages": None,
            "ingest_secs": 12.5,
            "pages_per_sec_ingest": None,
            "recall_5": 0.9,
            "ndcg_10": None,
        },
        "run_metadata": {
            "host": "builder-01",
            "gpu_count": 2,
            "cuda_driver": "550.54.15",
            "ray_version": "2.49.0",
            "python_version": "3.12.4",
        },
        "runtime_summary": {"elapsed_secs": 12.5},
        "detection_summary": {"total_detections": 7},
        "artifacts": {
            "command_file": str((artifact_dir / "command.txt").resolve()),
            "runtime_metrics_dir": str(runtime_dir.resolve()),
            "detection_summary_file": str(detection_file.resolve()),
        },
    }

    assert result == expected
    assert payload == expected


def test_run_single_allows_missing_optional_summary_files(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run_artifacts"
    artifact_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("q,s,p\nx,y,1\n", encoding="utf-8")

    runtime_dir = artifact_dir / "runtime_metrics"
    runtime_dir.mkdir()
    detection_file = runtime_dir / ".detection_summary.json"

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        query_csv=str(query_csv),
        write_detection_file=False,
        recall_required=False,
    )

    monkeypatch.setattr(
        harness_run,
        "_build_command",
        lambda _cfg, _artifact_dir, _run_id: (
            ["python", "-m", "nemo_retriever.examples.batch_pipeline", str(dataset_dir)],
            runtime_dir,
            detection_file,
            query_csv,
        ),
    )

    def _fake_run_subprocess(_cmd: list[str], metrics) -> int:
        metrics.rows_processed = 42
        metrics.rows_per_sec_ingest = 3.5
        metrics.ingest_secs = 12.0
        return 0

    monkeypatch.setattr(harness_run, "_run_subprocess_with_tty", _fake_run_subprocess)
    monkeypatch.setattr(harness_run, "now_timestr", lambda: "20260306_210000_UTC")
    monkeypatch.setattr(harness_run, "last_commit", lambda: "abc1234")
    monkeypatch.setattr(harness_run, "_collect_run_metadata", lambda: {"host": "builder-01"})

    result = harness_run._run_single(cfg, artifact_dir, run_id="jp20_single")

    assert result["success"] is True
    assert result["runtime_summary"] is None
    assert result["detection_summary"] is None
    assert result["metrics"]["rows_processed"] == 42
    assert result["metrics"]["rows_per_sec_ingest"] == 3.5
    assert result["metrics"]["pages"] is None
    assert result["summary_metrics"] == {
        "pages": None,
        "ingest_secs": 12.0,
        "pages_per_sec_ingest": None,
        "recall_5": None,
        "ndcg_10": None,
    }
    assert "detection_summary_file" not in result["artifacts"]


def test_resolve_summary_metrics_falls_back_to_dataset_page_count(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="earnings",
        preset="single_gpu",
        recall_required=False,
    )

    pdf_a = dataset_dir / "a.pdf"
    pdf_b = dataset_dir / "nested" / "b.pdf"
    pdf_b.parent.mkdir()
    pdf_a.write_text("placeholder", encoding="utf-8")
    pdf_b.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(harness_run, "_safe_pdf_page_count", lambda path: 3 if path.name == "a.pdf" else 7)

    summary = harness_run._resolve_summary_metrics(
        cfg,
        {"pages": None, "ingest_secs": 5.0, "pages_per_sec_ingest": None, "recall_5": 0.75},
        runtime_summary=None,
    )

    assert summary == {
        "pages": 10,
        "ingest_secs": 5.0,
        "pages_per_sec_ingest": 2.0,
        "recall_5": 0.75,
        "ndcg_10": None,
    }


def test_cli_run_accepts_repeated_tags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_entry(**kwargs) -> dict:
        captured.update(kwargs)
        return {
            "run_name": "jp20",
            "dataset": "jp20",
            "preset": "single_gpu",
            "artifact_dir": "/tmp/jp20",
            "success": True,
            "return_code": 0,
            "failure_reason": None,
            "metrics": {"files": 20, "pages": 100},
            "tags": ["nightly", "candidate"],
        }

    monkeypatch.setattr(harness_run, "_run_entry", _fake_run_entry)

    result = RUNNER.invoke(
        harness_app,
        ["run", "--dataset", "jp20", "--tag", "nightly", "--tag", "candidate"],
    )

    assert result.exit_code == 0
    assert captured["tags"] == ["nightly", "candidate"]
