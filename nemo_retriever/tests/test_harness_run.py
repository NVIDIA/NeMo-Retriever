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
    assert "--pdf-extract-workers" not in cmd
    assert "--pdf-extract-num-cpus" not in cmd
    assert "--page-elements-workers" not in cmd
    assert "--ocr-workers" not in cmd
    assert "--embed-workers" not in cmd
    assert "--gpu-page-elements" not in cmd
    assert "--gpu-ocr" not in cmd
    assert "--gpu-embed" not in cmd
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


def test_build_command_supports_inprocess_run_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        run_mode="inprocess",
        query_csv=str(query_csv),
        max_workers=8,
        gpu_devices="0,1",
    )
    cmd, _runtime_dir, _detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")

    assert "nemo_retriever.examples.inprocess_pipeline" in cmd
    assert "--max-workers" in cmd
    assert cmd[cmd.index("--max-workers") + 1] == "8"
    assert "--gpu-devices" in cmd
    assert cmd[cmd.index("--gpu-devices") + 1] == "0,1"
    assert "--query-csv" in cmd
    assert str(effective_query_csv) in cmd


def test_build_command_supports_fused_run_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")

    cfg = HarnessConfig(
        dataset_dir=str(dataset_dir),
        dataset_label="jp20",
        preset="single_gpu",
        run_mode="fused",
        query_csv=str(query_csv),
        fused_workers=2,
        fused_batch_size=32,
        fused_cpus_per_actor=2.0,
        fused_gpus_per_actor=1.0,
    )
    cmd, runtime_dir, detection_file, effective_query_csv = _build_command(cfg, tmp_path, run_id="r1")

    assert "nemo_retriever.examples.fused_pipeline" in cmd
    assert "--fused-workers" in cmd
    assert cmd[cmd.index("--fused-workers") + 1] == "2"
    assert "--fused-batch-size" in cmd
    assert cmd[cmd.index("--fused-batch-size") + 1] == "32"
    assert "--runtime-metrics-dir" in cmd
    assert str(runtime_dir) in cmd
    assert "--detection-summary-file" in cmd
    assert str(detection_file) in cmd
    assert "--query-csv" in cmd
    assert str(effective_query_csv) in cmd


def test_normalize_recall_metric_key_removes_duplicate_prefix() -> None:
    assert _normalize_recall_metric_key("recall@1") == "recall_1"
    assert _normalize_recall_metric_key("recall@10") == "recall_10"


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
