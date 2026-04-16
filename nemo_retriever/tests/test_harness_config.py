from pathlib import Path

import pytest

import nemo_retriever.harness.config as harness_config
from nemo_retriever.harness.config import load_harness_config, load_nightly_config, load_runs_config


def _write_harness_config(path: Path, dataset_dir: Path, query_csv: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "  query_csv: null",
                "  input_type: pdf",
                "  recall_required: false",
                "presets:",
                "  base:",
                "    pdf_extract_workers: 4",
                "    page_elements_workers: 2",
                "    page_elements_batch_size: 8",
                "    ocr_workers: 2",
                "    embed_workers: 2",
                "    gpu_page_elements: 0.1",
                "    gpu_ocr: 0.2",
                "    gpu_embed: 0.3",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    input_type: pdf",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )


def test_load_harness_config_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,source,page\nq,a,1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    _write_harness_config(cfg_path, dataset_dir, query_csv)

    monkeypatch.setenv("HARNESS_PRESET", "base")
    monkeypatch.setenv("HARNESS_GPU_EMBED", "0.9")

    cfg = load_harness_config(
        config_file=str(cfg_path),
        dataset="tiny",
        preset="base",
        sweep_overrides={"gpu_ocr": 0.7},
        cli_overrides=["gpu_page_elements=0.6"],
    )

    assert cfg.dataset_dir == str(dataset_dir.resolve())
    assert cfg.query_csv == str(query_csv.resolve())
    assert cfg.gpu_page_elements == 0.6  # CLI override
    assert cfg.gpu_ocr == 0.7  # sweep override
    assert cfg.gpu_embed == 0.9  # env override (highest)
    assert cfg.recall_required is True


def test_load_harness_config_supports_run_mode_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,source,page\nq,a,1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    _write_harness_config(cfg_path, dataset_dir, query_csv)

    monkeypatch.setenv("HARNESS_RUN_MODE", "inprocess")

    cfg = load_harness_config(
        config_file=str(cfg_path),
        dataset="tiny",
        preset="base",
    )
    assert cfg.run_mode == "inprocess"


def test_load_harness_config_rejects_invalid_run_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                f"  dataset_dir: {dataset_dir}",
                "  run_mode: invalid",
                "  preset: base",
                "  recall_required: false",
                "presets:",
                "  base: {}",
                "datasets: {}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_mode must be one of"):
        load_harness_config(config_file=str(cfg_path))


def test_load_harness_config_fails_when_recall_required_without_query(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                f"  dataset_dir: {dataset_dir}",
                "  preset: base",
                "  recall_required: true",
                "presets:",
                "  base: {}",
                "datasets: {}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recall_required=true requires query_csv"):
        load_harness_config(config_file=str(cfg_path))


def test_load_runs_config_parses_runs_list(tmp_path: Path) -> None:
    runs_path = tmp_path / "nightly.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "runs:",
                "  - name: r1",
                "    dataset: bo20",
                "    preset: single_gpu",
                "    overrides:",
                "      gpu_embed: 0.25",
                "  - name: r2",
                "    dataset: bo767",
            ]
        ),
        encoding="utf-8",
    )
    runs = load_runs_config(str(runs_path))
    assert len(runs) == 2
    assert runs[0]["name"] == "r1"
    assert runs[0]["overrides"]["gpu_embed"] == 0.25


def test_load_nightly_config_parses_slack_defaults(tmp_path: Path) -> None:
    runs_path = tmp_path / "nightly.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "preset: dgx_8gpu",
                "runs:",
                "  - name: r1",
                "    dataset: bo20",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_nightly_config(str(runs_path))

    assert cfg["runs"][0]["name"] == "r1"
    assert cfg["preset"] == "dgx_8gpu"
    assert cfg["slack"]["enabled"] is True
    assert cfg["slack"]["title"] == "nemo_retriever Nightly Harness"
    assert cfg["slack"]["post_artifact_paths"] is True
    assert "recall_5" in cfg["slack"]["metric_keys"]


def test_load_nightly_config_rejects_invalid_metric_keys(tmp_path: Path) -> None:
    runs_path = tmp_path / "nightly.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "runs:",
                "  - dataset: bo20",
                "slack:",
                "  metric_keys: invalid",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="slack.metric_keys"):
        load_nightly_config(str(runs_path))


def test_load_harness_config_supports_recall_adapter_and_match_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf,page\nq,doc,0\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    recall_adapter: page_plus_one",
                "    recall_match_mode: pdf_page",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.recall_adapter == "page_plus_one"
    assert cfg.recall_match_mode == "pdf_page"


def test_load_harness_config_supports_audio_recall_fields(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text(
        "query,expected_media_id,expected_start_time,expected_end_time\nq,clip,1.5,3.5\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny_audio",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny_audio:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    input_type: audio",
                "    segment_audio: true",
                "    audio_split_type: time",
                "    audio_split_interval: 30",
                "    recall_required: true",
                "    recall_adapter: none",
                "    recall_match_mode: audio_segment",
                "    audio_match_tolerance_secs: 3.5",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.input_type == "audio"
    assert cfg.segment_audio is True
    assert cfg.audio_split_type == "time"
    assert cfg.audio_split_interval == 30
    assert cfg.recall_adapter == "none"
    assert cfg.recall_match_mode == "audio_segment"
    assert cfg.audio_match_tolerance_secs == 3.5


def test_load_harness_config_supports_multimodal_embedding_options(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    embed_modality: text",
                "    embed_granularity: element",
                "    extract_page_as_image: false",
                "    extract_infographics: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.embed_modality == "text"
    assert cfg.embed_granularity == "element"
    assert cfg.extract_page_as_image is False
    assert cfg.extract_infographics is True


def test_load_harness_config_supports_beir_mode_without_recall_fields(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                "    evaluation_mode: beir",
                "    beir_loader: vidore_hf",
                "    beir_dataset_name: vidore_v3_computer_science",
                "    recall_required: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.evaluation_mode == "beir"
    assert cfg.beir_loader == "vidore_hf"
    assert cfg.beir_dataset_name == "vidore_v3_computer_science"
    assert cfg.query_csv is None


def test_load_harness_config_defaults_beir_dataset_name_from_dataset_label(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: vidore_v3_computer_science",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  vidore_v3_computer_science:",
                f"    path: {dataset_dir}",
                "    evaluation_mode: beir",
                "    beir_loader: vidore_hf",
                "    recall_required: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.beir_dataset_name == "vidore_v3_computer_science"


def test_load_harness_config_resolves_relative_query_csv_from_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                "    query_csv: query.csv",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )

    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.query_csv == str(query_csv.resolve())


def test_load_harness_config_falls_back_to_repo_root_for_query_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo_root"
    repo_root.mkdir()
    query_csv = repo_root / "data" / "financebench_train.json"
    query_csv.parent.mkdir()
    query_csv.write_text("[]", encoding="utf-8")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                "    query_csv: data/financebench_train.json",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(harness_config, "REPO_ROOT", repo_root)

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.query_csv == str(query_csv.resolve())


def test_resolve_query_csv_path_prefers_repo_root_for_default_harness_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo_root"
    repo_root.mkdir()
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(harness_config, "REPO_ROOT", repo_root)
    monkeypatch.setattr(harness_config, "DEFAULT_TEST_CONFIG_PATH", cfg_path)

    resolved = harness_config._resolve_query_csv_path("data/missing.csv", config_path=cfg_path)

    assert resolved == str((repo_root / "data" / "missing.csv").resolve())


def test_load_harness_config_rejects_invalid_recall_adapter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    recall_adapter: unknown_adapter",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recall_adapter must be one of"):
        load_harness_config(config_file=str(cfg_path))


def test_load_harness_config_rejects_invalid_beir_loader(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                "    evaluation_mode: beir",
                "    beir_loader: nope",
                "    recall_required: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="beir_loader must be one of"):
        load_harness_config(config_file=str(cfg_path))


def test_load_harness_config_rejects_invalid_embed_modality(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    embed_modality: invalid",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embed_modality must be one of"):
        load_harness_config(config_file=str(cfg_path))


def test_load_harness_config_rejects_image_text_alias(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    embed_modality: image_text",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embed_modality must be one of"):
        load_harness_config(config_file=str(cfg_path))


def test_load_harness_config_rejects_removed_image_elements_modality_key(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    image_elements_modality: text_image",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="image_elements_modality is no longer supported by the harness"):
        load_harness_config(config_file=str(cfg_path))


def test_load_harness_config_uses_financebench_repo_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    real_exists = Path.exists
    expected_query_csv = (harness_config.REPO_ROOT / "data" / "financebench_train.json").resolve()

    def _fake_exists(path_self: Path) -> bool:
        if path_self == Path("/datasets/nv-ingest/financebench"):
            return False
        if path_self == Path("/raid/tester/financebench"):
            return True
        if path_self == expected_query_csv:
            return True
        return real_exists(path_self)

    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(harness_config.Path, "exists", _fake_exists)

    cfg = load_harness_config(dataset="financebench", preset="single_gpu")
    assert cfg.dataset_dir == str(Path("/raid/tester/financebench").resolve())
    assert cfg.query_csv == str(expected_query_csv)
    assert cfg.recall_required is True


def test_load_harness_config_supports_bo767_beir_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    real_exists = Path.exists
    expected_dataset_dir = Path("/raid/cjarrett/nv-ingest/data/bo767").resolve()
    expected_query_csv = (harness_config.REPO_ROOT / "data" / "bo767_annotations.csv").resolve()

    def _fake_exists(path_self: Path) -> bool:
        if path_self == expected_dataset_dir:
            return True
        if path_self == expected_query_csv:
            return True
        return real_exists(path_self)

    monkeypatch.setattr(harness_config.Path, "exists", _fake_exists)

    cfg = load_harness_config(dataset="bo767", preset="single_gpu")

    assert cfg.dataset_dir == str(expected_dataset_dir)
    assert cfg.query_csv == str(expected_query_csv)
    assert cfg.recall_required is False
    assert cfg.evaluation_mode == "beir"
    assert cfg.beir_loader == "bo767_csv"
    assert cfg.beir_doc_id_field == "pdf_page"


def test_load_harness_config_supports_bo10k_beir_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    real_exists = Path.exists
    expected_dataset_dir = Path("/datasets/nv-ingest/bo10k").resolve()
    expected_query_csv = (harness_config.REPO_ROOT / "data" / "digital_corpora_10k_annotations.csv").resolve()

    def _fake_exists(path_self: Path) -> bool:
        if path_self == expected_dataset_dir:
            return True
        if path_self == expected_query_csv:
            return True
        return real_exists(path_self)

    monkeypatch.setattr(harness_config.Path, "exists", _fake_exists)

    cfg = load_harness_config(dataset="bo10k", preset="single_gpu")

    assert cfg.dataset_dir == str(expected_dataset_dir)
    assert cfg.query_csv == str(expected_query_csv)
    assert cfg.recall_required is False
    assert cfg.evaluation_mode == "beir"
    assert cfg.beir_loader == "bo10k_csv"
    assert cfg.beir_doc_id_field == "pdf_page"


def test_load_harness_config_supports_earnings_beir_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    real_exists = Path.exists
    expected_dataset_dir = Path("/datasets/nv-ingest/earnings_consulting").resolve()
    expected_query_csv = (harness_config.REPO_ROOT / "data" / "earnings_consulting_multimodal.csv").resolve()

    def _fake_exists(path_self: Path) -> bool:
        if path_self == expected_dataset_dir:
            return True
        if path_self == expected_query_csv:
            return True
        return real_exists(path_self)

    monkeypatch.setattr(harness_config.Path, "exists", _fake_exists)

    cfg = load_harness_config(dataset="earnings", preset="single_gpu")

    assert cfg.dataset_dir == str(expected_dataset_dir)
    assert cfg.query_csv == str(expected_query_csv)
    assert cfg.recall_required is False
    assert cfg.evaluation_mode == "beir"
    assert cfg.beir_loader == "earnings_csv"
    assert cfg.beir_doc_id_field == "pdf_page"


def test_load_harness_config_supports_store_options(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    store_images_uri: stored_images",
                "    store_text: true",
                "    strip_base64: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.store_images_uri == "stored_images"
    assert cfg.store_text is True
    assert cfg.strip_base64 is False
